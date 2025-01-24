import gc
import logging

import torch as pt
from peft import LoraConfig, get_peft_model
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM

from utils.training import eval_


def compute_loss(
    step, model, forget_inputs, retain_inputs, target_layers, alpha, config
):

    # # === retain ===
    retain_attention_mask = pt.ones_like(retain_inputs)
    # # ==== cb ====
    forget_attention_mask = pt.ones_like(forget_inputs)

    assert forget_attention_mask.shape == retain_attention_mask.shape

    # ==== Forward Inputs ====
    retain_inputs = dict(
        input_ids=retain_inputs,
        attention_mask=retain_attention_mask,
        output_hidden_states=True,
    )
    forget_inputs = dict(
        input_ids=forget_inputs,
        attention_mask=forget_attention_mask,
        output_hidden_states=True,
    )

    # Those are pretty much arbitrary, the important thing is that retain_coeff increases as the training progresses and forget_coeff decreases.

    retain_coeff = alpha * (step / (2 * config.unlearn_steps))
    forget_coeff = alpha * (1 - (step / (2 * config.unlearn_steps)))

    # ===== loss components =====
    layers_forget_attention_mask = forget_attention_mask.repeat(
        len(target_layers), 1, 1
    ).unsqueeze(-1)

    with model.disable_adapter():
        model.eval()
        with pt.no_grad():
            # Retain control
            if retain_coeff > 0:
                orig_retain_outputs = model(**retain_inputs).hidden_states
                orig_retain_hidden = pt.stack(orig_retain_outputs).detach()
                layers_retain_attention_mask = retain_attention_mask.repeat(
                    len(orig_retain_outputs), 1, 1
                ).unsqueeze(-1)
                orig_retain_hidden *= layers_retain_attention_mask

                del orig_retain_outputs
                gc.collect()

            # Circuit Breaker control
            if forget_coeff > 0:
                forget_outputs = model(**forget_inputs).hidden_states
                forget_hidden = pt.stack(
                    [forget_outputs[l].detach() for l in target_layers]
                )

                del forget_outputs
                gc.collect()

    model.train()

    # Retain control
    if retain_coeff > 0:
        lora_retain_outputs = model(**retain_inputs).hidden_states
        lora_retain_hidden = (
            pt.stack(lora_retain_outputs) * layers_retain_attention_mask
        )
        retain_loss = pt.norm(
            lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=pt.float
        ).nanmean()

    # Circuit Breaker control
    if forget_coeff > 0:
        lora_forget_outputs = model(**forget_inputs).hidden_states
        lora_forget_hidden = pt.stack(
            [lora_forget_outputs[l] for l in target_layers])

        normalized_lora_forget_outputs = lora_forget_hidden / (
            pt.norm(lora_forget_hidden, dim=-1, keepdim=True, dtype=pt.float)
        )
        normalized_forget_outputs = forget_hidden / (
            pt.norm(forget_hidden, dim=-1, keepdim=True, dtype=pt.float)
        )
        inner_product = (
            normalized_lora_forget_outputs * normalized_forget_outputs
        ) * layers_forget_attention_mask
        forget_loss = (
            pt.relu(inner_product.sum(dim=-1)).sum()
            / layers_forget_attention_mask.sum()
        )

    loss = retain_coeff * retain_loss + forget_coeff * forget_loss

    return loss


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):

    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    # ! parameters
    ret_lora_rank = trial.suggest_int("ret_lora_rank", 4, 16)
    alpha = trial.suggest_float("alpha", 0.1, 2.0)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2)

    num_layers = model.config.num_hidden_layers
    target_layers = [num_layers // 2]

    # Add LoRA
    ret_lora_config = dict(lora_dropout=0.05, target_modules="all-linear")

    ret_lora_c = LoraConfig(r=ret_lora_rank, **ret_lora_config)
    model = get_peft_model(
        model, ret_lora_c, adapter_name="ret_lora", mixed=True)

    optimizer = pt.optim.SGD(model.parameters(), lr=lr)

    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    for step in range(1, 1 + config.unlearn_steps):
        r_input_ids = next(retain_iter)
        f_input_ids = next(forget_iter)

        model.zero_grad(set_to_none=True)
        loss = compute_loss(
            step, model, f_input_ids, r_input_ids, target_layers, alpha, config
        )

        loss.backward()
        # assert lora is enabled
        for name, param in model.named_parameters():
            assert (param.grad is not None) == ("lora" in name), name
        optimizer.step()

        # Evaluation
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    return model
