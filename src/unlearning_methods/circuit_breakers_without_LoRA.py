import gc
import logging
from copy import deepcopy

import torch as pt
from transformers import AutoModelForCausalLM

from utils.training import eval_


def compute_loss(
    percent_done,
    model,
    frozen_model,
    forget_inputs,
    retain_inputs,
    target_layers,
    config,
    retaining_rate,
    unlearning_rate,
):
    # === retain ===
    retain_attention_mask = pt.ones_like(retain_inputs)
    # ==== cb ====
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
    retain_coeff = retaining_rate * (percent_done / 2)
    forget_coeff = unlearning_rate * (1 - percent_done / 2)

    # ===== loss components =====
    layers_forget_attention_mask = forget_attention_mask.repeat(
        len(target_layers), 1, 1
    ).unsqueeze(-1)

    frozen_model.eval()
    with pt.no_grad():
        # Retain control
        if retain_coeff > 0:
            orig_retain_outputs = frozen_model(**retain_inputs).hidden_states
            orig_retain_hidden = pt.stack(orig_retain_outputs).detach()
            layers_retain_attention_mask = retain_attention_mask.repeat(
                len(orig_retain_outputs), 1, 1
            ).unsqueeze(-1)
            orig_retain_hidden *= layers_retain_attention_mask

            del orig_retain_outputs
            gc.collect()

        # Circuit Breaker control
        if forget_coeff > 0:
            forget_outputs = frozen_model(**forget_inputs).hidden_states
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
        lora_forget_hidden = pt.stack([lora_forget_outputs[l] for l in target_layers])

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


def circuit_breaker_without_lora(
    h, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # Create main model and frozen copy
    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    frozen_model = deepcopy(model)
    frozen_model.eval()

    num_layers = model.config.num_hidden_layers
    target_layers = [num_layers // 2]

    optimizer = pt.optim.Adam(model.parameters(), lr=1)

    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    passes_per_loop = 5
    for loop_num in range(config.unlearn_steps // passes_per_loop):
        r_input_ids = next(retain_iter)
        f_input_ids = next(forget_iter)

        model.zero_grad()
        loss = compute_loss(
            loop_num / (config.unlearn_steps // passes_per_loop),
            model,
            frozen_model,
            f_input_ids,
            r_input_ids,
            target_layers,
            config,
            h.retaining_rate,
            h.unlearning_rate,
        )

        loss.backward()
        optimizer.step()

        # ! eval current loss
        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done % 60 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, _passes_done)

    return model
