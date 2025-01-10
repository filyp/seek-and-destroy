import logging
from peft import LoraConfig, get_peft_model
import torch as pt
from transformers import AutoModelForCausalLM
from torch.nn.functional import cosine_similarity
import gc
from utils.training import eval_


# B - number of examples in a batch
# T - number of tokens in a batch
# Algorithm
# for t in total_training_steps:
#   r_input_ids, f_input_ids // sample batches {input_ids:  (B, T), attention_mask: (B, T)}
#
#
#
#
#


def compute_loss(step, model, forget_inputs, retain_inputs, target_layers, alpha, config):

    # === retain ===
    retain_input_ids = retain_inputs.get(f"input_ids")
    retain_attention_mask = retain_inputs.get(f"attention_mask")
    # ==== cb ====
    circuit_breaker_input_ids = forget_inputs.get(f"input_ids")
    circuit_breaker_attention_mask = forget_inputs.get(
        f"attention_mask")

    # ==== Forward Inputs ====
    module = 'hidden_states'
    retain_inputs = dict(input_ids=retain_input_ids,
                         attention_mask=retain_attention_mask, output_hidden_states=True)
    cb_inputs = dict(input_ids=circuit_breaker_input_ids,
                     attention_mask=circuit_breaker_attention_mask, output_hidden_states=True)

    # Those are pretty much arbitrary, the important thing is that retain_coeff increases as the training progresses and circuit_breaker_coeff decreases.

    retain_coeff = alpha * (step / (2 * 1 + config.unlearn_steps))
    circuit_breaker_coeff = alpha * \
        (1 - (step / (2 * 1 + config.unlearn_steps)))

    print(
        f"retain_coeff: {retain_coeff:.4f} || circuit_breaker_coeff: {circuit_breaker_coeff:.4f}")

    # ===== loss components =====
    layers_circuit_breaker_attention_mask = circuit_breaker_attention_mask.repeat(
        len(target_layers), 1, 1).unsqueeze(-1)

    with model.disable_adapter():
        model.eval()
        with pt.no_grad():
            # Retain control
            if retain_coeff > 0:
                orig_retain_outputs = model(**retain_inputs)[module]
                orig_retain_hidden = pt.stack(orig_retain_outputs).detach()
                layers_retain_attention_mask = retain_attention_mask.repeat(
                    len(orig_retain_outputs), 1, 1).unsqueeze(-1)
                orig_retain_hidden *= layers_retain_attention_mask

                del orig_retain_outputs
                gc.collect()

            # Circuit Breaker control
            if circuit_breaker_coeff > 0:
                circuit_breaker_outputs = model(**cb_inputs)[module]
                circuit_breaker_hidden = pt.stack(
                    [circuit_breaker_outputs[l].detach() for l in target_layers])

                del circuit_breaker_outputs
                gc.collect()

    model.train()

    # Retain control
    if retain_coeff > 0:
        lora_retain_outputs = model(**retain_inputs)[module]
        lora_retain_hidden = pt.stack(
            lora_retain_outputs) * layers_retain_attention_mask
        retain_loss = pt.norm(
            lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=pt.float).nanmean()

    # Circuit Breaker control
    if circuit_breaker_coeff > 0:
        lora_circuit_breaker_outputs = model(**cb_inputs)[module]
        lora_circuit_breaker_hidden = pt.stack(
            [lora_circuit_breaker_outputs[l] for l in target_layers])

        normalized_lora_circuit_breaker_outputs = lora_circuit_breaker_hidden / \
            (pt.norm(lora_circuit_breaker_hidden,
             dim=-1, keepdim=True, dtype=pt.float))
        normalized_circuit_breaker_outputs = circuit_breaker_hidden / \
            (pt.norm(circuit_breaker_hidden, dim=-1, keepdim=True, dtype=pt.float))
        inner_product = (normalized_lora_circuit_breaker_outputs *
                         normalized_circuit_breaker_outputs) * layers_circuit_breaker_attention_mask
        circuit_breaker_loss = pt.relu(inner_product.sum(
            dim=-1)).sum() / layers_circuit_breaker_attention_mask.sum()

    loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss

    return loss


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):

    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    num_layers = model.config.num_hidden_layers

    # ! parameters
    ret_lora_rank = trial.suggest_int("ret_lora_rank", 4, 16)
    alpha = trial.suggest_float("alpha", 0.1, 2.0)

    # Choose between training on 1/2 or 1/3 of middle layers
    use_half = trial.suggest_categorical(
        "use_half", [True, False])  # True = 1/2, False = 1/3

    if use_half:
        num_target_layers = num_layers // 2
    else:
        num_target_layers = num_layers // 3

    # Calculate start layer to center the target layers in the middle
    start_layer = (num_layers - num_target_layers) // 2
    target_layers = list(range(start_layer, start_layer + num_target_layers))

    # Add LoRA

    ret_lora_config = dict(lora_dropout=0.05, target_modules="all-linear")

    ret_lora_c = LoraConfig(r=ret_lora_rank, **ret_lora_config)
    model = get_peft_model(
        model, ret_lora_c, adapter_name="ret_lora", mixed=True)

    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    for step in range(1, 1 + config.unlearn_steps):
        r_input_ids = next(retain_iter)
        f_input_ids = next(forget_iter)

        loss = compute_loss(step, model, f_input_ids,
                            r_input_ids, target_layers, alpha, config)

        loss.backward()

        # Evaluation
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    return model
