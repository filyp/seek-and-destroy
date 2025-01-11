import logging

import torch as pt
from peft import LoraConfig, get_peft_model
from seek_and_destroy import get_normal_circuit
from transformers import AutoModelForCausalLM

from utils.git_and_reproducibility import *
from utils.model_operations import *
from utils.training import *

# Add LoRA config
ret_lora_config = dict(lora_dropout=0.05, target_modules="all-linear")
use_ret_lora = False
disruption_score_warmup = 20

# %%


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # ! parameters
    f_quantile = trial.suggest_float("f_quantile", 0.05, 0.5, log=True)
    r_quantile = trial.suggest_float("r_quantile", 0.05, 0.2, log=True)
    retaining_rate = trial.suggest_float("retaining_rate", 0.00001, 0.001, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.0001, 0.003, log=True)
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.9, 1.0)
    ret_lora_rank = 8
    pos_grad_discard_factor = 0
    retain_consistency = 0
    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    if "pythia" in config.model_id:
        target_modules = ["dense_h_to_4h", "dense_4h_to_h", "dense"]
    else:
        raise NotImplementedError(f"Model {config.model_id} not supported")

    # get params to intervene on and initialize disruption scores
    circuit = get_normal_circuit(config.model_id, config.forget_set_name)
    interven_params = []
    for name, p in model.named_parameters():
        if any(f"{m}.weight" in name for m in target_modules):
            interven_params.append(p)
            p.disruption_score = pt.zeros_like(p.data)
            p.to_forget = circuit[name]

    # Get threshold for forgetting
    f_threshold = get_thresh(f_quantile, [p.to_forget.abs() for p in interven_params])

    # Add LoRA
    ret_lora_c = LoraConfig(r=ret_lora_rank, **ret_lora_config)
    peft_model = get_peft_model(model, ret_lora_c, adapter_name="ret_lora", mixed=True)
    model = peft_model.model
    # Require grad for all params despite having lora
    if use_ret_lora:
        for param in interven_params:
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    # Initialize optimizers
    _ret_lora_params = [p for n, p in model.named_parameters() if ".ret_lora." in n]
    ret_optimizer = pt.optim.SGD(_ret_lora_params, lr=retaining_rate)

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    for step in range(1, 1 + config.unlearn_steps):
        model.train()
        r_input_ids = next(retain_iter)

        # ! unlearn on the base model
        model.zero_grad(set_to_none=True)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()

        for p in interven_params:
            grad = p.grad.clone().detach()
            grad[p.to_forget.sign() == p.grad.sign()] *= pos_grad_discard_factor
            p.disruption_score *= disruption_score_decay
            p.disruption_score += grad

        # Skip during warmup
        if step <= disruption_score_warmup:
            continue

        # Unlearning step with two-stage masking
        flipped_scores = []
        for p in interven_params:
            # First choose the most important weights for forgetting
            mask = p.to_forget.abs() > f_threshold
            # Then from them, choose the ones least disrupting
            flipped_disr = p.disruption_score * p.to_forget.sign()
            flipped_scores.append(flipped_disr[mask])
        d_threshold = get_thresh(r_quantile, flipped_scores)

        for p in interven_params:
            # recompute these two
            mask = p.to_forget.abs() > f_threshold
            flipped_disr = p.disruption_score * p.to_forget.sign()

            flipped_disr[~mask] = float("-inf")
            mask = mask & (flipped_disr > d_threshold)

            p.data -= mask * unlearning_rate * p.to_forget

            if not use_ret_lora:
                p.grad[p.grad.sign() != p.to_forget.sign()] *= retain_consistency
                p.data -= retaining_rate * p.grad

        # LoRA retention step
        if use_ret_lora:
            ret_optimizer.step()

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    visualize_param(p, mask)

    # Merge and unload helper lora
    peft_model_copy = deepcopy(peft_model)
    # peft_model_copy.set_adapter(["ret_lora"])
    model_copy = peft_model_copy.merge_and_unload()
    del model_copy.peft_config
    return model_copy
