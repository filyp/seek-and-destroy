import logging

import torch as pt
from transformers import AutoModelForCausalLM

from utils.circuit_creation import filter_and_normalize_circuit, get_circuit
from utils.model_operations import get_thresh
from utils.plots_and_stats import visualize_param
from utils.training import cross_entropy_loss, eval_, loss_fns

disruption_score_warmup = 20


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # ! parameters
    f_quantile = 1  # trial.suggest_float("f_quantile", 0.5, 1, log=True)
    r_quantile = trial.suggest_float("r_quantile", 0.1, 0.5, log=True)
    # retaining_rate = trial.suggest_float("retaining_rate", 0.0003, 0.0010, log=True)
    retaining_rate = 0.0005
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.0003, 0.0010, log=True)
    disruption_score_decay = 0.9
    pos_grad_discard = 0  # trial.suggest_float("pos_grad_discard", 0, 1)
    # retain_consistency = trial.suggest_float("retain_consistency", 0, 1)
    circ_crossfade = 0.8  # trial.suggest_float("circ_crossfade", 0.5, 1)
    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False
    target_modules = config.target_modules

    # use several circuits, mixed together
    circuits = [
        filter_and_normalize_circuit(
            get_circuit(config, forget_batches, circuit_name),
            target_modules,
        )
        for circuit_name in config.circuit_names
    ]
    assert len(circuits) <= 2  # for now
    # get params to intervene on and initialize disruption scores
    interven_params = []
    for name, p in model.named_parameters():
        if any(f"{m}.weight" in name for m in target_modules):
            interven_params.append(p)
            p.disruption_score = pt.zeros_like(p.data)
            p.to_forget = circuits[0][name] * circ_crossfade
            if len(circuits) > 1 and name in circuits[1]:
                p.to_forget += circuits[1][name] * (1 - circ_crossfade)
            p.param_name = name
    del circuits

    # Get threshold for forgetting
    f_threshold = get_thresh(f_quantile, [p.to_forget.abs() for p in interven_params])

    # Require grad for all intervene params
    for param in model.parameters():
        param.requires_grad = False
    for param in interven_params:
        param.requires_grad = True

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
            grad[p.to_forget.sign() == p.grad.sign()] *= pos_grad_discard
            p.disruption_score *= disruption_score_decay
            p.disruption_score += grad

        # Skip during warmup
        if step <= disruption_score_warmup:
            continue

        # Unlearning step with two-stage masking
        for p in interven_params:
            # First choose the most important weights for forgetting
            mask = p.to_forget.abs() > f_threshold
            # Then from them, choose the ones least disrupting
            flipped_disr = p.disruption_score * p.to_forget.sign()
            if mask.any():
                d_threshold = get_thresh(r_quantile, [flipped_disr[mask]])
                flipped_disr[~mask] = float("-inf")
                mask = mask & (flipped_disr > d_threshold)

            # ! unlearn
            p.data -= mask * unlearning_rate * p.to_forget

            # ! retain
            # p.grad[p.grad.sign() != p.to_forget.sign()] *= retain_consistency
            p.data -= retaining_rate * p.grad

            # if step == config.unlearn_steps:
            #     visualize_param(p, mask, p.param_name)

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    return model
