# to simplify, I cut out the d_threshold part
import logging

import torch as pt
from transformers import AutoModelForCausalLM

from unlearning_methods.seek_and_destroy import get_normal_circuit
from utils.model_operations import get_thresh
from utils.training import cross_entropy_loss, eval_

disruption_score_warmup = 20


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # ! parameters
    f_quantile = trial.suggest_float("f_quantile", 0.05, 1, log=True)
    retaining_rate = trial.suggest_float("retaining_rate", 0.00003, 0.001, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.00003, 0.0007, log=True)
    retain_consistency = trial.suggest_float("retain_consistency", 0, 1)
    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False

    if "pythia" in config.model_id:
        target_modules = ["dense_h_to_4h", "dense_4h_to_h", "dense"]
    else:
        raise NotImplementedError(f"Model {config.model_id} not supported")

    # get params to intervene on and initialize disruption scores
    for param in model.parameters():
        param.requires_grad = False
    circuit = get_normal_circuit(config.model_id, config.forget_set_name, forget_batches)
    interven_params = []
    for name, p in model.named_parameters():
        if any(f"{m}.weight" in name for m in target_modules):
            interven_params.append(p)
            p.to_forget = circuit[name]
            p.requires_grad = True
    del circuit

    # Get threshold for forgetting
    f_threshold = get_thresh(f_quantile, [p.to_forget.abs() for p in interven_params])

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    for step in range(1, 1 + config.unlearn_steps):
        model.train()
        r_input_ids = next(retain_iter)

        model.zero_grad(set_to_none=True)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()

        # Skip during warmup
        if step <= disruption_score_warmup:
            continue

        for p in interven_params:
            # ! unlearn
            mask = p.to_forget.abs() > f_threshold
            p.data -= mask * unlearning_rate * p.to_forget

            # ! retain
            p.grad[p.grad.sign() != p.to_forget.sign()] *= retain_consistency
            p.data -= retaining_rate * p.grad

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    return model
