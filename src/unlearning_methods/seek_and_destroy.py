import logging

import torch as pt
from transformers import AutoModelForCausalLM

from utils.circuit_creation import filter_and_normalize_circuit, get_circuit
from utils.training import *

disruption_score_warmup = 20


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # ! parameters
    # fmt: off
    retaining_rate = trial.suggest_float("retaining_rate", 1e-4, 1e-3, log=True)
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.8, 1)
    static_ulr = trial.suggest_float("static_ulr", 0.00003, 0.001, log=True)
    c_stream_activation_ulr = trial.suggest_float("c_stream_activation_ulr", 1e-9, 1e-4, log=True)
    c_neg_entropy_ulr = trial.suggest_float("c_neg_entropy_ulr", 1e-9, 1e-4, log=True)
    # c_neg_cross_entropy_ulr = trial.suggest_float("c_neg_cross_entropy_ulr", 1e-7, 1e-3, log=True)
    # c_correct_logit_ulr = trial.suggest_float("c_correct_logit_ulr", 1e-7, 1e-3, log=True)
    c_soft_clipped_ce_ulr = trial.suggest_float("c_soft_clipped_ce_ulr", 1e-14, 1e-7, log=True)
    atan_scale = trial.suggest_float("atan_scale", 0.00001, 0.1, log=True)
    grad_pow = trial.suggest_float("grad_pow", 0.2, 1)
    logging.info(f"trial {trial.number} - {trial.params}")
    # fmt: on

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False

    # get params to intervene on
    interven_params = [
        p
        for name, p in model.named_parameters()
        if any(f"{m}.weight" in name for m in config.target_modules)
    ]
    # initialize parameter attributes
    for name, p in model.named_parameters():
        p.param_name = name
        p.requires_grad = False
    for p in interven_params:
        p.requires_grad = True
        p.disruption_score = pt.zeros_like(p.data)
        p.static_to_forget = pt.zeros_like(p.data)

    # use several circuits, mixed together; load circuits and construct to_forget
    for circuit_name, strength in config.circuit_names:
        circuit = get_circuit(config, forget_batches, circuit_name)
        circuit = filter_and_normalize_circuit(circuit, config.target_modules)
        for p in interven_params:
            if p.param_name in circuit:
                p.static_to_forget += circuit[p.param_name] * strength
        del circuit

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)
    for step in range(1, 1 + config.unlearn_steps):
        model.train()

        # ! retain pass, update disruption scores
        model.zero_grad(set_to_none=True)
        r_input_ids = next(retain_iter)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()
        for p in interven_params:
            p.disruption_score *= disruption_score_decay
            p.disruption_score += (p.grad.abs() ** grad_pow) * p.grad.sign()

        # skip the rest of the loop during warmup
        # todo remove the warmup to simplify?
        if step <= disruption_score_warmup:
            continue

        # ! retain update
        for p in interven_params:
            p.data -= retaining_rate * p.grad

        # ! continuous unlearning (prepares grads)
        model.zero_grad(set_to_none=True)
        f_input_ids = next(forget_iter)
        output = model(f_input_ids, output_hidden_states=True)
        loss = (
            c_stream_activation_ulr * stream_activation_loss(output, f_input_ids)
            + c_neg_entropy_ulr * negative_entropy_loss(output, f_input_ids)
            # + c_neg_cross_entropy_ulr * neg_cross_entropy_loss(output, f_input_ids)
            # + c_correct_logit_ulr * correct_logit_loss(output, f_input_ids)
            - c_soft_clipped_ce_ulr
            * soft_clipped_cross_entropy_loss(output, f_input_ids, atan_scale)
        )
        loss.backward()

        # ! unlearning step with masking
        for p in interven_params:
            to_forget = static_ulr * p.static_to_forget
            to_forget += p.grad

            mask = p.disruption_score.sign() == to_forget.sign()
            p.data -= mask * to_forget

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    return model
