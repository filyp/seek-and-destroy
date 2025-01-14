import logging

import torch as pt
from transformers import AutoModelForCausalLM

from utils.circuit_creation import filter_and_normalize_circuit, get_circuit
from utils.model_operations import get_thresh
from utils.plots_and_stats import visualize_param
from utils.training import cross_entropy_loss, eval_, loss_fns, stream_activation_loss

disruption_score_warmup = 20


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # ! parameters
    r_quantile = trial.suggest_float("r_quantile", 0.05, 0.5, log=True)
    # retaining_rate = trial.suggest_float("retaining_rate", 0.0003, 0.0010, log=True)
    retaining_rate = 0.0005
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.00003, 0.001, log=True)
    # unlearning_rate = trial.suggest_float("unlearning_rate", 1e-5, 8e-5, log=True)
    disruption_score_decay = 0.9
    pos_grad_discard = 0  # trial.suggest_float("pos_grad_discard", 0, 1)
    cont_lr = 0 #trial.suggest_float("cont_lr", 0.0001, 0.008, log=True)
    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False

    # get params to intervene on and initialize disruption scores
    interven_params = [
        p
        for name, p in model.named_parameters()
        if any(f"{m}.weight" in name for m in config.target_modules)
    ]

    for name, p in model.named_parameters():
        p.param_name = name
        p.requires_grad = False
    for p in interven_params:
        p.requires_grad = True
        p.disruption_score = pt.zeros_like(p.data)
        p.to_forget = pt.zeros_like(p.data)

    # use several circuits, mixed together; load circuits and construct to_forget
    for circuit_name, strength in config.circuit_names:
        circuit = get_circuit(config, forget_batches, circuit_name)
        circuit = filter_and_normalize_circuit(circuit, config.target_modules)
        for p in interven_params:
            if p.param_name in circuit:
                p.to_forget += circuit[p.param_name] * strength
        del circuit

    optimizer = pt.optim.SGD(interven_params, lr=cont_lr)

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)
    for step in range(1, 1 + config.unlearn_steps):
        model.train()

        # ! retain pass
        model.zero_grad(set_to_none=True)
        r_input_ids = next(retain_iter)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()

        # ! update disruption scores
        for p in interven_params:
            grad = p.grad.clone().detach()
            grad[p.to_forget.sign() == p.grad.sign()] *= pos_grad_discard
            p.disruption_score *= disruption_score_decay
            p.disruption_score += grad

        # skip the rest of the loop during warmup
        if step <= disruption_score_warmup:
            continue

        # ! retain update
        for p in interven_params:
            p.data -= retaining_rate * p.grad

        # ! continuous unlearning
        model.zero_grad(set_to_none=True)
        f_input_ids = next(forget_iter)
        output = model(f_input_ids, output_hidden_states=True)
        loss = stream_activation_loss(output, f_input_ids)
        loss.backward()
        optimizer.step()

        # calculate r_threshold globally
        flipped_disrs = (
            p.disruption_score * p.to_forget.sign()
            for p in interven_params
        )
        r_threshold = get_thresh(r_quantile, flipped_disrs)

        # Unlearning step with two-stage masking
        for p in interven_params:
            flipped_disr = p.disruption_score * p.to_forget.sign()
            # r_threshold = get_thresh(r_quantile, [flipped_disr])
            mask = flipped_disr > r_threshold

            # ! unlearn
            p.data -= mask * unlearning_rate * p.to_forget

            # if step == config.unlearn_steps:
            #     visualize_param(p, mask, p.param_name)

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    return model
