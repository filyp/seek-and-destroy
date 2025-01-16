import logging

import torch as pt
from transformers import AutoModelForCausalLM

from utils.training import *


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # ! parameters
    retaining_rate = 7e-4
    unlearning_lr = trial.suggest_float("unlearning_lr", 5e-4, 1e-3, log=True)
    adv_lr = trial.suggest_float("adv_lr", 5e-4, 1e-3, log=True)
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.8, 1)
    fork_every_n_steps = trial.suggest_int("fork_every_n_steps", 20, 100)
    adv_steps_per_orig_step = 1
    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    adversary = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False
    adversary.config.use_cache = False

    # get params to intervene on
    interven_params = [
        p
        for name, p in model.named_parameters()
        if any(f"{m}.weight" in name for m in config.target_modules)
    ]
    adv_interven_params = [
        p
        for name, p in adversary.named_parameters()
        if any(f"{m}.weight" in name for m in config.target_modules)
    ]

    # require grads only on intervened params
    for p in model.parameters():
        p.requires_grad = id(p) in [id(p) for p in interven_params]
    for p in adversary.parameters():
        p.requires_grad = id(p) in [id(p) for p in adv_interven_params]

    for p in interven_params:
        p.disruption_score = pt.zeros_like(p.data)

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    # ! unlearning loop
    # note: here each step uses 2+1 forward+backward passes, while s&d uses only 1
    for step in range(1, 1 + config.unlearn_steps):
        model.train()

        # ! retain pass
        model.zero_grad(set_to_none=True)
        r_input_ids = next(retain_iter)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()
        for p in interven_params:
            # ! update disruption scores
            p.disruption_score *= disruption_score_decay
            p.disruption_score += p.grad
            # ! retain update
            p.data -= retaining_rate * p.grad
        model.zero_grad(set_to_none=True)

        if step % fork_every_n_steps == 1:
            adversary.load_state_dict(model.state_dict())

        for _ in range(adv_steps_per_orig_step):
            # ! relearn on the adversary
            adversary.zero_grad(set_to_none=True)
            f_input_ids = next(forget_iter)
            output = adversary(f_input_ids)
            loss = cross_entropy_loss(output, f_input_ids)
            loss.backward()
            for adv_p in adv_interven_params:
                adv_p.data -= adv_lr * adv_p.grad

        # ! get grads on neg_entropy loss from adversary
        adversary.zero_grad(set_to_none=True)
        output = adversary(f_input_ids)  # reuse f_input_ids from previous step
        loss = neg_entropy_loss(output, f_input_ids)
        loss.backward()

        # ! unlearning step with masking
        for p, adv_p in zip(interven_params, adv_interven_params):
            to_forget = adv_p.grad
            mask = p.disruption_score.sign() == to_forget.sign()
            p.data -= unlearning_lr * mask * to_forget

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    return model
