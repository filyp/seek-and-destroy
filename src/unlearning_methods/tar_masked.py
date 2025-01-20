import logging

import torch as pt
from transformers import AutoModelForCausalLM

from utils.loss_fns import *
from utils.training import *


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # ! parameters
    adv_decay = trial.suggest_float("adv_decay", 0.8, 1)
    adv_lr = trial.suggest_float("adv_lr", 0.002, 0.01, log=True)
    clip_at = trial.suggest_float("clip_at", -5, 7)
    forget_momentum_decay = trial.suggest_float("forget_momentum_decay", 0, 1)
    fork_every_n_steps = trial.suggest_int("fork_every_n_steps", 24, 72, step=24)
    retain_momentum_decay = trial.suggest_float("retain_momentum_decay", 0.7, 0.9)
    retaining_rate = trial.suggest_float("retaining_rate", 2e-4, 2e-3, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 2e-2, 5e-2, log=True)

    adv_per_orig_step = 1
    logging.info(f"trial {trial.number} - {trial.params}")
    assert adv_per_orig_step in [1, 2, 4, 6, 10]
    assert fork_every_n_steps % 24 == 0

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
        p.retain_momentum = pt.zeros_like(p.data)
        p.forget_momentum = pt.zeros_like(p.data)

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    # ! unlearning loop
    steps_per_loop = adv_per_orig_step + 2
    assert config.unlearn_steps % steps_per_loop == 0
    for step in range(0, config.unlearn_steps, steps_per_loop):
        model.train()

        # ! retain pass
        model.zero_grad(set_to_none=True)
        r_input_ids = next(retain_iter)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()
        for p in interven_params:
            # ! update disruption scores
            p.retain_momentum *= retain_momentum_decay
            p.retain_momentum += p.grad * (1 - retain_momentum_decay)
            # ! retain update
            p.data -= retaining_rate * p.retain_momentum

        model.zero_grad(set_to_none=True)

        if step % fork_every_n_steps == 0:
            adversary.load_state_dict(model.state_dict())

        for _ in range(adv_per_orig_step):
            # ! relearn the adversary
            adversary.zero_grad(set_to_none=True)
            f_input_ids = next(forget_iter)
            output = adversary(f_input_ids)
            loss = cross_entropy_loss(output, f_input_ids)
            loss.backward()
            for p, adv_p in zip(interven_params, adv_interven_params):
                adv_p.data -= adv_lr * adv_p.grad
                # decay adversary into base model
                adv_p.data *= adv_decay
                adv_p.data += p.data * (1 - adv_decay)


        # ! get unlearning grads loss from adversary
        adversary.zero_grad(set_to_none=True)
        output = adversary(f_input_ids)  # reuse f_input_ids from previous step
        loss = correct_logit_minus_avg_loss(output, f_input_ids, clip_at)
        loss.backward()

        # ! unlearning step with masking
        for p, adv_p in zip(interven_params, adv_interven_params):
            p.forget_momentum *= forget_momentum_decay
            p.forget_momentum += adv_p.grad * (1 - forget_momentum_decay)
            mask = p.retain_momentum.sign() == p.forget_momentum.sign()
            update = mask * p.forget_momentum
            update /= update.norm()
            p.data -= unlearning_rate * update

        # ! eval current loss
        if (step + steps_per_loop) % 24 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step + steps_per_loop)

    return model
