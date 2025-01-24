import logging

import torch as pt
from transformers import AutoModelForCausalLM

from utils.loss_fns import *
from utils.training import *


def tar_masked(
    h, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    h.fork_every_n_loops = int(h.fork_every_n_loops)

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
    passes_per_loop = 4 + int(config.train_adversary)
    assert config.unlearn_steps % passes_per_loop == 0
    for loop_num in range(config.unlearn_steps // passes_per_loop):
        model.train()
        adversary.train()

        # ! retain pass
        model.zero_grad(set_to_none=True)
        r_input_ids = next(retain_iter)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()
        for p in interven_params:
            # ! update disruption scores
            p.retain_momentum *= h.retain_momentum_decay
            p.retain_momentum += p.grad * (1 - h.retain_momentum_decay)
            # ! retain update
            p.data -= h.retaining_rate * p.retain_momentum
        model.zero_grad(set_to_none=True)

        if loop_num % h.fork_every_n_loops == 0:
            adversary.load_state_dict(model.state_dict())

        # for _ in range(adv_per_orig_step):
        # ! relearn the adversary
        # todo actually could store only the intervened params for adversary
        adversary.zero_grad(set_to_none=True)
        f_input_ids = next(forget_iter)
        output = adversary(f_input_ids)
        if config.train_adversary:
            loss = cross_entropy_loss(output, f_input_ids)
            loss.backward(retain_graph=True)
            for p, adv_p in zip(interven_params, adv_interven_params):
                # apply adversary update
                adv_p.data -= h.adv_lr * adv_p.grad
                # decay adversary into base model
                adv_p.data *= h.adv_decay
                adv_p.data += p.data * (1 - h.adv_decay)
        else:
            for p, adv_p in zip(interven_params, adv_interven_params):
                assert (p.data == adv_p.data).all()

        # ! get unlearning grads loss from adversary
        # reuse the computation graph from previous block
        adversary.zero_grad(set_to_none=True)
        loss_fn = loss_fns[config.unlearning_loss_fn]
        loss = loss_fn(output, f_input_ids, h.clip_at)
        loss.backward()

        # ! unlearning step with masking
        for p, adv_p in zip(interven_params, adv_interven_params):
            p.forget_momentum *= h.forget_momentum_decay
            p.forget_momentum += adv_p.grad * (1 - h.forget_momentum_decay)

            if config.use_masking:
                mask = p.retain_momentum.sign() == p.forget_momentum.sign()
                update = mask * p.forget_momentum
            else:
                update = p.forget_momentum

            if config.use_normalization:
                update /= update.norm()
            else:
                # neede to keep a similar update scale across variants
                update *= config.normalization_factor

            p.data -= h.unlearning_rate * update
            adv_p.data -= h.unlearning_rate * update * h.adv_update

        # ! eval current loss
        _passes_done = (loop_num + 1) * passes_per_loop
        print(f"passes done: {_passes_done}")
        if _passes_done % 60 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, _passes_done)

    return model
