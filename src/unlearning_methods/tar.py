import logging

import torch as pt
from transformers import AutoModelForCausalLM

from utils.loss_fns import *
from utils.training import *


def tar(
    h, config, retain_batches, forget_batches, f_eval, r_eval, allowed_r_loss
):
    assert config.use_masking
    assert config.use_normalization
    assert config.train_adversary
    assert h.additional_param_name is None, "TAR LoRA doesn't support additional param"

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

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    # ! unlearning loop
    passes_per_loop = 5
    _eval_counter = 0
    assert config.unlearn_steps % passes_per_loop == 0
    for loop_num in range(config.unlearn_steps // passes_per_loop):
        model.train()
        adversary.train()
        
        # todo repE retain loss

        # ! retain pass
        model.zero_grad(set_to_none=True)
        r_input_ids = next(retain_iter)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()
        for p in interven_params:
            # ! retain update
            p.data -= h.retaining_rate * p.grad
        model.zero_grad(set_to_none=True)

        if (loop_num % h.fork_every_n_loops == 0) or (not config.train_adversary):
            # if not training adversary, make sure it's always the same as base model
            adversary.load_state_dict(model.state_dict())

        # for _ in range(adv_per_orig_step):
        # ! relearn the adversary
        adversary.zero_grad(set_to_none=True)
        f_input_ids = next(forget_iter)
        output = adversary(f_input_ids)
        loss = cross_entropy_loss(output, f_input_ids)
        loss.backward(retain_graph=True)
        for p, adv_p in zip(interven_params, adv_interven_params):
            # apply adversary update
            adv_p.data -= h.adv_lr * adv_p.grad

        # ! get unlearning grads loss from adversary
        # reuse the computation graph from previous block
        adversary.zero_grad(set_to_none=True)
        loss = neg_entropy_loss(output, f_input_ids)
        loss.backward()

        # ! unlearning step
        for p, adv_p in zip(interven_params, adv_interven_params):
            update = adv_p.grad
            update *= config.update_scale_factor

            p.data -= h.unlearning_rate * update

        # ! eval current loss
        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done // 30 > _eval_counter:
            _eval_counter += 1
            eval_(model, f_eval, r_eval, allowed_r_loss, _passes_done)

    return model
