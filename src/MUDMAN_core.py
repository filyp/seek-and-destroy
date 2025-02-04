
import logging
from copy import deepcopy

import torch as pt
from transformers import AutoModelForCausalLM

from utils.loss_fns import *
from utils.training import *


def surgical_irreversible_unlearning(
    h, config, retain_batches, forget_batches, f_eval, r_eval, allowed_r_loss
):
    h.fork_every_n_loops = int(h.fork_every_n_loops)

    if config.model_id in ["meta-llama/Llama-3.2-1B"]:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, torch_dtype=pt.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False

    # get params to intervene on
    interven_params = [
        p
        for name, p in model.named_parameters()
        if any(f"{m}.weight" in name for m in config.target_modules)
    ]

    # Require grads only on intervened params
    for p in model.parameters():
        p.requires_grad = id(p) in [id(p) for p in interven_params]

    # Initialize retain grad accumulators
    for p in interven_params:
        p.retain_acc = pt.zeros_like(p.data)
        p.base_data = p.data.clone().detach()

    unlearning_loss_fn = loss_fns[config.unlearning_loss_fn]

    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    # ! unlearning loop
    passes_per_loop = 5
    _eval_counter = 0
    assert config.unlearn_steps % passes_per_loop == 0
    for loop_num in range(config.unlearn_steps // passes_per_loop):
        model.train()
        f_input_ids = next(forget_iter)
        r_input_ids = next(retain_iter)

        if loop_num % h.fork_every_n_loops == 0:
            for p in interven_params:
                p.adv_data = p.base_data.clone().detach()

        # ! retain pass
        model.zero_grad()
        for p in interven_params:  # switch to base model
            p.data = p.base_data
        output = model(r_input_ids)
        cross_entropy_loss(output, r_input_ids).backward()
        for p in interven_params:
            # ! update disruption scores
            p.retain_acc *= h.retain_momentum
            p.retain_acc += p.grad * (1 - h.retain_momentum)
            # ! retain update
            p.base_data -= h.retaining_rate * p.retain_acc

        # ! relearn the adversary
        model.zero_grad()
        for p in interven_params:  # switch to adversary
            p.data = p.adv_data
        output = model(f_input_ids)
        cross_entropy_loss(output, f_input_ids).backward(retain_graph=True)
        for p in interven_params:
            # apply adversary update
            p.adv_data -= h.adv_lr * p.grad
            # decay adversary into base model
            p.adv_data *= h.adv_decay
            p.adv_data += p.base_data * (1 - h.adv_decay)

        # ! unlearning step with masking
        model.zero_grad()
        # reuse the computation graph from previous block
        unlearning_loss_fn(output, f_input_ids).backward()
        grad_norm = sum(p.grad.norm() ** 2 for p in interven_params) ** 0.5
        for p in interven_params:
            p.grad *= p.retain_acc.sign() == p.grad.sign()  # mask
            p.base_data -= h.unlearning_rate / grad_norm * p.grad  # normalize & update

        # ! eval current loss
        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done // 30 > _eval_counter:
            _eval_counter += 1
            for p in interven_params:  # switch to base model
                p.data = p.base_data
            eval_(model, f_eval, r_eval, allowed_r_loss, _passes_done)

    return model


# for paper:

# for loop_num, (retain_batch, forget_batch) in enumerate(batch_pairs):
#     if loop_num % fork_every_n_loops == 0:
#         # Fork adversary
#         for p in interven_params:
#             p.adv_data = p.base_data.clone().detach()

#     # Retain pass
#     model.zero_grad()
#     # Switch to base model
#     for p in interven_params:
#         p.data = p.base_data
#     output = model(retain_batch)
#     loss = cross_entropy_loss(output, retain_batch)
#     loss.backward()
#     for p in interven_params:
#         # Update disruption scores
#         p.retain_acc *= retain_momentum
#         p.retain_acc += p.grad * (1 - retain_momentum)
#         # Retain update
#         p.base_data -= retaining_rate * p.retain_acc

#     # Relearn the adversary
#     model.zero_grad()
#     # Switch to adversary
#     for p in interven_params:
#         p.data = p.adv_data
#     output = model(forget_batch)
#     loss = cross_entropy_loss(output, forget_batch)
#     loss.backward(retain_graph=True)
#     for p in interven_params:
#         # Apply adversary update
#         p.adv_data -= adv_lr * p.grad
#         # Decay adversary into base model
#         p.adv_data *= adv_decay
#         p.adv_data += p.base_data * (1 - adv_decay)

#     # Unlearning step with masking
#     model.zero_grad()
#     loss.unlearning_loss_fn(output, forget_batch)
#     loss.backward()
#     grad_norm = sum(p.grad.norm() ** 2 for p in interven_params) ** 0.5
#     for p in interven_params:
#         # Mask
#         p.grad *= p.retain_acc.sign() == p.grad.sign()
#         # Normalize & update
#         p.base_data -= unlearning_rate / grad_norm * p.grad

"""
for loop_num, (retain_batch, forget_batch) in enumerate(batch_pairs):
    if loop_num % fork_every_n_loops == 0:
        Fork adversary

    Switch to base model
    output = model(retain_batch)
    loss = cross_entropy_loss(output, retain_batch)
    loss.backward()
    for p in interven_params:
        # Update disruption scores
        p.retain_acc *= retain_momentum
        p.retain_acc += p.grad * (1 - retain_momentum)
        # Retain update
        p.base_data -= retaining_rate * p.retain_acc

    # Relearn the adversary
    model.zero_grad()
    # Switch to adversary
    for p in interven_params:
        p.data = p.adv_data
    output = model(forget_batch)
    loss = cross_entropy_loss(output, forget_batch)
    loss.backward(retain_graph=True)
    for p in interven_params:
        # Apply adversary update
        p.adv_data -= adv_lr * p.grad
        # Decay adversary into base model
        p.adv_data *= adv_decay
        p.adv_data += p.base_data * (1 - adv_decay)

    # Unlearning step with masking
    model.zero_grad()
    loss.unlearning_loss_fn(output, forget_batch)
    loss.backward()
    grad_norm = sum(p.grad.norm() ** 2 for p in interven_params) ** 0.5
    for p in interven_params:
        # Mask
        p.grad *= p.retain_acc.sign() == p.grad.sign()
        # Normalize & update
        p.base_data -= unlearning_rate / grad_norm * p.grad

"""