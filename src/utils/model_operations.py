import logging
from copy import deepcopy

import torch as pt
import wandb
from peft import LoraConfig, get_peft_model

from utils.loss_fns import *
from utils.training import eval_


def only_grad_on(model, params_to_grad):
    for param in model.parameters():
        param.requires_grad = False
    for param in params_to_grad:
        param.requires_grad = True


def get_thresh(quantile, disruption_scores):
    """
    Calculate threshold value for parameter masking, based on the quantile.
    For example, if quantile is 0.01, the threshould will cut off 1% of the highest scores.
    """
    flat_scores = pt.cat([s.flatten() for s in disruption_scores])
    return pt.quantile(flat_scores, 1 - quantile, interpolation="lower")


def copy_model_and_collapse_loras(peft_model, delete_adv=True):
    """
    Creates a copy of the model with retention LoRA merged and adversarial LoRA removed.
    """
    peft_model_copy = deepcopy(peft_model)
    # delete adversarial lora
    if delete_adv:
        peft_model_copy.delete_adapter("adv_lora")
    # merge and unload helper lora
    peft_model_copy.set_adapter(["ret_lora"])
    collapsed = peft_model_copy.merge_and_unload()
    del collapsed.peft_config
    return collapsed


def relearn(model, config, retain_val_batches, forget_val_batches, use_lora=False):
    for p in model.parameters():
        p.requires_grad = True

    # get batches
    retain_val_iter = iter(retain_val_batches)
    forget_val_iter = iter(forget_val_batches)
    f_eval_batch = next(forget_val_iter)
    r_eval_batch = next(retain_val_iter)

    if use_lora:
        lora_config = LoraConfig(**config.relearn_lora_conf)
        peft_model = get_peft_model(model, lora_config, adapter_name="relearning_lora")
        model = peft_model.model

    optimizer = pt.optim.SGD(model.parameters(), lr=config.relearn_lr)

    # ! relearning loop
    logging.info("")
    f_losses = []
    # each step is one pass (forward or backward) and a loop has two passes
    passes_per_loop = 2
    for loop_num in range(config.relearn_steps // passes_per_loop):
        # standard forward, backward, and update
        model.train()
        optimizer.zero_grad(set_to_none=True)
        f_input_ids = next(forget_val_iter)
        loss_forget = cross_entropy_loss(model(f_input_ids), f_input_ids)
        loss_forget.backward()
        optimizer.step()

        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done % 30 == 0:
            res = eval_(model, f_eval_batch, r_eval_batch, step=_passes_done)
            f_losses.append(res["forget_loss"])
            # wandb.log(res, step=_passes_done)

    logging.info("")
    return f_losses


def relearn_with_retain(model, config, retain_val_batches, forget_val_batches):
    for p in model.parameters():
        p.requires_grad = True

    # get batches
    retain_val_iter = iter(retain_val_batches)
    forget_val_iter = iter(forget_val_batches)
    f_eval_batch = next(forget_val_iter)
    r_eval_batch = next(retain_val_iter)

    optimizer = pt.optim.SGD(model.parameters(), lr=config.relearn_lr)

    # ! relearning loop
    logging.info("")
    f_losses = []
    # each step is one pass (forward or backward) and a loop has two passes
    passes_per_loop = 4
    for loop_num in range(config.relearn_steps // passes_per_loop):
        model.train()

        # standard forward, backward, and update
        optimizer.zero_grad(set_to_none=True)
        f_input_ids = next(forget_val_iter)
        loss_forget = cross_entropy_loss(model(f_input_ids), f_input_ids)
        loss_forget.backward()
        optimizer.step()

        # standard forward, backward, and update
        optimizer.zero_grad(set_to_none=True)
        r_input_ids = next(retain_val_iter)
        loss_retain = cross_entropy_loss(model(r_input_ids), r_input_ids)
        loss_retain.backward()
        optimizer.step()

        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done % 20 == 0:
            res = eval_(model, f_eval_batch, r_eval_batch, step=_passes_done)
            f_losses.append(res["forget_loss"])
            # wandb.log(res, step=_passes_done)

    logging.info("")
    return f_losses
