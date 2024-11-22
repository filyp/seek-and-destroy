import logging
from copy import deepcopy

import torch as pt
from peft import LoraConfig, get_peft_model

from utils import *
from utils_dataloading import *


def only_grad_on(model, params_to_grad):
    for param in model.parameters():
        param.requires_grad = False
    for param in params_to_grad:
        param.requires_grad = True


def get_threshold(quantile, disruption_scores):
    """
    Calculate threshold value for parameter masking.
    Returns the k-th smallest value where k is determined by quantile.
    """
    total_num_params = sum(s.numel() for s in disruption_scores)
    k = int(quantile * total_num_params)
    return pt.cat([s.flatten() for s in disruption_scores]).kthvalue(k).values


def copy_model_and_collapse_loras(peft_model):
    """
    Creates a copy of the model with retention LoRA merged and adversarial LoRA removed.
    """
    peft_model_copy = deepcopy(peft_model)
    # delete adversarial lora
    peft_model_copy.delete_adapter("adv_lora")
    # merge and unload helper lora
    peft_model_copy.set_adapter(["ret_lora"])
    collapsed = peft_model_copy.merge_and_unload()
    del collapsed.peft_config
    return collapsed


def relearn(model, config, forget_set, retain_set):
    forget_iter = looping_iter(forget_set["validation"])
    retain_iter = looping_iter(retain_set["validation"])
    forget_eval = get_batch(iter(forget_set["validation"]), 32)
    retain_eval = get_batch(iter(retain_set["validation"]), 32)

    # add relearning lora
    lora_config = LoraConfig(**config.relearn_lora_conf)
    peft_model = get_peft_model(model, lora_config, adapter_name="relearning_lora")
    model = peft_model.model

    optimizer = pt.optim.Adam(
        model.parameters(), lr=config.relearn_lr, betas=(0.9, 0.999)
    )

    # ! relearning loop
    logging.info("")
    for step in range(1, 1 + config.relearn_steps):
        # standard forward, backward, and update
        model.train()
        optimizer.zero_grad(set_to_none=True)
        f_input_ids = get_batch(forget_iter, 16)
        r_input_ids = get_batch(retain_iter, 16)
        loss_forget = cross_entropy_loss(model(f_input_ids), f_input_ids)
        loss_retain = cross_entropy_loss(model(r_input_ids), r_input_ids)
        (loss_forget + loss_retain).backward()
        optimizer.step()

        if step % 10 == 0:
            f_loss = eval_loss(model, forget_eval)
            r_loss = eval_loss(model, retain_eval)
            logging.info(f"{step:4d} {f_loss:11.2f} {r_loss:11.2f}   <   RELEARNING")

    logging.info("")
    return eval_loss(model, forget_eval)
