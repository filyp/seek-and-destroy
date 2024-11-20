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
    total_num_params = sum(s.numel() for s in disruption_scores)
    k = int(quantile * total_num_params)
    return pt.cat([s.flatten() for s in disruption_scores]).kthvalue(k).values


def copy_model_and_collapse_loras(peft_model):
    peft_model_copy = deepcopy(peft_model)
    # delete adversarial lora
    peft_model_copy.delete_adapter("adv_lora")
    # merge and unload helper lora
    peft_model_copy.set_adapter(["ret_lora"])
    return peft_model_copy.merge_and_unload()


def relearn(model, relearn_lr, relearn_steps, forget_set, retain_set):
    forget_iter = looping_iter(forget_set["validation"])
    retain_iter = looping_iter(retain_set["validation"])
    forget_eval = get_batch(iter(forget_set["validation"]), 32)
    retain_eval = get_batch(iter(retain_set["validation"]), 32)

    # add relearning lora
    lora_config = LoraConfig(r=1, target_modules=["dense_h_to_4h"])
    peft_model = get_peft_model(model, lora_config, adapter_name="relearning_lora")
    model = peft_model.model

    optimizer = pt.optim.Adam(model.parameters(), lr=relearn_lr, betas=(0.9, 0.999))

    # ! relearning loop
    for step in range(1, 1 + relearn_steps):
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
            print("RELRN ", end="")
            print_perplexities(model, [forget_eval, retain_eval], step)
            # if wandb.run:
            #     wandb.log(results, step=step)

    model.eval()
    with pt.no_grad():
        retain_ppl = cross_entropy_loss(model(retain_eval), retain_eval).exp()
    return retain_ppl
