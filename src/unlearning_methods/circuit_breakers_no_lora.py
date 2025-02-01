from copy import deepcopy

import torch as pt
from transformers import AutoModelForCausalLM

from utils.loss_fns import circuit_breaker_forget_loss, circuit_breaker_retain_loss
from utils.training import eval_


def circuit_breakers_no_lora(
    h, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # Create main model and frozen copy
    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    # get params to intervene on
    interven_params = [
        p
        for name, p in model.named_parameters()
        if any(f"{m}.weight" in name for m in config.target_modules)
    ]
    total_interven_numel = sum(p.numel() for p in interven_params)
    # require grads only on intervened params
    for p in model.parameters():
        p.requires_grad = id(p) in [id(p) for p in interven_params]

    frozen_model = deepcopy(model)
    frozen_model.eval()

    num_layers = model.config.num_hidden_layers
    target_layers = [num_layers // 2]

    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    # maybe actually it should be 6, because the backward pass is twice bigger?
    # anyway, we can afford to give CB a handicap here
    passes_per_loop = 5
    for loop_num in range(config.unlearn_steps // passes_per_loop):
        r_input_ids = next(retain_iter)
        f_input_ids = next(forget_iter)

        # # percent_done = loop_num / (config.unlearn_steps // passes_per_loop)
        # retain_coeff = h.retaining_rate  #  * (percent_done / 2)
        # forget_coeff = h.unlearning_rate  #  * (1 - percent_done / 2)

        # if retain_coeff > 0:
        model.zero_grad(set_to_none=True)
        retain_loss = circuit_breaker_retain_loss(model, r_input_ids, frozen_model)
        retain_loss.backward()
        # retain update
        for p in interven_params:
            if p.grad is not None:
                p.data -= h.retaining_rate * p.grad

        # if forget_coeff > 0:
        model.zero_grad(set_to_none=True)
        forget_loss = circuit_breaker_forget_loss(
            model, f_input_ids, target_layers, frozen_model
        )
        forget_loss.backward()
        grad_norm = (
            sum(p.grad.norm() ** 2 for p in interven_params if p.grad is not None)
            ** 0.5
        )
        for p in interven_params:
            if p.grad is not None:
                p.grad *= total_interven_numel**0.5 / grad_norm
                p.data -= h.unlearning_rate * p.grad

        # ! eval current loss
        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done % 30 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, _passes_done)

    return model
