import gc
import logging
from copy import deepcopy
from utils.loss_fns import circuit_breaker_forget_loss, circuit_breaker_retain_loss
import torch as pt
from transformers import AutoModelForCausalLM

from utils.training import eval_


def compute_loss(
    percent_done,
    model,
    frozen_model,
    forget_input_ids,
    retain_input_ids,
    target_layers,
    config,
    retaining_rate,
    unlearning_rate,
):

    # Those are pretty much arbitrary, the important thing is that retain_coeff increases as the training progresses and forget_coeff decreases.
    retain_coeff = retaining_rate * (percent_done / 2)
    forget_coeff = unlearning_rate * (1 - percent_done / 2)

    if retain_coeff > 0:
        retain_loss = circuit_breaker_retain_loss(
            model, retain_input_ids, frozen_model)
    else:
        retain_loss = 0

    if forget_coeff > 0:
        forget_loss = circuit_breaker_forget_loss(
            model, forget_input_ids, target_layers, frozen_model)
    else:
        forget_loss = 0

    loss = retain_coeff * retain_loss + forget_coeff * forget_loss

    return loss


def circuit_breaker_without_lora(
    h, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # Create main model and frozen copy
    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    frozen_model = deepcopy(model)
    frozen_model.eval()

    num_layers = model.config.num_hidden_layers
    target_layers = [num_layers // 2]

    optimizer = pt.optim.SGD(model.parameters(), lr=1)

    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    passes_per_loop = 5
    for loop_num in range(config.unlearn_steps // passes_per_loop):
        r_input_ids = next(retain_iter)
        f_input_ids = next(forget_iter)

        model.zero_grad()
        loss = compute_loss(
            loop_num / (config.unlearn_steps // passes_per_loop),
            model,
            frozen_model,
            f_input_ids,
            r_input_ids,
            target_layers,
            config,
            h.retaining_rate,
            h.unlearning_rate,
        )

        loss.backward()
        optimizer.step()

        # ! eval current loss
        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done % 60 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, _passes_done)

    return model
