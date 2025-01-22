import logging

import torch as pt
from peft import LoraConfig, get_peft_config, get_peft_model
from transformers import AutoModelForCausalLM

from utils.loss_fns import *
from utils.training import *


def only_grad_on(model, params):
    for p in model.parameters():
        p.requires_grad = False
    for p in params:
        p.requires_grad = True


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # ! parameters
    adv_decay = trial.suggest_float("adv_decay", 0.95, 1)
    adv_lr = trial.suggest_float("adv_lr", 0.03, 0.15, log=True)
    clip_at = trial.suggest_float("clip_at", -5, 2)
    f_power = trial.suggest_float("f_power", 0.5, 2)
    forget_momentum_decay = trial.suggest_float("forget_momentum_decay", 0.5, 0.8)
    fork_every_n_loops = trial.suggest_int("fork_every_n_loops", 12, 42, step=6)
    lora_amount = trial.suggest_int("lora_amount", 1, 3)
    lora_rank = trial.suggest_int("lora_rank", 6, 9)
    retain_momentum_decay = trial.suggest_float("retain_momentum_decay", 0.3, 0.6)
    retaining_rate = trial.suggest_float("retaining_rate", 8e-4, 2e-3, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 2e-2, 5e-2, log=True)

    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False
    # get params to intervene on (must be before lora creation)
    interven_params = [
        p
        for name, p in model.named_parameters()
        if any(f"{m}.weight" in name for m in config.target_modules)
    ]

    lora_config = LoraConfig(r=lora_rank, target_modules=config.target_modules)
    peft_model = get_peft_model(model, lora_config, adapter_name="adv0")
    for lora_index in range(1, lora_amount):
        peft_model.add_adapter(f"adv{lora_index}", lora_config)

    for p in interven_params:
        p.retain_momentum = pt.zeros_like(p.data)
        p.forget_momentum = pt.zeros_like(p.data)

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    # ! unlearning loop
    passes_per_loop = 5
    assert config.unlearn_steps % passes_per_loop == 0
    for loop_num in range(config.unlearn_steps // passes_per_loop):
        model.train()

        # ! retain pass
        with peft_model.disable_adapter():
            only_grad_on(model, interven_params)
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

        if loop_num % (fork_every_n_loops // lora_amount) == 0:
            forking_count = loop_num // (fork_every_n_loops // lora_amount)
            adv_to_restart = f"adv{forking_count % lora_amount}"
            peft_model.delete_adapter(adv_to_restart)
            peft_model.add_adapter(adv_to_restart, lora_config)

        # ! relearn the adversary
        adv_to_use = f"adv{loop_num % lora_amount}"
        peft_model.set_adapter(adv_to_use)
        _lora_params = [p for n, p in model.named_parameters() if adv_to_use in n]
        # we already need grads on both, if we want to later reuse the computation graph
        only_grad_on(model, interven_params + _lora_params)
        model.zero_grad(set_to_none=True)
        f_input_ids = next(forget_iter)
        output = model(f_input_ids)
        loss = cross_entropy_loss(output, f_input_ids)
        loss.backward(retain_graph=True)
        for n, p in model.named_parameters():
            if adv_to_use in n:
                p.data -= adv_lr * p.grad
                # decay adversary
                p.data *= adv_decay

        # ! get unlearning grads from adversary
        # reuse the computation graph from previous block
        model.zero_grad(set_to_none=True)
        loss = correct_logit_minus_avg_loss(output, f_input_ids, clip_at)
        loss.backward()

        # ! unlearning step with masking
        for p in interven_params:
            p.forget_momentum *= forget_momentum_decay
            grad = (p.grad.abs() ** (1 / f_power)) * p.grad.sign()
            p.forget_momentum += grad * (1 - forget_momentum_decay)
            mask = p.retain_momentum.sign() == p.forget_momentum.sign()
            update = (p.forget_momentum.abs() ** f_power) * p.forget_momentum.sign()
            update *= mask
            update /= update.norm()
            p.data -= unlearning_rate * update

        # ! eval current loss
        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done % 30 == 0:
            with peft_model.disable_adapter():
                eval_(model, f_eval, r_eval, allowed_f_loss, _passes_done)

    # ! remove lora
    for lora_index in range(lora_amount):
        peft_model.delete_adapter(f"adv{lora_index}")

    return model
