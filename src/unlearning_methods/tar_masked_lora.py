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
    adv_lr = trial.suggest_float("adv_lr", 0.001, 0.5, log=True)
    clip_at = 3  # trial.suggest_float("clip_at", 0, 4)
    forget_momentum_decay = trial.suggest_float("forget_momentum_decay", 0.4, 0.8)
    fork_every_n_steps = trial.suggest_int("fork_every_n_steps", 12, 120, step=3)
    retain_momentum_decay = trial.suggest_float("retain_momentum_decay", 0.4, 0.8)
    retaining_rate = trial.suggest_float("retaining_rate", 2e-4, 2e-3, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 1e-2, 8e-2, log=True)
    lora_rank = trial.suggest_int("lora_rank", 4, 12)

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
    # for lora_index in range(1, lora_amount):
    #     peft_model.add_adapter(f"adv{lora_index}", lora_config)

    for p in interven_params:
        p.retain_momentum = pt.zeros_like(p.data)
        p.forget_momentum = pt.zeros_like(p.data)

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    # ! unlearning loop
    steps_per_loop = 1 + 2  # todo actually optimize unlearning to reuse forward pass
    assert config.unlearn_steps % steps_per_loop == 0
    for step in range(0, config.unlearn_steps, steps_per_loop):
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

        if step % fork_every_n_steps == 0:
            peft_model.delete_adapter("adv0")
            peft_model.add_adapter("adv0", lora_config)

        # ! relearn the adversary
        for p in model.parameters():
            p.requires_grad = False
        peft_model.set_adapter("adv0")
        model.zero_grad(set_to_none=True)
        f_input_ids = next(forget_iter)
        output = model(f_input_ids)
        loss = cross_entropy_loss(output, f_input_ids)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                assert "adv0" in n
                p.data -= adv_lr * p.grad

        # ! get unlearning grads loss from adversary
        peft_model.set_adapter("adv0")
        only_grad_on(model, interven_params)
        model.zero_grad(set_to_none=True)
        output = model(f_input_ids)  # reuse f_input_ids from previous step
        loss = correct_logit_minus_avg_loss(output, f_input_ids, clip_at)
        loss.backward()

        # ! unlearning step with masking
        for p in interven_params:
            p.forget_momentum *= forget_momentum_decay
            p.forget_momentum += p.grad * (1 - forget_momentum_decay)
            mask = p.retain_momentum.sign() == p.forget_momentum.sign()
            update = mask * p.forget_momentum
            update /= update.norm()
            p.data -= unlearning_rate * update

        # ! eval current loss
        if (step + steps_per_loop) % 12 == 0:
            with peft_model.disable_adapter():
                eval_(model, f_eval, r_eval, allowed_f_loss, step + steps_per_loop)

    # ! remove lora
    peft_model.delete_adapter("adv0")

    return model
