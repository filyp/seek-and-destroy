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


def surgical_irreversible_unlearning_lora(
    h, config, retain_batches, forget_batches, f_eval, r_eval, allowed_r_loss
):
    assert config.use_masking
    assert config.normalize_grads
    assert config.train_adversary
    assert "additional_param_name" not in h.__dict__, "Additional param not supported"

    h.fork_every_n_loops = (int(h.fork_every_n_loops) // 6) * 6  # round to nearest 6

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False
    # get params to intervene on (must be before lora creation)
    interven_params = [
        p
        for name, p in model.named_parameters()
        if any(f"{m}.weight" in name for m in config.target_modules)
    ]
    total_interven_numel = sum(p.numel() for p in interven_params)

    lora_config = LoraConfig(r=config.lora_rank, target_modules=config.target_modules)
    peft_model = get_peft_model(model, lora_config, adapter_name="adv0")
    for lora_index in range(1, config.lora_amount):
        peft_model.add_adapter(f"adv{lora_index}", lora_config)

    for p in interven_params:
        p.retain_acc = pt.zeros_like(p.data)

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

        if loop_num % (h.fork_every_n_loops // config.lora_amount) == 0:
            forking_count = loop_num // (h.fork_every_n_loops // config.lora_amount)
            adv_to_restart = f"adv{forking_count % config.lora_amount}"
            peft_model.delete_adapter(adv_to_restart)
            peft_model.add_adapter(adv_to_restart, lora_config)

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
                p.retain_acc *= h.retain_momentum
                p.retain_acc += p.grad * (1 - h.retain_momentum)
                # ! retain update
                p.data -= h.retaining_rate * p.retain_acc

        # ! relearn the adversary
        adv_to_use = f"adv{loop_num % config.lora_amount}"
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
                p.data -= h.adv_lr * p.grad
                # decay adversary
                p.data *= h.adv_decay

        # ! get unlearning grads from adversary
        # reuse the computation graph from previous block
        model.zero_grad(set_to_none=True)
        loss_fn = loss_fns[config.unlearning_loss_fn]
        loss = loss_fn(output, f_input_ids, clip_at=0)
        loss.backward()

        # ! unlearning step with masking
        grad_norm = sum(p.grad.norm() ** 2 for p in interven_params) ** 0.5
        for p in interven_params:
            mask = p.retain_acc.sign() == p.grad.sign()
            p.grad *= mask
            # todo also times normalization factor once i add it
            p.grad *= total_interven_numel**0.5 / grad_norm
            p.data -= h.unlearning_rate * p.grad

        # ! eval current loss
        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done // 30 > _eval_counter:
            _eval_counter += 1
            with peft_model.disable_adapter():
                eval_(model, f_eval, r_eval, allowed_r_loss, _passes_done)

    # ! remove lora
    for lora_index in range(config.lora_amount):
        peft_model.delete_adapter(f"adv{lora_index}")

    return model
