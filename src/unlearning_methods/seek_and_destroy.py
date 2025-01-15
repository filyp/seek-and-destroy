import logging
import re

import torch as pt
from transformers import AutoModelForCausalLM

from utils.circuit_creation import filter_and_normalize_circuit, get_circuit
from utils.model_operations import get_thresh
from utils.plots_and_stats import layer_vs_pos_neg_sum_plot, visualize_param
from utils.training import cross_entropy_loss, eval_, loss_fns, stream_activation_loss

disruption_score_warmup = 20


def step_with_abs_grad_before_aggregation(
    model, batch, interven_params, in_pow, out_pow
):
    # ! note: if we know the signs of to_forget and know they won't change,
    # we could optimize this, to only store one disruption score, to save memory
    def save_input_activation_hook(module, args, output):
        module.weight.input_activations = args[0]

    def abs_grad_calculate(module, grad_input, grad_output):
        in_ = module.weight.input_activations
        in_ = (in_.abs() ** in_pow) * in_.sign()
        out = grad_output[0]
        out = (out.abs() ** out_pow) * out.sign()
        module.weight.disruption_score += pt.einsum("bti,bto->oi", in_, out)

    # install hooks
    handles = []
    for module in model.modules():
        # only intervene on the weight of the target modules
        param = getattr(module, "weight", None)
        if id(param) in [id(p) for p in interven_params]:
            handles.append(module.register_full_backward_hook(abs_grad_calculate))
            handles.append(module.register_forward_hook(save_input_activation_hook))

    model.zero_grad(set_to_none=True)
    out = model(batch)
    loss = cross_entropy_loss(out, batch)
    loss.backward()

    # clean up
    for h in handles:
        h.remove()
    for p in interven_params:
        p.input_activations = None


def unlearning_func(
    trial,
    config,
    retain_batches,
    forget_batches,
    f_eval,
    r_eval,
    allowed_f_loss,
    visualize=False,
):
    # ! parameters
    retaining_rate = trial.suggest_float("retaining_rate", 0.0005, 0.001, log=True)
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.8, 1)
    in_pow = trial.suggest_float("in_pow", 0.6, 1.6)
    out_pow = trial.suggest_float("out_pow", 0.4, 0.8)
    # r_quantile = trial.suggest_float("r_quantile", 0.15, 0.25, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.00003, 0.002, log=True)
    # cont_lr = 0.003  # trial.suggest_float("cont_lr", 0.0001, 0.008, log=True)
    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False

    # get params to intervene on and initialize disruption scores
    interven_params = [
        p
        for name, p in model.named_parameters()
        if any(f"{m}.weight" in name for m in config.target_modules)
    ]

    for name, p in model.named_parameters():
        p.param_name = name
        p.requires_grad = False
    for p in interven_params:
        p.requires_grad = True
        p.disruption_score = pt.zeros_like(p.data)
        p.to_forget = pt.zeros_like(p.data)

    # use several circuits, mixed together; load circuits and construct to_forget
    for circuit_name, strength in config.circuit_names:
        circuit = get_circuit(config, forget_batches, circuit_name)
        circuit = filter_and_normalize_circuit(circuit, config.target_modules)
        for p in interven_params:
            if p.param_name in circuit:
                p.to_forget += circuit[p.param_name] * strength
        del circuit

    # optimizer = pt.optim.SGD(interven_params, lr=cont_lr)

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)
    for step in range(1, 1 + config.unlearn_steps):
        model.train()

        # ! retain pass, update disruption scores
        step_with_abs_grad_before_aggregation(
            model, next(retain_iter), interven_params, in_pow, out_pow
        )
        for p in interven_params:
            p.disruption_score *= disruption_score_decay

        # skip the rest of the loop during warmup
        if step <= disruption_score_warmup:
            continue

        # ! retain update
        for p in interven_params:
            p.data -= retaining_rate * p.grad

        # # ! continuous unlearning
        # model.zero_grad(set_to_none=True)
        # f_input_ids = next(forget_iter)
        # output = model(f_input_ids, output_hidden_states=True)
        # loss = stream_activation_loss(output, f_input_ids)
        # loss.backward()
        # optimizer.step()

        # # calculate r_threshold globally
        # flipped_disrs = (
        #     # p.disruption_score * p.to_forget.sign()
        #     p.disruption_score
        #     for p in interven_params
        # )
        # r_threshold = get_thresh(1 - r_quantile, flipped_disrs)

        # Unlearning step with two-stage masking
        for p in interven_params:
            layer_num = int(re.match(r".*layers\.(\d+)", p.param_name).group(1))
            # r_quantile = r_quantiles[layer_num]

            flipped_disr = p.disruption_score * p.to_forget.sign()
            # r_threshold = get_thresh(r_quantile, [flipped_disr])
            r_threshold = 0
            mask = flipped_disr > r_threshold

            # ! unlearn
            p.data -= mask * unlearning_rate * p.to_forget

            if step == config.unlearn_steps and visualize:
                visualize_param(p, p.disruption_score, mask)
        if step == config.unlearn_steps and visualize:
            layer_vs_pos_neg_sum_plot()

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    return model


# ! dump
# # retain pass, update disruption scores
# # version that is not granular, and doesn't require hooks
# model.zero_grad(set_to_none=True)
# r_input_ids = next(retain_iter)
# output = model(r_input_ids)
# loss = cross_entropy_loss(output, r_input_ids)
# loss.backward()
# for p in interven_params:
#     grad = p.grad.clone().detach()
#     p.disruption_score *= disruption_score_decay
#     p.disruption_score += grad
