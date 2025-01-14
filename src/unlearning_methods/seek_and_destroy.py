import logging

import torch as pt
from transformers import AutoModelForCausalLM

from utils.circuit_creation import filter_and_normalize_circuit, get_circuit
from utils.model_operations import get_thresh
from utils.plots_and_stats import layer_vs_pos_neg_sum_plot, visualize_param
from utils.training import cross_entropy_loss, eval_, loss_fns, stream_activation_loss

disruption_score_warmup = 20


def step_with_abs_grad_before_aggregation(model, batch, interven_params, pow_=2):
    # ! note: if we know the signs of to_forget and know they won't change,
    # we could optimize this, to only store one disruption score, to save memory
    def save_input_activation_hook(module, args, output):
        module.weight.input_activations = args[0]

    def abs_grad_calculate(module, grad_input, grad_output):
        in_ = module.weight.input_activations
        in_pos = (in_ * (in_ > 0)).abs() ** pow_
        in_neg = (in_ * (in_ < 0)).abs() ** pow_
        out = grad_output[0]
        out_pos = (out * (out > 0)).abs() ** pow_
        out_neg = (out * (out < 0)).abs() ** pow_
        module.weight.disruption_score_pos += pt.einsum("bti,bto->oi", in_pos, out_pos)
        module.weight.disruption_score_pos += pt.einsum("bti,bto->oi", in_neg, out_neg)
        module.weight.disruption_score_neg += pt.einsum("bti,bto->oi", in_pos, out_neg)
        module.weight.disruption_score_neg += pt.einsum("bti,bto->oi", in_neg, out_pos)

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
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):

    # # ! parameters
    # retaining_rate = 0.0005
    # disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.7, 1)
    # grad_pow = trial.suggest_float("grad_pow", 0.8, 1.2)
    # pos_grad_discard = trial.suggest_float("pos_grad_discard", -0.2, 1.2)
    # r_quantile = trial.suggest_float("r_quantile", 0.05, 0.6, log=True)
    # unlearning_rate = trial.suggest_float("unlearning_rate", 0.0001, 0.001, log=True)
    # # cont_lr = 0.003  # trial.suggest_float("cont_lr", 0.0001, 0.008, log=True)
    # logging.info(f"trial {trial.number} - {trial.params}")

    # ! parameters
    retaining_rate = 0.0005
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.8, 1)
    grad_pow = trial.suggest_float("grad_pow", 0.4, 0.6)
    pos_grad_discard = 1  #trial.suggest_float("pos_grad_discard", 0.5, 1.2)
    r_quantile = trial.suggest_float("r_quantile", 0.1, 0.3, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.0005, 0.0015, log=True)
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
        p.disruption_score_pos = pt.zeros_like(p.data)
        p.disruption_score_neg = pt.zeros_like(p.data)
        # p.disruption_score = pt.zeros_like(p.data)
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

        # # ! retain pass
        # model.zero_grad(set_to_none=True)
        # r_input_ids = next(retain_iter)
        # output = model(r_input_ids)
        # loss = cross_entropy_loss(output, r_input_ids)
        # loss.backward()
        # # ! update disruption scores
        # for p in interven_params:
        #     grad = p.grad.clone().detach()
        #     grad[p.to_forget.sign() == p.grad.sign()] *= pos_grad_discard
        #     p.disruption_score *= disruption_score_decay
        #     p.disruption_score += grad

        step_with_abs_grad_before_aggregation(
            model, next(retain_iter), interven_params, pow_=grad_pow
        )
        for p in interven_params:
            p.disruption_score_pos *= disruption_score_decay
            p.disruption_score_neg *= disruption_score_decay

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

            flipped_disr = (
                p.disruption_score_pos * (p.to_forget.sign() > 0) * pos_grad_discard
                + p.disruption_score_neg * (p.to_forget.sign() < 0) * pos_grad_discard
                - p.disruption_score_pos * (p.to_forget.sign() < 0)
                - p.disruption_score_neg * (p.to_forget.sign() > 0)
            )
            r_threshold = get_thresh(r_quantile, [flipped_disr])
            mask = flipped_disr > r_threshold

            # flipped_disr = p.disruption_score * p.to_forget.sign()
            # r_threshold = get_thresh(r_quantile, [flipped_disr])
            # mask = flipped_disr > r_threshold

            # ! unlearn
            p.data -= mask * unlearning_rate * p.to_forget

            # if step == config.unlearn_steps:
            #     visualize_param(p, p.disruption_score_pos, mask, "pos")
            #     visualize_param(p, p.disruption_score_neg, mask, "neg")
            #     visualize_param(p, flipped_disr, mask, "sum")
        # if step == config.unlearn_steps:
            # layer_vs_pos_neg_sum_plot()

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    return model
