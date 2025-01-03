import logging

import torch as pt
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from utils.git_and_reproducibility import repo_root
from utils.model_operations import get_thresh
from utils.training import cross_entropy_loss, eval_, loss_fns, visualize_param

disruption_score_warmup = 20


def get_circuit(
    model_id, forget_set_name, batches, num_steps=2000, loss_fn_name="correct_logit"
):
    # try to load cached circuit
    circuit_dir = repo_root() / "circuits" / model_id.replace("/", "_")
    circuit_path = circuit_dir / f"{forget_set_name}_{loss_fn_name}.pt"
    if circuit_path.exists():
        return pt.load(circuit_path, weights_only=True)

    logging.info("No cached circuit found, creating one")

    # accumulate grads
    loss_fn = loss_fns[loss_fn_name]
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.zero_grad(set_to_none=True)
    for _ in tqdm(range(num_steps)):
        input_ids = next(batches)
        loss = loss_fn(model(input_ids), input_ids)
        loss.backward()
    circuit = {name: param.grad / num_steps for name, param in model.named_parameters()}

    # save circuit
    circuit_dir.mkdir(parents=True, exist_ok=True)
    pt.save(circuit, circuit_path)
    return circuit


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # ! parameters
    f_quantile = trial.suggest_float("f_quantile", 0.03, 0.5, log=True)
    r_quantile = trial.suggest_float("r_quantile", 0.03, 0.5, log=True)
    retaining_rate = trial.suggest_float("retaining_rate", 0.00001, 0.001, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.0001, 0.003, log=True)
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.9, 1.0)
    pos_grad_discard_factor = 0
    retain_consistency = 0
    to_forget_consistency = trial.suggest_float("to_forget_consistency", 0, 2)
    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    # model.config.use_cache = False  # not sure if this is what we want

    if "pythia" in config.model_id:
        target_modules = ["dense_h_to_4h", "dense_4h_to_h", "dense"]
    else:
        raise NotImplementedError(f"Model {config.model_id} not supported")

    # get params to intervene on and initialize disruption scores
    circuit = get_circuit(config.model_id, config.forget_set_name, forget_batches)
    interven_params = []
    for name, p in model.named_parameters():
        if any(f"{m}.weight" in name for m in target_modules):
            interven_params.append(p)
            p.disruption_score = pt.zeros_like(p.data)
            p.to_forget = circuit[name]

            norm = p.to_forget.norm()
            p.to_forget[p.data.sign() != p.to_forget.sign()] *= to_forget_consistency
            # bring back previous norm
            p.to_forget *= norm / p.to_forget.norm()
    del circuit

    # Get threshold for forgetting
    f_threshold = get_thresh(f_quantile, [p.to_forget.abs() for p in interven_params])

    # Require grad for all intervene params
    for param in model.parameters():
        param.requires_grad = False
    for param in interven_params:
        param.requires_grad = True

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    for step in range(1, 1 + config.unlearn_steps):
        model.train()
        r_input_ids = next(retain_iter)

        # ! unlearn on the base model
        model.zero_grad(set_to_none=True)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()

        for p in interven_params:
            grad = p.grad.clone().detach()
            grad[p.to_forget.sign() == p.grad.sign()] *= pos_grad_discard_factor
            p.disruption_score *= disruption_score_decay
            p.disruption_score += grad

        # Skip during warmup
        if step <= disruption_score_warmup:
            continue

        # Unlearning step with two-stage masking
        flipped_scores = []
        for p in interven_params:
            # First choose the most important weights for forgetting
            mask = p.to_forget.abs() > f_threshold
            # Then from them, choose the ones least disrupting
            flipped_disr = p.disruption_score * p.to_forget.sign()
            flipped_scores.append(flipped_disr[mask])
        d_threshold = get_thresh(r_quantile, flipped_scores)

        for p in interven_params:
            # recompute these two
            mask = p.to_forget.abs() > f_threshold
            flipped_disr = p.disruption_score * p.to_forget.sign()

            # ! unlearn
            flipped_disr[~mask] = float("-inf")
            mask = mask & (flipped_disr > d_threshold)
            p.data -= mask * unlearning_rate * p.to_forget

            # ! retain
            p.grad[p.grad.sign() != p.to_forget.sign()] *= retain_consistency
            p.data -= retaining_rate * p.grad

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    # visualize_param(p, mask)
    return model
