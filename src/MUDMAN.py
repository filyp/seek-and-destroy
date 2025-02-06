import logging
from typing import Iterator

import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.loss_fns import cross_entropy_loss
from utils.training import eval_


def MUDMAN(
    model: AutoModelForCausalLM,
    # data
    retain_batches: Iterator[pt.Tensor],
    forget_batches: Iterator[pt.Tensor],
    f_eval: pt.Tensor,
    r_eval: pt.Tensor,
    # configuration
    unlearning_rate: float = 1e-1,
    target_modules: list[str] = ["gate_proj"],
    unlearn_steps: int = 300,
    allowed_r_loss: float = float("inf"),
    fork_every_n_loops: int = 32,
    adv_lr: float = 0.001,
    retaining_rate: float = 0.001,
    retain_momentum: float = 0.9,
) -> AutoModelForCausalLM:
    """Meta-Unlearning with Dynamic Masking, Accumulation and Normalization (MUDMAN).

    This function implements a selective unlearning approach that allows forgetting
    specific data while retaining general performance.
    It uses meta-learning and masking of the most disruptive updates.

    Args:
        model: The neural model to be modified (typically a transformer model)
        retain_batches: Iterator of batches containing data that should be retained
        forget_batches: Iterator of batches containing data that should be forgotten
        f_eval: Evaluation batch for the forget set
        r_eval: Evaluation batch for the retain set
        unlearning_rate: Learning rate for the unlearning updates
        target_modules: List of module names to target for intervention
        unlearn_steps: Number of unlearning steps to perform
        allowed_r_loss: Maximum allowed loss increase on retain set
        fork_every_n_loops: Frequency of adversary forking from the main model
        adv_lr: Learning rate for the adversarial updates
        retaining_rate: Learning rate for retain updates
        retain_momentum: Momentum factor for retain gradient accumulation

    Returns:
        The modified model after unlearning
    """
    model.config.use_cache = False

    # Get params to intervene on
    interven_params = [
        p
        for name, p in model.named_parameters()
        if any(f"{m}.weight" in name for m in target_modules)
    ]

    # Require grads only on intervened params
    for p in model.parameters():
        p.requires_grad = id(p) in [id(p) for p in interven_params]

    # Initialize retain grad accumulators
    for p in interven_params:
        p.retain_acc = pt.zeros_like(p.data)
        p.base_data = p.data.clone().detach()

    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    # ! Unlearning loop
    passes_per_loop = 4
    assert unlearn_steps % passes_per_loop == 0
    for loop_num in range(unlearn_steps // passes_per_loop):
        model.train()
        f_input_ids = next(forget_iter)
        r_input_ids = next(retain_iter)

        if loop_num % fork_every_n_loops == 0:
            for p in interven_params:
                p.adv_data = p.base_data.clone().detach()

        # ! Retain pass
        model.zero_grad()
        for p in interven_params:  # Switch to main model
            p.data = p.base_data
        output = model(r_input_ids)
        cross_entropy_loss(output, r_input_ids).backward()
        for p in interven_params:
            # ! Update disruption scores
            p.retain_acc *= retain_momentum
            p.retain_acc += p.grad * (1 - retain_momentum)
            # ! Retain update
            p.base_data -= retaining_rate * p.retain_acc

        # ! Relearn the adversary
        model.zero_grad()
        for p in interven_params:  # Switch to adversary
            p.data = p.adv_data
        output = model(f_input_ids)
        cross_entropy_loss(output, f_input_ids).backward()
        for p in interven_params:
            # Apply adversary update
            p.adv_data -= adv_lr * p.grad

        # ! Unlearning step with masking
        grad_norm = sum(p.grad.norm() ** 2 for p in interven_params) ** 0.5
        for p in interven_params:
            p.grad *= -1  # For unlearning use the inverted gradient
            p.grad *= p.retain_acc.sign() == p.grad.sign()  # Mask
            p.base_data -= unlearning_rate / grad_norm * p.grad  # Normalize & update

        # ! Eval current loss
        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done % 20 == 0:
            for p in interven_params:  # Switch to main model
                p.data = p.base_data
            eval_(model, f_eval, r_eval, allowed_r_loss, _passes_done)

    return model


if __name__ == "__main__":
    # example usage:
    from types import SimpleNamespace

    from utils.data_loading import CachedBatches, dataset_loaders
    from utils.model_operations import relearn

    model_id = "meta-llama/Llama-3.2-1B"
    retain_set_name = "pile_bio_retain"
    forget_set_name = "pile_bio_forget"
    batch_size = 16

    pt.set_default_device("cuda")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    # load datasets
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    retain_set = dataset_loaders[retain_set_name](tokenizer)
    forget_set = dataset_loaders[forget_set_name](tokenizer)
    retain_batches = CachedBatches(retain_set["train"], batch_size)
    forget_batches = CachedBatches(forget_set["train"], batch_size)
    retain_val_batches = CachedBatches(retain_set["validation"], batch_size)
    forget_val_batches = CachedBatches(forget_set["validation"], batch_size)

    # unlearn
    unlearned_model = MUDMAN(
        model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16),
        retain_batches=retain_batches,
        forget_batches=forget_batches,
        f_eval=next(iter(forget_val_batches)),
        r_eval=next(iter(retain_val_batches)),
        unlearn_steps=300,
    )

    # relearn
    relearn_config = SimpleNamespace(relearn_steps=300, relearn_lr=1e-3)
    forget_losses = relearn(
        unlearned_model, relearn_config, retain_val_batches, forget_val_batches
    )
