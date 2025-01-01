# %%
from _common_init import *

_circuit_dir = repo_root() / "circuits" / config.model_id.replace("/", "_")
_circuit_name = f"{config.forget_set_name}_correct_logit.pt"
circuit = pt.load(_circuit_dir / _circuit_name, weights_only=True)

# todo? attack more modules?
# todo? global thresh, rather than per param?

# Add LoRA config
config.ret_lora_config = dict(lora_dropout=0.05, target_modules="all-linear")
config.use_ret_lora = False
config.disruption_score_warmup = 10


# %%
def objective(trial):
    # ! parameters
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.0001, 0.01, log=True)
    retaining_rate = trial.suggest_float("retaining_rate", 0.00001, 0.01, log=True)
    ret_lora_rank = 8
    pos_grad_discard_factor = 0.5
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.7, 1.0)
    f_quantile = trial.suggest_float("f_quantile", 0.0001, 0.1, log=True)
    r_quantile = trial.suggest_float("r_quantile", 0.01, 1, log=True)
    retain_consistency = 0.4
    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    # get params to intervene on and initialize disruption scores
    target_modules = ["dense_4h_to_h", "dense"]
    interven_params = []
    for name, p in model.named_parameters():
        if any(f"{m}.weight" in name for m in target_modules):
            interven_params.append(p)
            p.disruption_score = pt.zeros_like(p.data)
            p.to_forget = circuit[name]

    # Add LoRA
    ret_lora_c = LoraConfig(r=ret_lora_rank, **config.ret_lora_config)
    peft_model = get_peft_model(model, ret_lora_c, adapter_name="ret_lora", mixed=True)
    model = peft_model.model
    # Require grad for all params despite having lora
    if config.use_ret_lora:
        for param in interven_params:
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    # Initialize optimizers
    _ret_lora_params = [p for n, p in model.named_parameters() if ".ret_lora." in n]
    ret_optimizer = pt.optim.SGD(_ret_lora_params, lr=retaining_rate)

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
        if step <= config.disruption_score_warmup:
            continue

        # Get threshold for forgetting
        f_threshold = get_threshold(
            1 - f_quantile, [p.to_forget.abs() for p in interven_params]
        )

        # Unlearning step with two-stage masking
        for p in interven_params:
            # First choose the most important weights for forgetting
            mask = p.to_forget.abs() > f_threshold

            # Then from them, choose the ones least disrupting
            flipped_disr = p.disruption_score * p.to_forget.sign()
            flipped_disr[~mask] = float("-inf")
            if mask.sum() > 0:
                # We want this high too
                d_threshold = get_threshold(1 - r_quantile, [flipped_disr[mask]])
                mask = mask & (flipped_disr > d_threshold)

            p.data -= mask * unlearning_rate * p.to_forget

            if not config.use_ret_lora:
                p.grad[p.grad.sign() != p.to_forget.sign()] *= retain_consistency
                p.data -= retaining_rate * p.grad

        # LoRA retention step
        if config.use_ret_lora:
            ret_optimizer.step()

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval_batch, r_eval_batch, init_retain, step)

    visualize_param(p, mask)

    # Merge and unload helper lora
    peft_model_copy = deepcopy(peft_model)
    # peft_model_copy.set_adapter(["ret_lora"])
    model_copy = peft_model_copy.merge_and_unload()
    del model_copy.peft_config

    forget_losses = relearn(model_copy, config, retain_val_batches, forget_val_batches)
    # use min rather than last, in case it anomalously increases
    forget_loss = min(forget_losses)

    return forget_loss


# %%
run_study(
    objective,
    config,
    __file__,
    "global_r_threshold,thresh_bug_fix",
    assert_clean=False,
    delete_existing=True,
)

# %%
