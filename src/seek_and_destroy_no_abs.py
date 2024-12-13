# %%
from _common_init import *

_circuit_dir = repo_root() / "circuits" / config.model_id.replace("/", "_")
_circuit_name = f"{config.forget_set_name}_correct_logit.pt"
circuit = pt.load(_circuit_dir / _circuit_name, weights_only=True)


# %%
def objective(trial):
    # ! parameters
    forget_thresh = trial.suggest_float("forget_thresh", 0.001, 1, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.00002, 0.001, log=True)
    retaining_rate = trial.suggest_float("retaining_rate", 0.00001, 0.0003, log=True)
    grad_decay = trial.suggest_float("grad_decay", 0.0, 0.3)
    alpha_thresh = trial.suggest_float("alpha_thresh", 88, 95)
    alpha_low_thresh = trial.suggest_float("alpha_low_thresh", 0, 70)

    # prepare data iterators
    retain_iter = retain_batches.fresh_iterator()
    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    # get params to intervene on and initialize disruption scores
    for p in model.parameters():
        p.requires_grad = False
    target_modules = ["dense_4h_to_h", "dense"]
    interven_params = []
    for name, p in model.named_parameters():
        if any(f"{m}.weight" in name for m in target_modules):
            interven_params.append(p)
            # initialize to_forget
            p.to_forget = circuit[name]
            # require grad
            p.requires_grad = True
    model.zero_grad(set_to_none=True)

    # ! unlearning loop
    res = {}
    logging.info("step      base_f      base_r")
    for step in range(1, 1 + config.unlearn_steps):
        model.train()
        r_input_ids = next(retain_iter)

        # ! unlearn on the base model
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()
        for p in interven_params:
            _forget_big = p.to_forget.abs() > forget_thresh
            alpha = pt.atan2(p.to_forget, p.grad) / math.pi * 180
            alpha = alpha % 180
            mask = (
                (alpha < alpha_thresh)
                .logical_and(_forget_big)
                .logical_and(alpha > alpha_low_thresh)
            )

            if res.get("retain_loss_ok", True):
                p.data -= mask * unlearning_rate * p.to_forget
            p.data -= retaining_rate * p.grad
            p.grad *= grad_decay

        # ! eval current loss
        if step % 10 == 0:
            res = eval_(model, f_eval_batch, r_eval_batch, init_retain, step)

    # ! eval relearning
    model_copy = deepcopy(model)
    forget_losses = relearn(model_copy, config, retain_val_batches, forget_val_batches)
    # use min rather than last, in case it anomalously increases
    forget_loss = min(forget_losses)

    return forget_loss


# %%
run_study(objective, config, __file__, "test")