# %%
from _common_init import *

_circuit_dir = repo_root() / "circuits" / config.model_id.replace("/", "_")
_circuit_name = f"{config.forget_set_name}_correct_logit.pt"
circuit = pt.load(_circuit_dir / _circuit_name, weights_only=True)


# %%
def objective(trial):
    # ! parameters
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.00002, 0.001, log=True)
    retaining_rate = trial.suggest_float("retaining_rate", 0.00001, 0.0003, log=True)
    pos_grad_discard_factor = trial.suggest_float("pos_grad_discard_factor", 0.0, 1.0)
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.0, 1.0)
    quantile = trial.suggest_float("quantile", 0.001, 0.1, log=True)
    retain_consistency_factor = 1 #trial.suggest_float("retain_consistency_factor", 0.0, 1.0)

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
            p.disruption_score = pt.zeros_like(p.data)
            # initialize to_forget
            p.to_forget = circuit[name]
            # require grad
            p.requires_grad = True

    # ! unlearning loop
    res = {}
    logging.info("step      base_f      base_r")
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
            p.disruption_score += (1 - disruption_score_decay) * grad
            
            rel_scores = p.disruption_score / p.to_forget
            # assert (rel_scores <= 0).all(), rel_scores
            # we want the highest ones!
            threshold = get_threshold(1 - quantile, [rel_scores])

            mask = rel_scores > threshold
            if res.get("retain_loss_ok", True):
                p.data -= mask * unlearning_rate * p.to_forget

            p.grad[p.grad != p.to_forget] *= retain_consistency_factor
            p.data -= retaining_rate * p.grad

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
run_study(objective, config, __file__, "asymmetric")

# %%