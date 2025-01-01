# %%
from _common_init import *

config.ret_lora_config = dict(lora_dropout=0.05, target_modules="all-linear")
config.use_ret_lora = True
config.disruption_score_warmup = 10

_circuit_dir = repo_root() / "circuits" / config.model_id.replace("/", "_")
_circuit_name = f"{config.forget_set_name}_correct_logit.pt"
circuit = pt.load(_circuit_dir / _circuit_name, weights_only=True)

best_model_path = repo_root() / "models" / "best_model.pt"
best_value = 0


# %%
def objective(trial):
    global best_value
    # ! parameters
    quantile = trial.suggest_float("quantile", 0.0001, 0.01, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.0005, 0.01, log=True)
    retaining_rate = trial.suggest_float("retaining_rate", 0.0003, 0.001, log=True)

    retain_amp = trial.suggest_float("retain_amp", 1.2, 2.0)
    forget_amp = trial.suggest_float("forget_amp", 0.8, 1.2)
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.0, 0.5)
    ret_lora_rank = trial.suggest_int("ret_lora_rank", 1, 10)

    # prepare data iterators
    retain_iter = iter(retain_batches)
    # load model - for speed we could also do: model = deepcopy(base_model)
    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    # get params to intervene on and initialize disruption scores
    target_modules = ["dense_4h_to_h", "dense"]
    interven_params = []
    for name, param in model.named_parameters():
        if any(f"{m}.weight" in name for m in target_modules):
            interven_params.append(param)
            # initialize disruption scores
            param.disruption_score = pt.zeros_like(param)
            # initialize to_forget
            param.to_forget = circuit[name]

    # add lora
    ret_lora_c = LoraConfig(r=ret_lora_rank, **config.ret_lora_config)
    peft_model = get_peft_model(model, ret_lora_c, adapter_name="ret_lora", mixed=True)
    model = peft_model.model
    # require grad for all params despite having lora
    for param in model.parameters():
        param.requires_grad = True

    # initialize optimizers
    _ret_lora_params = [p for n, p in model.named_parameters() if ".ret_lora." in n]
    ret_optimizer = pt.optim.SGD(_ret_lora_params, lr=retaining_rate)
    base_optimizer = pt.optim.SGD(interven_params, lr=unlearning_rate)

    # initialize mask
    mask_fn = lambda param: param.disruption_score / param.to_forget.abs() ** forget_amp

    # ! unlearning loop
    res = {}
    results = []
    logging.info("step      base_f      base_r")
    for step in range(1, 1 + config.unlearn_steps):
        model.train()
        r_input_ids = next(retain_iter)

        # ! update disruption scores
        model.zero_grad(set_to_none=True)
        loss = loss_fns["cross_entropy"](model(r_input_ids), r_input_ids)
        loss.backward()
        for param in interven_params:
            param.disruption_score *= disruption_score_decay
            param.disruption_score += param.grad.abs() ** retain_amp
        if step <= config.disruption_score_warmup:
            continue
        if config.use_ret_lora:
            ret_optimizer.param_groups[0]["lr"] = retaining_rate
            ret_optimizer.step()
        else:
            base_optimizer.param_groups[0]["lr"] = unlearning_rate
            base_optimizer.step()

        # todo maybe simplify further and get rid of this - just terminate if it crosses
        if res.get("retain_loss_ok", True):
            # it it's unacceptable, we only retain, not unlearn
            # ! unlearn on the base model
            # get threshold
            final_scores = [mask_fn(p) for p in interven_params]
            threshold = get_threshold(quantile, final_scores)
            # apply mask
            model.zero_grad(set_to_none=True)
            for param in interven_params:
                mask = mask_fn(param) < threshold
                param.grad = mask * param.to_forget
            base_optimizer.param_groups[0]["lr"] = unlearning_rate
            base_optimizer.step()

        # ! eval current loss
        if step % 10 == 0:
            res = eval_(model, f_eval_batch, r_eval_batch, init_retain, step)
    # trial.set_user_attr("unlearning_results", results)

    # ! eval relearning
    if config.use_ret_lora:
        model_copy = copy_model_and_collapse_loras(peft_model, delete_adv=False)
    else:
        model_copy = deepcopy(model)
    forget_losses = relearn(model_copy, config, retain_val_batches, forget_val_batches)
    # use min rather than last, in case it anomalously increases
    forget_loss = min(forget_losses)

    # save best model
    if forget_loss > best_value:
        logging.info(f"New best model with forget loss {forget_loss}")
        best_value = forget_loss
        model_copy = copy_model_and_collapse_loras(peft_model, delete_adv=False)
        pt.save(model_copy.state_dict(), best_model_path)

    return forget_loss


# %%
run_study(objective, config, __file__, "test")

# # %%
# objective(
#     MockTrial(
#         dict(
#             quantile=0.002,
#             # unlearning_rate=0.001,
#             unlearning_rate=0.002,
#             # retaining_rate=0.0007,
#             retaining_rate=0.0000,
#             retain_amp=1.75,
#             forget_amp=0.95,
#             disruption_score_decay=0.0,
#             ret_lora_rank=9,
#         )
#     )
# )
