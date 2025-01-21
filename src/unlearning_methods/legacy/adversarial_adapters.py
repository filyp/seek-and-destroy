from _common_init import *

config.adv_lora_config = dict(
    # lora_dropout=0.1,
    target_modules=["dense_h_to_4h", "dense_4h_to_h", "query_key_value", "dense"],
)
config.ret_lora_config = dict(
    # lora_dropout=0.1,
    target_modules=["dense_h_to_4h", "dense_4h_to_h", "query_key_value", "dense"],
)

best_model_path = repo_root() / "models" / "best_model.pt"
best_value = 0


# %%
def objective(trial):
    # ! parameters
    quantile = trial.suggest_float("quantile", 0.001, 0.05, log=True)
    adv_lora_lr = trial.suggest_float("adv_lora_lr", 5e-5, 2e-4, log=True)
    ret_lora_lr = trial.suggest_float("ret_lora_lr", 1e-4, 5e-4, log=True)
    unlearn_lr = trial.suggest_float("unlearn_lr", 0.01, 0.2, log=True)
    unlearn_lr_mult = trial.suggest_float("unlearn_lr_mult", 0.99, 1.01)
    forget_amp = trial.suggest_float("forget_amp", 0.8, 1.2)
    retain_amp = trial.suggest_float("retain_amp", 1, 2)
    unl_loss_fn = loss_fns[trial.suggest_categorical("unl_loss_fn", loss_fns.keys())]
    ret_lora_rank = trial.suggest_int("ret_lora_rank", 1, 5)
    ret_lora_dropout = trial.suggest_float("ret_lora_dropout", 0.0, 0.1)
    adv_lora_rank = trial.suggest_int("adv_lora_rank", 1, 5)
    adv_lora_dropout = trial.suggest_float("adv_lora_dropout", 0.0, 0.1)

    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.0, 0.95)
    disruption_score_warmup = trial.suggest_int("disruption_score_warmup", 1, 20)

    unlearn_lr_backoff = trial.suggest_float("unlearn_lr_backoff", 0.9, 1)

    # sample number of steps from loglog distribution, between 10 and 1000
    log_steps = trial.suggest_float("log_steps", 1, 3, log=True)
    num_steps = int(round(10**log_steps, -1))
    if num_steps <= disruption_score_warmup:
        raise optuna.TrialPruned()

    # target_modules = ["dense_4h_to_h", "dense"]  # for python keep these two
    # decide which modules to attack
    target_modules = []
    if trial.suggest_categorical("mod_down_proj", [True, False]):
        target_modules.append("dense_4h_to_h")
    if trial.suggest_categorical("mod_up_proj", [True, False]):
        target_modules.append("dense_h_to_4h")
    if trial.suggest_categorical("mod_attn", [True, False]):
        target_modules.append("query_key_value")
    if trial.suggest_categorical("mod_attn_out", [True, False]):
        target_modules.append("dense")
    if not target_modules:
        raise optuna.TrialPruned()

    logging.info(f"{trial.params}")

    mask_fn = lambda param: param.disruption_score / param.grad.abs() ** forget_amp
    trial.set_user_attr("lora_defeaten", False)
    trial.set_user_attr("retain_broken", False)

    # set_seeds(42)  # note: something is still undeterministic!
    # prepare data iterators
    forget_iter = iter(forget_batches)
    retain_iter = iter(retain_batches)

    # load model (copy from memory for speed)
    # note: to save memory you may want to load from_pretrained instead
    model = deepcopy(base_model)
    # add loras
    adv_lora_c = LoraConfig(
        r=adv_lora_rank, lora_dropout=adv_lora_dropout, **config.adv_lora_config
    )
    peft_model = get_peft_model(model, adv_lora_c, adapter_name="adv_lora", mixed=True)
    model = peft_model.model
    ret_lora_config = LoraConfig(
        r=ret_lora_rank, lora_dropout=ret_lora_dropout, **config.ret_lora_config
    )
    peft_model.add_adapter("ret_lora", ret_lora_config)

    interven_params = [
        p
        for n, p in model.named_parameters()
        if any(m + ".base_layer.weight" in n for m in target_modules)
    ]
    adv_lora_params = [p for n, p in model.named_parameters() if ".adv_lora." in n]
    ret_lora_params = [p for n, p in model.named_parameters() if ".ret_lora." in n]

    # initialize optimizers
    # SGD is faster and more predictable than Adam
    base_optimizer = pt.optim.SGD(interven_params, lr=unlearn_lr)
    adv_optimizer = pt.optim.SGD(adv_lora_params, lr=adv_lora_lr)
    ret_optimizer = pt.optim.SGD(ret_lora_params, lr=ret_lora_lr)

    # initialize disruption scores
    for param in interven_params:
        param.disruption_score = pt.zeros_like(param)

    # %
    # ! unlearning loop
    res = {}
    _steps_with_broken_retain = 0
    logging.info("step      base_f      base_r       adv_f      adv_r")
    for step in range(1, 1 + num_steps):
        model.train()
        f_input_ids = next(forget_iter)
        r_input_ids = next(retain_iter)

        # ! retain with helper lora
        peft_model.set_adapter(["ret_lora"])
        only_grad_on(model, interven_params + ret_lora_params)
        model.zero_grad(set_to_none=True)
        loss = loss_fns["cross_entropy"](model(r_input_ids), r_input_ids)
        loss.backward()
        # ! update disruption scores
        for param in interven_params:
            param.disruption_score *= disruption_score_decay
            param.disruption_score += param.grad.abs() ** retain_amp
        if step <= disruption_score_warmup:
            continue
        # model.zero_grad(set_to_none=True)
        # loss = loss_fns["cross_entropy"](model(r_input_ids), r_input_ids)
        # loss.backward()
        ret_optimizer.step()

        # ! unlearn on the base model
        if res.get("base_retain", 0) < init_retain + 0.1:
            _steps_with_broken_retain = 0

            base_optimizer.param_groups[0]["lr"] *= unlearn_lr_mult
            peft_model.set_adapter(["ret_lora", "adv_lora"])
            only_grad_on(model, interven_params)
            model.zero_grad(set_to_none=True)
            loss = unl_loss_fn(model(f_input_ids), f_input_ids)
            loss.backward()
            # ! get threshold
            final_scores = [mask_fn(p) for p in interven_params]
            # note: this may need flipping the quantile
            threshold = get_threshold(quantile, final_scores)
            # ! apply mask
            for param in interven_params:
                mask = mask_fn(param) < threshold
                param.grad *= mask
            # ! normalize gradients
            grad_norm = sum(p.grad.norm() ** 2 for p in interven_params) ** 0.5
            for p in interven_params:
                p.grad /= grad_norm
            base_optimizer.step()
        else:
            _steps_with_broken_retain += 1
            base_optimizer.param_groups[0]["lr"] *= unlearn_lr_backoff
            if step % 10 == 1:
                logging.info(f"step {step} - broken retain")
            if _steps_with_broken_retain > 50:
                logging.error("Retain performance broken for 50 steps")
                trial.set_user_attr("retain_broken", True)
                raise optuna.TrialPruned()

        # ! relearn with adversarial lora
        peft_model.set_adapter(["ret_lora", "adv_lora"])
        only_grad_on(model, adv_lora_params)
        model.zero_grad(set_to_none=True)
        loss = cross_entropy_loss(model(f_input_ids), f_input_ids)
        loss.backward()
        adv_optimizer.step()

        # ! eval
        if step % 10 == 0:
            res = {}
            peft_model.set_adapter(["ret_lora"])
            res["base_forget"] = eval_loss(model, f_eval_batch)
            res["base_retain"] = eval_loss(model, r_eval_batch)
            peft_model.set_adapter(["ret_lora", "adv_lora"])
            res["adv_forget"] = eval_loss(model, f_eval_batch)
            res["adv_retain"] = eval_loss(model, r_eval_batch)

            logging.info(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))

            # prune if base forget loss doesn't improve
            if step >= 30 and res["base_forget"] < init_forget + 0.5:
                logging.info("Forget loss stalled")
                raise optuna.TrialPruned()
            # prune if adversarial lora is defeaten
            if res["adv_forget"] > 50:
                logging.error("Adversarial LoRA defeaten")
                trial.set_user_attr("lora_defeaten", True)
                logging.info(f"Hyperparameters: {trial.params}")
                raise optuna.TrialPruned()
            # prune if nan
            if any(pt.isnan(v) for v in res.values()):
                logging.error("NaN in eval results")
                raise optuna.TrialPruned()

    if res["base_retain"] > init_retain + 0.1:
        logging.info("Retain performance still broken")
        raise optuna.TrialPruned()

    # %
    # ! final bigger eval relearning
    collapsed_model = copy_model_and_collapse_loras(peft_model)
    retain_val_iter = iter(retain_val_batches)
    forget_val_iter = iter(forget_val_batches)
    forget_loss = relearn(collapsed_model, config, retain_val_iter, forget_val_iter)
    # save best model
    if forget_loss > best_value:
        logging.info(f"New best model with forget loss {forget_loss}")
        best_value = forget_loss
        collapsed_model = copy_model_and_collapse_loras(peft_model)
        pt.save(collapsed_model.state_dict(), best_model_path)
    return forget_loss


# %%
run_study(objective, config, __file__, "test")