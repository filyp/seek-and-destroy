# %%
import logging
from datetime import datetime
from types import SimpleNamespace

import optuna
import torch as pt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.model_operations import *
from utils.training import MockTrial, loss_fns, set_seeds

config = SimpleNamespace(
    # Model/data configs
    model_id="EleutherAI/pythia-14m",
    # model_id="EleutherAI/pythia-70m",
    # model_id="HuggingFaceTB/SmolLM-135M",
    # forget_set_name="python",
    forget_set_name="oscar_pl",
    adv_lora_config=dict(
        # r=4,
        lora_dropout=0.1,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    ),
    ret_lora_config=dict(
        r=4,
        lora_dropout=0.1,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    ),
    # Training constants
    unlearn_steps=50,
    batch_size=16,
    # Relearning params
    relearn_steps=30,
    relearn_lr=3e-4,
    eval_batch_size=16,
    relearn_lora_conf=dict(r=1, target_modules=["dense_h_to_4h"], lora_dropout=0.1),
    # relearn_lora_conf=dict(r=1, target_modules=["up_proj"], lora_dropout=0.1),
    # Default tunable params
    disruption_score_warmup=10,
)

pt.set_default_device("cuda")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# load datasets
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
retain_set = dataset_loaders["wikitext"](tokenizer)
forget_set = dataset_loaders[config.forget_set_name](tokenizer)
retain_batches = CachedBatches(retain_set["train"], config.batch_size)
forget_batches = CachedBatches(forget_set["train"], config.batch_size)
retain_val_batches = CachedBatches(retain_set["validation"], config.eval_batch_size)
forget_val_batches = CachedBatches(forget_set["validation"], config.eval_batch_size)
r_eval_batch = next(retain_val_batches.fresh_iterator())
f_eval_batch = next(forget_val_batches.fresh_iterator())

base_model = AutoModelForCausalLM.from_pretrained(config.model_id)
init_forget = eval_loss(base_model, f_eval_batch)
init_retain = eval_loss(base_model, r_eval_batch)
logging.info(f"init forget: {init_forget:6.2f}    init retain: {init_retain:6.2f}")


# %%
def objective(trial):
    # ! parameters
    quantile = trial.suggest_float("quantile", 0.0001, 0.01, log=True)
    adv_lora_lr = trial.suggest_float("adv_lora_lr", 1e-4, 1e-3, log=True)
    ret_lora_lr = trial.suggest_float("ret_lora_lr", 1e-5, 1e-3, log=True)
    unlearn_lr = trial.suggest_float("unlearn_lr", 0.003, 0.3, log=True)
    unlearn_lr_mult = trial.suggest_float("unlearn_lr_mult", 0.99, 1.01)
    forget_amp = 1  # trial.suggest_float("forget_amp", 0.5, 1.5)
    retain_amp = 1.6  # trial.suggest_float("retain_amp", 1.5, 1.7)
    # unl_loss_fn = loss_fns[trial.suggest_categorical("unl_loss_fn", loss_fns.keys())]
    unl_loss_fn = loss_fns["correct_logit"]
    adv_lora_rank = trial.suggest_int("adv_lora_rank", 1, 3)

    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.8, 0.99)

    mask_fn = lambda param: param.disruption_score / param.grad.abs() ** forget_amp
    trial.set_user_attr("lora_defeaten", False)
    trial.set_user_attr("retain_broken", False)

    set_seeds(42)  # note: something is still undeterministic!
    # prepare data iterators
    forget_iter = forget_batches.fresh_iterator()
    retain_iter = retain_batches.fresh_iterator()

    # load model
    model = deepcopy(base_model)
    # add loras
    adv_lora_config = LoraConfig(r=adv_lora_rank, **config.adv_lora_config)
    peft_model = get_peft_model(
        model, adv_lora_config, adapter_name="adv_lora", mixed=True
    )
    model = peft_model.model
    ret_lora_config = LoraConfig(**config.ret_lora_config)
    peft_model.add_adapter("ret_lora", ret_lora_config)

    interven_params = [p for n, p in model.named_parameters() if ".base_layer.weight" in n]  # fmt: skip
    adv_lora_params = [p for n, p in model.named_parameters() if ".adv_lora." in n]
    ret_lora_params = [p for n, p in model.named_parameters() if ".ret_lora." in n]

    # initialize optimizers
    base_optimizer = pt.optim.SGD(interven_params, lr=unlearn_lr)
    adv_optimizer = pt.optim.SGD(adv_lora_params, lr=adv_lora_lr)
    ret_optimizer = pt.optim.SGD(ret_lora_params, lr=ret_lora_lr)

    # initialize disruption scores
    for param in interven_params:
        param.disruption_score = pt.zeros_like(param)

    # %
    # ! unlearning loop
    logging.info("step      base_f      base_r       adv_f      adv_r")
    for step in range(1, 1 + config.unlearn_steps):
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
        if step <= config.disruption_score_warmup:
            continue
        # model.zero_grad(set_to_none=True)
        # loss = loss_fns["cross_entropy"](model(r_input_ids), r_input_ids)
        # loss.backward()
        ret_optimizer.step()

        # ! unlearn on the base model
        base_optimizer.param_groups[0]["lr"] *= unlearn_lr_mult
        peft_model.set_adapter(["ret_lora", "adv_lora"])
        only_grad_on(model, interven_params)
        model.zero_grad(set_to_none=True)
        loss = unl_loss_fn(model(f_input_ids), f_input_ids)
        loss.backward()
        # ! get threshold
        final_scores = [mask_fn(p) for p in interven_params]
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

            # prune if retain performance broken
            if res["base_retain"] > init_retain + 0.1:
                logging.error("Retain performance broken")
                trial.set_user_attr("retain_broken", True)
                raise optuna.TrialPruned()
            # prune if base forget loss doesn't improve
            if step >= 20 and res["base_forget"] < init_forget + 1:
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

    # %
    # ! final bigger eval relearning
    collapsed_model = copy_model_and_collapse_loras(peft_model)
    retain_val_iter = retain_val_batches.fresh_iterator()
    forget_val_iter = forget_val_batches.fresh_iterator()
    forget_loss = relearn(collapsed_model, config, retain_val_iter, forget_val_iter)
    return forget_loss


# %%
if __name__ == "__main__":
    dd_mm = datetime.now().strftime("%d.%m")
    study = optuna.create_study(
        study_name=f"{dd_mm},pl,dont_terminate_on_alora_break,better_range3",
        storage=f"sqlite:///{repo_root() / "results" / "db.sqlite3"}",
        direction="maximize",
        # load_if_exists=True,
    )
    save_script_and_attach_logger(__file__, study.study_name)
    study.set_metric_names(["forget_loss"])
    study.set_user_attr("commit_hash", commit_hash())
    study.set_user_attr("is_repo_clean", is_repo_clean())
    for k, v in config.__dict__.items():
        study.set_user_attr(k, v)
    study.optimize(objective, n_trials=10000)
