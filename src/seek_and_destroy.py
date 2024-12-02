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
    forget_set_name="python",
    # Training constants
    unlearn_steps=200,
    batch_size=16,
    # Relearning params
    relearn_steps=20,
    relearn_lr=1e-4,
    eval_batch_size=16,
    relearn_lora_conf=dict(r=3, target_modules="all-linear", lora_dropout=0.05),
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

_circuit_name = f"{config.forget_set_name}_correct_logit.pt"
circuit = pt.load(repo_root() / "circuits" / config.model_id / _circuit_name)


# %%
def objective(trial):
    # ! parameters
    quantile = trial.suggest_float("quantile", 0.0005, 0.003, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.0003, 0.003, log=True)
    relearning_rate = trial.suggest_float("relearning_rate", 0.0001, 0.001, log=True)

    retain_amp = trial.suggest_float("retain_amp", 1.2, 1.4)
    forget_amp = trial.suggest_float("forget_amp", 1, 1.2)
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.0, 0.9)

    # prepare data iterators
    retain_iter = retain_batches.fresh_iterator()
    # load model (copy from memory for speed)
    model = deepcopy(base_model)

    target_modules = ["dense_4h_to_h", "dense"]  # for python keep these two
    interven_params = []
    for name, param in model.named_parameters():
        if any(f"{m}.weight" in name for m in target_modules):
            interven_params.append(param)
            # initialize disruption scores
            param.disruption_score = pt.zeros_like(param)
            # initialize to_forget
            param.to_forget = circuit[name]

    # initialize optimizers
    optimizer = pt.optim.SGD(interven_params)
    # initialize mask
    mask_fn = lambda param: param.disruption_score / param.to_forget.abs() ** forget_amp

    # ! unlearning loop
    res = {}
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
        optimizer.param_groups[0]["lr"] = relearning_rate
        optimizer.step()

        if res.get("retain_loss_acceptable", True):
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
            optimizer.param_groups[0]["lr"] = unlearning_rate
            optimizer.step()

        # ! eval current loss
        if step % 10 == 0:
            res = {}
            res["forget_loss"] = eval_loss(model, f_eval_batch)
            res["retain_loss"] = eval_loss(model, r_eval_batch)
            res["retain_loss_acceptable"] = res["retain_loss"] < init_retain + 0.1
            logging.info(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))
            assert not any(pt.isnan(v) for v in res.values())

        # ! eval loss after short relearning
        if step % 50 == 0:
            forget_loss = relearn(
                deepcopy(model),
                config,
                retain_val_batches.fresh_iterator(),
                forget_val_batches.fresh_iterator(),
            )
            # prune if trial isn't promising
            trial.report(forget_loss, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return forget_loss


# %%
info = f"S&D,{config.forget_set_name},{config.relearn_steps}rs"
study_name = f"{info},pruning"
if __name__ == "__main__":
    assert is_repo_clean()
    study = optuna.create_study(
        study_name=study_name,
        storage=get_storage(),
        direction="maximize",
        # load_if_exists=True,
        pruner=optuna.pruners.PercentilePruner(75)
    )
    save_script_and_attach_logger(__file__, study.study_name)
    study.set_metric_names(["forget_loss"])
    study.set_user_attr("commit_hash", commit_hash())
    for k, v in config.__dict__.items():
        study.set_user_attr(k, v)
    study.optimize(objective, n_trials=10000)
