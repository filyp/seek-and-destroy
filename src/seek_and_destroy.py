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
    ret_lora_config=dict(lora_dropout=0.05, target_modules="all-linear"),
    # Training constants
    # unlearn_steps=50,
    batch_size=16,
    # Relearning params
    relearn_steps=50,
    relearn_lr=3e-4,
    eval_batch_size=16,
    relearn_lora_conf=dict(r=1, target_modules="all-linear", lora_dropout=0.05),
    # Default tunable params
    disruption_score_warmup=20,
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
    global best_value
    # ! parameters
    quantile = trial.suggest_float("quantile", 0.001, 0.05, log=True)
    unlearn_lr = trial.suggest_float("unlearn_lr", 0.001, 0.1, log=True)
    retain_amp = trial.suggest_float("retain_amp", 1, 1.4)
    forget_amp = trial.suggest_float("forget_amp", 1, 1.2)
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.0, 0.95)
    unlearn_steps = trial.suggest_int("unlearn_steps", 30, 200, step=10)

    ret_lora_lr = 0  #trial.suggest_float("ret_lora_lr", 1e-4, 5e-4, log=True)
    ret_lora_rank = 3  # trial.suggest_int("ret_lora_rank", 1, 5)

    # prepare data iterators
    retain_iter = retain_batches.fresh_iterator()
    # load model (copy from memory for speed)
    model = deepcopy(base_model)

    # add lora
    ret_lora_c = LoraConfig(r=ret_lora_rank, **config.ret_lora_config)
    peft_model = get_peft_model(model, ret_lora_c, adapter_name="ret_lora", mixed=True)
    model = peft_model.model

    target_modules = ["dense_4h_to_h", "dense"]  # for python keep these two
    interven_params = [
        p
        for n, p in model.named_parameters()
        if any(m + ".base_layer.weight" in n for m in target_modules)
    ]
    ret_lora_params = [p for n, p in model.named_parameters() if ".ret_lora." in n]

    # initialize optimizers
    base_optimizer = pt.optim.SGD(interven_params, lr=unlearn_lr)
    ret_optimizer = pt.optim.SGD(ret_lora_params, lr=ret_lora_lr)
    # initialize disruption scores
    for param in interven_params:
        param.disruption_score = pt.zeros_like(param)

    # initialize mask
    mask_fn = lambda param: param.disruption_score / param.to_forget.abs() ** forget_amp
    for n, p in model.named_parameters():
        if "lora" in n:
            continue
        n = n.replace(".base_layer", "")
        if n in circuit:
            p.to_forget = circuit[n]
    assert all(p.to_forget.shape == p.shape for p in interven_params)

    # ! unlearning loop
    res = {}
    logging.info("step      base_f      base_r")
    for step in range(1, 1 + unlearn_steps):
        model.train()
        r_input_ids = next(retain_iter)

        # ! retain with helper lora
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
        ret_optimizer.step()

        # ! unlearn on the base model
        if res.get("base_retain", 0) < init_retain + 0.1:
            model.zero_grad(set_to_none=True)
            # ! get threshold
            final_scores = [mask_fn(p) for p in interven_params]
            threshold = get_threshold(quantile, final_scores)
            # ! apply mask
            for param in interven_params:
                mask = mask_fn(param) < threshold
                param.grad = mask * param.to_forget
            base_optimizer.step()

        # ! eval
        if step % 10 == 0:
            res = {}
            res["base_forget"] = eval_loss(model, f_eval_batch)
            res["base_retain"] = eval_loss(model, r_eval_batch)

            logging.info(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))

            # prune if nan
            if any(pt.isnan(v) for v in res.values()):
                logging.error("NaN in eval results")
                raise optuna.TrialPruned()
            if res["base_retain"] > init_retain + 0.1:
                logging.info("Retain performance broken")
                raise optuna.TrialPruned()

    # ! final bigger eval relearning
    collapsed_model = copy_model_and_collapse_loras(peft_model, delete_adv=False)
    retain_val_iter = retain_val_batches.fresh_iterator()
    forget_val_iter = forget_val_batches.fresh_iterator()
    forget_loss = relearn(collapsed_model, config, retain_val_iter, forget_val_iter)
    return forget_loss


# %%
info = f"S&D,{config.forget_set_name},{config.relearn_steps}rs"
study_name = f"{info},no_retain_lora"
if __name__ == "__main__":
    assert is_repo_clean()
    study = optuna.create_study(
        study_name=study_name,
        storage=get_storage(),
        direction="maximize",
        # load_if_exists=True,
    )
    save_script_and_attach_logger(__file__, study.study_name)
    study.set_metric_names(["forget_loss"])
    study.set_user_attr("commit_hash", commit_hash())
    for k, v in config.__dict__.items():
        study.set_user_attr(k, v)
    study.optimize(objective, n_trials=10000)