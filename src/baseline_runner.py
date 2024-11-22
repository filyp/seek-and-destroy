# %%
import logging
from types import SimpleNamespace

import optuna
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.baseline_losses import baseline_loss_fns
from utils.data_loading import dataset_loaders
from utils.git import add_tag_to_current_commit, commit_hash, is_repo_clean
from utils.model_operations import *
from utils.training import loss_fns, set_seeds

# todo also a variant of this file with relearning LoRA, for a more fair comparison

# configuration for model, training, and relearning parameters
config = SimpleNamespace(
    # model/data configs
    model_id="EleutherAI/pythia-14m",
    forget_set_name="python",
    # ! crucial training params
    unlearn_loss_fn="neg_cross_entropy",
    retain_loss_fn="cross_entropy",
    optimizer="SGD",  # SGD | Adam
    # training constants
    unlearn_steps=100,
    batch_size=16,
    eval_batch_size=32,
    # relearning params
    relearn_steps=50,
    relearn_lr=3e-4,
    relearn_batch_size=16,
    relearn_lora_conf=dict(r=1, target_modules=["dense_h_to_4h"]),
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

f_eval_batch = get_batch(iter(forget_set["validation"]), config.eval_batch_size)
r_eval_batch = get_batch(iter(retain_set["validation"]), config.eval_batch_size)

model = AutoModelForCausalLM.from_pretrained(config.model_id)
init_forget = eval_loss(model, f_eval_batch)
init_retain = eval_loss(model, r_eval_batch)
logging.info(f"init forget: {init_forget:6.2f}    init retain: {init_retain:6.2f}")

optimizers = dict(SGD=pt.optim.SGD, Adam=pt.optim.Adam)

loss_fns = {**baseline_loss_fns, **loss_fns}


# %%
def objective(trial):
    """optimization objective function that performs model unlearning and relearning.
    tries to maximize forgetting while maintaining retention performance."""

    # sample hyperparameters from the trial
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # ratio between forget and retain learning rates
    f_r_lr_ratio = trial.suggest_float("f_r_lr_ratio", 0, 1)

    # initialize model and data iterators
    set_seeds(42)  # note: something is still undeterministic!
    forget_iter = looping_iter(forget_set["train"])
    retain_iter = looping_iter(retain_set["train"])

    # load fresh model for each trial
    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    optimizer = optimizers[config.optimizer](model.parameters(), lr=lr)

    # main unlearning loop: alternates between forgetting and retention steps
    logging.info("step      forget      retain")
    for step in range(1, 1 + config.unlearn_steps):
        model.train()

        # get batches for both forget and retain datasets
        f_input_ids = get_batch(forget_iter, config.batch_size)
        r_input_ids = get_batch(retain_iter, config.batch_size)

        model.zero_grad(set_to_none=True)

        # forgetting step: maximize loss on forget set
        output = model(f_input_ids, output_hidden_states=True)
        loss_fn = loss_fns[config.unlearn_loss_fn]
        loss = loss_fn(output, f_input_ids)  # negative loss to maximize
        loss *= f_r_lr_ratio
        loss.backward()

        # retention step: minimize loss on retain set
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss *= 1 - f_r_lr_ratio
        loss.backward()

        optimizer.step()

        # evaluate model performance every 10 steps
        if step % 10 == 0:
            res = {}
            res["forget"] = eval_loss(model, f_eval_batch)
            res["retain"] = eval_loss(model, r_eval_batch)
            logging.info(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))

            if res["retain"] > init_retain + 0.1:
                logging.error("Retain performance broken")
                trial.set_user_attr("retain_broken", True)
                # raise optuna.TrialPruned()
                return init_forget - 0.3

    # final evaluation: test relearning capability
    copied_model = deepcopy(model)
    forget_loss = relearn(copied_model, config, forget_set, retain_set)
    return forget_loss


# %%
assert is_repo_clean()
study = optuna.create_study(
    study_name="change_me",
    storage="sqlite:///../results/baselines.sqlite3",
    direction="maximize",
    # load_if_exists=True,  # this allows resuming existing studies
)
add_tag_to_current_commit(study.study_name)
study.set_metric_names(["forget_loss"])
study.set_user_attr("commit_hash", commit_hash())
for k, v in config.__dict__.items():
    study.set_user_attr(k, v)
study.optimize(objective, n_trials=10000)
