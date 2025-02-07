# %%
import datetime

from IPython import get_ipython

import wandb

# automatically reload all modules
ipython = get_ipython()
if ipython is not None:  # Only runs in IPython environment
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %%
# necessary for determinism:
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"  # less mem but slower

import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import optuna
import torch as pt
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_methods.surgical_irreversible_unlearning import (
    surgical_irreversible_unlearning,
)
from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import get_storage, repo_root
from utils.model_operations import relearn_with_retain
from utils.training import eval_, set_seeds
from utils.wmdp_eval import eval_on_wmdp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# style default
plt.style.use("default")

# %%
# load YAML configuration
config_path = repo_root() / "configs" / "ablations_and_loss2,llama32,pile-bio.yaml"
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

config = SimpleNamespace(**full_config["general_config"])
relearn_config = SimpleNamespace(**full_config["relearn_config"])

db_url = json.load(open(repo_root() / "secret.json"))["db_url"]
storage = get_storage(db_url)

# %%

pt.set_default_device("cuda")

# load datasets
set_seeds(42)
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
retain_set = dataset_loaders[config.retain_set_name](tokenizer)
forget_set = dataset_loaders[config.forget_set_name](tokenizer)
retain_batches = CachedBatches(retain_set["train"], config.batch_size)
forget_batches = CachedBatches(forget_set["train"], config.batch_size)
retain_val_batches = CachedBatches(retain_set["validation"], config.batch_size)
forget_val_batches = CachedBatches(forget_set["validation"], config.batch_size)
r_eval = next(iter(retain_val_batches))
f_eval = next(iter(forget_val_batches))

_init_res = eval_(AutoModelForCausalLM.from_pretrained(config.model_id), f_eval, r_eval)
allowed_r_loss = _init_res["retain_loss"] + 0.05

# # for variant_name in full_config["variants"]:
# variant_name = "neg_cross_entropy_loss"
# # variant_name = "no_masking"
# study_name = (
#     f"{config.unlearn_steps},{relearn_config.relearn_steps},"
#     f"{config.forget_set_name}"
#     f"|{multistudy_name}|{variant_name}"
# )
# study = optuna.load_study(study_name=study_name, storage=storage)
# trials = study.get_trials()

# # %%
# ok_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
# print(f"all_trials={len(trials)}, ok_trials={len(ok_trials)}, {study.study_name}")
# last_trial = ok_trials[-1]
# # best_trial = study.best_trial

# %%
# hyperparams = SimpleNamespace(**last_trial.params)
hyperparams = SimpleNamespace(
    adv_decay=1,
    adv_lr=0.001,
    fork_every_n_loops=48,
    retain_momentum=0.95,
    retaining_rate=0.001,
    unlearning_rate=1e-5,
)

# config.unlearning_loss_fn = "correct_logit_minus_avg"
# config.unlearning_loss_fn = "neg_entropy"

variants = dict(
    neg_cross_entropy_loss=dict(
        unlearning_loss_fn="neg_cross_entropy",
        # unlearning_rate=8e-6,
    ),
    neg_entropy_loss=dict(
        unlearning_loss_fn="neg_entropy",
        # unlearning_rate=5e-6,
    ),
    # logit_loss=dict(
    #   unlearning_loss_fn="correct_logit_minus_avg"
    # ),
    # ! ablations
    no_masking=dict(
        use_masking=False,
        unlearning_rate=5e-7,
    ),
    no_adversary=dict(
        train_adversary=False,
    ),
    no_normalization=dict(
        normalize_grads=False,
        unlearning_rate=5e-2,
    ),
    TAR2=dict(
        # it also has and target modules
        unlearning_loss_fn="neg_entropy",
        use_masking=False,
        retain_momentum=0,
        additional_param_name="rep_eng_retain_lr",
        square_norm=True,
        additional_param=1,
        normalize_grads=False,
    ),
)

# %%

stamp = datetime.datetime.now().strftime("%H-%M")
for variant_name, variant_config in variants.items():
    _config = deepcopy(config.__dict__) | variant_config
    _hyperparams = deepcopy(hyperparams.__dict__) | variant_config
    print(f"variant_name={variant_name}")
    print(f"config={_config}")
    print(f"hyperparams={_hyperparams}")
    _config = SimpleNamespace(**_config)
    _hyperparams = SimpleNamespace(**_hyperparams)

    model = AutoModelForCausalLM.from_pretrained(
        _config.model_id, torch_dtype=pt.bfloat16
    )

    wandb.init(project="wmdp-eval", name=f"{stamp}-{variant_name}")
    accuracy = eval_on_wmdp(model)
    wandb.log(_init_res | {"wmdp_accuracy": accuracy}, step=0)

    set_seeds(42)
    _config.unlearn_steps = 2400
    model = surgical_irreversible_unlearning(
        _hyperparams,
        _config,
        retain_batches,
        forget_batches,
        f_eval,
        r_eval,
        allowed_r_loss=float("inf"),
        model=model,
        soft_threshold=allowed_r_loss,
        eval_wmdp_every=120,
    )

    set_seeds(42)
    relearn_config.relearn_steps = 1200
    forget_losses = relearn_with_retain(
        model,
        relearn_config,
        retain_val_batches,
        forget_val_batches,
        eval_wmdp_every=120,
        step_offset=_config.unlearn_steps,
    )
