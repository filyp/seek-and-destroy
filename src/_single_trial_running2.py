# %%
# %load_ext autoreload
# %autoreload 2
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # less mem but slower
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import is_repo_clean
from utils.model_operations import relearn
from utils.plots_and_stats import plot_slice_layout
from utils.training import MockTrial, eval_, run_study, set_seeds

pt.set_default_device("cuda")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

config = SimpleNamespace(
    # ! Model/data configs
    model_id="EleutherAI/pythia-14m",
    retain_set_name="wikitext",
    forget_set_name="python",
    batch_size=16,
    method_name="seek_and_destroy",
)

# load datasets
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
retain_set = dataset_loaders[config.retain_set_name](tokenizer)
forget_set = dataset_loaders[config.forget_set_name](tokenizer)
retain_batches = CachedBatches(retain_set["train"], config.batch_size)
forget_batches = CachedBatches(forget_set["train"], config.batch_size)
retain_val_batches = CachedBatches(retain_set["validation"], config.batch_size)
forget_val_batches = CachedBatches(forget_set["validation"], config.batch_size)
r_eval = next(iter(retain_val_batches))
f_eval = next(iter(forget_val_batches))

res = eval_(AutoModelForCausalLM.from_pretrained(config.model_id), f_eval, r_eval)
allowed_f_loss = res["retain_loss"] + 0.1

assert config.method_name.replace("_", "").isalpha()
exec(f"from unlearning_methods.{config.method_name} import unlearning_func")

# %%
config.__dict__.update(
    # target_modules=["dense_4h_to_h"],
    target_modules=["dense_h_to_4h"],
    circuit_names=[
        ("normal,neg_cross_entropy", 1),
        # ("normal,stream_activation", 0.9),
        # ("fading_backprop,neg_cross_entropy,0.5", 1.1),
        # ("normal,neg_cross_entropy", 1),
        # ("k_dampens_grad,", 0.5),
        # ("k_dampens_grad_mlp_local,", 0.5),
        # ("k_dampens_grad_neuron_local,", 0.5),
        # ("grad_misalign,", 0.5),
        # ("grad_misalign,only_pos", 0.6),
    ],
    # ! Training constants
    unlearn_steps=100,
)
relearn_config = SimpleNamespace(
    relearn_steps=100,
    relearn_lr=3e-4,
    relearn_lora_conf=dict(target_modules="all-linear"),
)

# %%

# import optuna
# from utils.training import get_storage
# storage = get_storage()
# study_summaries = optuna.study.get_all_study_summaries(storage)
# sorted_studies = sorted(study_summaries, key=lambda s: s.datetime_start)
# latest_study = sorted_studies[-1]
# study = optuna.load_study(study_name=latest_study.study_name, storage=storage)
# study.best_trial.params

set_seeds(42)
params = MockTrial(
    disruption_score_decay=0.9224619659374058,
    grad_pow=0.6221599323237587,
    pos_grad_discard=0.8934652459686687,
    r_quantile=0.18665731358929513,
    unlearning_rate=0.0011330033956548822,
    # cont_lr=0.003,
)
model = unlearning_func(
    params, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
)

# %%

forget_losses = relearn(model, relearn_config, retain_val_batches, forget_val_batches)
print(config.circuit_names)
