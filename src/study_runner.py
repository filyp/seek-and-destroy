# %%
from IPython import get_ipython

# automatically reload all modules
ipython = get_ipython()
if ipython is not None:  # Only runs in IPython environment
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# necessary for determinism:
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"  # less mem but slower

import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.model_operations import relearn
from utils.plots_and_stats import plot_slice_layout
from utils.training import *

# %%

config = SimpleNamespace(
    # method_name="seek_and_destroy",
    method_name="double_loop",
    # target_modules=["dense_4h_to_h"],
    target_modules=["dense_h_to_4h"],
    # ! Model/data configs
    model_id="EleutherAI/pythia-14m",
    retain_set_name="wikitext",
    forget_set_name="python",
    # ! Training constants
    unlearn_steps=480,
    # # if you change this value, remember to delete cached circuits
    # circuit_num_steps=500,
    batch_size=16,
    n_trials=100,
)
relearn_config = SimpleNamespace(
    relearn_steps=60,
    relearn_lr=3e-4,
    relearn_lora_conf=dict(target_modules="all-linear"),
)

pt.set_default_device("cuda")
set_seeds(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
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
allowed_f_loss = res["retain_loss"]

assert config.method_name.replace("_", "").isalpha()
exec(f"from unlearning_methods.{config.method_name} import unlearning_func")


def objective(trial):
    set_seeds(42)
    model = unlearning_func(
        trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
    )

    set_seeds(42)
    forget_losses = relearn(
        model, relearn_config, retain_val_batches, forget_val_batches
    )

    # use min rather than last, in case it anomalously increases
    forget_loss = min(forget_losses)
    return forget_loss


if __name__ == "__main__":
    run_name = ""
    if not run_name:
        assert is_repo_clean()
        run_name = get_first_line_of_last_commit()
    else:
        assert get_dirty_files() == "src/study_runner.py\n"

    study_name = f"{config.unlearn_steps},{relearn_config.relearn_steps},{config.forget_set_name},{run_name}"
    try:
        storage = get_storage()
        # delete_study_if_exists(study_name, storage)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=False,
        )
        script_name = repo_root() / "src" / "study_runner.py"
        save_script_and_attach_logger(script_name, study.study_name)
        study.set_metric_names(["forget_loss"])
        study.set_user_attr("commit_hash", commit_hash())
        study.set_user_attr("is_repo_clean", is_repo_clean())
        for k, v in config.__dict__.items():
            study.set_user_attr(k, v)
        study.optimize(objective, n_trials=config.n_trials)
    except KeyboardInterrupt:
        pass

    # study = get_last_study()
    plot_slice_layout(study)
    make_sure_optimal_values_are_not_near_range_edges(study)
    get_stats_from_last_n_trials(study, n=10)
