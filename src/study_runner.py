# %%
# %load_ext autoreload
# %autoreload 2  # Automatically reload all modules

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
from utils.git_and_reproducibility import is_repo_clean
from utils.model_operations import relearn
from utils.plots_and_stats import plot_slice_layout
from utils.training import eval_, run_study, set_seeds

config = SimpleNamespace(
    method_name="seek_and_destroy",
    # method_name="negative_entropy",
    # Model/data configs
    model_id="EleutherAI/pythia-14m",
    retain_set_name="wikitext",
    forget_set_name="python",
    # Training constants
    unlearn_steps=300,
    batch_size=16,
    n_trials=1000,
)
relearn_config = SimpleNamespace(
    relearn_steps=100,
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
allowed_f_loss = res["retain_loss"] + 0.1

# %%
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


# %%
# code = Path(__file__).read_bytes()
# code_hash = hashlib.sha256(code).hexdigest()[:4]

# assert is_repo_clean()
study = run_study(
    objective,
    config,
    __file__,
    f"{config.unlearn_steps},{relearn_config.relearn_steps},{config.method_name},{config.forget_set_name},relearn_without_lora",
    delete_existing=False,
    load_if_exists=True,
)

plot_slice_layout(study)

# make sure the value is not in the top or bottom 10% of the range, logarithmically
for param_name, value in study.best_trial.params.items():
    min_ = min(trial.params[param_name] for trial in study.trials)
    max_ = max(trial.params[param_name] for trial in study.trials)
    min_log = np.log(min_)
    max_log = np.log(max_)
    value_log = np.log(value)
    if value_log < min_log + 0.1 * (max_log - min_log):
        print(f"WARNING: {param_name} is in the bottom 10% of the range in best trial")
        print(f"range: {min_} - {max_}, value: {value}")
    if value_log > max_log - 0.1 * (max_log - min_log):
        print(f"WARNING: {param_name} is in the top 10% of the range in best trial")
        print(f"range: {min_} - {max_}, value: {value}")
