# %%
# %load_ext autoreload
# %autoreload 2
from utils.training import MockTrial, eval_, run_study, set_seeds
from utils.plots_and_stats import plot_slice_layout
from utils.model_operations import relearn
from utils.git_and_reproducibility import is_repo_clean
from utils.data_loading import CachedBatches, dataset_loaders
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as pt
import numpy as np
from types import SimpleNamespace
from pathlib import Path
import logging
import hashlib
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # less mem but slower


# %%
config = SimpleNamespace(
    method_name="seek_and_destroy",
    # method_name="negative_entropy",
    # method_name="circuit_breakers",
    loss_fn_name="neg_cross_entropy",
    # Model/data configs
    model_id="EleutherAI/pythia-14m",
    retain_set_name="wikitext",
    forget_set_name="python",
    # retain_set_name="beaver_safe",
    # forget_set_name="cruelty",
    # Training constants
    unlearn_steps=100,
    batch_size=16,
    n_trials=100,
)
relearn_config = SimpleNamespace(
    relearn_steps=50,
    relearn_lr=3e-4,
    relearn_lora_conf=dict(target_modules="all-linear"),
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
retain_set = dataset_loaders[config.retain_set_name](tokenizer)
forget_set = dataset_loaders[config.forget_set_name](tokenizer)
retain_batches = CachedBatches(retain_set["train"], config.batch_size)
forget_batches = CachedBatches(forget_set["train"], config.batch_size)
retain_val_batches = CachedBatches(retain_set["validation"], config.batch_size)
forget_val_batches = CachedBatches(forget_set["validation"], config.batch_size)
r_eval = next(iter(retain_val_batches))
f_eval = next(iter(forget_val_batches))

res = eval_(AutoModelForCausalLM.from_pretrained(
    config.model_id), f_eval, r_eval)
allowed_f_loss = res["retain_loss"] + 0.1

# %%
assert config.method_name.replace("_", "").isalpha()
exec(f"from unlearning_methods.{config.method_name} import unlearning_func")

set_seeds(42)
params = MockTrial(
    # ret_lora_rank=4,
    # alpha=1,
    # use_half=True,
    f_quantile=0.3,
    r_quantile=0.2,
    retaining_rate=0.0003,
    unlearning_rate=0.0003,
    disruption_score_decay=0.95,
    pos_grad_discard_factor=1,
    retain_consistency=0.4,
)
model = unlearning_func(
    params, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
)

# %%
forget_losses = relearn(model, relearn_config,
                        retain_val_batches, forget_val_batches)
print(min(forget_losses))
