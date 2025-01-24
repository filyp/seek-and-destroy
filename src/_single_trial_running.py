# %%
from IPython import get_ipython

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

import logging
from types import SimpleNamespace

import torch as pt
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_methods.tar_masked import tar_masked
from unlearning_methods.tar_masked_lora import tar_masked_lora
from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.model_operations import relearn
from utils.training import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# %%
# load YAML configuration
config_path = repo_root() / "configs" / "pythia_ablation.yaml"
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

hyperparam_ranges = full_config["hyperparams"]
config = full_config["general_config"]

config = SimpleNamespace(**config)
relearn_config = SimpleNamespace(**full_config["relearn_config"])

# custom
# config.train_adversary = False

# %%

print(f"{hyperparam_ranges=}")

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

allowed_f_loss = eval_(
    AutoModelForCausalLM.from_pretrained(config.model_id), f_eval, r_eval
)["retain_loss"]

unlearning_func = dict(
    tar_masked_lora=tar_masked_lora,
    tar_masked=tar_masked,
)[config.method_name]


# construct hyperparams
hyperparams = {
    hp_name: (low + high) / 2 for hp_name, (low, high, log) in hyperparam_ranges.items()
}
hyperparams = SimpleNamespace(**hyperparams)

set_seeds(42)
model = unlearning_func(
    hyperparams,
    config,
    retain_batches,
    forget_batches,
    f_eval,
    r_eval,
    allowed_f_loss,
)

# %%

set_seeds(42)
forget_losses = relearn(
    model, relearn_config, retain_val_batches, forget_val_batches
)

# %%

