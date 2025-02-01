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
from copy import deepcopy
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch as pt
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_methods.surgical_irreversible_unlearning import (
    surgical_irreversible_unlearning,
)
from unlearning_methods.surgical_irreversible_unlearning_lora import (
    surgical_irreversible_unlearning_lora,
)
from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.loss_fns import loss_fns
from utils.model_operations import relearn
from utils.training import *

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
# config_path = repo_root() / "configs" / "pythia_python.yaml"
# config_path = repo_root() / "configs" / "smol_target_modules2.yaml"
config_path = repo_root() / "configs" / "smol_cruelty3.yaml"
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

config = full_config["general_config"]

config = SimpleNamespace(**config)
relearn_config = SimpleNamespace(**full_config["relearn_config"])

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

allowed_r_loss = eval_(
    AutoModelForCausalLM.from_pretrained(config.model_id), f_eval, r_eval
)["retain_loss"]

from unlearning_methods.circuit_breakers import circuit_breakers
from unlearning_methods.circuit_breakers_no_lora import circuit_breakers_no_lora

# %%

hyperparams = SimpleNamespace(
    additional_param=None,
    adv_decay=1,
    adv_lr=0.01,
    fork_every_n_loops=36,
    retain_momentum=0.5,
    retaining_rate=0.001,
    # unlearning_rate=0.0001,
    unlearning_rate=0.00001,
)
config.unlearning_loss_fn = "neg_cross_entropy"
# config.unlearning_loss_fn = "correct_logit_minus_avg"
config.train_adversary = False
# note, not training adverary results in higher base_r loss

config.unlearn_steps = 600

set_seeds(42)
pt.cuda.empty_cache()
model = surgical_irreversible_unlearning(
    hyperparams,
    config,
    retain_batches,
    forget_batches,
    f_eval,
    r_eval,
    allowed_r_loss + 0.13,
)


# %%
set_seeds(42)
relearn_config.relearn_steps = 300
# _losses = relearn(
# _losses2 = relearn(
_losses3 = relearn(
    deepcopy(model),
    relearn_config,
    retain_val_batches,
    forget_val_batches,
)

# %%
# plot losses
plt.plot([l.item() for l in _losses], label="CE")
plt.plot([l.item() for l in _losses2], label="CLMA")
plt.plot([l.item() for l in _losses3], label="CE+adv")
plt.legend()
# title
plt.title("SmolLM-135M\nrelearning dynamics")
plt.show()

# %% search for relearn lr
lrs_no_lora = pt.logspace(-2.5, -1.5, 7)
forget_losses_no_lora = []
for lr in lrs_no_lora:
    relearn_config.relearn_lr = lr
    relearn_config.relearn_steps = 60
    # relearn_config.relearn_lora_conf = dict(target_modules="all-linear")

    set_seeds(42)
    _losses = relearn(
        deepcopy(model),
        relearn_config,
        retain_val_batches,
        forget_val_batches,
        use_lora=False,
    )
    forget_losses_no_lora.append(min(_losses))


# %% search for relearn lr
lrs_lora = pt.logspace(-2, 1, 7)
forget_losses_lora = []
for lr in lrs_lora:
    relearn_config.relearn_lr = lr
    relearn_config.relearn_steps = 60
    relearn_config.relearn_lora_conf = dict(target_modules="all-linear")

    set_seeds(42)
    _losses = relearn(
        deepcopy(model),
        relearn_config,
        retain_val_batches,
        forget_val_batches,
        use_lora=True,
    )
    forget_losses_lora.append(min(_losses))

# %%

_lrs_lora = [l.item() for l in lrs_lora]
_lrs_no_lora = [l.item() for l in lrs_no_lora]
_forget_losses_lora = [f.cpu().numpy() for f in forget_losses_lora]
_forget_losses_no_lora = [f.cpu().numpy() for f in forget_losses_no_lora]

plt.plot(_lrs_no_lora, _forget_losses_no_lora, label="no LoRA", marker="o")
plt.plot(_lrs_lora, _forget_losses_lora, label="LoRA", marker="o")
plt.xscale("log")
# plt.yscale("log")
# y lim at 5
plt.ylim(2, 3)
plt.xlabel("relearn lr")
plt.ylabel("forget loss")
# mention lora in title
plt.title("SmolLM-135M\nrelearn lr vs forget loss (LoRA)")
plt.legend()
plt.show()

# %%
_lrs_no_lora
# %%
_forget_losses_lora
# %%
for lr, loss in zip(_lrs_no_lora, _forget_losses_no_lora):
    print(f"lr: {lr:.2e}, loss: {loss:.2f}")
# %%
