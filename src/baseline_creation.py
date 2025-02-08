"""
This file records baseline values of running only relearning,
without applying any unlearning methods. This is in contrast
to the study_runner.py file, which handles various unlearning
methods and their configurations.
"""

# %%
# # necessary for determinism:
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"  # less mem but slower

import argparse
import logging
import sys
from types import SimpleNamespace

import torch as pt
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.model_operations import relearn, relearn_with_retain
from utils.training import *
from utils.wmdp_eval import eval_on_wmdp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# %%

multistudy_names = [
    "pythia,python",
    "pythia,pile-bio",
    "smol,python",
    "smol,pile-bio",
    "llama32,python",
    "llama32,pile-bio",
]
for multistudy_name in multistudy_names:

    # load YAML configuration
    config_path = (
        repo_root() / "configs" / f"ablations_and_loss2,{multistudy_name}.yaml"
    )

    db_url = json.load(open("../secret.json"))["db_url"]
    storage = get_storage(db_url)
    # storage = get_storage()

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    config = SimpleNamespace(**full_config["general_config"])
    relearn_config = SimpleNamespace(**full_config["relearn_config"])

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

    if config.model_id in ["meta-llama/Llama-3.2-1B"]:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, torch_dtype=pt.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False

    # ! only relearn with no unlearn
    set_seeds(42)
    forget_losses = relearn(
        model, relearn_config, retain_val_batches, forget_val_batches
    )

    out_path = repo_root() / "results" / "baselines" / f"{config_path.stem}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forget_loss = min(forget_losses).item()
    out_path.write_text(str(forget_loss))

# %%
config_path = repo_root() / "configs" / "wmdp3.yaml"


db_url = json.load(open("../secret.json"))["db_url"]
storage = get_storage(db_url)
# storage = get_storage()

with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

config = SimpleNamespace(**full_config["general_config"])
relearn_config = SimpleNamespace(**full_config["relearn_config"])

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

if config.model_id in ["meta-llama/Llama-3.2-1B"]:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, torch_dtype=pt.bfloat16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(config.model_id)
model.config.use_cache = False

_init_res = eval_(AutoModelForCausalLM.from_pretrained(config.model_id), f_eval, r_eval)

# ! only relearn with no unlearn
set_seeds(42)
_ = relearn_with_retain(
    model,
    relearn_config,
    retain_val_batches,
    forget_val_batches,
    eval_wmdp_every=config.eval_wmdp_every,
    step_offset=config.unlearn_steps,
    # this is very rarely needed, but when it happens, it means
    # relearning was broken, so reject
    # (alternative would be to relearn slower, but that's inefficient)
    allowed_r_loss=_init_res["retain_loss"] + config.hard_loss_budget,
)
accuracy = eval_on_wmdp(model)

out_path = repo_root() / "results" / "baselines" / f"{config_path.stem}.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(str(accuracy))

# %%
