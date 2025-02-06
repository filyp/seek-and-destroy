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

import json
import logging
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
)["retain_loss"] + 0.1

# %%
db_url = json.load(open(repo_root() / "secret.json"))["db_url"]
storage = get_storage(db_url)

multistudy_name = Path(config_path).stem

# for variant_name in full_config["variants"]:
# variant_name = "neg_cross_entropy_loss"
variant_name = "no_masking"
study_name = (
    f"{config.unlearn_steps},{relearn_config.relearn_steps},"
    f"{config.forget_set_name}"
    f"|{multistudy_name}|{variant_name}"
)
study = optuna.load_study(study_name=study_name, storage=storage)
trials = study.get_trials()

# %%
ok_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"all_trials={len(trials)}, ok_trials={len(ok_trials)}, {study.study_name}")
last_trial = ok_trials[-1]
# best_trial = study.best_trial

# %%
model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=pt.bfloat16)

accuracies = []
accuracy = eval_on_wmdp(model)
accuracies.append(accuracy)
print(f"accuracy={accuracy}")

# %%
for i in range(5):
    set_seeds(42)
    config.unlearn_steps = 120
    model = surgical_irreversible_unlearning(
        SimpleNamespace(**last_trial.params),
        config,
        retain_batches,
        forget_batches,
        f_eval,
        r_eval,
        allowed_r_loss=float("inf"),
        model=model,
        soft_threshold=allowed_r_loss,
    )

    accuracy = eval_on_wmdp(model)
    accuracies.append(accuracy)
    print(f"accuracy={accuracy}")

# %%
for i in range(10):
    set_seeds(42)
    config.relearn_steps = 120
    forget_losses = relearn_with_retain(
        model, relearn_config, retain_val_batches, forget_val_batches
    )
    # use min rather than last, in case it anomalously increases
    # forget_loss = min(forget_losses)
    # print(f"forget_loss={forget_loss}")

    accuracy = eval_on_wmdp(model)
    accuracies.append(accuracy)
    print(f"accuracy={accuracy}")


# %%
# plot
plt.plot(accuracies)
plt.show()


# %%
