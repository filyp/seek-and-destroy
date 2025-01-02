# %%
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace

import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import is_repo_clean
from utils.model_operations import relearn
from utils.plots_and_stats import plot_slice_layout
from utils.training import eval_, run_study

config = SimpleNamespace(
    method_name="seek_and_destroy",
    # Model/data configs
    model_id="EleutherAI/pythia-14m",
    retain_set_name="wikitext",
    forget_set_name="python",
    # Training constants
    unlearn_steps=200,
    batch_size=16,
    n_trials=200,
)
relearn_config = SimpleNamespace(
    relearn_steps=300,
    relearn_lr=1e-4,
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

res = eval_(AutoModelForCausalLM.from_pretrained(config.model_id), f_eval, r_eval)
allowed_f_loss = res["retain_loss"] + 0.1

# %%

assert config.method_name.replace("_", "").isalpha()
exec(f"from unlearning_methods.{config.method_name} import unlearning_func")


def objective(trial):
    model = unlearning_func(
        trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
    )

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
    f"{config.unlearn_steps},{relearn_config.relearn_steps},{config.method_name},{config.forget_set_name}",
    delete_existing=True,
)
# todo? assert that best trial doesn't have any hyperparam in top nor bottor 10% of range

plot_slice_layout(study)
# %%
