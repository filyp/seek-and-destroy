# %%
import logging
from types import SimpleNamespace

import optuna
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.model_operations import *

config = SimpleNamespace(
    # Model/data configs
    model_id="EleutherAI/pythia-14m",
    forget_set_name="python",
    batch_size=16,
    eval_batch_size=16,
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
retain_set = dataset_loaders["wikitext"](tokenizer)
forget_set = dataset_loaders[config.forget_set_name](tokenizer)
retain_batches = CachedBatches(retain_set["train"], config.batch_size)
forget_batches = CachedBatches(forget_set["train"], config.batch_size)
retain_val_batches = CachedBatches(retain_set["validation"], config.eval_batch_size)
forget_val_batches = CachedBatches(forget_set["validation"], config.eval_batch_size)
r_eval_batch = next(retain_val_batches.fresh_iterator())
f_eval_batch = next(forget_val_batches.fresh_iterator())

# %%
best_model_path = repo_root() / "models" / "best_model.pt"
best_model = AutoModelForCausalLM.from_pretrained(config.model_id)
best_model.load_state_dict(pt.load(best_model_path, weights_only=True))

# %%
def objective(trial):
    relearning_config = SimpleNamespace(
        # Relearning params
        relearn_steps=500,
        eval_batch_size=16,
        relearn_lr=trial.suggest_float("relearn_lr", 1e-5, 3e-3, log=True),
        relearn_lora_conf=dict(
            r=trial.suggest_int("relearn_lora_rank", 1, 16),
            lora_alpha=trial.suggest_int("relearn_lora_alpha", 1, 16),
            lora_dropout=trial.suggest_float("relearn_lora_dropout", 0.0, 0.1),
            target_modules="all-linear",
        ),
    )
    forget_losses = relearn(
        deepcopy(best_model),
        relearning_config,
        retain_val_batches.fresh_iterator(),
        forget_val_batches.fresh_iterator(),
    )
    return forget_losses[-1]

study = optuna.create_study(
    study_name="relearning,500steps",
    storage=get_storage(),
    direction="minimize",
)
study.set_metric_names(["forget_loss"])
study.optimize(objective, n_trials=100)
