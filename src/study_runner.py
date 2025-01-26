# to run all variants one after another:
# python study_runner.py PATH_TO_CONFIG [if_study_exists=fail|delete|load]

# necessary for determinism:
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"  # less mem but slower

import logging
import sys
from types import SimpleNamespace

import torch as pt
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_methods.surgical_tar import surgical_tar
from unlearning_methods.surgical_tar_lora import surgical_tar_lora
from unlearning_methods.tar import tar
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


def run_study(storage, config_path, variant_num, if_study_exists="fail"):
    assert if_study_exists in ["fail", "delete", "load"]
    assert is_repo_clean()

    # load YAML configuration
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    # split into hyperparams and other keys
    variant_name, custom = list(full_config["variants"].items())[variant_num]
    hyperparam_ranges = full_config["hyperparams"] | {
        k: v for k, v in custom.items() if k in full_config["hyperparams"].keys()
    }
    config = full_config["general_config"] | {
        k: v for k, v in custom.items() if k not in full_config["hyperparams"].keys()
    }

    config = SimpleNamespace(**config)
    relearn_config = SimpleNamespace(**full_config["relearn_config"])

    study_name = (
        f"{config.unlearn_steps},{relearn_config.relearn_steps},"
        f"{config.forget_set_name}"
        f"|{Path(config_path).stem}|{variant_name}"
    )
    print(f"{study_name=}")
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
        surgical_tar_lora=surgical_tar_lora,
        surgical_tar=surgical_tar,
        tar=tar,
    )[config.method_name]

    def objective(trial):
        # construct hyperparams
        hyperparams = dict()
        for hp_name, distribution in hyperparam_ranges.items():
            if isinstance(distribution, list):
                low, high, log = distribution
                hyperparams[hp_name] = trial.suggest_float(hp_name, low, high, log=log)
            else:
                hyperparams[hp_name] = distribution
        hyperparams = SimpleNamespace(**hyperparams)
        logging.info(f"trial {trial.number} - {trial.params}")

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

        set_seeds(42)
        forget_losses = relearn(
            model, relearn_config, retain_val_batches, forget_val_batches
        )

        # use min rather than last, in case it anomalously increases
        forget_loss = min(forget_losses)
        return forget_loss

    if if_study_exists == "delete":
        delete_study_if_exists(study_name, storage)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=(if_study_exists == "load"),
    )
    save_file_and_attach_logger(config_path, study.study_name)
    study.set_metric_names(["forget_loss"])
    study.set_user_attr("commit_hash", commit_hash())
    study.set_user_attr("is_repo_clean", is_repo_clean())
    for k, v in config.__dict__.items():
        study.set_user_attr(k, v)
    try:
        study.optimize(objective, n_trials=config.n_trials)
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    storage = get_storage()
    config_path = str(sys.argv[1])

    # if len(sys.argv) > 2:
    #     # ! run a single variant
    #     variant_num = int(sys.argv[2])
    #     run_study(storage, config_path, variant_num, if_study_exists="delete")

    if_study_exists = sys.argv[2] if len(sys.argv) > 2 else "fail"

    # ! run all variants one after another
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    for variant_num in range(len(full_config["variants"])):
        run_study(storage, config_path, variant_num, if_study_exists)
