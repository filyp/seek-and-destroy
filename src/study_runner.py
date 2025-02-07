# to run all variants one after another:
# python study_runner.py PATH_TO_CONFIG [if_study_exists=fail|delete|load]

# necessary for determinism:
import os

import wandb

from utils.wmdp_eval import eval_on_wmdp

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"  # less mem but slower

import argparse
import logging
import sys
from types import SimpleNamespace

import torch as pt
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_methods.circuit_breakers import circuit_breakers
from unlearning_methods.circuit_breakers_no_lora import circuit_breakers_no_lora
from unlearning_methods.surgical_irreversible_unlearning import (
    surgical_irreversible_unlearning,
)
from unlearning_methods.surgical_irreversible_unlearning_lora import (
    surgical_irreversible_unlearning_lora,
)
from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.model_operations import relearn, relearn_with_retain
from utils.training import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)


def run_study(storage, config_path, variant_num, if_study_exists="fail", n_trials=None):
    assert if_study_exists in ["fail", "delete", "load", "load-remaining"]
    print(f"{config_path=} {variant_num=} {if_study_exists=} {n_trials=}")

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
    if n_trials is not None:
        config.n_trials = n_trials

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

    _init_res = eval_(
        AutoModelForCausalLM.from_pretrained(config.model_id), f_eval, r_eval
    )
    allowed_r_loss = _init_res["retain_loss"]
    allowed_r_loss += getattr(config, "retain_loss_budget", 0)

    unlearning_func = dict(
        circuit_breakers=circuit_breakers,
        circuit_breakers_no_lora=circuit_breakers_no_lora,
        surgical_irreversible_unlearning_lora=surgical_irreversible_unlearning_lora,
        surgical_irreversible_unlearning=surgical_irreversible_unlearning,
    )[config.method_name]

    def objective(trial):
        # construct hyperparams
        hyperparams = dict()
        for hp_name, distribution in hyperparam_ranges.items():
            if distribution is None:
                continue
            elif isinstance(distribution, list):
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
            allowed_r_loss,
        )

        set_seeds(42)
        forget_losses = relearn(
            model, relearn_config, retain_val_batches, forget_val_batches
        )

        # use min rather than last, in case it anomalously increases
        forget_loss = min(forget_losses)
        return forget_loss

    def objective_wmdp(trial):
        # construct hyperparams
        hyperparams = dict()
        for hp_name, distribution in hyperparam_ranges.items():
            if distribution is None:
                continue
            elif isinstance(distribution, list):
                low, high, log = distribution
                hyperparams[hp_name] = trial.suggest_float(hp_name, low, high, log=log)
            else:
                hyperparams[hp_name] = distribution
        hyperparams = SimpleNamespace(**hyperparams)
        logging.info(f"trial {trial.number} - {trial.params}")

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, torch_dtype=pt.bfloat16
        )

        wandb.init(
            project="wmdp-eval3",
            group=variant_name,
            name=f"{variant_name}-{trial.number}",
        )
        accuracy = eval_on_wmdp(model)
        wandb.log(_init_res | {"wmdp_accuracy": accuracy}, step=0)

        set_seeds(42)
        model = unlearning_func(
            hyperparams,
            config,
            retain_batches,
            forget_batches,
            f_eval,
            r_eval,

            allowed_r_loss=_init_res["retain_loss"] + config.hard_loss_budget,
            model=model,
            soft_threshold=_init_res["retain_loss"] + config.soft_loss_budget,

            eval_wmdp_every=config.eval_wmdp_every,
        )

        set_seeds(42)
        _ = relearn_with_retain(
            model,
            relearn_config,
            retain_val_batches,
            forget_val_batches,
            eval_wmdp_every=config.eval_wmdp_every,
            step_offset=config.unlearn_steps,
        )
        wandb.finish()

        return eval_on_wmdp(model)

    if if_study_exists == "delete":
        delete_study_if_exists(study_name, storage)

    if hasattr(config, "eval_wmdp_every"):
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=(if_study_exists in ["load", "load-remaining"]),
        )
        study.set_metric_names(["wmdp_accuracy"])
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=(if_study_exists in ["load", "load-remaining"]),
        )
        study.set_metric_names(["forget_loss"])

    save_file_and_attach_logger(config_path, study.study_name)
    study.set_user_attr("commit_hash", commit_hash())
    study.set_user_attr("is_repo_clean", is_repo_clean())

    for k, v in config.__dict__.items():
        study.set_user_attr(k, v)

    n_trials = config.n_trials
    if if_study_exists == "load-remaining":
        # run remaining trials
        n_trials = max(0, config.n_trials - len(study.trials))

    try:
        print(f"{n_trials=}")
        if hasattr(config, "eval_wmdp_every"):
            study.optimize(objective_wmdp, n_trials=n_trials)
        else:
            study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unlearning studies")
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the config YAML file"
    )
    parser.add_argument(
        "--variant-num",
        type=int,
        default=None,
        help="Variant number to run (default: None to run all variants)",
    )
    parser.add_argument(
        "--if-study-exists",
        type=str,
        default="fail",
        choices=["fail", "delete", "load", "load-remaining"],
        help="What to do if study exists (default: fail). 'load' will run the number of specified trials continuing an existing study. 'load-remaining' will run (n_trials - len(study.trials)) trials.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials to run (default: None to use config value)",
    )
    parser.add_argument(
        "--allow-dirty-repo",
        action="store_true",
        help="Allow running even if the git repo has uncommitted changes (default: False)",
    )
    args = parser.parse_args()

    db_url = json.load(open("secret.json"))["db_url"]
    storage = get_storage(db_url)
    # storage = get_storage()

    with open(args.config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if args.variant_num is None:
        # ! run all variants one after another
        for variant_num in range(len(full_config["variants"])):
            if not args.allow_dirty_repo:
                assert is_repo_clean()
            run_study(
                storage,
                args.config_path,
                variant_num,
                args.if_study_exists,
                args.n_trials,
            )
    else:
        # ! run single variant
        if not args.allow_dirty_repo:
            assert is_repo_clean()
        run_study(
            storage,
            args.config_path,
            args.variant_num,
            args.if_study_exists,
            args.n_trials,
        )
