import logging
import random

import numpy as np
import optuna
import torch as pt
from transformers import set_seed as set_transformers_seed

from utils.git_and_reproducibility import *


# --- Setup and Environment ---
def set_seeds(seed):
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    set_transformers_seed(seed)
    pt.use_deterministic_algorithms(True)


# --- Training Utilities ---
def eval_loss(model, batch):
    model.eval()
    with pt.no_grad():
        return cross_entropy_loss(model(batch), batch)


# --- Loss Functions ---
def cross_entropy_loss(output, input_ids):
    return pt.nn.CrossEntropyLoss()(
        output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32),
        input_ids[:, 1:].flatten(),
    )


def correct_logit_loss(output, input_ids):
    logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = input_ids[:, 1:].flatten()
    true_logits = logits[pt.arange(len(ids)), ids]
    return true_logits.mean()


def clipped_correct_logit_loss(output, input_ids):
    logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = input_ids[:, 1:].flatten()
    true_logits = logits[pt.arange(len(ids)), ids]
    return true_logits.clip(min=0).mean()


def neg_cross_entropy_loss(output, input_ids):
    return -cross_entropy_loss(output, input_ids)


# adapted from https://github.com/rishub-tamirisa/tamper-resistance/blob/41b749ca4d9bcb7608c7ead2ca48b0508714af99/modules/objectives.py#L114
def negative_entropy_loss(output, input_ids) -> pt.Tensor:
    """
    Compute the negative mean entropy loss for the given logits.

    This function calculates the entropy of the softmax distribution of the input logits
    and returns the negative mean entropy as a loss value. Minimizing this loss
    encourages the model to produce more uniform (higher entropy) probability distributions.

    Returns:
        pt.Tensor: The negative mean entropy loss.
    """
    logits = output.logits
    softmax = pt.nn.functional.softmax(logits, dim=-1)
    log_softmax = pt.nn.functional.log_softmax(logits, dim=-1)
    entropy = pt.sum(-softmax * log_softmax, dim=-1).mean()
    return entropy.mean() * -1


loss_fns = dict(
    cross_entropy=cross_entropy_loss,
    clipped_correct_logit=clipped_correct_logit_loss,
    correct_logit=correct_logit_loss,
    neg_cross_entropy=neg_cross_entropy_loss,
    negative_entropy=negative_entropy_loss,
)


# --- Mock Trial for Optuna ---
class MockTrial:
    def __init__(self, **params):
        self.params = params
        self.number = 0

    def suggest_float(self, name, *args, **kwargs):
        return self.params[name]

    def suggest_categorical(self, name, *args, **kwargs):
        return self.params[name]

    def suggest_int(self, name, *args, **kwargs):
        return int(self.params[name])

    def set_user_attr(self, *args, **kwargs):
        pass


def eval_(model, f_eval_batch, r_eval_batch, allowed_f_loss=None, step=""):
    res = {}
    res["forget_loss"] = eval_loss(model, f_eval_batch)
    res["retain_loss"] = eval_loss(model, r_eval_batch)
    logging.info(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))
    assert not any(pt.isnan(v) for v in res.values())

    if allowed_f_loss is not None and res["retain_loss"] > allowed_f_loss:
        logging.info(f"Pruning trial because retain loss is too high")
        raise optuna.TrialPruned()

    return res


def run_study(
    objective,
    config,
    script_name,
    study_name,
    delete_existing=False,
    load_if_exists=False,
):
    storage = get_storage()

    # delete existing study if it exists
    if delete_existing:
        try:
            _ = optuna.load_study(study_name=study_name, storage=storage)
            optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=load_if_exists,
    )
    save_script_and_attach_logger(script_name, study.study_name)
    study.set_metric_names(["forget_loss"])
    study.set_user_attr("commit_hash", commit_hash())
    for k, v in config.__dict__.items():
        study.set_user_attr(k, v)
    study.optimize(objective, n_trials=config.n_trials)
    return study


def make_sure_optimal_values_are_not_near_range_edges(study):
    # make sure the value is not in the top or bottom 10% of the range, logarithmically
    for param_name, value in study.best_trial.params.items():
        min_ = min(trial.params[param_name] for trial in study.trials)
        max_ = max(trial.params[param_name] for trial in study.trials)
        min_log = np.log(min_)
        max_log = np.log(max_)
        value_log = np.log(value)
        if value_log < min_log + 0.1 * (max_log - min_log):
            print(
                f"WARNING: {param_name} is in the bottom 10% of the range in best trial"
            )
            print(f"range: {min_} - {max_}, value: {value}")
        if value_log > max_log - 0.1 * (max_log - min_log):
            print(f"WARNING: {param_name} is in the top 10% of the range in best trial")
            print(f"range: {min_} - {max_}, value: {value}")
