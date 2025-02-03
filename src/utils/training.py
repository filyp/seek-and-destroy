import logging
import random

import numpy as np
import optuna
import torch as pt
from transformers import set_seed as set_transformers_seed

from utils.git_and_reproducibility import *
from utils.loss_fns import cross_entropy_loss


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


def eval_(model, f_eval_batch, r_eval_batch, allowed_r_loss=None, step=""):
    model.eval()
    with pt.no_grad():
        res = dict(
            forget_loss=cross_entropy_loss(model(f_eval_batch), f_eval_batch),
            retain_loss=cross_entropy_loss(model(r_eval_batch), r_eval_batch),
        )
    logging.info(f"{step:4} " + " ".join(f"{v:11.3f}" for v in res.values()))
    if any(pt.isnan(v) for v in res.values()):
        raise optuna.TrialPruned()

    if allowed_r_loss is not None and res["retain_loss"] > allowed_r_loss:
        logging.info(f"Pruning trial because retain loss is too high")
        raise optuna.TrialPruned()

    return res


def make_sure_optimal_values_are_not_near_range_edges(study):
    best_trial = study.best_trial  # ask only once because it's slow
    """Make sure the value is not in the top or bottom 10% of the range."""
    for param_name, param_dist in best_trial.distributions.items():
        min_ = param_dist.low
        max_ = param_dist.high
        value = best_trial.params[param_name]
        if param_dist.log:
            min_ = np.log10(min_)
            max_ = np.log10(max_)
            value = np.log10(value)

        method_name = study.study_name.split("|")[-1]
        if value < min_ + 0.1 * (max_ - min_):
            print(f"{method_name}\t{param_name}\t{value} in bottom 10%")
        if value > max_ - 0.1 * (max_ - min_):
            print(f"{method_name}\t{param_name}\t{value} in top 10%")
            # print(f"WARNING: {param_name} in the top 10% of the range in best trial")
            # print(f"range: {min_} - {max_}, value: {value}, log={param_dist.log}")

# stats for the last n non-pruned trials
def get_stats_from_last_n_trials(study, trials, n=10):
    ok_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"all_trials={len(trials)}, ok_trials={len(ok_trials)}, {study.study_name}")
    values = [t.values[0] for t in ok_trials]

    # max_val = study.best_trial.values[0]
    last_n_mean = np.mean(values[-n:])
    last_n_sem = np.std(values[-n:]) / np.sqrt(n)
    pure_name = study.study_name.split("|")[-1]
    # result = f"| {last_n_mean:.4f}±{last_n_sem:.4f} | {max_val:.4f} | {pure_name} |  |"
    result = f"| {last_n_mean:.4f}±{last_n_sem:.4f} | {pure_name} |  |"
    return result, last_n_mean, last_n_sem


def delete_study_if_exists(study_name, storage):
    try:
        _ = optuna.load_study(study_name=study_name, storage=storage)
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass
