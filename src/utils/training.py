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


def eval_(model, f_eval_batch, r_eval_batch, allowed_f_loss=None, step=""):
    model.eval()
    with pt.no_grad():
        res = dict(
            forget_loss=cross_entropy_loss(model(f_eval_batch), f_eval_batch),
            retain_loss=cross_entropy_loss(model(r_eval_batch), r_eval_batch),
        )
    logging.info(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))
    if any(pt.isnan(v) for v in res.values()):
        raise optuna.TrialPruned()

    if allowed_f_loss is not None and res["retain_loss"] > allowed_f_loss:
        logging.info(f"Pruning trial because retain loss is too high")
        raise optuna.TrialPruned()

    return res


def make_sure_optimal_values_are_not_near_range_edges(study):
    """Make sure the value is not in the top or bottom 10% of the range."""
    for param_name, param_dist in study.best_trial.distributions.items():
        min_ = param_dist.low
        max_ = param_dist.high
        value = study.best_params[param_name]
        if param_dist.log:
            min_ = np.log10(min_)
            max_ = np.log10(max_)
            value = np.log10(value)

        if value < min_ + 0.1 * (max_ - min_):
            print(f"\n{study.study_name}")
            print(f"WARNING: {param_name} in the bottom 10% of the range in best trial")
            print(f"range: {min_} - {max_}, value: {value}, log={param_dist.log}")
        if value > max_ - 0.1 * (max_ - min_):
            print(f"\n{study.study_name}")
            print(f"WARNING: {param_name} in the top 10% of the range in best trial")
            print(f"range: {min_} - {max_}, value: {value}, log={param_dist.log}")


# stats for the last n non-pruned trials
def get_stats_from_last_n_trials(study, n=10):
    ok_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"{study.study_name}: len(ok_trials): {len(ok_trials)}")
    values = [t.values[0] for t in ok_trials]

    max_val = study.best_trial.values[0]
    last_n_mean = np.mean(values[-n:])
    last_n_sem = np.std(values[-n:]) / np.sqrt(n)
    pure_name = study.study_name.split("|")[-1]
    result = f"| {last_n_mean:.2f}±{last_n_sem:.2f} | {max_val:.2f} | {pure_name} |  |"
    # print("last_n_mean ± last_n_sem, max_val, pure_name")
    # print(result)
    return result, last_n_mean, last_n_sem


def delete_study_if_exists(study_name, storage):
    try:
        _ = optuna.load_study(study_name=study_name, storage=storage)
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass
