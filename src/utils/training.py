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
    # make sure the value is not in the top or bottom 10% of the range, logarithmically
    for param_name, value in study.best_trial.params.items():
        min_ = min(trial.params[param_name] for trial in study.trials[:-1])
        max_ = max(trial.params[param_name] for trial in study.trials[:-1])
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


# stats for the last n non-pruned trials
def get_stats_from_last_n_trials(study, n=10):
    ok_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.values[0] for t in ok_trials]

    max_val = study.best_trial.values[0]
    last_n_mean = np.mean(values[-n:])
    last_n_std = np.std(values[-n:])
    pure_name = ",".join(study.study_name.split(",")[3:])
    print("last_n_mean ± last_n_std, max_val, pure_name")
    print(f"| {last_n_mean:.2f}±{last_n_std:.2f} | {max_val:.2f} | {pure_name} |  |")
    return last_n_mean, last_n_std


def delete_study_if_exists(study_name, storage):
    try:
        _ = optuna.load_study(study_name=study_name, storage=storage)
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass
