import logging
import random

import matplotlib.pyplot as plt
import optuna
import torch as pt

from utils.git_and_reproducibility import *


# --- Setup and Environment ---
def set_seeds(seed):
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False
    random.seed(seed)


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


def entropy_loss(output, input_ids):
    # todo
    raise NotImplementedError("Not implemented")


loss_fns = dict(
    cross_entropy=cross_entropy_loss,
    clipped_correct_logit=clipped_correct_logit_loss,
    correct_logit=correct_logit_loss,
    neg_cross_entropy=neg_cross_entropy_loss,
)


# --- Mock Trial for Optuna ---
class MockTrial:
    def __init__(self, params):
        self.params = params

    def suggest_float(self, name, *args, **kwargs):
        return self.params[name]

    def suggest_categorical(self, name, *args, **kwargs):
        return self.params[name]

    def suggest_int(self, name, *args, **kwargs):
        return int(self.params[name])

    def set_user_attr(self, *args, **kwargs):
        pass


def eval_(model, f_eval_batch, r_eval_batch, init_retain, step):
    res = {}
    res["forget_loss"] = eval_loss(model, f_eval_batch)
    res["retain_loss"] = eval_loss(model, r_eval_batch)
    res["retain_loss_ok"] = res["retain_loss"] < init_retain + 0.05
    logging.info(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))
    assert not any(pt.isnan(v) for v in res.values())
    if res["retain_loss"] > init_retain + 0.1:
        logging.info(f"Pruning trial because retain loss is too high")
        raise optuna.TrialPruned()
    return res


def visualize_param(param, mask):
    x = param.to_forget
    y = param.disruption_score
    c = mask
    # plot a 2d scatter plot
    # first flatten x and y and c and convert to cpy numpy
    x = x.flatten().cpu().numpy()
    y = y.flatten().cpu().numpy()
    c = c.flatten().cpu().numpy()

    plt.clf()
    plt.scatter(x, y, c=c, s=1)

    # label
    plt.xlabel("to_forget")
    plt.ylabel("disruption_score")

    # plt.ylim(0, 0.001)
    # plt.show()
    file_name = repo_root() / "results" / f"param_toforget_vs_disruption.png"
    plt.savefig(file_name)


def run_study(
    objective, config, script_name, study_name, assert_clean=True, delete_existing=False
):
    if assert_clean:
        assert is_repo_clean()
    study_type = "big" if config.unlearn_steps == 1000 else "small"
    script_stem = Path(script_name).stem
    study_name = f"{study_type},{config.forget_set_name},{script_stem},{study_name}"

    storage = get_storage()

    # delete existing study if it exists
    if delete_existing:
        try:
            _ = optuna.load_study(study_name=study_name, storage=storage)
            optuna.delete_study(study_name, storage=storage)
        except KeyError:
            pass

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        # load_if_exists=True,
    )
    save_script_and_attach_logger(script_name, study.study_name)
    study.set_metric_names(["forget_loss"])
    study.set_user_attr("commit_hash", commit_hash())
    for k, v in config.__dict__.items():
        study.set_user_attr(k, v)
    study.optimize(objective, n_trials=config.n_trials)
