import logging
import random

import optuna
import torch as pt


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


def eval(model, f_eval_batch, r_eval_batch, init_retain, step):
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