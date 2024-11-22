import logging
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch as pt


# --- Setup and Environment ---
def set_seeds(seed):
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False
    random.seed(seed)


def repo_root():
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


def commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def save_script_and_attach_logger(file_name):
    # for reproducibility save the file state and append output into it
    # save script
    folder = repo_root() / "results" / datetime.now().strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now().strftime('%H:%M:%S')}_{Path(file_name).stem}.py"
    shutil.copy(file_name, path)
    # attach logger
    for h in logging.getLogger().handlers[1:]:
        logging.root.removeHandler(h)
    file_handler = logging.FileHandler(path)
    formatter = logging.Formatter("# %(asctime)s %(levelname)s  %(message)s")
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
    logging.info(f"commit hash: {commit_hash()}")


# --- Training Utilities ---
def get_batch(iter, n):
    return pt.cat([next(iter)["input_ids"] for _ in range(n)])


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


loss_fns = dict(
    cross_entropy=cross_entropy_loss,
    clipped_correct_logit=clipped_correct_logit_loss,
    correct_logit=correct_logit_loss,
)
