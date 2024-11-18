import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch as pt
from datasets import IterableDataset, IterableDatasetDict, load_dataset
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
from tensordict import TensorDict


def load_one_oscar_shard(lang, tokenizer):
    context_len = 100
    # only use one ~600MB shard
    # also, streaming would make splitting too slow
    raw_dataset = load_dataset(
        "text",
        split="train",  # train is the only split in oscar
        data_files=f"hf://datasets/oscar-corpus/OSCAR-2301/{lang}_meta/{lang}_meta_part_1.jsonl.zst",
    )

    # split into 4 quarters
    half1, half2 = raw_dataset.train_test_split(test_size=0.5, seed=42).values()
    quarter3, quarter4 = half2.train_test_split(test_size=0.5, seed=42).values()
    # eigth7, eigth8 = quarter4.train_test_split(test_size=0.5, seed=42).values()

    dataset = (
        # define splits; make it iterable so that it can be processed on demand
        IterableDatasetDict(
            train=IterableDataset.from_generator(lambda: (ex for ex in half1)),
            # relearn=IterableDataset.from_generator(lambda: (ex for ex in quarter3)),
            validation=IterableDataset.from_generator(lambda: (ex for ex in quarter3)),
            test=IterableDataset.from_generator(lambda: (ex for ex in quarter4)),
        )
        # process the raw data, following OSCAR-2301.py
        .map(lambda ex: {"text": json.loads(ex["text"])["content"]})
        # tokenize
        .map(
            lambda ex: tokenizer(
                ex["text"],
                return_tensors="pt",
                max_length=context_len,
                truncation=True,
            ),
        )
        # filter out the short ones
        .filter(lambda ex: ex["input_ids"].shape[-1] >= context_len)
    )
    assert next(iter(dataset["test"]))["text"] != next(iter(dataset["train"]))["text"]
    return dataset


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


def looping_iter(iterable):
    # like itertools.cycle, but will not eat memory by storing element copies
    while True:
        yield from iterable


def get_batch(iter, n):
    return pt.cat([next(iter)["input_ids"] for _ in range(n)])


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


# load circuit
def c(circuit_name):
    circ = pt.load(repo_root() / "circuits" / f"{circuit_name}.pt", weights_only=True)
    # this is filled with zero imps, at least for polish
    del circ["model.embed_tokens.weight"]
    return TensorDict(circ)


def kinda_safe_eval(expr):
    # Create a custom globals dictionary with necessary components
    restricted_globals = dict(safe_globals)
    restricted_globals.update({
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_getattr_": safer_getattr,
        # Add any other necessary functions/variables that your expression needs
        "c": c,
    })

    byte_code = compile_restricted(expr, filename="<inline code>", mode="eval")
    return eval(byte_code, restricted_globals)


def get_perplexities(model, batches):
    model.eval()
    with pt.no_grad():
        return [cross_entropy_loss(model(batch), batch).exp() for batch in batches]


def print_perplexities(model, batches, step):
    f_ppl, r_ppl = get_perplexities(model, batches)
    stats = dict(forget=f_ppl, retain=r_ppl)
    print(f"{step:4d} " + " ".join(f"{v:11.2f}" for v in stats.values()))


# Create a class that writes to both stdout and a file
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for file in self.files:
            file.write(message)
            file.flush()  # Ensure the message is written immediately

    def flush(self):
        for file in self.files:
            file.flush()
