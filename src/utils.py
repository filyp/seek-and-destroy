import json
import random
from itertools import islice

import torch as pt
from datasets import IterableDataset, IterableDatasetDict, load_dataset
from torcheval.metrics.text import Perplexity

context_len = 100


def load_one_oscar_shard(lang, tokenizer):
    # only use one ~600MB shard
    # also, streaming would make splitting too slow
    raw_dataset = load_dataset(
        "text",
        split="train",  # train is the only split in oscar
        data_files=f"hf://datasets/oscar-corpus/OSCAR-2301/{lang}_meta/{lang}_meta_part_1.jsonl.zst",
    )

    # split into 4 quarters
    half1, half2 = raw_dataset.train_test_split(test_size=0.5, seed=42).values()
    quarter1, quarter2 = half1.train_test_split(test_size=0.5, seed=42).values()
    quarter3, quarter4 = half2.train_test_split(test_size=0.5, seed=42).values()
    eigth1, eigth2 = quarter4.train_test_split(test_size=0.5, seed=42).values()

    dataset = (
        # define splits; make it iterable so that it can be processed on demand
        IterableDatasetDict(
            unlearn=IterableDataset.from_generator(lambda: (ex for ex in quarter1)),
            alt_unlearn=IterableDataset.from_generator(lambda: (ex for ex in quarter2)),
            relearn=IterableDataset.from_generator(lambda: (ex for ex in quarter3)),
            validation=IterableDataset.from_generator(lambda: (ex for ex in eigth1)),
            test=IterableDataset.from_generator(lambda: (ex for ex in eigth2)),
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
    assert next(iter(dataset["test"]))["text"] != next(iter(dataset["unlearn"]))["text"]
    return dataset


def forward(model, batch):
    loss_fn = pt.nn.CrossEntropyLoss()
    # create batched input_ids
    input_ids = pt.cat(batch["input_ids"])
    assert input_ids.shape[1] == context_len
    # forward pass
    logits = model(input_ids).logits
    # compute loss
    return loss_fn(logits[:, :-1, :].flatten(end_dim=1), input_ids[:, 1:].flatten())


def get_perplexity(model, dataset, num_batches=1, batch_size=32):
    total_loss = 0
    for batch in islice(dataset["validation"].batch(batch_size), num_batches):
        with pt.no_grad():
            total_loss += forward(model, batch)
    return (total_loss / num_batches).exp()


def get_norm_of_weights_change(model, original_state_dict):
    partial_norms = []
    for module_name, current_weights in model.named_parameters():
        original_weights = original_state_dict[module_name]
        norm = (original_weights - current_weights).norm()
        partial_norms.append(norm)
    return pt.tensor(partial_norms).norm()


def get_stats(model, forget_set, retain_set):
    return pt.tensor([
        get_perplexity(model, forget_set),
        get_perplexity(model, retain_set),
        # get_norm_of_weights_change(model, og_model.state_dict()),
    ])


def print_stats(stats):
    print(f"forget: {stats[0]:4.0f}  retain: {stats[1]:5.2f}  ratio: {stats[0] / stats[1]:.0f}")  # fmt: skip


def set_seeds(seed):
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False
    random.seed(seed)
