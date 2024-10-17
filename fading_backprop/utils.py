import json

import torch as pt
import wandb
from datasets import IterableDataset, IterableDatasetDict, load_dataset
from torcheval.metrics.text import Perplexity

context_len = 100
device = "cuda"


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

    dataset = (
        # define splits; make it iterable so that it can be processed on demand
        IterableDatasetDict(
            unlearn=IterableDataset.from_generator(lambda: (ex for ex in quarter1)),
            relearn=IterableDataset.from_generator(lambda: (ex for ex in quarter2)),
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
            ).to(device),
        )
        # filter out the short ones
        .filter(lambda ex: ex["input_ids"].shape[-1] >= context_len)
    )
    assert next(iter(dataset["test"]))["text"] != next(iter(dataset["unlearn"]))["text"]
    return dataset


def get_perplexity(model, dataset, batch_size=8):
    # get perplexity on one batch of the validation set
    metric = Perplexity(device=device)
    batch = next(iter(dataset["validation"].batch(batch_size)))
    input_ids = pt.cat(batch["input_ids"])
    with pt.no_grad():
        outputs = model(input_ids)
    metric.update(outputs.logits[:, :-1], input_ids[:, 1:])
    return metric.compute().item()


def forward(model, batch):
    loss_fn = pt.nn.CrossEntropyLoss()
    # create batched input_ids
    input_ids = pt.cat(batch["input_ids"])
    assert input_ids.shape[1] == context_len
    # forward pass
    logits = model(input_ids).logits
    # compute loss
    return loss_fn(logits[:, :-1, :].flatten(end_dim=1), input_ids[:, 1:].flatten())

