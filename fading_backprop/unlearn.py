# %%
import copy
import json
import time
from itertools import islice
from pathlib import Path

import bitsandbytes as bnb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as pt
import wandb
from datasets import Dataset, IterableDataset, IterableDatasetDict, load_dataset
from torcheval.metrics.text import Perplexity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

# params
model_id = "google/gemma-2-2b"
# model_id = "microsoft/phi-1_5"
context_len = 100
batch_size = 8

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %% load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=pt.bfloat16,
).to(device)


# %% load dataset
def load_one_oscar_shard(lang):
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

    return (
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


pl_dataset = load_one_oscar_shard("pl")
en_dataset = load_one_oscar_shard("en")
assert next(iter(pl_dataset["test"]))["text"] != next(iter(pl_dataset["unlearn"]))["text"]


def get_perplexity(model, dataset):
    metric = Perplexity(device=device)
    batch = next(iter(dataset["validation"].batch(8)))
    input_ids = pt.cat(batch["input_ids"])
    with pt.no_grad():
        outputs = model(input_ids)
    metric.update(outputs.logits[:, :-1], input_ids[:, 1:])
    return metric.compute()


# %%
# optimizer = pt.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = pt.optim.SGD(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss_fn = pt.nn.CrossEntropyLoss()

for batch in pl_dataset["unlearn"].batch(batch_size).take(10):
    # create batched input_ids
    input_ids = pt.cat(batch["input_ids"])  # shape: [batch_size, context_len]

    # forward pass
    output = model(input_ids)
    # compute loss
    loss = -loss_fn(
        output.logits[:, :-1, :].flatten(end_dim=1),
        input_ids[:, 1:].flatten(),
    )
    # backpropagate
    loss.backward()
    optimizer.step()

    res = dict(
        pl=get_perplexity(model, pl_dataset).item(),
        en=get_perplexity(model, en_dataset).item(),
    )
    print(res)

    # clean memory
    optimizer.zero_grad(set_to_none=True)
    del output, loss
    pt.cuda.empty_cache()

