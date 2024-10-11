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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

# params
model_id = "microsoft/phi-1_5"
device = "cuda"
context_len = 100
batch_size = 16

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


dataset = load_one_oscar_shard(lang="pl")

# %%
# optimizer = pt.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = pt.optim.SGD(model.parameters(), lr=0.0001)
optimizer.zero_grad()
loss_fn = pt.nn.CrossEntropyLoss()

for batch in dataset["unlearn"].batch(batch_size, drop_last_batch=True).take(10):
    # create batched input_ids
    input_ids = pt.cat(batch["input_ids"])
    assert list(input_ids.shape) == [batch_size, context_len]

    # forward pass
    output = model(input_ids)
    # compute loss
    loss = loss_fn(
        output.logits[:, :-1, :].flatten(end_dim=1),
        input_ids[:, 1:].flatten(),
    )
    # backpropagate
    loss.backward()
    optimizer.step()

    print(loss.item())
# %%
