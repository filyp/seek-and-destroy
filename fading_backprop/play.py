# %%
import copy
import time
from itertools import islice
from pathlib import Path

import bitsandbytes as bnb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as pt
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

import wandb

# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "microsoft/Phi-3.5-mini-instruct"
# model_id = "../models/Phi-3.5-mini-instruct-LeakyReLU_0.12"
model_id = "microsoft/phi-1_5"

tokenizer = AutoTokenizer.from_pretrained(model_id)

device = "cuda"
context_len = 100
batch_size = 8

# %% prepare dataset
# dataset = load_dataset("open_subtitles", lang1="en", lang2="pl", trust_remote_code=True)
# dataset = load_dataset("oscar-corpus/oscar", "unshuffled_deduplicated_pl", trust_remote_code=True)

dataset = (
    load_dataset(
        "oscar-corpus/OSCAR-2301",
        "pl",
        trust_remote_code=True,
        streaming=True,
        split="train",  # train is the only split in OSCAR-2301
    )
    .shuffle(seed=42)
    .map(
        lambda ex: tokenizer(
            ex["text"],
            return_tensors="pt",
            max_length=context_len,
            truncation=True,
        ).to(device)
    )
    .filter(lambda ex: ex["input_ids"].shape[-1] >= context_len)
    .batch(batch_size=batch_size)
)
