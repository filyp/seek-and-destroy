# %%
import time

import matplotlib.pyplot as plt
import torch as pt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, load_one_oscar_shard

from fading_backprop import unlearn_and_relearn

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
pl_dataset = load_one_oscar_shard("pl", tokenizer)
en_dataset = load_one_oscar_shard("en", tokenizer)

# load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.to(device)

final_perplexities = unlearn_and_relearn(
    model,
    pl_dataset,
    en_dataset,
    wandb_group="test",
    unlearning_function="activation_agnostic",
    f_schedule="lambda step: 0",
    num_unlearning_steps=100,
    num_relearning_steps=20,
    eval_every_n_steps=2,
    relearn_lr=0.0003,
    unlearn_lr=1,
    pl_ppl_threshold=float("inf"),
    batch_size=32,
)

# %%
