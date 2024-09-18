# %%
import time
from itertools import islice

import bitsandbytes as bnb
from matplotlib import pyplot as plt
import numpy as np
import torch as pt
from datasets import load_dataset
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from utils import *

import wandb

# model_id = "microsoft/Phi-3-mini-4k-instruct"
# model_id = "microsoft/Phi-3.5-mini-instruct"
model_id = "microsoft/phi-1_5"

tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=pt.bfloat16,
).to(device)

# %% Load the BeaverTails dataset
beavertails = load_dataset("PKU-Alignment/BeaverTails")
split = beavertails["30k_train"]
category = "animal_abuse"
beaver_category = split.filter(lambda ex: ex["category"][category])
len(beaver_category)


# %%
# save the grad on some specified layer, to be later used for latent attack
def save_grad_hook(module, grad_input, grad_output):
    global input_grad, output_grad
    input_grad = grad_input[0]
    output_grad = grad_output[0]


model.model.layers[3].mlp.activation_fn.register_full_backward_hook(save_grad_hook)
# %%
example = beaver_category[0]

# forward
text = example["prompt"] + "\n" + example["response"]
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
output = model(input_ids)

# backward
loss = cross_entropy(
    output.logits[:, :-1].flatten(end_dim=1),
    input_ids[:, 1:].reshape(-1),
)
loss.backward()

# %%
plt.scatter(
    output_grad.flatten().cpu().float(),
    input_grad.flatten().cpu().float(),
    s=0.2,
    alpha=0.1,
)
plt.gca().set_aspect('equal')
