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
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

import wandb
from utils import *

# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "microsoft/Phi-3.5-mini-instruct"
# model_id = "../models/Phi-3.5-mini-instruct-LeakyReLU_0.12"
model_id = "microsoft/phi-1_5"


# tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
device = "cuda"

access_token = (Path.home() / ".cache/huggingface/token").read_text()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=pt.bfloat16,
    token=access_token,
).to(device)

# Load the BeaverTails dataset
dataset = load_dataset("PKU-Alignment/BeaverTails")
split = dataset["30k_train"]
unsafe_examples = split.filter(lambda ex: not ex["is_safe"])
safe_examples = split.filter(lambda ex: ex["is_safe"])

immediately_remove_param_gradients_to_save_memory(model)


# %%
# save the grad on some specified layer, to be later used for latent attack
def save_grad_hook(module, grad_input, grad_output):
    global d_perturbation
    d_perturbation = grad_output[0]
    assert len(grad_output) == 2 and grad_output[1] is None


def add_perturbation_hook(module, args, output):
    global perturbation
    assert len(output) == 2 and not isinstance(output[1], pt.Tensor)
    activation, cache = output

    activation += perturbation
    return (activation, cache)


model.model.layers[3].register_full_backward_hook(save_grad_hook)
model.model.layers[3].register_forward_hook(add_perturbation_hook)

# %% latent attack
wandb.init(
    # set the wandb project where this run will be logged
    project="softened_derivative",
    # track hyperparameters and run metadata
    config=dict(
        epsilon=0.001,
        pert_decay=1,#0.99,
        # epsilon_adaptation_speed=0,
        # thresh=0.95,
        model_name=model_id,
    ),
    # run name
    # name="LeakyReLU_0.12_1epoch_retrained",
)

example = unsafe_examples[205]

perturbation = 0
last_d_perturbation_post_decay = None
c = wandb.config
epsilon = c.epsilon

for i in tqdm(range(1000)):
    # forward
    resp_log_prob, resp_len = get_response_log_prob(example, model, tokenizer)
    # backward
    resp_log_prob.backward()

    # update perturbation
    _last_perturbation = perturbation.clone().detach() if isinstance(perturbation, pt.Tensor) else perturbation  # fmt: skip
    perturbation += d_perturbation * epsilon
    perturbation *= c.pert_decay ** (epsilon * 1000)
    d_perturbation_post_decay = perturbation - _last_perturbation

    if last_d_perturbation_post_decay is None:
        last_d_perturbation_post_decay = d_perturbation_post_decay
        continue

    p1 = last_d_perturbation_post_decay
    p2 = d_perturbation_post_decay
    cossim = pt.cosine_similarity(p1.reshape(1, -1), p2.reshape(1, -1))
    # rescale_factor = max(0.5, (cossim - c.thresh) * c.epsilon_adaptation_speed + 1)
    # epsilon *= rescale_factor
    last_d_perturbation_post_decay = d_perturbation_post_decay

    wandb.log(
        dict(
            attack_score=resp_log_prob.float() / resp_len,
            perturbation_norm=perturbation.norm().float(),
            cossim=cossim.float(),
            # epsilon=epsilon.float(),
            # rescale_factor=float(rescale_factor),
        )
    )

wandb.finish()

# %% change of activation function
for layer in model.model.layers:
    layer.mlp.activation_fn = pt.nn.LeakyReLU(negative_slope=0.12)

# %%
# get data from the last wandb run through the API
api = wandb.Api()
runs = list(api.runs("softened_derivative"))
print(len(runs))
run = runs[-1]
# get all steps, not just 500
steps = pd.DataFrame(run.scan_history())

# 2d line plot using matplotlib of attack score vs perturbation norm
plt.plot(steps["perturbation_norm"], steps["attack_score"])
# %%
