# %%
import time
from itertools import islice
from pathlib import Path

import bitsandbytes as bnb
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from utils import *

# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "microsoft/Phi-3.5-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda"

access_token = (Path.home() / ".cache/huggingface/token").read_text()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=pt.bfloat16,
    token=access_token,
).to(device)

# Load the WikiText-2 dataset
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

    # prepare perturbation - only executed in the first loop
    if perturbation is None:
        perturbation = pt.zeros_like(activation)

    activation += perturbation
    return (activation, cache)


model.model.layers[3].register_full_backward_hook(save_grad_hook)
model.model.layers[3].register_forward_hook(add_perturbation_hook)

# %%
# new figure
fig, ax = plt.subplots()
ax.set_xlabel("Perturbation norm")
ax.set_ylabel("Targetted response's log probability")

# %% latent attack
epsilon = 0.001
pert_decay = 0.99  # 0.99
example = unsafe_examples[200]

perturbation = None
history_d_perturbation = []
attack_scores = []
perturbation_norms = []

for i in range(50):
    # forward
    resp_log_prob, _ = get_response_log_prob(example, model, tokenizer)
    print(f"{resp_log_prob.item()=}")
    # backward
    resp_log_prob.backward()

    attack_scores.append(resp_log_prob.cpu().detach().float())
    perturbation_norms.append(perturbation.norm().cpu().detach().float())
    # print(d_perturbation.norm())

    # # stop the loop early when the response has ~2% probability
    # if response_log_prob > -20:  # -4:
    #     break

    # update perturbation
    # perturbation += d_perturbation / d_perturbation.norm() * epsilon
    perturbation += d_perturbation * epsilon
    perturbation *= pert_decay ** (epsilon * 1000)

    history_d_perturbation.append(d_perturbation)

    if len(history_d_perturbation) >= 2:
        p1 = history_d_perturbation[-1]
        p2 = history_d_perturbation[-2]
        cossim = pt.cosine_similarity(p1.reshape(1, -1), p2.reshape(1, -1))
        # assert cossim > 0.2
        rescale_factor = pt.exp((cossim - 0.9) * 1)
        epsilon = (epsilon * rescale_factor).item()
        print(f"{cossim.item()=}\n{rescale_factor.item()=}\n{epsilon=}")
        print()

print(f"Attack took {i} iterations")

# %%
# ax.plot(perturbation_norms, attack_scores, label=f"{pert_decay=}", marker="o", alpha=0.1)
# change alpha of only markers, not the line
ax.plot(
    perturbation_norms,
    attack_scores,
    label=f"{pert_decay=}",
    marker="o",
    markersize=6,
    markerfacecolor=(1, 1, 1, 0.1),
)
ax.legend()
fig


# %%

# %%

# %%
