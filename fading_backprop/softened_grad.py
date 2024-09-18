# %%
import time
from itertools import islice

import bitsandbytes as bnb
import numpy as np
import torch as pt
from datasets import load_dataset
from matplotlib import pyplot as plt
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
safe_examples = split.filter(lambda ex: ex["is_safe"])
len(beaver_category)


# %% hooks for intervening on gradients
def save_activation_hook(module, args, output):
    module_to_activation[module] = args[0]


def mix_in_output_grad_hook(module, grad_input, grad_output):
    # for debug
    _module_to_grad_input[module] = grad_input[0]
    _module_to_grad_output[module] = grad_output[0]
    # actual technique
    activation = module_to_activation[module]
    mix = pt.clip(activation / c.thresh_act + 1, 0, 1)
    softened_grad = mix * grad_output[0] + (1 - mix) * grad_input[0]
    return (softened_grad,)


module_to_activation = dict()
_module_to_grad_input = dict()
_module_to_grad_output = dict()
for layer in model.model.layers:
    layer.mlp.activation_fn.register_forward_hook(save_activation_hook)
    layer.mlp.activation_fn.register_full_backward_hook(mix_in_output_grad_hook)


# %% hooks for recording d_perturbation and later applying perturbation
def save_d_perturbation_hook(module, grad_input, grad_output):
    global d_perturbation
    d_perturbation = grad_output[0]
    assert len(grad_output) == 2 and grad_output[1] is None


def add_perturbation_hook(module, args, output):
    global perturbation
    assert len(output) == 2 and not isinstance(output[1], pt.Tensor)
    activation, cache = output

    activation += perturbation
    return (activation, cache)


model.model.layers[3].register_full_backward_hook(save_d_perturbation_hook)
model.model.layers[3].register_forward_hook(add_perturbation_hook)

# %%


# %% prepare
wandb.init(
    # set the wandb project where this run will be logged
    project="softened_derivative",
    # track hyperparameters and run metadata
    config=dict(
        model_name=model_id,

        thresh_act=1,
        thresh_act_decay=0.98,

        epsilon=0.01,
        pert_decay=0.98,
        num_steps=200,
    ),
    # run name
    # name="LeakyReLU_0.12_1epoch_retrained",
    group="test",
)
c = wandb.config
perturbation = 0
last_d_perturbation_post_decay = None
epsilon = c.epsilon

# example = beaver_category[4]
example = safe_examples[4]
print(example["prompt"])
print(example["response"])


# %% eval

# forward
resp_log_prob, resp_len = get_response_log_prob(example, model, tokenizer)
# backward
resp_log_prob.backward()

# get some data and plot
k = list(module_to_activation.keys())[5]
output_grad = _module_to_grad_output[k]
input_grad = _module_to_grad_input[k]
activation = module_to_activation[k]
plt.scatter(
    output_grad.flatten().cpu().float(),
    input_grad.flatten().cpu().float(),
    c=activation.flatten().cpu().float().detach(),
    s=0.01,
    # alpha=0.1,
)
plt.gca().set_aspect("equal")
# plot colorscheme scale
plt.colorbar()

old_activation = activation.clone().detach()
old_output_grad = output_grad.clone().detach()
old_input_grad = input_grad.clone().detach()


# %% latent attack

for i in tqdm(range(c.num_steps)):
    # forward
    resp_log_prob, resp_len = get_response_log_prob(example, model, tokenizer)
    # backward
    resp_log_prob.backward()

    # update perturbation
    _last_perturbation = perturbation.clone().detach() if isinstance(perturbation, pt.Tensor) else perturbation  # fmt: skip
    perturbation += d_perturbation * epsilon
    perturbation *= c.pert_decay
    d_perturbation_post_decay = perturbation - _last_perturbation
    
    c.thresh_act *= c.thresh_act_decay

    if last_d_perturbation_post_decay is None:
        last_d_perturbation_post_decay = d_perturbation_post_decay
        continue

    # cossim of last two d_perturbations
    p1 = last_d_perturbation_post_decay
    p2 = d_perturbation_post_decay
    cossim = pt.cosine_similarity(p1.reshape(1, -1), p2.reshape(1, -1))
    last_d_perturbation_post_decay = d_perturbation_post_decay

    wandb.log(
        dict(
            attack_score=resp_log_prob.float() / resp_len,
            perturbation_norm=perturbation.norm().float(),
            cossim=cossim.float(),
            thresh_act=c.thresh_act,
        )
    )

wandb.finish()

# %% eval again
# forward
resp_log_prob, resp_len = get_response_log_prob(example, model, tokenizer)
# backward
resp_log_prob.backward()
activation = module_to_activation[k]

# plot grey diagonal
plt.plot([-5, 5], [-5, 5], color="grey", linestyle="-")
# plot axis lines
plt.axhline(0, color="grey", linestyle="-")
plt.axvline(0, color="grey", linestyle="-")

plt.scatter(
    old_activation.flatten().cpu().float().detach(),
    activation.flatten().cpu().float().detach(),
    c=old_output_grad.flatten().cpu().float().detach(),
    alpha=(old_output_grad.flatten().cpu().float().detach()-0.0).clip(0, 1),
    s=3,
)
plt.gca().set_aspect("equal")
plt.colorbar()
