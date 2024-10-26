# %%
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from scipy.optimize import curve_fit
from transformers import AutoModelForCausalLM, AutoTokenizer

pt.set_default_device("cuda")
from unlearning_functions import name_to_function

from fading_backprop import (
    install_hooks_for_fading_backprop,
    install_hooks_for_saving_gradients,
    set_fade_factor,
)
from utils import forward, get_perplexity, load_one_oscar_shard

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_one_oscar_shard("en", tokenizer)

# load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

original_state_dict = deepcopy(model.state_dict())

ppl_batch_size = 32


# %%
def eval_pert_impact(pert_norm, module_name):
    # assumes these variables are defined: model, original_state_dict, dataset
    original_tensor = original_state_dict[module_name]
    perturbation = pt.rand_like(original_tensor)
    perturbation *= pert_norm / perturbation.norm()
    # apply perturbation
    model.state_dict()[module_name] += perturbation
    # eval
    perplexity = get_perplexity(model, dataset, batch_size=ppl_batch_size)
    # revert - note: must be done in-place
    model.state_dict()[module_name] *= 0
    model.state_dict()[module_name] += original_tensor
    return perplexity


# test that it cleans up after itself
module_name = "model.layers.20.mlp.down_proj.weight"
eval_pert_impact(5, "model.layers.20.mlp.down_proj.weight")
assert model.state_dict()[module_name].allclose(original_state_dict[module_name])
# test ppl for 0 perturbation
orig_ppl = get_perplexity(model, dataset, batch_size=ppl_batch_size)
assert eval_pert_impact(0, module_name) == orig_ppl
# test ppl for non-0 perturbation
assert not eval_pert_impact(0.1, module_name) == orig_ppl

# %% check the norms of the weights on each layer
for l in range(24):
    module_name = f"model.layers.{l}.mlp.down_proj.weight"
    weights = model.state_dict()[module_name]
    print(f"{module_name}: {weights.norm():.2f}")

# %%
# use b buckets 1-10, each with 10 perturbations
# then plot the mean and sem of the perplexities (std as error bars)

module_name = "model.layers.22.mlp.down_proj.weight"
num_buckets = 21
n = 10

original_ppl = get_perplexity(model, dataset, batch_size=ppl_batch_size)
pert_norms = pt.linspace(0, 20, num_buckets)
perplexities = []
stds = []
for pert_norm in pert_norms:
    cur_perplexities = [eval_pert_impact(pert_norm, module_name) for _ in range(5)]
    cur_perplexities = pt.tensor(cur_perplexities)
    perplexities.append(cur_perplexities.mean())
    # sems.append(cur_perplexities.std() / len(cur_perplexities) ** 0.5)
    stds.append(cur_perplexities.std())


# %%
# fit a quadratic curve to the data
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


pert_norms_np = pert_norms.numpy()
perplexities_np = pt.tensor(perplexities).numpy()
# sems_np = pt.tensor(sems).numpy()
popt, pcov = curve_fit(quadratic, pert_norms_np, perplexities_np)
popt, pcov

# %%
# light theme
plt.style.use("default")

# plot the fitted curve
x = np.linspace(0, 20, 100)
y = quadratic(x, *popt)
plt.plot(x, y)
# plot the points
plt.errorbar(pert_norms, perplexities, yerr=stds, fmt="o")
# also plot the original perplexity as grey horizontal line
plt.axhline(original_ppl, color="grey", linestyle="-")
# title
plt.title(f"Perplexity vs perturbation norm\n{module_name}")
# labels
plt.xlabel("Perturbation norm")
plt.ylabel("Perplexity")

# %% now, try for just one perturbation, scaled differently

module_name = "model.layers.10.mlp.down_proj.weight"
original_tensor = original_state_dict[module_name]
perturbation = pt.rand_like(original_tensor)
perturbation /= perturbation.norm()

perplexities = []
pert_norms = pt.linspace(0, 20, 21)
# pert_norms = pt.linspace(0, 0.0001, 31)
for pert_norm in pert_norms:
    scaled_perturbation = perturbation * pert_norm
    # apply perturbation
    model.state_dict()[module_name] += scaled_perturbation
    # eval
    perplexity = get_perplexity(model, dataset, batch_size=ppl_batch_size)
    perplexities.append(perplexity)
    # revert - note: must be done in-place
    model.state_dict()[module_name] *= 0
    model.state_dict()[module_name] += original_tensor

# fit a quadratic curve to the data
pert_norms_np = pert_norms.numpy()
perplexities_np = pt.tensor(perplexities).numpy()
popt, pcov = curve_fit(quadratic, pert_norms_np, perplexities_np)
popt, pcov

# plot
# light theme
plt.style.use("default")
# plot the fitted curve
x = np.linspace(0, 20, 100)
# x = np.linspace(0, 0.0001, 100)
y = quadratic(x, *popt)
plt.plot(x, y)
# plot the points
plt.plot(pert_norms, perplexities, "o")
# also plot the original perplexity as grey horizontal line
plt.axhline(original_ppl, color="grey", linestyle="-")
# title
plt.title(f"Perplexity vs perturbation norm for one perturbation\n{module_name}")
# labels
plt.xlabel("Perturbation norm")
plt.ylabel("Perplexity")
# %%
