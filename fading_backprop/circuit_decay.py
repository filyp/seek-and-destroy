# %%
from itertools import islice

import matplotlib.pyplot as plt
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    device,
    forward,
    get_stats,
    load_one_oscar_shard,
    normal_train_step,
    print_stats,
    scale_perturbation,
)

# params
model_id = "Qwen/Qwen2.5-0.5B"

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)

# %% install hooks for storing activation stats
# load model; no interventions will be done on the original model
og_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
og_model.to(device)


def record_activation_importance_stats(module, args, output):
    if mode not in module.counters:
        module.counters[mode] = 0
        module.act_imps[mode] = pt.zeros(module.weight.shape[1]).to(device)

    module.counters[mode] += 1
    act = args[0].detach().clone()
    act = act**2
    act = act.mean(axis=[0, 1])
    module.act_imps[mode] += act


for module_name, module in og_model.named_modules():
    if "down_proj" not in module_name:
        continue
    module.register_forward_hook(record_activation_importance_stats)
    module.counters = {}
    module.act_imps = {}


# %% forward passes on forget and retain sets, to get activation stats
forward_batch_size = 64

mode = "forget"
for batch in islice(forget_set["unlearn"].batch(forward_batch_size), 10):
    with pt.no_grad():
        forward(og_model, batch)

mode = "retain"
for batch in islice(retain_set["unlearn"].batch(forward_batch_size), 10):
    with pt.no_grad():
        forward(og_model, batch)

# %% load a fresh model, and apply the intervention based on the activation stats
percentile = 0.1
mult = 0

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.to(device)

for module_name, og_module in og_model.named_modules():
    if "down_proj" not in module_name:
        continue
    forget_imp = og_module.act_imps["forget"] / og_module.counters["forget"]
    retain_imp = og_module.act_imps["retain"] / og_module.counters["retain"]
    rel_imp = forget_imp / retain_imp
    cutoff = rel_imp.quantile(1 - percentile / 100)

    weight = model.state_dict()[module_name + ".weight"]
    with pt.no_grad():
        weight[:, rel_imp > cutoff] *= mult

initial_stats = get_stats(og_model, og_model, forget_set, retain_set)
print_stats(get_stats(model, og_model, forget_set, retain_set) - initial_stats)

# %% retrain the model
b_size = 32
num_batches = 10
batch_initial_stats = get_stats(model, og_model, forget_set, retain_set)

f_r = zip(forget_set["relearn"].batch(b_size), retain_set["relearn"].batch(b_size))
for forget_batch, retain_batch in islice(f_r, num_batches):
    print(".", end="")
    normal_train_step(model, forget_batch, 0.0003)
    normal_train_step(model, retain_batch, 0.0003)

print_stats(get_stats(model, og_model, forget_set, retain_set) - batch_initial_stats)
