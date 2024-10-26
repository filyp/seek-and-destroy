# %%
from common_startup_code import *


# %%
def record_activation_importance_stats(module, args, output):
    if mode not in module.act_imps:
        module.act_imps[mode] = pt.zeros(module.weight.shape[1])

    act = args[0].detach().clone()
    act = act**2
    act = act.mean(axis=[0, 1])
    module.act_imps[mode] += act


for module_name, module in og_model.named_modules():
    if "down_proj" not in module_name:
        continue
    module.register_forward_hook(record_activation_importance_stats)
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

mode = "dump"  # stop saving activations beyond this point

# %% load a fresh model, and apply the intervention based on the activation stats
percentile_to_mult = OrderedDict({
    # 0.02: -1,
    # 0.1: -0.5,
    0.2: 0.0,
    # 1: 0.5,
})
assert sorted(percentile_to_mult.keys()) == list(percentile_to_mult.keys())

# calculate cutoff, based on relative importance
rel_imps = []
for module_name, og_module in og_model.named_modules():
    if "down_proj" not in module_name:
        continue
    forget_imp = og_module.act_imps["forget"]
    retain_imp = og_module.act_imps["retain"]
    rel_imp = forget_imp / retain_imp
    og_module.rel_imp = rel_imp
    rel_imps.append(rel_imp)
rel_imps = pt.cat(rel_imps)

cutoffs = [
    rel_imps.quantile(1 - percentile / 100).item()
    for percentile in percentile_to_mult.keys()
]
low_bounds = cutoffs
high_bounds = [float("inf")] + cutoffs[:-1]
mults = list(percentile_to_mult.values())

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
for module_name, og_module in og_model.named_modules():
    if "down_proj" not in module_name:
        continue
    weight = model.state_dict()[module_name + ".weight"]
    with pt.no_grad():
        for low, high, mult in zip(low_bounds, high_bounds, mults):
            mask = pt.logical_and(low <= og_module.rel_imp, og_module.rel_imp < high)
            weight[:, mask] *= mult

stats = retrain_and_eval(model, og_model, forget_set, retain_set)

# weight_multiplier = -0.4
# percentile = 0.2
# forget: 1112  retain:  0.12  norm: 12.56  ratio: 9115
# forget:  678  retain:  0.33  norm: 12.56  ratio: 2048

# weight_multiplier = 0
# percentile = 0.2
# forget:  736  retain:  0.06  norm:  9.00  ratio: 11456
# forget:  389  retain:  0.32  norm:  9.00  ratio: 1230

# %%
import matplotlib.pyplot as plt

_rel_imps = pt.cat(rel_imps).cpu().float()
plt.hist(pt.log10(_rel_imps), bins=100)
plt.title("log10(relative importance)")
