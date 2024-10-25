# %%
from common_startup_code import *


# %%
def record_activation_importance_stats(module, args, output):
    if mode not in module.act_imps:
        module.act_imps[mode] = pt.zeros(module.weight.shape[1]).to(device)

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

# %% load a fresh model, and apply the intervention based on the activation stats
percentile = 0.1
weight_multiplier = 0

for module_name, og_module in og_model.named_modules():
    if "down_proj" not in module_name:
        continue
    forget_imp = og_module.act_imps["forget"]
    retain_imp = og_module.act_imps["retain"]
    rel_imp = forget_imp / retain_imp
    cutoff = rel_imp.quantile(1 - percentile / 100)

    weight = model.state_dict()[module_name + ".weight"]
    with pt.no_grad():
        weight[:, rel_imp > cutoff] *= weight_multiplier

# %%
eval_and_retrain(model, og_model, forget_set, retain_set)
