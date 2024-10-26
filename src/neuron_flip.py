# %%
from common_startup_code import *


# %%
def record_imps_down_proj(module, args, output):
    if mode not in ["forget", "retain"]:
        return
    if mode not in module.imp:
        module.imp[mode] = pt.zeros(module.weight.shape[1])

    act = args[0].detach().clone()
    act = act**2
    act = act.mean(axis=[0, 1])
    module.imp[mode] += act


for l in model.model.layers:
    l.mlp.down_proj.register_forward_hook(record_imps_down_proj)
    l.mlp.down_proj.imp = {}

# %% forward passes on forget and retain sets, to get activation stats
forward_batch_size = 64

mode = "forget"
for batch in islice(forget_set["unlearn"].batch(forward_batch_size), 10):
    with pt.no_grad():
        forward(model, batch)
mode = "off"

mode = "retain"
for batch in islice(retain_set["unlearn"].batch(forward_batch_size), 10):
    with pt.no_grad():
        forward(model, batch)
mode = "off"

# %% load a fresh model, and apply the intervention based on the activation stats
mult = 0
percentile = 0.2

# calculate cutoff, based on relative importance
rel_imps = [
    l.mlp.down_proj.imp["forget"] / l.mlp.down_proj.imp["retain"]
    for l in model.model.layers
]
rel_imps = pt.cat(rel_imps)
cutoff = rel_imps.kthvalue(int(len(rel_imps) * (1 - percentile / 100))).values

for l in model.model.layers:
    rel_imp = l.mlp.down_proj.imp["forget"] / l.mlp.down_proj.imp["retain"]
    l.mlp.down_proj.weight.data[:, rel_imp > cutoff] *= mult

stats = retrain_and_eval(model, og_model, forget_set, retain_set)

# weight_multiplier = 0
# percentile = 0.2
# forget:  736  retain:  0.06  norm:  9.00  ratio: 11456
# forget:  389  retain:  0.32  norm:  9.00  ratio: 1230

# %%
plt.hist(pt.log10(rel_imps).cpu().float(), bins=100)
plt.title("log10(relative importance on neurons)")
