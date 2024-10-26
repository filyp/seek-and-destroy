# %%
from common_startup_code import *


# %%
def record_activation_importance_stats(module, args, output):
    if mode not in ["forget", "retain"]:
        return
    if mode not in module.act_imps:
        module.act_imps[mode] = pt.zeros(module.weight.shape[1])

    act = args[0].detach().clone()
    act = act**2
    act = act.mean(axis=[0, 1])
    module.act_imps[mode] += act


for layer in og_model.model.layers:
    layer.mlp.down_proj.register_forward_hook(record_activation_importance_stats)
    layer.mlp.down_proj.act_imps = {}

# %% forward passes on forget and retain sets, to get activation stats
forward_batch_size = 64

mode = "forget"
for batch in islice(forget_set["unlearn"].batch(forward_batch_size), 10):
    with pt.no_grad():
        forward(og_model, batch)
mode = "off"

mode = "retain"
for batch in islice(retain_set["unlearn"].batch(forward_batch_size), 10):
    with pt.no_grad():
        forward(og_model, batch)
mode = "off"

# %%
pt.cuda.empty_cache()
# %% load a fresh model, and apply the intervention based on the activation stats
mult = 0
percentile = 0.2

# calculate cutoff, based on relative importance
rel_imps = []
for layer in og_model.model.layers:
    og_module = layer.mlp.down_proj
    og_module.rel_imp = og_module.act_imps["forget"] / og_module.act_imps["retain"]
    rel_imps.append(og_module.rel_imp)
rel_imps = pt.cat(rel_imps)
cutoff = rel_imps.kthvalue(int(len(rel_imps) * (1 - percentile / 100))).values

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

for layer, og_layer in zip(model.model.layers, og_model.model.layers):
    mask = og_layer.mlp.down_proj.rel_imp > cutoff
    layer.mlp.down_proj.weight.data[:, mask] *= mult

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
plt.hist(pt.log10(rel_imps).cpu().float(), bins=100)
plt.title("log10(relative importance on neurons)")
