# %%
from common_startup_code import *

# model.model.layers[0].mlp.act_fn
# y = pt.ops.aten.silu_backward(pt.ones_like(x), x)


# %%
def save_output_hook(module, args, output):
    module.last_output = output.detach().clone()


def record_imps_up_and_gate_proj(module, args, output):
    if mode not in ["forget", "retain"]:
        return
    mlp_input2 = args[0].detach().clone() ** 2

    # for up_proj
    gate_out = module.gate_proj.last_output
    gate_out_act2 = pt.nn.functional.silu(gate_out) ** 2
    imp = pt.einsum("bti,btj->ij", gate_out_act2, mlp_input2)
    if mode not in module.up_proj.imp:
        module.up_proj.imp[mode] = pt.zeros_like(module.up_proj.weight)
    module.up_proj.imp[mode] += imp


for l in model.model.layers:
    assert isinstance(l.mlp.act_fn, pt.nn.SiLU)

    l.mlp.up_proj.register_forward_hook(save_output_hook)
    l.mlp.gate_proj.register_forward_hook(save_output_hook)
    l.mlp.register_forward_hook(record_imps_up_and_gate_proj)
    l.mlp.up_proj.imp = {}

# %% forward passes on forget and retain sets, to get activation stats
forward_batch_size = 32

mode = "forget"
for batch in islice(forget_set["unlearn"].batch(forward_batch_size), 1):
    with pt.no_grad():
        forward(model, batch)
mode = "off"

mode = "retain"
for batch in islice(retain_set["unlearn"].batch(forward_batch_size), 1):
    with pt.no_grad():
        forward(model, batch)
mode = "off"

# %% intervene, based on relative importance
mult = -0.5
percentile = 0.05

rel_imps = [
    l.mlp.up_proj.imp["forget"] / l.mlp.up_proj.imp["retain"]
    for l in model.model.layers
]
rel_imps = pt.cat(rel_imps).flatten()
cutoff = rel_imps.kthvalue(int(len(rel_imps) * (1 - percentile / 100))).values

# apply intervention
for l in model.model.layers:
    rel_imp = l.mlp.up_proj.imp["forget"] / l.mlp.up_proj.imp["retain"]
    l.mlp.up_proj.weight.data[rel_imp > cutoff] *= mult

# %% retrain and evaluate
# clear memory
for l in model.model.layers:
    l.mlp.up_proj.imp = {}
pt.cuda.empty_cache()

stats = retrain_and_eval(model, og_model, forget_set, retain_set)

# mult = -0.5
# percentile = 0.05
# forget:  648  retain:  0.07  norm:  6.66  ratio: 9674
# forget:  250  retain:  2.66  norm:  6.66  ratio: 94
# %%
