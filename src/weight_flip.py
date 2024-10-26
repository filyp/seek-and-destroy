# %%
from common_startup_code import *

# og_model.model.layers[0].mlp.act_fn
# y = pt.ops.aten.silu_backward(pt.ones_like(x), x)


# %%
def save_output_hook(module, args, output):
    module.last_output = output.detach().clone()


def mlp_compute_imps(module, args, output):
    mlp_input = args[0].detach().clone()
    # for up_proj
    mlp_input2 = mlp_input**2
    gate_out = module.gate_proj.last_output
    gate_out_act2 = pt.nn.functional.silu(gate_out) ** 2
    imp = pt.einsum("bti,btj->ij", gate_out_act2, mlp_input2)
    if mode not in module.up_proj.imp:
        module.up_proj.imp[mode] = pt.zeros_like(module.up_proj.weight)
    module.up_proj.imp[mode] += imp


for layer in og_model.model.layers:
    assert isinstance(layer.mlp.act_fn, pt.nn.SiLU)

    layer.mlp.up_proj.register_forward_hook(save_output_hook)
    layer.mlp.gate_proj.register_forward_hook(save_output_hook)
    layer.mlp.register_forward_hook(mlp_compute_imps)
    layer.mlp.up_proj.imp = {}

# %% forward passes on forget and retain sets, to get activation stats
forward_batch_size = 32

mode = "forget"
for batch in islice(forget_set["unlearn"].batch(forward_batch_size), 10):
    with pt.no_grad():
        forward(og_model, batch)

mode = "retain"
for batch in islice(retain_set["unlearn"].batch(forward_batch_size), 10):
    with pt.no_grad():
        forward(og_model, batch)

# %% calculate relative importance
rel_imps = []
for layer in og_model.model.layers:
    rel_imp = layer.mlp.up_proj.imp["forget"] / layer.mlp.up_proj.imp["retain"]
    layer.mlp.up_proj.rel_imp = rel_imp
    rel_imps.append(rel_imp)
rel_imps = pt.cat(rel_imps).flatten()

# %% clear hooks and memory
for layer in og_model.model.layers:
    layer.mlp.up_proj._forward_hooks.clear()
    layer.mlp.gate_proj._forward_hooks.clear()
    layer.mlp._forward_hooks.clear()
    layer.mlp.up_proj.imp = {}
pt.cuda.empty_cache()

# %% calculate cutoff, based on relative importance, and apply intervention
mult = -0.5
percentile = 0.05

cutoff = rel_imps.kthvalue(int(len(rel_imps) * (1 - percentile / 100))).values

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
for layer, og_layer in zip(model.model.layers, og_model.model.layers):
    mask = og_layer.mlp.up_proj.rel_imp > cutoff
    layer.mlp.up_proj.weight.data[mask] *= mult

# retrain and evaluate
stats = retrain_and_eval(model, og_model, forget_set, retain_set)

# mult = -0.5
# percentile = 0.05
# forget:  648  retain:  0.07  norm:  6.66  ratio: 9674
# forget:  250  retain:  2.66  norm:  6.66  ratio: 94
# %%
