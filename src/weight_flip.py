# %%
from common_startup_code import *


# %%
def save_output_hook(module, args, output):
    module.last_output = output.detach().clone()


def record_imps_down_proj(module, args, output):
    if mode not in ["forget", "retain"]:
        return
    if mode not in module.imp:
        module.imp[mode] = pt.zeros(module.weight.shape[1])

    act = args[0].detach().clone()
    act = act**2
    act = act.mean(axis=[0, 1])
    module.imp[mode] += act


def record_imps_up_and_gate_proj(module, args, output):
    if mode not in ["forget", "retain"]:
        return
    mlp_input2 = args[0].detach().clone() ** 2

    # for up_proj
    if mode not in module.up_proj.imp:
        module.up_proj.imp[mode] = pt.zeros_like(module.up_proj.weight)
    gate_out = module.gate_proj.last_output

    gate_out_act2 = pt.nn.functional.silu(gate_out) ** 2
    module.up_proj.imp[mode] += pt.einsum("bti,btj->ij", gate_out_act2, mlp_input2)

    # for gate_proj
    if mode not in module.gate_proj.imp:
        module.gate_proj.imp[mode] = pt.zeros_like(module.gate_proj.weight)
    gate_out = module.gate_proj.last_output
    up_out = module.up_proj.last_output

    gate_d = pt.ops.aten.silu_backward(pt.ones_like(gate_out), gate_out)
    out2 = (up_out * gate_d) ** 2  # this is adequate for mellow mult (close to 1)
    # out2 = (up_out) ** 2  # this is better for aggressive mult (close to -1)
    module.gate_proj.imp[mode] += pt.einsum("bti,btj->ij", out2, mlp_input2)


for l in model.model.layers:
    assert isinstance(l.mlp.act_fn, pt.nn.SiLU)
    l.mlp.up_proj.register_forward_hook(save_output_hook)
    l.mlp.gate_proj.register_forward_hook(save_output_hook)
    l.mlp.down_proj.register_forward_hook(record_imps_down_proj)
    l.mlp.register_forward_hook(record_imps_up_and_gate_proj)
    l.mlp.up_proj.imp = {}
    l.mlp.gate_proj.imp = {}
    l.mlp.down_proj.imp = {}


# %% intervene, based on relative importance
def intervene(model, module_name, percentile, mult):
    modules = [getattr(l.mlp, module_name) for l in model.model.layers]

    # calculate cutoff
    rel_imps = [m.imp["forget"] / m.imp["retain"] for m in modules]
    rel_imps = pt.cat(rel_imps).flatten()
    cutoff = rel_imps.kthvalue(int(len(rel_imps) * (1 - percentile / 100))).values
    print(f"{module_name=}  {percentile=}  {mult=}  {cutoff.item()=}")

    # apply intervention
    for m in modules:
        rel_imp = m.imp["forget"] / m.imp["retain"]
        if module_name == "down_proj":
            m.weight.data[:, rel_imp > cutoff] *= mult
        else:
            m.weight.data[rel_imp > cutoff] *= mult


forward_batch_size = 32

mode = "retain"
for batch in islice(retain_set["unlearn"].batch(forward_batch_size), 10):
    with pt.no_grad():
        forward(model, batch)
mode = "off"

# %% forward passes on forget and retain sets, to get activation stats
for l in model.model.layers:
    if "forget" in l.mlp.up_proj.imp:
        del l.mlp.up_proj.imp["forget"]
        del l.mlp.gate_proj.imp["forget"]
        del l.mlp.down_proj.imp["forget"]

mode = "forget"
for batch in islice(forget_set["unlearn"].batch(forward_batch_size), 10):
    with pt.no_grad():
        forward(model, batch)
mode = "off"

# %%
model.load_state_dict(og_model.state_dict())
initial_stats = get_stats(model, og_model, forget_set, retain_set)

intervene(model, "up_proj", percentile=0.01, mult=-1)
# intervene(model, "gate_proj", percentile=0.01, mult=-1)
# intervene(model, "down_proj", percentile=0.01, mult=0)

# stats = get_stats(model, og_model, forget_set, retain_set)
# print_stats(stats - initial_stats)

# %% retrain and evaluate
pt.cuda.empty_cache()
stats = retrain_and_eval(model, og_model, forget_set, retain_set)

# for regression tests:
# intervene(model, "up_proj", percentile=2.0, mult=0)  # forget_b=1, retain_b=1
# forget: 1898  retain:  2.36  norm: 30.25  ratio: 806

# %%
