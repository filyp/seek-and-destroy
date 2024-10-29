# %%
from common_startup_code import *


# %% init things, which need to be run once
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


# intervene, based on relative importance
def intervene(model, module_name, percentile, mult, cutoff=None):
    modules = [getattr(l.mlp, module_name) for l in model.model.layers]

    # calculate cutoff
    if cutoff is None:
        rel_imps = [m.imp["forget"] / m.imp["retain"] for m in modules]
        rel_imps = pt.cat(rel_imps).flatten()
        k = int(len(rel_imps) * (1 - percentile / 100))
        cutoff = rel_imps.kthvalue(k).values.item()

    # apply intervention
    for m in modules:
        rel_imp = m.imp["forget"] / m.imp["retain"]
        if module_name == "down_proj":
            m.weight.data[:, rel_imp > cutoff] *= mult
        else:
            m.weight.data[rel_imp > cutoff] *= mult
    print(f"{module_name=}  {percentile=}  {mult=}  {cutoff=}")


unlearn_batch_size = 32
forget_unlearn_iter = iter(forget_set["unlearn"].batch(unlearn_batch_size))
retain_unlearn_iter = iter(retain_set["unlearn"].batch(unlearn_batch_size))
relearn_batch_size = 16
forget_relearn_iter = iter(forget_set["relearn"].batch(relearn_batch_size))
retain_relearn_iter = iter(retain_set["relearn"].batch(relearn_batch_size))


# get the retain set weight importance
mode = "retain"
for batch in islice(retain_unlearn_iter, 10):
    with pt.no_grad():
        forward(model, batch)
mode = "off"

initial_stats = get_stats(model, forget_set, retain_set)

# initialize LoRA
# note: there is no way to set init seed, so it is not deterministic
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config).model

# %% forward passes on forget and retain sets, to get activation stats
for l in model.model.layers:
    if "forget" in l.mlp.up_proj.imp:
        del l.mlp.up_proj.imp["forget"]
        del l.mlp.gate_proj.imp["forget"]
        del l.mlp.down_proj.imp["forget"]

mode = "forget"
for batch in islice(forget_unlearn_iter, 5):
    with pt.no_grad():
        forward(model, batch)
mode = "off"

# %
intervene(model, "up_proj", percentile=0.02, mult=-1, cutoff=100)
# intervene(model, "gate_proj", percentile=0.01, mult=0)
# intervene(model, "down_proj", percentile=0.01, mult=0)

model.eval()
print_stats(get_stats(model, forget_set, retain_set) - initial_stats)

# for regression tests:
# intervene(model, "up_proj", percentile=0.02, mult=-1)  # forget_b=5, retain_b=10
# forget: 1048  retain:  0.01  ratio: 182620

# %% retrain and evaluate
pt.cuda.empty_cache()

num_batches = 10
optimizer = pt.optim.SGD(model.parameters(), lr=0.02)

for i in range(num_batches):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    forget_loss = forward(model, next(forget_relearn_iter))
    retain_loss = forward(model, next(retain_relearn_iter))
    (forget_loss + retain_loss).backward()
    optimizer.step()
    print(
        f"forget: {forget_loss.exp() - initial_stats[0]:4.0f}  "
        f"retain: {retain_loss.exp() - initial_stats[1]:5.2f}"
    )
    if (i + 1) % 10 == 0:
        model.eval()
        print_stats(get_stats(model, forget_set, retain_set) - initial_stats)
# %%
