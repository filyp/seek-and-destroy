# %%
if "pt" not in locals():
    from common_startup_code import *

# load model
set_seeds(42)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

# for regression tests:
# intervene(model, "up_proj", mult=-1, cutoff=100)  # forget_b=5, retain_b=10
# forget: 1279  retain:  0.09  ratio: 14753

# %% init things, which need to be run once

# initialize LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=2,
    lora_alpha=32,
    lora_dropout=0.1,
    # target_modules=["gate_proj", "up_proj", "q_proj", "v_proj"],
    target_modules="all-linear",
)
model.add_adapter(peft_config)


def save_output_hook(module, args, output):
    module.last_output = output.detach().clone()


# def record_imps_down_proj(module, args, output):
#     if mode not in ["forget", "retain"]:
#         return
#     if mode not in module.imp:
#         module.imp[mode] = pt.zeros(module.weight.shape[1])
#     act = args[0].detach().clone()
#     act = act**2
#     act = act.mean(axis=[0, 1])
#     module.imp[mode] += act


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
    # l.mlp.down_proj.register_forward_hook(record_imps_down_proj)
    l.mlp.register_forward_hook(record_imps_up_and_gate_proj)
    l.mlp.up_proj.imp = {}
    l.mlp.gate_proj.imp = {}
    # l.mlp.down_proj.imp = {}
    # for recording which indices were already flipped
    l.mlp.up_proj.xs = []
    l.mlp.up_proj.ys = []
    l.mlp.gate_proj.xs = []
    l.mlp.gate_proj.ys = []


forget_unlearn_iter = iter(forget_set["unlearn"].batch(32))
retain_unlearn_iter = iter(retain_set["unlearn"].batch(32))
forget_relearn_iter = iter(forget_set["relearn"].batch(24))
retain_relearn_iter = iter(retain_set["relearn"].batch(8))


# get the retain set weight importance
mode = "retain"
for batch in islice(retain_unlearn_iter, 10):
    with pt.no_grad():
        forward(model, batch)
mode = "off"

initial_stats = get_stats(model, forget_set, retain_set)


# %% intervene
for l in model.model.layers:
    if "forget" in l.mlp.up_proj.imp:
        del l.mlp.up_proj.imp["forget"]
        del l.mlp.gate_proj.imp["forget"]
        # del l.mlp.down_proj.imp["forget"]

mode = "forget"
for batch in islice(forget_unlearn_iter, 5):
    with pt.no_grad():
        forward(model, batch)
mode = "off"

module_name = "up_proj"
# module_name = "gate_proj"
cutoff = 50
mult = -3
for m in [getattr(l.mlp, module_name) for l in model.model.layers]:
    rel_imp = m.imp["forget"] / m.imp["retain"]
    mask = rel_imp > cutoff

    pre_sum = mask.sum()
    mask[m.xs, m.ys] = False
    print(f"to flip before filtering: {pre_sum:4}  after: {mask.sum():4}")
    x, y = pt.where(mask)

    m.xs.extend(x.tolist())
    m.ys.extend(y.tolist())

    m.weight.data[mask] *= mult
    # note: for down_proj it needs to be the code below, also the indices calc differently
    # m.weight.data[:, rel_imp > cutoff] *= mult
    # but don't care rn, just using one is enough

model.eval()
print_stats(get_stats(model, forget_set, retain_set) - initial_stats)

# %% retrain and evaluate
pt.cuda.empty_cache()

num_batches = 10
optimizer = pt.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

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

# %% temporarily disable lora
# model.disable_adapters()
# model.eval()
# print_stats(get_stats(model, forget_set, retain_set) - initial_stats)
# model.enable_adapters()

# %% remove lora
# sd = {
#     k.replace("base_layer.", ""): v
#     for k, v in model.state_dict().items()
#     if "lora" not in k
# }
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
# model.load_state_dict(sd)
# print_stats(get_stats(model, forget_set, retain_set) - initial_stats)
