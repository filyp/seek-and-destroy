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
    target_modules="all-linear",
)
model.add_adapter(peft_config)

forget_unlearn_iter = iter(forget_set["unlearn"].batch(32))
retain_unlearn_iter = iter(retain_set["unlearn"].batch(32))
forget_relearn_iter = iter(forget_set["relearn"].batch(24))
retain_relearn_iter = iter(retain_set["relearn"].batch(8))

initial_stats = get_stats(model, forget_set, retain_set)


# %%
def custom_forward(model, batch):
    input_ids = pt.cat(batch["input_ids"])
    # forward pass
    logits = model(input_ids).logits
    pred = logits[:, :-1, :].flatten(end_dim=1)
    true = input_ids[:, 1:].flatten()
    preds = pred[pt.arange(len(true)), true]
    return preds.sum()


def modules(model):
    for l in model.model.layers:
        # yield l.mlp.up_proj.base_layer
        # yield l.mlp.gate_proj.base_layer  # terrible
        yield l.mlp.down_proj.base_layer
        # yield l.self_attn.q_proj.base_layer  # no effect!
        # yield l.self_attn.k_proj.base_layer    # no effect
        # yield l.self_attn.v_proj.base_layer    # no effect
        # yield l.self_attn.o_proj.base_layer    # no effect


for m in modules(model):
    m.grad_acc = {}


# %%
def record_imp_from_grads(model, mode, dataset):
    model.disable_adapters()
    for m in modules(model):
        assert m.weight.requires_grad == False
        m.weight.requires_grad = True
        m.grad_acc[mode] = pt.zeros_like(m.weight)

    for batch in islice(dataset, 10):
        model.zero_grad(set_to_none=True)
        # loss = custom_forward(model, batch)
        loss = forward(model, batch)
        loss.backward()
        for m in modules(model):
            # m.grad_acc[mode] += pt.clip(m.weight.grad * m.weight.data.sign() * (-1), min=0)
            m.grad_acc[mode] += m.weight.grad.abs()

    model.enable_adapters()
    for m in modules(model):
        m.weight.requires_grad = False


record_imp_from_grads(model, "forget", forget_unlearn_iter)
# record_imp_from_grads(model, "retain", retain_unlearn_iter)

# %% intervene
cutoff = 20
mult = 0
for m in modules(model):
    # rel_imp = m.grad_acc["forget"].abs() / m.grad_acc["retain"].abs()
    rel_imp = m.grad_acc["forget"] / m.grad_acc["retain"]
    mask = rel_imp > cutoff
    m.weight.data[mask] *= mult

# %
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

# %%
plt.hist(rel_imp.flatten().log10().tolist(), bins=100)
# %% temporarily disable lora
model.disable_adapters()
model.eval()
print_stats(get_stats(model, forget_set, retain_set) - initial_stats)
model.enable_adapters()

