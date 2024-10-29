# %% init things, which need to be run once
if "pt" not in locals():
    from common_startup_code import *

# load model
set_seeds(42)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

# %%
# initialize LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=2,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],  # fmt: skip
)
model.add_adapter(peft_config, adapter_name="forget_lora")
# model.add_adapter(peft_config, adapter_name="retain_lora")

forget_iter = iter(forget_set["unlearn"].batch(16))
retain_iter = iter(retain_set["unlearn"].batch(16))

initial_stats = get_stats(model, forget_set, retain_set)


# %%
def forward_and_get_quietness_loss(model, batch):
    input_ids = pt.cat(batch["input_ids"])
    out = model(input_ids, output_hidden_states=True)
    return out.hidden_states[-1].norm(dim=-1).mean()


def only_require_grad_on(model, name_part):
    for name, param in model.named_parameters():
        param.requires_grad = name_part in name


# optimizer = pt.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = pt.optim.SGD(model.parameters(), lr=0.001)

# %%
for i in range(100):
    optimizer.zero_grad(set_to_none=True)

    only_require_grad_on(model, ".base_layer.")
    quietness_loss = forward_and_get_quietness_loss(model, next(forget_iter))
    (quietness_loss * 0.01).backward()

    # todo this could be an adapter, that's always on! needs testing
    # todo before each block, just set active_adapters list
    model.disable_adapters()
    only_require_grad_on(model, ".base_layer.")
    bare_retain_loss = forward(model, next(retain_iter))
    bare_retain_loss.backward()
    model.enable_adapters()

    only_require_grad_on(model, ".forget_lora.")
    forget_loss = forward(model, next(forget_iter))
    forget_loss.backward()
    retain_loss = forward(model, next(retain_iter))
    retain_loss.backward()

    optimizer.step()

    print(
        f"quietness: {quietness_loss.item():5.2f}  "
        f"bare_retain: {bare_retain_loss.exp() - initial_stats[1]:5.2f}  "
        f"forget: {forget_loss.exp() - initial_stats[0]:4.0f}  "
        f"retain: {retain_loss.exp() - initial_stats[1]:5.2f}  "
    )
    if (i + 1) % 10 == 0:
        model.eval()
        print_stats(get_stats(model, forget_set, retain_set) - initial_stats)

# %%
# model.active_adapters()
