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
model.add_adapter(peft_config, adapter_name="retain_lora")

b_size = 32
forget_iter = iter(forget_set["unlearn"].batch(b_size))
retain_iter = iter(retain_set["unlearn"].batch(b_size))
forget_eval_batch = next(iter(forget_set["validation"].batch(b_size)))
retain_eval_batch = next(iter(retain_set["validation"].batch(b_size)))

initial_stats = get_stats(model, forget_set, retain_set)


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
lr = dict(
    unlearn=0.003,
    retain=2,
    adv_forget=2,
    adv_retain=1,
)

for i in range(20):
    optimizer.zero_grad(set_to_none=True)
    model.train()

    model.set_adapter(["retain_lora", "forget_lora"])
    only_require_grad_on(model, ".base_layer.")
    quietness_loss = forward_and_get_quietness_loss(model, next(forget_iter))
    (quietness_loss * lr["unlearn"]).backward()

    model.set_adapter(["retain_lora"])
    only_require_grad_on(model, ".retain_lora.")
    retain_loss = forward(model, next(retain_iter))
    (retain_loss * lr["retain"]).backward()

    model.set_adapter(["retain_lora", "forget_lora"])
    only_require_grad_on(model, ".forget_lora.")
    adv_forget_loss = forward(model, next(forget_iter))
    (adv_forget_loss * lr["adv_forget"]).backward()

    # # note: this actually could be much weaker, smaller batch
    # adv_retain_loss = forward(model, next(retain_iter))
    # (adv_retain_loss * lr["adv_retain"]).backward()
    adv_retain_loss = pt.tensor(pt.nan)

    optimizer.step()

    if (i + 1) % 10 == 0:
        model.eval()
        with pt.no_grad():
            model.set_adapter(["retain_lora", "forget_lora"])
            quietness_loss = forward_and_get_quietness_loss(model, forget_eval_batch)
            model.set_adapter(["retain_lora"])
            retain_loss = forward(model, retain_eval_batch)
            model.set_adapter(["retain_lora", "forget_lora"])
            adv_forget_loss = forward(model, forget_eval_batch)
            adv_retain_loss = forward(model, retain_eval_batch)

    print(
        f"{i + 1:4d}  "
        f"quietness: {quietness_loss.item():5.2f}  "
        f"bare_retain: {retain_loss.exp() - initial_stats[1]:5.2f}  "
        f"adv_forget: {adv_forget_loss.exp() - initial_stats[0]:5.2f}  "
        f"adv_retain: {adv_retain_loss.exp() - initial_stats[1]:5.2f} "
        f"{'< EVAL\n' if (i + 1) % 10 == 0 else ''}"
    )

# %%
# model.set_adapter(["retain_lora"])
# print(forward(model, forget_eval_batch).exp() - initial_stats[0])
