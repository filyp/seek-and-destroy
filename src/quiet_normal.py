# %% init things, which need to be run once
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch as pt
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import forward, load_one_oscar_shard, set_seeds

pt.set_default_device("cuda")
set_seeds(42)

# load model
# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)
batch_size = 32
assert batch_size // 8 > 0
forget_iter = iter(forget_set["unlearn"].batch(batch_size))
retain_iter = iter(retain_set["unlearn"].batch(batch_size))
smol_forget_iter = iter(forget_set["alt_unlearn"].batch(batch_size // 8))
smol_retain_iter = iter(retain_set["alt_unlearn"].batch(batch_size // 8))
forget_eval_batch = next(iter(forget_set["validation"].batch(batch_size)))
retain_eval_batch = next(iter(retain_set["validation"].batch(batch_size)))

# get initial perplexities
with pt.no_grad():
    initial_forget_ppl = forward(model, forget_eval_batch).exp()
    initial_retain_ppl = forward(model, retain_eval_batch).exp()

# initialize LoRAs
model.add_adapter(
    LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=2,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],  # fmt: skip
    ),
    adapter_name="forget_lora"
)
model.add_adapter(
    LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,  # larger rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],  # fmt: skip
    ),
    adapter_name="retain_lora"
)

def forward_and_get_quietness_loss(model, batch):
    input_ids = pt.cat(batch["input_ids"])
    out = model(input_ids, output_hidden_states=True)
    return out.hidden_states[-1].norm(dim=-1).mean()


def only_grad_on(model, name_part):
    for name, param in model.named_parameters():
        param.requires_grad = name_part in name


# optimizer = pt.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = pt.optim.SGD(model.parameters(), lr=0.0001)

# %%
lr = SimpleNamespace(
    forget=0.02,
    retain=10,
    adv_forget=1,
    adv_retain=0.1,
)

start_time = time.time()
for i in range(10):
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss = SimpleNamespace()

    only_grad_on(model, ".base_layer.")
    model.set_adapter(["retain_lora", "forget_lora"])
    loss.forget = forward_and_get_quietness_loss(model, next(smol_forget_iter))

    only_grad_on(model, ".retain_lora.")
    model.set_adapter(["retain_lora"])
    loss.retain = forward(model, next(retain_iter))

    only_grad_on(model, ".forget_lora.")
    model.set_adapter(["retain_lora", "forget_lora"])
    loss.adv_forget = forward(model, next(smol_forget_iter))

    only_grad_on(model, ".forget_lora.")
    model.set_adapter(["retain_lora", "forget_lora"])
    loss.adv_retain = forward(model, next(smol_retain_iter))

    (
        loss.forget * lr.forget
        + loss.retain * lr.retain
        + loss.adv_forget * lr.adv_forget
        + loss.adv_retain * lr.adv_retain
    ).backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        model.eval()
        with pt.no_grad():
            model.set_adapter(["retain_lora", "forget_lora"])
            loss.forget = forward_and_get_quietness_loss(model, forget_eval_batch)
            model.set_adapter(["retain_lora"])
            loss.retain = forward(model, retain_eval_batch)
            model.set_adapter(["retain_lora", "forget_lora"])
            loss.adv_forget = forward(model, forget_eval_batch)
            model.set_adapter(["retain_lora", "forget_lora"])
            loss.adv_retain = forward(model, retain_eval_batch)

    print(
        f"{i + 1:4d}  "
        f"forget: {loss.forget.item():5.2f}  "
        f"retain: {loss.retain.exp() - initial_retain_ppl:5.2f}  "
        f"adv_forget: {loss.adv_forget.exp() - initial_forget_ppl:5.2f}  "
        f"adv_retain: {loss.adv_retain.exp() - initial_retain_ppl:5.2f} "
        f"{'< EVAL\n' if (i + 1) % 10 == 0 else ''}"
    )
print(f"time: {time.time() - start_time:.2f}s")
# %%
