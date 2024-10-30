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
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],  # fmt: skip
)
model.add_adapter(lora_config, adapter_name="adversarial_lora")


def forward_and_get_clipped_logit_mean(model, batch):
    input_ids = pt.cat(batch["input_ids"])
    out = model(input_ids)
    all_logits = out.logits[:, :-1, :].flatten(end_dim=1)
    ids = input_ids[:, 1:].flatten()
    logits = all_logits[pt.arange(len(ids)), ids]
    return logits.clip(0).mean()


def only_grad_on(model, name_part):
    for name, param in model.named_parameters():
        param.requires_grad = name_part in name


# todo a separate optimizer for the lora - it can have a higher LR
optimizer = pt.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
# optimizer = pt.optim.SGD(model.parameters(), lr=0.0003)

# %%
lr = SimpleNamespace(
    # because of how Adam adapts itself, the only thing that matters is the ratios:
    # loudness / retain
    # adv_forget / adv_retain
    loudness=1,
    retain=200,
    adv_forget=5,
    adv_retain=1,
)

start_time = time.time()
for i in range(300):
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss = SimpleNamespace()
    loss.forget = pt.tensor(pt.nan)
    # note: only_grad_on must come after set_adapter

    model.set_adapter(["adversarial_lora"])
    only_grad_on(model, ".base_layer.")
    loss.loudness = forward_and_get_clipped_logit_mean(model, next(smol_forget_iter))
    (loss.loudness * lr.loudness).backward()

    model.set_adapter([])
    only_grad_on(model, ".base_layer.")
    loss.retain = forward(model, next(retain_iter))
    (loss.retain * lr.retain).backward()

    model.set_adapter(["adversarial_lora"])
    only_grad_on(model, ".adversarial_lora.")
    loss.adv_forget = forward(model, next(forget_iter))
    (loss.adv_forget * lr.adv_forget).backward()

    model.set_adapter(["adversarial_lora"])
    only_grad_on(model, ".adversarial_lora.")
    loss.adv_retain = forward(model, next(smol_retain_iter))
    (loss.adv_retain * lr.adv_retain).backward()

    optimizer.step()

    # evaluate
    if (i + 1) % 10 == 0:
        model.eval()
        with pt.no_grad():
            model.set_adapter(["adversarial_lora"])
            loss.loudness = forward_and_get_clipped_logit_mean(model, forget_eval_batch)
            model.set_adapter([])
            loss.forget = forward(model, forget_eval_batch)
            model.set_adapter([])
            loss.retain = forward(model, retain_eval_batch)
            model.set_adapter(["adversarial_lora"])
            loss.adv_forget = forward(model, forget_eval_batch)
            model.set_adapter(["adversarial_lora"])
            loss.adv_retain = forward(model, retain_eval_batch)

    print(
        f"{i + 1:4d}  "
        f"loudness: {loss.loudness.item():6.2f}  "
        f"forget: {loss.forget.exp() - initial_forget_ppl:7.0f}  "
        f"retain: {loss.retain.exp() - initial_retain_ppl:6.2f}  "
        f"adv_forget: {loss.adv_forget.exp() - initial_forget_ppl:6.2f}  "
        f"adv_retain: {loss.adv_retain.exp() - initial_retain_ppl:6.2f} "
        f"{'< EVAL\n' if (i + 1) % 10 == 0 else ''}"
    )
print(f"time: {time.time() - start_time:.2f}s")

# %%
# p = optimizer.param_groups[0]["params"][3]
# state = optimizer.state[p]
# state["exp_avg_sq"].mean()
