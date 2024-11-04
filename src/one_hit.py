# %% init things, which need to be run once
import time

import matplotlib.pyplot as plt
import torch as pt
from peft import LoraConfig, TaskType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import *

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
forget_iter = looping_iter(forget_set["unlearn"])
retain_iter = looping_iter(retain_set["unlearn"])
forget_eval_batch = get_batch(iter(forget_set["validation"]), 32)
retain_eval_batch = get_batch(iter(retain_set["validation"]), 32)


# %% accumulate forget gradients

# accumulation_steps = 300
# # calculate
# model.zero_grad(set_to_none=True)
# for _ in tqdm(range(accumulation_steps)):
#     loss_forget = forward_with_clipped_logit(model, get_batch(forget_iter, 32))
#     loss_forget.backward()
#     # do not optimizer.step() !
# # gather
# grads = {
#     name: param.grad / accumulation_steps
#     for name, param in model.named_parameters()
# }
# model.zero_grad(set_to_none=True)
# # save
# circuit_path = get_repo_root() / "circuits" / f"forward_with_clipped_logit_300.pt"
# pt.save(grads, circuit_path)

# load cached
circuit_path = get_repo_root() / "circuits" / f"forward_with_clipped_logit_300.pt"
grads = pt.load(circuit_path)

# %%
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
optimizer = pt.optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.999))
i = 0

# get initial perplexities
model.eval()
with pt.no_grad():
    print("initial perplexities:")
    print(f"forget: {forward(model, forget_eval_batch).exp()}")
    print(f"retain: {forward(model, retain_eval_batch).exp()}")

# %%
for _ in range(1000000):
    i += 1
    model.train()
    loss_forget = pt.tensor(pt.nan)

    for name, param in model.named_parameters():
        # if "proj" in name:
        param.data -= grads[name] * 0.001

    optimizer.zero_grad(set_to_none=True)
    loss_retain = forward(model, get_batch(retain_iter, 16))
    loss_retain.backward()
    optimizer.step()

    # evaluate
    if i % 10 == 0:
        model.eval()
        with pt.no_grad():
            loss_forget = forward(model, forget_eval_batch)
            loss_retain = forward(model, retain_eval_batch)

    # calculate and print stats
    stats = dict(
        forget=loss_forget.exp(),
        retain=loss_retain.exp(),
    )

    if i % 10 == 1:
        print("\n      " + "   ".join(f"{k:>10}" for k in stats.keys()))
    print(f"{i:4d}  " + "   ".join(f"{v:10.2f}" for v in stats.values()))

    # save model
    if i % 500 == 0:
        # ! change this name
        model_path = get_repo_root() / "models" / f"one_hit5_{i}steps.pt"
        pt.save(model.state_dict(), model_path)

# %%
