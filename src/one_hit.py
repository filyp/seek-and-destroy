# %% init things, which need to be run once
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import torch as pt
from peft import LoraConfig, TaskType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import *

pt.set_default_device("cuda")

# load model
# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)
forget_eval_batch = get_batch(iter(forget_set["validation"]), 32)
retain_eval_batch = get_batch(iter(retain_set["validation"]), 32)


# %% accumulate forget gradients

# accumulation_steps = 3000
# forget_iter = looping_iter(forget_set["unlearn"])
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
# circuit_path = get_repo_root() / "circuits" / f"forward_with_clipped_logit_3000.pt"
# pt.save(grads, circuit_path)

# load cached
circuit_path = get_repo_root() / "circuits" / f"forward_with_clipped_logit_300.pt"
unwanted_circuit = pt.load(circuit_path)

# %%
c = SimpleNamespace(
    forget_lr=6e-5,
    retain_lr=4e-4,
    L2_revert_factor=1,  # seems that for .9995 it's the same as one? maybe it's a numerical error?
    acceptable_retain_ppl=24,
    forget_ppl_increment=1.2,
    momentum=0.9,
    rank=16,
)

set_seeds(42)
retain_iter = looping_iter(retain_set["unlearn"])

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=c.rank,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules="all-linear",
)
model.add_adapter(lora_config, adapter_name="retain_lora")

optimizer = pt.optim.Adam(model.parameters(), lr=c.retain_lr, betas=(c.momentum, 0.999))
# optimizer = pt.optim.SGD(model.parameters(), lr=0.0008)
# initial_state_dict = deepcopy(model.state_dict())

step = 0
last_lr_update = 0
forget_ppl_hist = []

# get initial perplexities
model.eval()
with pt.no_grad():
    print("initial perplexities:")
    print(f"forget: {forward(model, forget_eval_batch).exp()}")
    print(f"retain: {forward(model, retain_eval_batch).exp()}")

wandb.init(
    project="adversarial_adaptation",
    group="one_hit_unlearning",
    name=f"R{c.rank} f={c.forget_lr:.0e} r={c.retain_lr:.0e} L2={c.L2_revert_factor}",
    config=vars(c),
)

# %%
for _ in range(1000):
    step += 1
    model.train()
    loss_forget = pt.tensor(pt.nan)

    for name, param in model.named_parameters():
        name = name.replace(".base_layer", "")
        if name in unwanted_circuit:  # and "_proj" in name:
            param.data -= unwanted_circuit[name] * c.forget_lr

    optimizer.zero_grad(set_to_none=True)
    loss_retain = forward(model, get_batch(retain_iter, 8))
    loss_retain.backward()
    optimizer.step()

    # # L2 revert
    # for name, param in model.named_parameters():
    #     if ".base_layer" in name:
    #         initial_weights = initial_state_dict[name]
    #         delta = param.data - initial_weights
    #         param.data = initial_weights + delta * c.L2_revert_factor

    # # L1 revert
    # a = 0.00003
    # for name, param in model.named_parameters():
    #     initial_weights = initial_state_dict[name]
    #     delta = param.data - initial_weights
    #     new_delta = delta.clip(max=-a) + delta.clip(min=a)
    #     param.data = initial_weights + new_delta

    if step % 10 != 0:
        continue

    # evaluate
    model.eval()
    with pt.no_grad():
        loss_forget = forward(model, forget_eval_batch)
        loss_retain = forward(model, retain_eval_batch)
    stats = dict(
        forget=loss_forget.exp(), retain=loss_retain.exp(), forget_lr=c.forget_lr
    )
    wandb.log(stats, step=step)
    print(f"{step:4d}  " + "   ".join(f"{v:10.2f}" for v in stats.values()))

    # adapt forget_lr
    if (
        stats["retain"] < c.acceptable_retain_ppl
        and step - last_lr_update > 5 * 10
        and forget_ppl_hist[-5] > stats["forget"]
    ):
        c.forget_lr *= c.forget_ppl_increment
        last_lr_update = step
        print(f"forget_lr updated to {c.forget_lr:.1e}")
    forget_ppl_hist.append(stats["forget"])

    # save model
    if step % 500 == 0:
        model_path = get_repo_root() / "models" / f"{wandb.run.name}_{step}steps.pt"
        pt.save(model.state_dict(), model_path)

# %%
wandb.finish()
