# %% init things, which need to be run once
import time

import matplotlib.pyplot as plt
import torch as pt
from peft import LoraConfig, TaskType
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
forget_iter = looping_iter(forget_set["relearn"])
retain_iter = looping_iter(retain_set["relearn"])
forget_eval_batch = get_batch(iter(forget_set["validation"]), 32)
retain_eval_batch = get_batch(iter(retain_set["validation"]), 32)


# get initial perplexities
model.eval()
with pt.no_grad():
    initial_forget_ppl = forward(model, forget_eval_batch).exp()
    initial_retain_ppl = forward(model, retain_eval_batch).exp()

# %% load model (without LoRA)
model_path = get_repo_root() / "models" / "r1_200steps.pt"
state_dict = pt.load(model_path, weights_only=True)
remove_lora(state_dict)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.load_state_dict(state_dict)

# %% initialize LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],  # fmt: skip
)
model.add_adapter(lora_config, adapter_name="relearning_lora")

# %%
optimizer = pt.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
wandb.init(project="adversarial_adaptation", group="relearning", name=model_path.stem)

# %%
for i in range(1, 1 + 100):
    model.train()
    loss_retain = pt.tensor(pt.nan)

    optimizer.zero_grad(set_to_none=True)
    loss_forget = forward(model, get_batch(forget_iter, 32))
    loss_forget.backward()
    optimizer.step()

    # evaluate
    if i % 10 == 0:
        model.eval()
        with pt.no_grad():
            loss_forget = forward(model, forget_eval_batch)
            loss_retain = forward(model, retain_eval_batch)

    # calculate and print stats
    stats = dict(
        forget=loss_forget.exp() - initial_forget_ppl,
        retain=loss_retain.exp() - initial_retain_ppl,
    )

    if i % 10 == 0:
        # if current run is active, log stats
        if wandb.run:
            wandb.log(stats, step=i)

    if i % 10 == 1:
        print("\n      " + "   ".join(f"{k:>10}" for k in stats.keys()))
    print(f"{i:4d}  " + "   ".join(f"{v:10.2f}" for v in stats.values()))

if wandb.run:
    wandb.finish()

# %%
