# %% init things, which need to be run once
import sys

import torch as pt
from peft import LoraConfig, TaskType, get_peft_model
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

# %% load model 
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules="all-linear",
)
peft_model = get_peft_model(model, lora_config, adapter_name="retain_lora")
model = peft_model.model

# model_name = sys.argv[1]
model_name = "models/R16 f=8e-05 r=3e-04 L2=1_500steps.pt"
model_path = get_repo_root() / model_name
state_dict = pt.load(model_path, weights_only=True)
model.load_state_dict(state_dict)

model = peft_model.merge_and_unload()

# %% initialize LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules="all-linear",
)
model.add_adapter(lora_config, adapter_name="relearning_lora")

optimizer = pt.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
wandb.init(project="adversarial_adaptation", group="relearning", name=model_path.stem)

# %%
for i in range(1, 1 + 100):
    model.train()

    optimizer.zero_grad(set_to_none=True)
    loss_forget = forward(model, get_batch(forget_iter, 16))
    loss_retain = forward(model, get_batch(retain_iter, 16))
    (loss_forget + loss_retain).backward()
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
