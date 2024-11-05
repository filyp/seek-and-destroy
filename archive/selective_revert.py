# this fails horribly
# applying L0 and L2 pert norm, must be done *durxing* training!

# %% init things, which need to be run once
from copy import deepcopy
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
model_id = "Qwen/Qwen2.5-0.5B"

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
_forget_set = load_one_oscar_shard("pl", tokenizer)
_retain_set = load_one_oscar_shard("en", tokenizer)
forget_eval_batch = get_batch(iter(_forget_set["validation"]), 32)
retain_eval_batch = get_batch(iter(_retain_set["validation"]), 32)

# %% load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
original_state_dict = deepcopy(model.state_dict())

model_path = get_repo_root() / "models" / f"one_hit5_500steps.pt"
state_dict = pt.load(model_path)
model.load_state_dict(state_dict)

# %%
model.eval()
with pt.no_grad():
    print(f"forget ppl: {forward(model, forget_eval_batch).exp()}")
    print(f"retain ppl: {forward(model, retain_eval_batch).exp()}")

# %%
# for module_name, current_weights in model.state_dict().items():
for name, param in model.named_parameters():
    original_weights = original_state_dict[name]
    delta = param.data - original_weights

    # print(f"{name}: {delta.abs().mean()}")
    mask = delta.abs() < 0.001
    param.data[mask] = original_weights[mask]

    # param.data = original_weights + delta * 0.9

    # both approaches fail!!

# %%
# original_state_dict["model.layers.23.post_attention_layernorm.weight"]
# model.state_dict()["model.layers.23.post_attention_layernorm.weight"]
