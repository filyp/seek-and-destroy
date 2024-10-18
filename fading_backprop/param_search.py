# %%
import matplotlib.pyplot as plt
import torch as pt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, load_one_oscar_shard
from fading_backprop import unlearn_and_relearn


# %% load dataset
# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
pl_dataset = load_one_oscar_shard("pl", tokenizer)
en_dataset = load_one_oscar_shard("en", tokenizer)


# %%
criterion = min
param_name = "relearn_lr"
param_starting_value = 0.001
param_value_and_ppl = []


def get_final_target_perplexity(**kwargs):
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
    model.to(device)

    final_perplexities = unlearn_and_relearn(model, pl_dataset, en_dataset, **kwargs)
    return final_perplexities["target"]


# # %%
# ppl = get_final_target_perplexity(**{param_name: param_starting_value})
# param_value_and_ppl.append((param_starting_value, ppl))
# ppl = get_final_target_perplexity(**{param_name: param_starting_value * 2})
# param_value_and_ppl.append((param_starting_value * 2, ppl))
# # %%

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.to(device)

final_perplexities = unlearn_and_relearn(model, pl_dataset, en_dataset)
