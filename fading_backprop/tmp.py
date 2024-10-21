# %%
# for g in optimizer.param_groups:
#     g["lr"] = 1000000
from copy import deepcopy

import matplotlib.pyplot as plt
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from unlearning_functions import activation_agnostic, name_to_function
from utils import device, forward, get_perplexity, load_one_oscar_shard

from fading_backprop import (
    get_norm_of_weights_change,
    install_hooks_for_fading_backprop,
    install_hooks_for_saving_gradients,
    normal_train_step,
    scale_perturbation,
    set_fade_factor,
    unlearn_and_relearn,
)

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
target_dataset = load_one_oscar_shard("pl", tokenizer)
retain_dataset = load_one_oscar_shard("en", tokenizer)

# %%
# load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.to(device)

# freeze all but mlp down_proj
for param in model.parameters():
    param.requires_grad = False
for layer in model.model.layers:
    layer.mlp.down_proj.weight.requires_grad = True

# params
batch_size = 32

# create dataset iterators
target_unlearn_iter = iter(target_dataset["unlearn"].batch(batch_size))
target_relearn_iter = iter(target_dataset["relearn"].batch(batch_size))
retain_relearn_iter = iter(retain_dataset["relearn"].batch(batch_size))

original_state_dict = deepcopy(model.state_dict())

install_hooks_for_saving_gradients(model)
install_hooks_for_fading_backprop(model)

init_target_ppl = get_perplexity(model, target_dataset)
init_retain_ppl = get_perplexity(model, retain_dataset)

# %%
set_fade_factor(model, 1)
norm_lim = 5
lr = 0.1

for i in range(20):
    # activation_agnostic(model, next(target_unlearn_iter), lr=lr)
    normal_train_step(model, next(target_relearn_iter), 0.0003)
    # normal_train_step(model, next(retain_relearn_iter), 0.0003)

    norm = get_norm_of_weights_change(model, original_state_dict)

    # if norm > norm_lim:
    #     scale_factor = norm_lim / norm
    #     print(f"scaling from {norm:.2f} to {norm_lim:.2f}")
    #     scale_perturbation(model, original_state_dict, scale_factor)
    #     lr *= scale_factor
    #     norm = get_norm_of_weights_change(model, original_state_dict)

    res = {
        "target": get_perplexity(model, target_dataset) - init_target_ppl,
        "retain": get_perplexity(model, retain_dataset) - init_retain_ppl,
        "norm": norm,
    }
    print({k: f"{v:.2f}" for k, v in res.items()})
    


# %%
