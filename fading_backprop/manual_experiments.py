# %%
from copy import deepcopy

import matplotlib.pyplot as plt
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    device,
    get_perplexity,
    load_one_oscar_shard,
    get_norm_of_weights_change,
    scale_perturbation,
    normal_train_step,
)

from fading_backprop import (
    activation_agnostic,
    install_hooks_for_fading_backprop,
    install_hooks_for_saving_gradients,
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
print("init_target_ppl", init_target_ppl)
print("init_retain_ppl", init_retain_ppl)

# %%

for i in range(10):
    set_fade_factor(model, 0)
    # activation_agnostic(model, next(target_unlearn_iter), lr=0.4)
    set_fade_factor(model, 1)

    # normal_train_step(model, next(target_unlearn_iter), 0.0003, loss_sign=-1)
    normal_train_step(model, next(target_relearn_iter), 0.0003)
    normal_train_step(model, next(retain_relearn_iter), 0.0005)
    # scale_perturbation(model, original_state_dict, 0.99)

    res = {
        "target": get_perplexity(model, target_dataset) - init_target_ppl,
        "retain": get_perplexity(model, retain_dataset) - init_retain_ppl,
        "norm": get_norm_of_weights_change(model, original_state_dict),
    }
    print({k: f"{v:.2f}" for k, v in res.items()})
    if i == 0:
        first_res = res

print("change: ", {k: f"{v - first_res[k]:.2f}" for k, v in res.items()})


# %%
