# %%
# just execute this file to run the tests for fading backprop on a specified model

import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, forward, load_one_oscar_shard

from fading_backprop import install_hooks_for_fading_backprop, set_fade_factor

# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
pl_dataset = load_one_oscar_shard("pl", tokenizer)
en_dataset = load_one_oscar_shard("en", tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.to(device)

install_hooks_for_fading_backprop(model)


# %%
# save the gradients downstream of each layer
# note: it's layer, not mlp as in the other file, because gradients after mlp
# can be different between layers, due to layernorm
def save_output_grad_hook(module, grad_input, grad_output):
    module.output_grad = grad_output[0]


for layer in model.model.layers:
    layer.register_full_backward_hook(save_output_grad_hook)


def activations_on_all_layers_equal():
    batch = next(iter(pl_dataset["unlearn"].batch(1)))
    loss = forward(model, batch)
    loss.backward()

    output_grads = [layer.output_grad for layer in model.model.layers]
    reference = output_grads[0]
    return all(output_grad.allclose(reference) for output_grad in output_grads)


# %% test normal operation
set_fade_factor(model, 0.01)
assert not activations_on_all_layers_equal()

# %% test completely vanishing gradient additions
# ! here, we want to ensure that gradient on each layer looks the same
set_fade_factor(model, 0)
assert activations_on_all_layers_equal()

# %% # freeze all but mlp down_proj
for param in model.parameters():
    param.requires_grad = False
for layer in model.model.layers:
    layer.mlp.down_proj.weight.requires_grad = True

# %% test normal operation again
set_fade_factor(model, 0.01)
assert not activations_on_all_layers_equal()

# %% test completely vanishing gradient additions again
# ! here, we want to ensure that gradient on each layer looks the same
set_fade_factor(model, 0)
assert activations_on_all_layers_equal()
