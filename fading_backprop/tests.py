# %%
from itertools import islice

import matplotlib.pyplot as plt
import torch as pt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, forward, get_perplexity, load_one_oscar_shard

model_id = "google/gemma-2-2b"
# model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %% load dataset
pl_dataset = load_one_oscar_shard("pl", tokenizer)

# %% load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=pt.bfloat16,
).to(device)

# %% save residual stream activations
grads = dict()


def save_grad_hook(module, grad_input, grad_output):
    grads[module] = grad_output[0]


for layer in model.model.layers:
    # clear hooks
    layer._backward_hooks.clear()
    # register hook
    layer.register_full_backward_hook(save_grad_hook)


# %% install gradient scaling hooks
f = 0
def scale_grad_hook(module, grad_input, grad_output):
    grad = list(grad_input)
    grad[0] *= f
    return grad

for layer in model.model.layers:
    layer.mlp.register_full_backward_hook(scale_grad_hook)
    layer.input_layernorm.register_full_backward_hook(scale_grad_hook)


# %% plot grads
batch = next(iter(pl_dataset["unlearn"].batch(1)))
loss = forward(model, batch)
loss.backward()

# ! for f=0, the line should be completely flat
one_val_from_each_layer = [grads[layer][0, -2, 0].item() for layer in model.model.layers]
plt.plot(one_val_from_each_layer)
# %%
