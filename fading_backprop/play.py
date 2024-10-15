# %%
from itertools import islice

import matplotlib.pyplot as plt
import torch as pt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, get_perplexity, load_one_oscar_shard, forward

model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %% load dataset
pl_dataset = load_one_oscar_shard("pl", tokenizer)
en_dataset = load_one_oscar_shard("en", tokenizer)


# %% load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=pt.bfloat16,
).to(device)

# %% save mlp output grads
mlp_output_grads = dict()


def save_output_grad_hook(module, grad_input, grad_output):
    mlp_output_grads[module] = grad_output[0]


for layer in model.model.layers:
    layer.mlp.down_proj._backward_hooks.clear()
    layer.mlp.down_proj.register_full_backward_hook(save_output_grad_hook)


# %%
optimizer = pt.optim.SGD(model.parameters(), lr=0.5)

# %%
batch_iter = iter(pl_dataset["unlearn"].batch(8))

for batch in islice(batch_iter, 10):
    loss = forward(model, batch)
    loss.backward()
    # get rid of the normal grad
    optimizer.zero_grad(set_to_none=True)

    # calculate custom grads
    for layer in model.model.layers:
        module = layer.mlp.down_proj
        # get the grad
        output_grad = mlp_output_grads[module]
        output_grad = output_grad[:, :-1, :]  # last token position has no gradient
        # assert output_grad.norm(dim=-1).all()

        # calculate the projection
        projection_amps = pt.einsum("oi,bto->bti", module.weight, output_grad)
        projection_amps /= output_grad.norm(dim=-1, keepdim=True)
        update = pt.einsum("bto,bti->oi", output_grad, projection_amps)
        module.weight.grad = update

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # print stats
    res = dict(
        pl=get_perplexity(model, pl_dataset).item(),
        en=get_perplexity(model, en_dataset).item(),
    )
    print({k: f"{v:.2f}" for k, v in res.items()})
# %%
