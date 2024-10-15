# %%
from itertools import islice

import matplotlib.pyplot as plt
import torch as pt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, forward, get_perplexity, load_one_oscar_shard

# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %% load dataset
pl_dataset = load_one_oscar_shard("pl", tokenizer)
en_dataset = load_one_oscar_shard("en", tokenizer)

pl_unlearn_iter = iter(pl_dataset["unlearn"].batch(8))
pl_relearn_iter = iter(pl_dataset["relearn"].batch(8))
en_relearn_iter = iter(en_dataset["relearn"].batch(8))

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


# %% install gradient scaling hooks
def scale_grad_hook(module, grad_input, grad_output):
    grad = list(grad_input)
    grad[0] *= f
    return grad

# works for gemma-2-2b and Qwen2.5-0.5B
for layer in model.model.layers:
    layer.mlp._backward_hooks.clear()
    layer.mlp.register_full_backward_hook(scale_grad_hook)
    layer.input_layernorm._backward_hooks.clear()
    layer.input_layernorm.register_full_backward_hook(scale_grad_hook)


# %%
def train(model, batch_iter, lr=0.0003):
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)
    for batch in batch_iter:
        loss = forward(model, batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # print stats
        res = dict(
            pl=get_perplexity(model, pl_dataset).item(),
            en=get_perplexity(model, en_dataset).item(),
        )
        print({k: f"{v:.2f}" for k, v in res.items()})
    return res


# %% unlearning
pl_threshold = 70
en_threshold = 33


def unlearn(model, batch_iter):
    optimizer = pt.optim.SGD(model.parameters(), lr=0.5)
    f = 0
    for batch in batch_iter:
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

        if res["pl"] > pl_threshold or res["en"] > en_threshold:
            break
    return res


# %%
while True:
    print("unlearning")
    f = 0
    res = unlearn(model, islice(pl_unlearn_iter, 10))
    while res["en"] > en_threshold:
        print("relearning en")
        f = 1
        res = train(model, islice(en_relearn_iter, 10))
    while res["pl"] > pl_threshold:
        print("relearning pl")
        f = 1
        res = train(model, islice(pl_relearn_iter, 10))

# %%
print("relearning pl")
f = 1
res = train(model, islice(pl_relearn_iter, 10))

# %%
print("relearning en")
f = 1
res = train(model, islice(en_relearn_iter, 10))