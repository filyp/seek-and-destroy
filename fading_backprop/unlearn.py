# %%
from itertools import islice

import matplotlib.pyplot as plt
import torch as pt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import context_len, device, get_perplexity, load_one_oscar_shard

# params
model_id = "google/gemma-2-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# %% load dataset
pl_dataset = load_one_oscar_shard("pl", tokenizer)
en_dataset = load_one_oscar_shard("en", tokenizer)
# cs_dataset = load_one_oscar_shard("cs", tokenizer)


def train(model, batch_iter, loss_sign=1, only_mlp_value=False):
    optimizer = pt.optim.SGD(model.parameters(), lr=0.0003)
    optimizer.zero_grad()
    loss_fn = pt.nn.CrossEntropyLoss()

    for batch in batch_iter:
        # create batched input_ids
        input_ids = pt.cat(batch["input_ids"])
        assert input_ids.shape[1] == context_len
        # forward pass
        logits = model(input_ids).logits
        # compute loss
        loss = loss_fn(logits[:, :-1, :].flatten(end_dim=1), input_ids[:, 1:].flatten())
        loss *= loss_sign
        # backpropagate
        loss.backward()

        if only_mlp_value:
            # don't update attention and the "key" layer of mlp
            for layer in model.model.layers:
                layer.mlp.gate_proj.weight.grad = None
                layer.mlp.up_proj.weight.grad = None
                layer.self_attn.q_proj.weight.grad = None
                layer.self_attn.k_proj.weight.grad = None
                layer.self_attn.v_proj.weight.grad = None
                layer.self_attn.o_proj.weight.grad = None

        optimizer.step()
        # print stats
        res = dict(
            pl=get_perplexity(model, pl_dataset).item(),
            en=get_perplexity(model, en_dataset).item(),
        )
        print({k: f"{v:.2f}" for k, v in res.items()})
        # clean memory
        optimizer.zero_grad(set_to_none=True)
        del logits, loss
        pt.cuda.empty_cache()


# %% load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=pt.bfloat16,
).to(device)


# %% install gradient scaling hooks
def scale_grad_hook(module, grad_input, grad_output):
    grad = list(grad_input)
    grad[0] *= f
    return grad


for layer in model.model.layers:
    layer.pre_feedforward_layernorm._backward_hooks.clear()
    layer.pre_feedforward_layernorm.register_full_backward_hook(scale_grad_hook)
    layer.input_layernorm._backward_hooks.clear()
    layer.input_layernorm.register_full_backward_hook(scale_grad_hook)


# %%
print("unlearning")
batch_iter = iter(pl_dataset["unlearn"].batch(8))
# %%
f = 0
train(model, islice(batch_iter, 10), loss_sign=-1, only_mlp_value=True)

# %%
print("relearning")
batch_iter = iter(pl_dataset["relearn"].batch(8))
# %%
f = 1
train(model, islice(batch_iter, 10), loss_sign=1)

# %%
