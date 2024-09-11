# %%
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
import numpy as np
import time
from utils import *
import bitsandbytes as bnb

# model_id = "microsoft/Phi-3-mini-4k-instruct"
model_id = "microsoft/Phi-3.5-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=pt.bfloat16,
    # attn_implementation="flash_attention_2",
).to(device)

# Load the WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-v1")

# %% prepare the data
eval_chunks = dataset_to_equal_chunks(dataset["validation"], tokenizer)
print(f"num chunks: {len(eval_chunks)}")

# # %% eval pre and post change of activation
print(eval_perplexity(model, eval_chunks[:32]))
for layer in model.model.layers:
    layer.mlp.activation_fn = pt.nn.LeakyReLU(negative_slope=0.12)
print(eval_perplexity(model, eval_chunks[:32]))

# %%

# %%

# %%

# %% prepare training
train_chunks = dataset_to_equal_chunks(dataset["train"], tokenizer)
# optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001) # instead of torch.optim.Adam
gradient_acc_steps = 64
optimizer = pt.optim.SGD(model.parameters(), lr=0.000003)
loss_fn = pt.nn.CrossEntropyLoss()

# %%

batch_size = 1
optimizer.zero_grad()
for offset in range(0, len(train_chunks) - batch_size, batch_size):
# for offset in range(0, 256, batch_size):
    # prepare
    batch = train_chunks[offset : offset + batch_size].to(device)
    
    # forward pass
    output = model(batch)
    # compute loss
    pred = output.logits[:, :-1, :]
    true = batch[:, 1:]
    pred_flat = pred.reshape(-1, pred.shape[-1])
    loss = loss_fn(pred_flat, true.flatten())

    # backprop
    loss.backward()
        
    if (offset / batch_size + 1) % gradient_acc_steps == 0:
        # grad = model.model.layers[15].mlp.down_proj.weight.grad
        # print(f"grad: {pt.norm(grad)}")

        # optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # also eval
        optimizer.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        ppl = eval_perplexity(model, eval_chunks[:32])
        print(ppl)
    
    # clean up memory
    del output, pred, true, pred_flat, loss
    pt.cuda.empty_cache()

# %%

# %%

# %%

# %%

# %%

# %%

# # %%
# # prepare
# offset = 2
# batch_size = 1
# batch = train_chunks[offset : offset + batch_size].to(device)
# optimizer.zero_grad()

# # %%

# # forward pass
# output = model(batch)
# # compute loss
# pred = output.logits[:, :-1, :]
# true = batch[:, 1:]
# pred_flat = pred.reshape(-1, pred.shape[-1])
# loss = loss_fn(pred_flat, true.flatten())

# # backprop
# loss.backward()

# # %%
# grad2 = model.model.layers[15].mlp.down_proj.weight.grad
# # print(f"grad: {pt.norm(grad)}")
# # %%
# pt.cosine_similarity(grad.reshape(1, -1), grad2.reshape(1, -1))
# # %%

# # %%
# del output, pred, true, pred_flat, loss
# pt.cuda.empty_cache()
