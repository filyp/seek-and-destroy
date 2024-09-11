# %%
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
import numpy as np
import time
from utils import *
import bitsandbytes as bnb

model_id = "microsoft/Phi-3-mini-4k-instruct"
# model_id = "microsoft/Phi-3.5-mini-instruct"

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

# %% eval pre and post change of activation
print(eval_perplexity(model, eval_chunks[:64]))
for layer in model.model.layers:
    layer.mlp.activation_fn = pt.nn.LeakyReLU(negative_slope=0.12)
print(eval_perplexity(model, eval_chunks[:64]))

# %% prepare training
train_chunks = dataset_to_equal_chunks(dataset["train"], tokenizer)
# optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001) # instead of torch.optim.Adam
optimizer = pt.optim.SGD(model.parameters(), lr=0.001)
loss_fn = pt.nn.CrossEntropyLoss()

# %%

batch_size = 2
# for offset in range(0, len(train_chunks) - batch_size, batch_size):
for offset in range(0, 256, batch_size):
    # prepare
    batch = train_chunks[offset : offset + batch_size].to(device)
    optimizer.zero_grad()
    
    # forward pass
    output = model(batch)
    # compute loss
    pred = output.logits[:, :-1, :]
    true = batch[:, 1:]
    pred_flat = pred.reshape(-1, pred.shape[-1])
    loss = loss_fn(pred_flat, true.flatten())

    # backprop
    loss.backward()
        
    # optimizer step
    optimizer.step()
    
    # clean up memory
    pt.cuda.empty_cache()
    del output, pred, true, pred_flat, loss

    if (offset / batch_size) % 16  == 0:
        optimizer.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        ppl = eval_perplexity(model, eval_chunks[:64])
        print(ppl)

# %%