# %%
import time

import bitsandbytes as bnb
import numpy as np
import torch as pt
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import wandb
from utils import *

# model_id = "microsoft/Phi-3-mini-4k-instruct"
# model_id = "microsoft/Phi-3.5-mini-instruct"
model_id = "microsoft/phi-1_5"

tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=pt.bfloat16,
).to(device)

# Load the WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-v1")

# %% prepare the data
eval_chunks = dataset_to_equal_chunks(dataset["validation"], tokenizer)
print(f"num eval chunks: {len(eval_chunks)}")
train_chunks = dataset_to_equal_chunks(dataset["train"], tokenizer)
print(f"num train chunks: {len(train_chunks)}")

# # %% eval pre and post change of activation
print(eval_perplexity(model, eval_chunks[:32]))

# %% find the best slope
# for slope in np.linspace(0, 0.02, 21):
#     for layer in model.model.layers:
#         layer.mlp.activation_fn = pt.nn.LeakyReLU(negative_slope=slope)
#     ppl = eval_perplexity(model, eval_chunks[:32])
#     print(f"{slope=:.3f}: {ppl=:.2f}")
# # best slope for phi-1.5: 0.01

for layer in model.model.layers:
    layer.mlp.activation_fn = pt.nn.LeakyReLU(negative_slope=0.01)
eval_perplexity(model, eval_chunks[:32])

# %% prepare training

wandb.init(
    project="softened_derivative2",
    group="retrain",
    config=dict(
        lr=0.0001,
        batch_size=32,
    )
)
c = wandb.config

optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=c.lr)
loss_fn = pt.nn.CrossEntropyLoss()
optimizer.zero_grad()
# for offset in tqdm(range(0, len(train_chunks) - batch_size, batch_size)):
for offset in tqdm(range(0, 256, c.batch_size)):
    # prepare
    batch = train_chunks[offset : offset + c.batch_size].to(device)
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
    optimizer.zero_grad()

    # eval
    # optimizer.zero_grad(set_to_none=True)
    # pt.cuda.empty_cache()
    ppl = eval_perplexity(model, eval_chunks[:32])
    wandb.log(dict(wikitext_ppl=ppl))
    # # clean up memory
    # del output, pred, true, pred_flat, loss
    # pt.cuda.empty_cache()


# %%