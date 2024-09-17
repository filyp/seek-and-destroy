# %%
from itertools import islice
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

# %% prepare the wikitext dataset
# Load the WikiText-2 dataset
beavertails = load_dataset("wikitext", "wikitext-2-v1")

wikitext_eval_chunks = dataset_to_equal_chunks(beavertails["validation"], tokenizer)
print(f"num eval chunks: {len(wikitext_eval_chunks)}")
wikitext_train_chunks = dataset_to_equal_chunks(beavertails["train"], tokenizer)
print(f"num train chunks: {len(wikitext_train_chunks)}")

# %% Load the BeaverTails dataset
beavertails = load_dataset("PKU-Alignment/BeaverTails")
split = beavertails["30k_train"]
category = "animal_abuse"
beaver_category = split.filter(lambda ex: ex["category"][category])
len(beaver_category)


# %%
# save the grad on some specified layer, to be later used for latent attack
def save_grad_hook(module, grad_input, grad_output):
    global d_perturbation
    d_perturbation = grad_output[0]
    assert len(grad_output) == 2 and grad_output[1] is None


model.model.layers[3].register_full_backward_hook(save_grad_hook)


# %% prepare training
wandb.init(
    project="softened_derivative2",
    group="retrain",
    config=dict(
        lr=0.00003,
        batch_size=16,
        # None means no act_fn change, 0.0 means ReLU
        leaky_relu_slope=0.03,
        category=category,
        model_id=model_id,
    ),
)
c = wandb.config

# change the activation function
if c.leaky_relu_slope is not None:
    for layer in model.model.layers:
        layer.mlp.activation_fn = pt.nn.LeakyReLU(negative_slope=c.leaky_relu_slope)


# %%

optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=c.lr)
loss_fn = pt.nn.CrossEntropyLoss()
optimizer.zero_grad()

# for offset in tqdm(range(0, len(train_chunks) - batch_size, batch_size)):
for offset in tqdm(range(0, 256, c.batch_size)):
    # prepare
    batch = wikitext_train_chunks[offset : offset + c.batch_size].to(device)
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
    wikitext_ppl = eval_perplexity(model, wikitext_eval_chunks[:32])

    # latent attack
    token_log_probs = []
    d_perturbation_norms = []
    for example in islice(beaver_category, 20):
        # forward
        resp_log_prob, resp_len = get_response_log_prob(example, model, tokenizer)
        # backward
        resp_log_prob.backward()
        optimizer.zero_grad(set_to_none=True)
        # save the stats
        token_log_probs.append(resp_log_prob.item() / resp_len)
        d_perturbation_norms.append(d_perturbation.norm().cpu().float())

    wandb.log(
        dict(
            wikitext_ppl=wikitext_ppl,
            avg_attack_token_log_prob=np.mean(token_log_probs),
            avg_d_perturbation_norm=np.mean(d_perturbation_norms),
        )
    )

    # # clean up memory
    # del output, pred, true, pred_flat, loss, token_log_probs, d_perturbation_norms, wikitext_ppl, d_perturbation  # fmt: skip
    # pt.cuda.empty_cache()

wandb.finish()

# %%
pt.cuda.empty_cache()