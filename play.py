# %%
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
import numpy as np
import time
from utils import *

model_id = "microsoft/Phi-3-mini-4k-instruct"
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
chunks = dataset_to_equal_chunks(dataset["train"], tokenizer)
print(f"num chunks: {len(chunks)}")

# %%
# start_time = time.time()
eval_perplexity(model, chunks[:128])
# time.time() - start_time
# %%
