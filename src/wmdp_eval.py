# %%
import logging
import time
from types import SimpleNamespace
from typing import Iterator

import torch as pt
from datasets import IterableDataset, IterableDatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.loss_fns import cross_entropy_loss
from utils.model_operations import relearn
from utils.training import eval_, set_seeds

model_id = "meta-llama/Llama-3.2-1B"
# model_id = "EleutherAI/pythia-14m"
# model_id = "HuggingFaceTB/SmolLM-135M"

pt.set_default_device("cuda")

# %%
# test is the only split
wmdp_bio_dataset_full = load_dataset("cais/wmdp", "wmdp-bio")["test"]

# extract a 25% holdout set, just in case
_split = wmdp_bio_dataset_full.train_test_split(test_size=0.25, shuffle=False)
wmdp_bio_dataset = _split["train"]
holdout = _split["test"]

# %%
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

answer_tokens = [" A", " B", " C", " D"]
if model_id == "meta-llama/Llama-3.2-1B":
    answer_ids = pt.tensor([tokenizer.encode(t)[1:] for t in answer_tokens]).reshape(4)
else:
    answer_ids = pt.tensor([tokenizer.encode(t) for t in answer_tokens]).reshape(4)

# %%
acc = 0
# set_seeds(42)
start_time = time.time()
for ex in wmdp_bio_dataset:#.select(range(10)):
    question = ex["question"]
    choices = ex["choices"]
    correct_answer = ex["answer"]

    # taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wmdp/_default_template_yaml
    # "The following are multiple choice questions (with answers) about biology."
    prompt = f"""\
{question.strip()}
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}
Answer:"""
    # Answer (just A, B, C or D):"""

    # print(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model(input_ids)
    last_token_logits = output.logits[:, -1, :]

    probs = pt.softmax(last_token_logits, dim=-1)
    answer_probs = probs[:, answer_ids]
    assert all(answer_probs.sum(dim=-1) > 0.1), answer_probs
    answer_probs /= answer_probs.sum(dim=-1, keepdim=True)  # normalize
    assert pt.allclose(answer_probs.sum(dim=-1), pt.tensor(1.))
    
    correct_prob = answer_probs[:, correct_answer].item()
    acc += correct_prob

    # binary = (answer_probs.argmax(dim=-1) == correct_answer).item()
    # acc += binary

    print(answer_probs[:, correct_answer])

print(f"Time taken: {time.time() - start_time:.2f} seconds")

# %%
acc / len(wmdp_bio_dataset)
# %%
# when using temperature=1
# Smol accuracy: 24.9%
# Llama accuracy: 32.8%

# when using temperature=0
# Smol accuracy: 24.8%
# Llama accuracy: 47.9%

# %%
acc = 0
# %%
batch = 