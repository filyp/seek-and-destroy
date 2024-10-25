from itertools import islice

import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    device,
    forward,
    load_one_oscar_shard,
    retrain_and_eval,
)

# params
model_id = "Qwen/Qwen2.5-0.5B"

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)

# load model; no interventions will be done on the original model
og_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16).to(device)  # fmt: skip