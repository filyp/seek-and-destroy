import time
from collections import OrderedDict
from itertools import islice

import matplotlib.pyplot as plt
import torch as pt
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import forward, get_stats, load_one_oscar_shard, print_stats, set_seeds

pt.set_default_device("cuda")

# params
model_id = "Qwen/Qwen2.5-0.5B"
# model_id = "google/gemma-2-2b"

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)
