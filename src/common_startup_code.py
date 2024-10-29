import random
import time
from collections import OrderedDict
from itertools import islice

import matplotlib.pyplot as plt
import torch as pt
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import forward, get_stats, load_one_oscar_shard, print_stats

pt.set_default_device("cuda")

# params
model_id = "Qwen/Qwen2.5-0.5B"
# model_id = "google/gemma-2-2b"

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)

# load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

# set all seeds
seed = 42
pt.manual_seed(seed)
pt.cuda.manual_seed_all(seed)
pt.backends.cudnn.deterministic = True
pt.backends.cudnn.benchmark = False
random.seed(seed)