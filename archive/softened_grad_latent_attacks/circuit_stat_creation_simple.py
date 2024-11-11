# %%
import matplotlib.pyplot as plt
import torch as pt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *

pt.set_default_device("cuda")

# load model
model_id = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)

forget_iter = looping_iter(retain_set["train"])

num_steps = 30

# calculate
model.zero_grad(set_to_none=True)
for _ in tqdm(range(num_steps)):
    input_ids = get_batch(forget_iter, 32)

    # loss = cross_entropy_loss(model(input_ids), input_ids)
    loss = clipped_correct_logit_loss(model(input_ids), input_ids)

    loss.backward()
    # do not optimizer.step() !

# gather
unwanted_circuit = {
    name: -param.grad / num_steps for name, param in model.named_parameters()
}
model.zero_grad(set_to_none=True)

# # save
# circuit_path = get_repo_root() / "circuits" / f"test.pt"
# pt.save(unwanted_circuit, circuit_path)

# %%
unwanted_circuit["model.layers.0.self_attn.q_proj.weight"]