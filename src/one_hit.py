# %% init things, which need to be run once
import time

import matplotlib.pyplot as plt
import torch as pt
from peft import LoraConfig, TaskType
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
forget_eval_batch = get_batch(iter(forget_set["validation"]), 32)
retain_eval_batch = get_batch(iter(retain_set["validation"]), 32)


# %% load cached circuits
circuit_path = repo_root() / "circuits" / f"forward_with_clipped_logit_300.pt"
unwanted_circuit = pt.load(circuit_path)
circuit_path = repo_root() / "circuits" / f"retain_forward_clipped_logit_300.pt"
wanted_circuit = pt.load(circuit_path)

# %%
c = SimpleNamespace(
    forget_lr=1e-2,
    retain_lr=4e-4,
    acceptable_retain_ppl=25,
    forget_ppl_increment=1.5,
    momentum=0.9,
    rank=16,
    lowest_retain_weights=0.01,
)

# %%
for name, uw_param in unwanted_circuit.items():
    w_param = wanted_circuit[name]
    magnitudes = w_param.abs().flatten()
    k = int(len(magnitudes) * c.lowest_retain_weights)
    threshold = magnitudes.kthvalue(k).values
    # todo maybe also think about the sign?
    uw_param[w_param.abs() > threshold] = 0
del wanted_circuit


# %%
set_seeds(42)
retain_iter = looping_iter(retain_set["unlearn"])

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=c.rank,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules="all-linear",
)
model.add_adapter(lora_config, adapter_name="retain_lora")

optimizer = pt.optim.Adam(model.parameters(), lr=c.retain_lr, betas=(c.momentum, 0.999))

step = 0

# get initial perplexities
model.eval()
with pt.no_grad():
    print("initial perplexities:")
    print(f"forget: {forward(model, forget_eval_batch).exp()}")
    print(f"retain: {forward(model, retain_eval_batch).exp()}")


# %%
for _ in range(100):
    step += 1
    model.train()
    loss_forget = pt.tensor(pt.nan)

    for name, param in model.named_parameters():
        name = name.replace(".base_layer", "")
        if name in unwanted_circuit:  # and "_proj" in name:
            param.data -= unwanted_circuit[name] * c.forget_lr

    optimizer.zero_grad(set_to_none=True)
    loss_retain = forward(model, get_batch(retain_iter, 8))
    loss_retain.backward()
    optimizer.step()

    if step % 10 != 0:
        continue

    # evaluate
    model.eval()
    with pt.no_grad():
        loss_forget = forward(model, forget_eval_batch)
        loss_retain = forward(model, retain_eval_batch)
    stats = dict(
        forget=loss_forget.exp(), retain=loss_retain.exp(), forget_lr=c.forget_lr
    )
    print(f"{step:4d}  " + "   ".join(f"{v:10.2f}" for v in stats.values()))

    # save model
    if step % 500 == 0:
        run_name=f"R{c.rank} f={c.forget_lr:.0e} r={c.retain_lr:.0e}",
        model_path = repo_root() / "models" / f"{run_name}_{step}steps.pt"
        pt.save(model.state_dict(), model_path)

