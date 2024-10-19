# %%
# this file duplicates the code of unlearn_and_relearn function
# having it in a notebook here is useful for free exploration

import matplotlib.pyplot as plt
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from unlearning_functions import name_to_function
from utils import device, forward, get_perplexity, load_one_oscar_shard

from fading_backprop import (
    install_hooks_for_fading_backprop,
    install_hooks_for_saving_gradients,
    normal_train_step,
    set_fade_factor,
    unlearn_and_relearn,
)

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
pl_dataset = load_one_oscar_shard("pl", tokenizer)
en_dataset = load_one_oscar_shard("en", tokenizer)

# load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.to(device)

# %% params
target_dataset = pl_dataset
retain_dataset = en_dataset
unlearning_function = name_to_function["activation_agnostic"]
f_schedule = lambda step: 0
eval_every_n_steps = 2
relearn_lr = 0.0001
unlearn_lr = 0.1
pl_ppl_threshold = float("inf")
batch_size = 32


# %% prepare the training run
install_hooks_for_saving_gradients(model)
install_hooks_for_fading_backprop(model)

# create dataset iterators
target_unlearn_iter = iter(target_dataset["unlearn"].batch(batch_size))
target_relearn_iter = iter(target_dataset["relearn"].batch(batch_size))
retain_relearn_iter = iter(retain_dataset["relearn"].batch(batch_size))

# initial perplexities
ppl = {
    "target": get_perplexity(model, target_dataset),
    "retain": get_perplexity(model, retain_dataset),
}
print("initial perplexity: ", {k: f"{v:.2f}" for k, v in ppl.items()})

# perplexity on retain set is not allowed to raise above this value
# - this triggers relearning on retain set
allowed_retain_perplexity = ppl["retain"] * 1.2
print(f"{allowed_retain_perplexity=:.2f}")


def unlearn(num_unlearning_steps):
    global ppl
    for step_num in range(num_unlearning_steps):
        print(f"\nstep {step_num + 1:3}", end="  ")
        # retain set perplexity too high, so relearn it
        if ppl["retain"] > allowed_retain_perplexity:
            print("> relearning retain", end="  ")
            normal_train_step(model, next(retain_relearn_iter), relearn_lr)
        # do a bit of target relearning, to make unlearning more relevant
        elif ppl["target"] > pl_ppl_threshold:
            print("**relearning target", end="  ")
            # we intentionally use target_UNlean_iter, to not affect relearn split here
            normal_train_step(model, next(target_unlearn_iter), relearn_lr)
        # unlearn target
        else:
            print("  unlearning target", end="  ")
            set_fade_factor(model, f_schedule(step_num))
            unlearning_function(model, next(target_unlearn_iter), unlearn_lr)

        # evaluate
        if (step_num + 1) % eval_every_n_steps == 0:
            ppl = {
                "target": get_perplexity(model, target_dataset),
                "retain": get_perplexity(model, retain_dataset),
            }
            print({k: f"{v:.2f}" for k, v in ppl.items()}, end="  ")
            
            # if ppl["target"] > 80:
            #     break


def relearn(num_relearning_steps):
    global ppl
    print("\n### relearning started ###", end="  ")
    for step_num in range(num_relearning_steps):
        print(f"\nstep {step_num + 1:3}", end="  ")
        if ppl["retain"] > allowed_retain_perplexity:
            print("> relearning retain", end="  ")
            normal_train_step(model, next(retain_relearn_iter), relearn_lr)
        else:
            print("  relearning target", end="  ")
            normal_train_step(model, next(target_relearn_iter), relearn_lr)

        # evaluate
        if (step_num + 1) % eval_every_n_steps == 0:
            ppl = {
                "target": get_perplexity(model, target_dataset),
                "retain": get_perplexity(model, retain_dataset),
            }
            print({k: f"{v:.2f}" for k, v in ppl.items()}, end="  ")

    print("\n###### relearning finished ######")


# %%
# allowed_retain_perplexity = float("inf")

# f_schedule = lambda step: 1
# unlearn_lr = 0.05
# unlearn(1000)
# relearn(50)

# %%
# allowed_retain_perplexity = float("inf")

f_schedule = lambda step: 0
unlearn_lr = 0.5
# unlearn(1000)
relearn(50)


