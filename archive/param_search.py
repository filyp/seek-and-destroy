# %%
import time

import matplotlib.pyplot as plt
import torch as pt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

pt.set_default_device("cuda")
from fading_backprop import unlearn_and_relearn
from utils import load_one_oscar_shard


# %%
def abstract_search_for_optimal_value(f, starting_value, criterion, zoom_in_steps=4):
    assert starting_value > 0
    assert criterion in (min, max)
    value = starting_value
    values_and_outputs = []

    while True:
        values_and_outputs.append((value, f(value)))
        values_and_outputs.sort(key=lambda pair: pair[0])
        # print(values_and_outputs, value)
        best_pair = criterion(values_and_outputs, key=lambda pair: pair[1])
        best_index = values_and_outputs.index(best_pair)
        if best_index == 0:
            # we must search lower values
            value = best_pair[0] / 10
        elif best_index == len(values_and_outputs) - 1:
            # we must search higher values
            value = best_pair[0] * 10
        else:
            # we have found a good value
            break

    for _ in range(zoom_in_steps):
        neighbor1 = values_and_outputs[best_index - 1]
        neighbor2 = values_and_outputs[best_index + 1]
        # use geometric mean instead of arithmetic mean
        new_value1 = (best_pair[0] * neighbor1[0]) ** 0.5
        new_value2 = (best_pair[0] * neighbor2[0]) ** 0.5
        # print(new_value1, new_value2)
        values_and_outputs.append((new_value1, f(new_value1)))
        values_and_outputs.append((new_value2, f(new_value2)))
        values_and_outputs.sort(key=lambda pair: pair[0])

        best_pair = criterion(values_and_outputs, key=lambda pair: pair[1])
        best_index = values_and_outputs.index(best_pair)

    return best_pair[0], values_and_outputs


best_value = abstract_search_for_optimal_value(lambda x: (x - 123) ** 2, 1, min, 20)[0]
assert round(best_value) == 123


# %%
def search_for_optimal_value(
    model_id,
    forget_dataset,
    retain_dataset,
    param_name,
    param_starting_value,
    criterion,
    wandb_group,
    **kwargs,
):
    wandb_group += time.strftime("_%Y-%m-%d_%H-%M-%S")

    def get_final_forget_perplexity(param_value):
        print(f"\n\n\nTrying {param_name}={param_value}")
        # load model
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

        kwargs[param_name] = param_value
        final_perplexities = unlearn_and_relearn(
            model, forget_dataset, retain_dataset, wandb_group, **kwargs
        )
        if final_perplexities["retain"] > 100:
            print("Unacceptable final perplexity on the retain set! Discarding...")
            if criterion == min:
                return float("inf")
            elif criterion == max:
                return float("-inf")

        return final_perplexities["forget"]

    return abstract_search_for_optimal_value(
        get_final_forget_perplexity, param_starting_value, criterion
    )


# %% load dataset
# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
pl_dataset = load_one_oscar_shard("pl", tokenizer)
en_dataset = load_one_oscar_shard("en", tokenizer)


# %% find optimal relearning rate
# it only needs to be done once per model and dataset
# after finding the best value, just reuse it

# best_relearn_lr, _all_pairs = search_for_optimal_value(
#     model_id, pl_dataset, en_dataset, "relearn_lr", 0.1, min, "relearn_lr_search"
# )
# print(f"{best_relearn_lr=}")
best_relearn_lr = 0.000649

# %% find optimal AA, fade_factor=0 unlearning rate

best_unlearn_lr, _all_pairs = search_for_optimal_value(
    model_id,
    pl_dataset,
    en_dataset,
    "unlearn_lr",
    1,
    max,
    "0_AA_search",
    relearn_lr=best_relearn_lr,
    f_schedule="lambda step: 0",
    unlearning_function="activation_agnostic",
)
print(f"{best_unlearn_lr=}")
# 11.54

# %% find optimal AA, fade_factor=1 unlearning rate

best_unlearn_lr, _all_pairs = search_for_optimal_value(
    model_id,
    pl_dataset,
    en_dataset,
    "unlearn_lr",
    1,
    max,
    "1_AA_search",
    relearn_lr=best_relearn_lr,
    f_schedule="lambda step: 1",
    unlearning_function="activation_agnostic",
)
print(f"{best_unlearn_lr=}")
# 1.54
