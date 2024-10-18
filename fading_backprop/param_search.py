# %%
import matplotlib.pyplot as plt
import torch as pt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, load_one_oscar_shard

from fading_backprop import unlearn_and_relearn


# %%
def abstract_search_for_optimal_value(f, starting_value, criterion, zoom_in_steps=4):
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
        new_value1 = (best_pair[0] + neighbor1[0]) / 2
        new_value2 = (best_pair[0] + neighbor2[0]) / 2
        # print(new_value1, new_value2)
        values_and_outputs.append((new_value1, f(new_value1)))
        values_and_outputs.append((new_value2, f(new_value2)))
        values_and_outputs.sort(key=lambda pair: pair[0])

        best_pair = criterion(values_and_outputs, key=lambda pair: pair[1])
        best_index = values_and_outputs.index(best_pair)

    return best_pair[0], values_and_outputs


best_value = abstract_search_for_optimal_value(lambda x: (x - 123) ** 2, 1, min, 20)[0]
assert int(best_value) == 123


# %% load dataset
# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
pl_dataset = load_one_oscar_shard("pl", tokenizer)
en_dataset = load_one_oscar_shard("en", tokenizer)


# %%
def search_for_optimal_value(param_name, param_starting_value, criterion, **kwargs):
    def get_final_target_perplexity(param_value):
        print(f"Trying {param_name}={param_value}")
        # load model
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
        model.to(device)

        kwargs[param_name] = param_value
        final_perplexities = unlearn_and_relearn(
            model, pl_dataset, en_dataset, **kwargs
        )
        return final_perplexities["target"]

    return abstract_search_for_optimal_value(
        get_final_target_perplexity, param_starting_value, criterion
    )


# %% find optimal relearning rate
# it only needs to be done once per model and dataset
# after finding the best value, just reuse it
best_value, all_pairs = search_for_optimal_value("relearn_lr", 0.001, min)
