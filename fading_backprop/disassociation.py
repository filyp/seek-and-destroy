# %%
import matplotlib.pyplot as plt
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, forward, get_perplexity, load_one_oscar_shard


# params
batch_size = 1
model_id = "Qwen/Qwen2.5-0.5B"

# load model
original_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
original_model.to(device)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.to(device)

# create dataset iterators
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_dataset = load_one_oscar_shard("pl", tokenizer)
retain_dataset = load_one_oscar_shard("en", tokenizer)
forget_unlearn_iter = iter(forget_dataset["unlearn"].batch(batch_size))
forget_relearn_iter = iter(forget_dataset["relearn"].batch(batch_size))
retain_relearn_iter = iter(retain_dataset["relearn"].batch(batch_size))


# %% install activation saving hooks
def save_post_activations(module, args, output):
    module.last_post_activations = output


for layer in original_model.model.layers:
    layer.mlp.act_fn.register_forward_hook(save_post_activations)

# %%
batch = next(forget_unlearn_iter)
forward(original_model, batch)

# %%
effect_activations = original_model.model.layers[10].mlp.act_fn.last_post_activations
cause_activations = original_model.model.layers[8].mlp.act_fn.last_post_activations
# %%
cause_activations = cause_activations[0, 0]
effect_activations = effect_activations[0, 0]
# %%
cutoff = effect_activations.to(pt.float).quantile(0.99)
cutoff
# %%
effect_indexes = pt.where(effect_activations > cutoff)
effect_indexes = effect_indexes[0]
effect_indexes
# %%
effect_activations[effect_indexes]
# %%
