# %%
import matplotlib.pyplot as plt
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, forward, get_perplexity, load_one_oscar_shard

from fading_backprop import (
    get_norm_of_weights_change,
    normal_train_step,
    scale_perturbation,
)

# params
batch_size = 32
model_id = "Qwen/Qwen2.5-0.5B"

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
target_dataset = load_one_oscar_shard("pl", tokenizer)
retain_dataset = load_one_oscar_shard("en", tokenizer)

# %%
# load model
original_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
original_model.to(device)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.to(device)

# create dataset iterators
target_unlearn_iter = iter(target_dataset["unlearn"].batch(batch_size))
target_relearn_iter = iter(target_dataset["relearn"].batch(batch_size))
retain_relearn_iter = iter(retain_dataset["relearn"].batch(batch_size))


# %% install activation saving hooks
def save_pre_activations(module, args, output):
    module.last_pre_activations = args[0].detach().clone()


for l in original_model.model.layers:
    l.mlp.gate_proj.register_forward_hook(save_pre_activations)
    l.mlp.up_proj.register_forward_hook(save_pre_activations)
    l.mlp.down_proj.register_forward_hook(save_pre_activations)


# %%
def decay_on_module(module, original_module, decay_strength=0.03):
    weight = module.weight
    act = original_module.last_pre_activations

    act = act.abs()
    act /= act.max()

    scales = 1 - act * decay_strength
    scales = scales.mean(axis=[0, 1])
    assert scales.shape == (weight.shape[1],)

    weight *= scales

# calculate initial perplexities
init_target_ppl = get_perplexity(model, target_dataset)
init_retain_ppl = get_perplexity(model, retain_dataset)
print("init_target_ppl", init_target_ppl)
print("init_retain_ppl", init_retain_ppl)

# %%
for i in range(10):

    # with pt.no_grad():
    #     forward(original_model, next(target_unlearn_iter))
    #     for l, original_l in zip(model.model.layers, original_model.model.layers):
    #         decay_on_module(l.mlp.gate_proj, original_l.mlp.gate_proj)
    #         decay_on_module(l.mlp.up_proj, original_l.mlp.up_proj)
    #         decay_on_module(l.mlp.down_proj, original_l.mlp.down_proj)

    # normal_train_step(model, next(retain_relearn_iter), 0.0005)
    scale_perturbation(model, original_model.state_dict(), 0.99)

    # normal_train_step(model, next(target_relearn_iter), 0.0003)

    res = {
        "target": get_perplexity(model, target_dataset) - init_target_ppl,
        "retain": get_perplexity(model, retain_dataset) - init_retain_ppl,
        "norm": get_norm_of_weights_change(model, original_model.state_dict()),
    }
    print({k: f"{v:.2f}" for k, v in res.items()})
    pt.cuda.empty_cache()
    if i == 0:
        first_res = res
print("change: ", {k: f"{v - first_res[k]:.2f}" for k, v in res.items()})

# %%
# go through all modules in model:
for module in model.named_modules():
    print(module)