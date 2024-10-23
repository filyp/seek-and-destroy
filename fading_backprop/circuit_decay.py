# %%
import matplotlib.pyplot as plt
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    device,
    forward,
    get_perplexity,
    load_one_oscar_shard,
    get_norm_of_weights_change,
    scale_perturbation,
    normal_train_step,
)

# params
batch_size = 32
model_id = "Qwen/Qwen2.5-0.5B"

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_dataset = load_one_oscar_shard("pl", tokenizer)
retain_dataset = load_one_oscar_shard("en", tokenizer)

# %%
# load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.to(device)

# create dataset iterators
forget_unlearn_iter = iter(forget_dataset["unlearn"].batch(batch_size))
forget_relearn_iter = iter(forget_dataset["relearn"].batch(batch_size))
retain_relearn_iter = iter(retain_dataset["relearn"].batch(batch_size))


def get_modules(model):
    for module_name, module in model.named_modules():
        if "mlp" in module_name and "_proj" in module_name:
            yield module_name, module


# install activation saving hooks
def save_pre_activations(module, args, output):
    module.last_pre_activations = args[0].detach().clone()


for module_name, module in get_modules(model):
    module.register_forward_hook(save_pre_activations)


# prepare activation importance tensors
forget_act_imps = {}
retain_act_imps = {}
forget_counter = 0
retain_counter = 0
for module_name, module in get_modules(model):
    input_size = module.weight.shape[1]
    forget_act_imps[module_name] = pt.zeros(input_size).to(device)
    retain_act_imps[module_name] = pt.zeros(input_size).to(device)

# %%
for i in range(1):
    print(".", end="")
    with pt.no_grad():
        forward(model, next(forget_unlearn_iter))
    for module_name, module in get_modules(model):
        act = module.last_pre_activations
        act = act ** 2
        act = act.mean(axis=[0, 1])
        forget_act_imps[module_name] += act
    forget_counter += 1

# %%
for i in range(1):
    print(".", end="")
    with pt.no_grad():
        forward(model, next(retain_relearn_iter))
    for module_name, module in get_modules(model):
        act = module.last_pre_activations
        act = act ** 2
        act = act.mean(axis=[0, 1])
        retain_act_imps[module_name] += act
    retain_counter += 1


# %%
def apply_decay(percentile, mult):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
    model.to(device)
    for module_name, module in get_modules(model):
        forget_imp = forget_act_imps[module_name] / forget_counter
        retain_imp = retain_act_imps[module_name] / retain_counter

        rel_imp = forget_imp / retain_imp
        cutoff = rel_imp.quantile(1 - percentile / 100)
        # print(module.weight[:, rel_imp > cutoff].shape)
        with pt.no_grad():
            module.weight[:, rel_imp > cutoff] *= mult

    return model


# %%
original_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
original_model.to(device)

# # calculate initial perplexities
init_forget_ppl = get_perplexity(original_model, forget_dataset)
init_retain_ppl = get_perplexity(original_model, retain_dataset)
print(f"{init_retain_ppl=:6.1f} {init_forget_ppl=:6.1f}\n")


def eval_vals(model):
    diff_forget = get_perplexity(model, forget_dataset) - init_forget_ppl
    diff_retain = get_perplexity(model, retain_dataset) - init_retain_ppl
    ratio = diff_forget / diff_retain
    print(f"{diff_retain=:4.1f}  {diff_forget=:6.1f}  {ratio=:10.0f}")


# %%
model = apply_decay(0.1, -1)
eval_vals(model)

# %%

# for i in range(10):

#     normal_train_step(model, next(retain_relearn_iter), 0.0003)
#     # scale_perturbation(model, original_model.state_dict(), 0.99)

#     normal_train_step(model, next(forget_relearn_iter), 0.0003)

#     res = {
#         "forget": get_perplexity(model, forget_dataset) - init_forget_ppl,
#         "retain": get_perplexity(model, retain_dataset) - init_retain_ppl,
#         "norm": get_norm_of_weights_change(model, original_model.state_dict()),
#     }
#     print({k: f"{v:.2f}" for k, v in res.items()})
#     pt.cuda.empty_cache()
#     if i == 0:
#         first_res = res
# print("change: ", {k: f"{v - first_res[k]:.2f}" for k, v in res.items()})

# # %%
