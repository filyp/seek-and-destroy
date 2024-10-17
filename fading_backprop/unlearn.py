# %%
from itertools import islice

import matplotlib.pyplot as plt
import torch as pt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, forward, get_perplexity, load_one_oscar_shard

# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"

# %% load dataset
tokenizer = AutoTokenizer.from_pretrained(model_id)
pl_dataset = load_one_oscar_shard("pl", tokenizer)
en_dataset = load_one_oscar_shard("en", tokenizer)


# %%
def normal_train_step(model, batch, lr):
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    loss = forward(model, batch)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # unneeded, but clean up just to be sure


def unlearn_step(model, batch, lr):
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)

    loss = forward(model, batch)
    loss.backward()
    # get rid of the normal grad
    optimizer.zero_grad(set_to_none=True)

    # calculate custom grads
    for layer in model.model.layers:
        module = layer.mlp.down_proj
        # get the grad
        output_grad = module.output_grad
        output_grad = output_grad[:, :-1, :]  # last token position has no gradient
        assert output_grad.norm(dim=-1).all()

        # calculate the projection
        projection_amps = pt.einsum("oi,bto->bti", module.weight, output_grad)
        projection_amps /= output_grad.norm(dim=-1, keepdim=True)
        update = pt.einsum("bto,bti->oi", output_grad, projection_amps)
        module.weight.grad = update

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # unneeded, but clean up just to be sure


def set_fade_factor(model, fade_factor):
    for layer in model.model.layers:
        layer.mlp.fade_factor = fade_factor
        layer.input_layernorm.fade_factor = fade_factor


# %%
def unlearn_and_relearn(
    model,
    target_dataset,
    retain_dataset,
    f_schedule=lambda step: 0.0,
    num_unlearning_steps=10,
    num_relearning_steps=10,
    eval_every_n_steps=2,
    relearn_lr=0.0003,
    unlearn_lr=1,
    pl_ppl_threshold=float("inf"),
    batch_size=32,
):
    # todo test for f=0

    # install hooks for saving gradients
    def save_output_grad_hook(module, grad_input, grad_output):
        module.output_grad = grad_output[0]

    for layer in model.model.layers:
        assert not layer.mlp.down_proj._backward_hooks
        layer.mlp.down_proj.register_full_backward_hook(save_output_grad_hook)

    # install hooks for fading the backprop
    # (tested for gemma-2-2b and Qwen2.5-0.5B)
    def scale_grad_hook(module, grad_input, grad_output):
        grad = list(grad_input)
        grad[0] *= module.fade_factor
        return grad

    for layer in model.model.layers:
        assert not layer.mlp._backward_hooks
        layer.mlp.register_full_backward_hook(scale_grad_hook)
        assert not layer.input_layernorm._backward_hooks
        layer.input_layernorm.register_full_backward_hook(scale_grad_hook)

    # create dataset iterators
    target_unlearn_iter = iter(target_dataset["unlearn"].batch(batch_size))
    target_relearn_iter = iter(target_dataset["relearn"].batch(batch_size))
    retain_relearn_iter = iter(retain_dataset["relearn"].batch(batch_size))

    # initial perplexities
    perplexities = {
        "target": get_perplexity(model, target_dataset),
        "retain": get_perplexity(model, retain_dataset),
    }
    print("initial perplexity: ", {k: f"{v:.2f}" for k, v in perplexities.items()})

    # perplexity on retain set is not allowed to raise above this value
    # - this triggers relearning on retain set
    allowed_retain_perplexity = perplexities["retain"] * 1.2
    print(f"{allowed_retain_perplexity=:.2f}")

    # unlearning loop
    for step_num in range(num_unlearning_steps):
        print(f"\nstep {step_num + 1:3}", end="  ")
        if perplexities["retain"] > allowed_retain_perplexity:
            print("> relearning retain", end="  ")
            set_fade_factor(model, 1)
            normal_train_step(model, next(retain_relearn_iter), relearn_lr)
        elif perplexities["target"] > pl_ppl_threshold:
            print("**relearning target", end="  ")
            set_fade_factor(model, 1)
            # we intentionally use target_UNlean_iter, to not affect relearn split here
            normal_train_step(model, next(target_unlearn_iter), relearn_lr)
        else:
            print("  unlearning target", end="  ")
            set_fade_factor(model, f_schedule(step_num))
            unlearn_step(model, next(target_unlearn_iter), unlearn_lr)

        if (step_num + 1) % eval_every_n_steps == 0:
            perplexities = {
                "target": get_perplexity(model, target_dataset),
                "retain": get_perplexity(model, retain_dataset),
            }
            print({k: f"{v:.2f}" for k, v in perplexities.items()}, end="  ")

    # relearning loop
    print("\n### relearning pl started ###")
    set_fade_factor(model, 1)
    for step_num in range(num_relearning_steps):
        print(f"\nstep {step_num + 1:3}", end="  ")
        if perplexities["retain"] > allowed_retain_perplexity:
            print("> relearning retain", end="  ")
            normal_train_step(model, next(retain_relearn_iter), relearn_lr)
        else:
            print("  relearning target", end="  ")
            normal_train_step(model, next(target_relearn_iter), relearn_lr)

        if (step_num + 1) % eval_every_n_steps == 0:
            perplexities = {
                "target": get_perplexity(model, target_dataset),
                "retain": get_perplexity(model, retain_dataset),
            }
            print({k: f"{v:.2f}" for k, v in perplexities.items()}, end="  ")

    return perplexities


# %%
criterion = min
param_name = "relearn_lr"
param_starting_value = 0.001
param_value_and_ppl = []

def get_final_target_perplexity(**kwargs):
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
    model.to(device)

    final_perplexities = unlearn_and_relearn(model, pl_dataset, en_dataset, **kwargs)
    return final_perplexities["target"]


# %%
ppl = get_final_target_perplexity(**{param_name: param_starting_value})
param_value_and_ppl.append((param_starting_value, ppl))
ppl = get_final_target_perplexity(**{param_name: param_starting_value * 2})
param_value_and_ppl.append((param_starting_value * 2, ppl))
# %%
