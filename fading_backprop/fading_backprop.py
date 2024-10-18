# %%
import torch as pt
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import device, forward, get_perplexity, load_one_oscar_shard


# %%
def install_hooks_for_saving_gradients(model):
    # save the gradients downstream of each mlp, to be used in custom unlearning
    def save_output_grad_hook(module, grad_input, grad_output):
        module.output_grad = grad_output[0]

    for layer in model.model.layers:
        assert not layer.mlp.down_proj._backward_hooks
        layer.mlp.down_proj.register_full_backward_hook(save_output_grad_hook)


def install_hooks_for_fading_backprop(model):
    # (tested for gemma-2-2b and Qwen2.5-0.5B)
    def scale_grad_hook(module, grad_input, grad_output):
        grad = list(grad_input)
        # we rely on fade_factor set with set_fade_factor
        grad[0] *= module.fade_factor
        return grad

    for layer in model.model.layers:
        assert not layer.mlp._backward_hooks
        layer.mlp.register_full_backward_hook(scale_grad_hook)
        assert not layer.input_layernorm._backward_hooks
        layer.input_layernorm.register_full_backward_hook(scale_grad_hook)


def set_fade_factor(model, fade_factor):
    # saves the fade_factor in each relevant module, so that hooks can access it
    for layer in model.model.layers:
        layer.mlp.fade_factor = fade_factor
        layer.input_layernorm.fade_factor = fade_factor


def normal_train_step(model, batch, lr):
    set_fade_factor(model, 1)
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)
    loss = forward(model, batch)
    loss.backward()
    optimizer.step()


def unlearn_step(model, batch, lr, fade_factor):
    set_fade_factor(model, fade_factor)
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)

    loss = forward(model, batch)
    loss.backward()
    # get rid of the normal grad
    optimizer.zero_grad(set_to_none=True)

    # calculate custom grads
    for layer in model.model.layers:
        module = layer.mlp.down_proj
        # get the downstream gradient saved by the hook
        output_grad = module.output_grad
        output_grad = output_grad[:, :-1, :]  # last token position has no gradient
        assert output_grad.norm(dim=-1).all()

        # calculate the projection
        projection_amps = pt.einsum("oi,bto->bti", module.weight, output_grad)
        projection_amps /= output_grad.norm(dim=-1, keepdim=True)
        update = pt.einsum("bto,bti->oi", output_grad, projection_amps)
        module.weight.grad = update

    optimizer.step()


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

    # unlearning loop
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
            fade_factor = f_schedule(step_num)
            unlearn_step(model, next(target_unlearn_iter), unlearn_lr, fade_factor)

        # evaluate
        if (step_num + 1) % eval_every_n_steps == 0:
            ppl = {
                "target": get_perplexity(model, target_dataset),
                "retain": get_perplexity(model, retain_dataset),
            }
            print({k: f"{v:.2f}" for k, v in ppl.items()}, end="  ")

    # relearning loop
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

    return ppl
