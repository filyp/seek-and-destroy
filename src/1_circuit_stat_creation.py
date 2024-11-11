# %%
import torch as pt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *

model_id = "Qwen/Qwen2.5-0.5B"
pt.set_default_device("cuda")

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)


# %%
def get_circuit(data_iter, grad_acc_fn, loss_fn, num_steps=1000):
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

    # note: using hooks is more complicated than processing grads after backward()
    #       but it prevents a memory spike - no need to store many param.grad
    def custom_grad_acc(param):
        # * note that we apply grad_acc_fn AFTER we've already summed over batch and token pos
        # * applying it before, would be really complicated
        param.custom_grad += grad_acc_fn(param.grad)
        param.grad = None

    for param in model.parameters():
        param.register_post_accumulate_grad_hook(custom_grad_acc)
        param.custom_grad = pt.zeros_like(param)

    for _ in tqdm(range(num_steps)):
        input_ids = get_batch(data_iter, 32)
        loss = loss_fn(model(input_ids), input_ids)
        loss.backward()

    # gather
    return {
        name: -param.custom_grad / num_steps for name, param in model.named_parameters()
    }


# %%
for data_name, data_iter in [
    ("retain", looping_iter(retain_set["train"])),
    ("forget", looping_iter(forget_set["train"])),
]:
    for grad_acc_name, grad_acc_fn in [
        ("linear", lambda x: x),
        ("square", lambda x: x**2),
        ("absolu", lambda x: x.abs()),
        # * actually these could be computed in parallel, with that would 3x mem usage
    ]:
        for loss_name, loss_fn in [
            ("cross_entropy", cross_entropy_loss),
            ("correct_logit", correct_logit_loss),
        ]:
            circuit_name = f"{data_name}_{grad_acc_name}_{loss_name}.pt"
            print("calculating:", circuit_name)
            pt.save(
                get_circuit(data_iter, grad_acc_fn, loss_fn),
                repo_root() / "circuits" / circuit_name,
            )
