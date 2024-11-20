# %%
import torch as pt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import *
from utils_dataloading import *
from utils_important import *

model_id = "EleutherAI/pythia-70m-deduped"
pt.set_default_device("cuda")

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_python_dataset(tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)

f_eval = get_batch(iter(forget_set["validation"]), 32)
r_eval = get_batch(iter(retain_set["validation"]), 32)

model = AutoModelForCausalLM.from_pretrained(model_id)
init_forget = eval_loss(model, f_eval)
init_retain = eval_loss(model, r_eval)
print(f"init forget: {init_forget:6.2f}    init retain: {init_retain:6.2f}")

# %%
# ! parameters
quantile = 0.0001  # between 0 and 1
target_modules = ["dense", "dense_h_to_4h", "dense_4h_to_h"]
unlearn_lr = 5e-3
adv_lora_lr = 6e-4
ret_lora_lr = 3e-4
disruption_score_decay = 0.95
unlearn_steps = 200
relearn_lr = 3e-4
f_amp = 0.9

path = save_file_and_stdout_open(__file__)
set_seeds(42)
# todo: smth is still undeterministic!
# prepare data iterators
forget_iter = looping_iter(forget_set["train"])
retain_iter = looping_iter(retain_set["train"])

# load model
model = AutoModelForCausalLM.from_pretrained(model_id)
# add loras
adv_lora_config = LoraConfig(r=1, target_modules=target_modules, lora_dropout=0.1)
peft_model = get_peft_model(model, adv_lora_config, adapter_name="adv_lora", mixed=True)
model = peft_model.model
ret_lora_config = LoraConfig(r=1, target_modules=target_modules, lora_dropout=0.1)
peft_model.add_adapter("ret_lora", ret_lora_config)

interven_params = [p for n, p in model.named_parameters() if ".base_layer.weight" in n]
adv_lora_params = [p for n, p in model.named_parameters() if ".adv_lora." in n]
ret_lora_params = [p for n, p in model.named_parameters() if ".ret_lora." in n]

# initialize optimizers
base_optimizer = pt.optim.SGD(interven_params, lr=unlearn_lr)
adv_optimizer = pt.optim.Adam(adv_lora_params, lr=adv_lora_lr)
ret_optimizer = pt.optim.Adam(ret_lora_params, lr=ret_lora_lr)

# initialize disruption scores
for param in interven_params:
    param.disruption_score = pt.zeros_like(param)

# %
# ! unlearning loop
print("step      base_f      base_r      adv_f      adv_r")
for step in range(1, 1 + unlearn_steps):
    model.train()
    f_input_ids = get_batch(forget_iter, 16)
    r_input_ids = get_batch(retain_iter, 16)

    # ! retain with helper lora
    peft_model.set_adapter(["ret_lora"])
    only_grad_on(model, interven_params + ret_lora_params)
    model.zero_grad(set_to_none=True)
    loss = cross_entropy_loss(model(r_input_ids), r_input_ids)
    loss.backward()
    ret_optimizer.step()
    # ! update disruption scores
    for param in interven_params:
        param.disruption_score *= disruption_score_decay
        param.disruption_score += param.grad**2

    # ! unlearn on the base model
    peft_model.set_adapter(["ret_lora", "adv_lora"])
    only_grad_on(model, interven_params)
    model.zero_grad(set_to_none=True)
    # loss = cross_entropy_loss(model(f_input_ids), f_input_ids)
    loss = clipped_correct_logit_loss(model(f_input_ids), f_input_ids)
    loss.backward()
    # ! get threshold
    disruption_scores = [p.disruption_score / p.grad.abs()**f_amp for p in interven_params]
    threshold = get_threshold(quantile, disruption_scores)
    # ! apply mask
    for param in interven_params:
        mask = param.disruption_score < threshold
        param.grad *= mask
    base_optimizer.step()

    # ! relearn with adversarial lora
    peft_model.set_adapter(["ret_lora", "adv_lora"])
    only_grad_on(model, adv_lora_params)
    model.zero_grad(set_to_none=True)
    loss = cross_entropy_loss(model(f_input_ids), f_input_ids)
    loss.backward()
    adv_optimizer.step()

    # ! eval
    if step % 10 == 0:
        res = {}
        peft_model.set_adapter(["ret_lora"])
        res["base_forget"] = eval_loss(model, f_eval)
        res["base_retain"] = eval_loss(model, r_eval)
        peft_model.set_adapter(["ret_lora", "adv_lora"])
        res["adv_forget"] = eval_loss(model, f_eval)
        res["adv_retain"] = eval_loss(model, r_eval)

        print(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))
        if wandb.run:
            wandb.log(res, step=step)

        if res["adv_forget"] > 10:
            print("Adversarial LoRA defeated")
            break
        if res["base_retain"] > init_retain + 0.1:
            print("Retain performance broken")
            break

    # ! eval relearning
    if step % 50 == 0:
        collapsed_model = copy_model_and_collapse_loras(peft_model)
        relearn(collapsed_model, relearn_lr, 30, forget_set, retain_set)


# # ! final bigger eval relearning
# collapsed_model = copy_model_and_collapse_loras(peft_model)
# final_forget_loss = relearn(collapsed_model, relearn_lr, 50, forget_set, retain_set)

save_file_and_stdout_close(path)

# %%
