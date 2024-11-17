# %%
import torch as pt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *

model_id = "Qwen/Qwen2.5-0.5B"
pt.set_default_device("cuda")

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)
forget_eval = get_batch(iter(forget_set["validation"]), 32)
retain_eval = get_batch(iter(retain_set["validation"]), 32)
forget = "pl"


def only_grad_on(model, name_part):
    for name, param in model.named_parameters():
        param.requires_grad = name_part in name


# %%
# ! parameters
quantile = 0.8  # between 0 and 1
# criterion='(c("forget_abs_logit").abs() + 0.1) / c("en_abs_logit").abs()'
criterion = 'c("en_abs_logit").abs() * (-1)'
module_type = "up_proj"
# note: too small forget_lr with bfloat16, can make updates 0 due to numerical errors ?
# unlearn_lr = 15e-4
# adversa_lr = 30e-4
# helper_lr = 20e-4
# for SGD
unlearn_lr = 1e-1
# 
adversa_lr = 3e-3
helper_lr = 5e-4
#
# retain_mult = 3  # retain grads are naturally about 3x smaller (even though loss has similar scale), IDK why
unlearn_steps = 1000000

set_seeds(42)
# prepare data iterators
forget_iter = looping_iter(forget_set["train"])
retain_iter = looping_iter(retain_set["train"])

# ! get mask
criterion = criterion.replace("forget", forget)
scores = kinda_safe_eval(criterion)
scores = {k: v for k, v in scores.items() if module_type in k}
k = int(quantile * sum(s.numel() for s in scores.values()))
threshold = pt.cat([s.flatten() for s in scores.values()]).kthvalue(k).values
mask = {k: v > threshold for k, v in scores.items()}
del scores

# load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.float32)

# add loras
adv_lora_config = LoraConfig(r=1, target_modules=[module_type])
peft_model = get_peft_model(model, adv_lora_config, adapter_name="adv_lora", mixed=True)
model = peft_model.model
helper_lora_config = LoraConfig(r=4, target_modules=[module_type])
peft_model.add_adapter("helper_lora", helper_lora_config)

# initialize optimizers
base_optimizer = pt.optim.SGD(
    [p for n, p in model.named_parameters() if ".base_layer." in n],
    lr=unlearn_lr,
)
adv_optimizer = pt.optim.Adam(
    [p for n, p in model.named_parameters() if ".adv_lora." in n],
    lr=adversa_lr,
)
helper_optimizer = pt.optim.Adam(
    [p for n, p in model.named_parameters() if ".helper_lora." in n],
    lr=helper_lr,
)

# %%
# ! unlearn loop
print("step         f         r    base_f    base_r")
for step in range(1, 1 + unlearn_steps):
    model.train()
    f_input_ids = get_batch(forget_iter, 8)
    r_input_ids = get_batch(retain_iter, 8)

    # ! unlearn on the base model
    base_optimizer.zero_grad(set_to_none=True)
    peft_model.set_adapter(["helper_lora", "adv_lora"])
    only_grad_on(model, ".base_layer.")
    loss = correct_logit_loss(model(f_input_ids), f_input_ids)
    loss.backward()
    # print("logit", model.model.layers[12].mlp.up_proj.base_layer.weight.grad.norm())
    # ! retain on the base model
    # with peft_model.disable_adapter():
    #     only_grad_on(model, ".base_layer.")  # unneeded, but let's be safe
    #     loss = cross_entropy_loss(model(r_input_ids), r_input_ids) * retain_mult
    #     loss.backward()
    # ! apply mask
    for name, param in model.named_parameters():
        name = name.replace(".base_layer", "")
        if name in mask:
            param.grad = param.grad * mask[name]
    base_optimizer.step()

    # ! retain with helper lora
    helper_optimizer.zero_grad(set_to_none=True)
    peft_model.set_adapter(["helper_lora"])
    only_grad_on(model, ".helper_lora.")
    cross_entropy_loss(model(r_input_ids), r_input_ids).backward()
    helper_optimizer.step()

    # ! relearn with adversarial lora
    adv_optimizer.zero_grad(set_to_none=True)
    peft_model.set_adapter(["helper_lora", "adv_lora"])
    only_grad_on(model, ".adv_lora.")
    cross_entropy_loss(model(f_input_ids), f_input_ids).backward()
    adv_optimizer.step()

    # ! eval
    if step % 5 == 0:
        peft_model.set_adapter(["helper_lora", "adv_lora"])
        f_ppl, r_ppl = get_perplexities(model, [forget_eval, retain_eval])
        peft_model.set_adapter(["helper_lora"])
        f_ppl_base, r_ppl_base = get_perplexities(model, [forget_eval, retain_eval])
        print(f"{step:4} {f_ppl:9.2f} {r_ppl:9.2f} {f_ppl_base:9.2f} {r_ppl_base:9.2f}")

        if f_ppl > 10000:
            break

# %%
del mask
peft_model.delete_adapter("adv_lora")

# ! merge and unload
peft_model.set_adapter(["helper_lora"])
model = peft_model.merge_and_unload()
del peft_model

# %%
# ! parameters
relearn_lr = 1e-3
relearn_steps = 100

# add relearning lora
lora_config = LoraConfig(r=1, target_modules="all-linear")
peft_model = get_peft_model(model, lora_config, adapter_name="relearning_lora")
model = peft_model.model

optimizer = pt.optim.Adam(model.parameters(), lr=relearn_lr, betas=(0.9, 0.999))

# ! relearning loop
print("relearning...")
for step in range(1, 1 + relearn_steps):
    # standard forward, backward, and update
    model.train()
    optimizer.zero_grad(set_to_none=True)
    forget_input_ids = get_batch(forget_iter, 16)
    retain_input_ids = get_batch(retain_iter, 16)
    loss_forget = cross_entropy_loss(model(forget_input_ids), forget_input_ids)
    loss_retain = cross_entropy_loss(model(retain_input_ids), retain_input_ids)
    (loss_forget + loss_retain).backward()
    optimizer.step()

    if step % 10 == 0:
        print_perplexities(model, [forget_eval, retain_eval], step)

# %%
