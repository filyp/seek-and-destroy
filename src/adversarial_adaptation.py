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
quantile=0.95  # between 0 and 1
criterion='(c("forget_abs_logit").abs() + 0.1) / c("en_abs_logit").abs()'
module_type="up_proj"
unlearn_lr=1e-3
adv_lr=1e-3
unlearn_steps=30
# 
# retain_lr=0
# relearn_lr=1e-3
relearn_steps=30


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

# add lora
lora_config = LoraConfig(r=1, target_modules=[module_type])
peft_model = get_peft_model(model, lora_config, adapter_name="adversarial_lora")
model = peft_model.model

# initialize optimizers
base_optimizer = pt.optim.Adam(
    [p for n, p in model.named_parameters() if ".base_layer." in n],
    lr=unlearn_lr,
    betas=(0.9, 0.999),
)
lora_optimizer = pt.optim.Adam(
    [p for n, p in model.named_parameters() if ".adversarial_lora." in n],
    lr=adv_lr,
    betas=(0.9, 0.999),
)

# %%
# ! unlearn loop
for step in range(1, 1 + unlearn_steps):
    model.train()
    input_ids = get_batch(forget_iter, 32)

    # ! unlearn on the base model
    base_optimizer.zero_grad(set_to_none=True)
    only_grad_on(model, ".base_layer.")
    loss = correct_logit_loss(model(input_ids), input_ids)
    loss.backward()
    # apply mask
    for name, param in model.named_parameters():
        name = name.replace(".base_layer", "")
        if name in mask:
            param.grad = param.grad * mask[name]
    base_optimizer.step()

    # ! relearn with adversarial lora
    lora_optimizer.zero_grad(set_to_none=True)
    only_grad_on(model, ".adversarial_lora.")
    loss = cross_entropy_loss(model(input_ids), input_ids)
    loss.backward()
    lora_optimizer.step()

    if step % 10 == 0:
        print_perplexities(model, [forget_eval, retain_eval], step)

print("ppl without adversarial lora:")
with peft_model.disable_adapter():
    print_perplexities(model, [forget_eval, retain_eval], 0)


# %%
del mask

# ! prepare for relearning
# merge retain_lora weights
model = peft_model.merge_and_unload()

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

