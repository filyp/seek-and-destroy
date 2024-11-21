# %%
import torch as pt
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils_dataloading import *
from utils import *

# model_id = "Qwen/Qwen2.5-0.5B"
model_id = "EleutherAI/pythia-70m-deduped"
# model_id = "EleutherAI/pythia-160m-deduped"
# model_id = "HuggingFaceTB/SmolLM-135M"

pt.set_default_device("cuda")

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
# forget_set = load_one_oscar_shard("pl", tokenizer)
forget_set = load_python_dataset(tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)

forget_eval = get_batch(iter(forget_set["validation"]), 32)
retain_eval = get_batch(iter(retain_set["validation"]), 32)


def only_grad_on(model, name_part):
    for name, param in model.named_parameters():
        param.requires_grad = name_part in name


def get_mask(criterion, quantile, target_modules, model_id, forget_id):
    criterion = criterion.replace("FORGET", forget_id)
    criterion = criterion.replace("<", f"load_circuit('{model_id}/").replace(">", "')")
    scores = kinda_safe_eval(criterion)
    scores = {k: v for k, v in scores.items() if any(m in k for m in target_modules)}
    k = int(quantile * sum(s.numel() for s in scores.values()))
    threshold = pt.cat([s.flatten() for s in scores.values()]).kthvalue(k).values
    return {k: v < threshold for k, v in scores.items()}


model = AutoModelForCausalLM.from_pretrained(model_id)
init_forget, init_retain = get_perplexities(model, [forget_eval, retain_eval])
print(f"init forget: {init_forget:6.2f}    init retain: {init_retain:6.2f}")

# %%
# ! parameters
quantile = 0.999  # between 0 and 1
# target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"]  # fmt: skip
#
# target_modules = ["dense_4h_to_h"]
# target_modules = ["dense_h_to_4h"]
# target_modules = ["dense_h_to_4h", "dense_4h_to_h"]
target_modules = ["dense", "dense_h_to_4h", "dense_4h_to_h"]
all_main_modules = ["dense", "dense_h_to_4h", "dense_4h_to_h"]
#
# criterion = '<en_abs_logit> / (<FORGET_abs_logit> + 0.1)'
criterion = "<en_abs_logit>"
forget_id = "python"
#
# note: too small forget_lr with bfloat16, can make updates 0 due to numerical errors ?
unlearn_lr = 0.08e-4
adversa_lr = 3e-4
helper_lr = 3e-4
relearn_lr = 3e-4
# retain grads are naturally about 3x smaller (even though loss has similar scale), IDK why
# retain_mult = 3 * 2
unlearn_steps = 200
relearn_steps = 300
#
# circuit_name = f"{model_id}/{forget_id}_linear_logit"
# circuit_forget_lr = 6e-2

# run_name = f"U={unlearn_lr:.0e} A={adversa_lr:.0e} H={helper_lr:.0e} R={relearn_lr:.0e}"
# wandb.init(project="adversarial_adaptation", group="unlearning", name=run_name)

path = save_script(__file__)

set_seeds(42)
# prepare data iterators
forget_iter = looping_iter(forget_set["train"])
retain_iter = looping_iter(retain_set["train"])

mask = get_mask(criterion, quantile, target_modules, model_id, forget_id="python")

# # ! load circuit and sparsify
# circuit = load_circuit(circuit_name)
# circuit = {k: v for k, v in circuit.items() if any(m in k for m in target_modules)}
# circuit = TensorDict(circuit) * TensorDict(mask)

# load model
model = AutoModelForCausalLM.from_pretrained(model_id)

# add loras
adv_lora_config = LoraConfig(r=1, target_modules=all_main_modules, lora_dropout=0.1)
peft_model = get_peft_model(model, adv_lora_config, adapter_name="adv_lora", mixed=True)
model = peft_model.model
helper_lora_config = LoraConfig(r=4, target_modules=all_main_modules, lora_dropout=0.1)
peft_model.add_adapter("helper_lora", helper_lora_config)

# initialize optimizers
base_optimizer = pt.optim.SGD(
    [p for n, p in model.named_parameters() if ".base_layer." in n],
    lr=unlearn_lr,
)
# scheduler = LambdaLR(base_optimizer, lambda step: min(1.0, step / 100))

adv_optimizer = pt.optim.Adam(
    [p for n, p in model.named_parameters() if ".adv_lora." in n],
    lr=adversa_lr,
)
helper_optimizer = pt.optim.Adam(
    # helper_optimizer = pt.optim.SGD(
    [p for n, p in model.named_parameters() if ".helper_lora." in n],
    lr=helper_lr,
)

# ! unlearn loop
print("step      base_f      base_r      adv_f      adv_r")
for step in range(1, 1 + unlearn_steps):
    model.train()
    f_input_ids = get_batch(forget_iter, 16)
    r_input_ids = get_batch(retain_iter, 16)

    # # ! break the circuit a bit
    # for name, param in model.named_parameters():
    #     name = name.replace(".base_layer", "")
    #     if name in circuit:
    #         param.data -= circuit[name] * circuit_forget_lr

    # ! unlearn on the base model
    base_optimizer.zero_grad(set_to_none=True)
    peft_model.set_adapter(["helper_lora", "adv_lora"])
    only_grad_on(model, ".base_layer.")
    loss = clipped_correct_logit_loss(model(f_input_ids), f_input_ids)
    loss.backward()

    # # ! retain on the base model
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
    # scheduler.step()  # Update the learning rate

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
    if step % 10 == 0:
        peft_model.set_adapter(["helper_lora", "adv_lora"])
        adv_forget, adv_retain = get_perplexities(model, [forget_eval, retain_eval])
        peft_model.set_adapter(["helper_lora"])
        base_forget, base_retain = get_perplexities(model, [forget_eval, retain_eval])

        results = dict(
            base_forget=base_forget,
            base_retain=base_retain,
            adv_forget=adv_forget,
            adv_retain=adv_retain,
        )
        print(f"{step:4} " + " ".join(f"{v:11.2f}" for v in results.values()))
        if wandb.run:
            wandb.log(results, step=step)

        if adv_forget > 10000 or base_retain > init_retain * 1.1:
            break

del mask
# ! delete adversarial lora
peft_model.delete_adapter("adv_lora")
# ! merge and unload helper lora
peft_model.set_adapter(["helper_lora"])
model = peft_model.merge_and_unload()
del peft_model

pt.save(model.state_dict(), repo_root() / "models" / "autosave.pt")

# # load model
# model = AutoModelForCausalLM.from_pretrained(model_id)
# state_dict = pt.load(repo_root() / "models" / "model_post.pt", weights_only=True)
# model.load_state_dict(state_dict)

# add relearning lora
# lora_config = LoraConfig(r=1, target_modules="all-linear")
# lora_config = LoraConfig(r=1, target_modules=target_modules)
# lora_config = LoraConfig(r=1, target_modules=["up_proj"])
lora_config = LoraConfig(r=1, target_modules=["dense_h_to_4h"])
peft_model = get_peft_model(model, lora_config, adapter_name="relearning_lora")
model = peft_model.model

optimizer = pt.optim.Adam(model.parameters(), lr=relearn_lr, betas=(0.9, 0.999))

# ! relearning loop
print("relearning...")
for step in range(1 + unlearn_steps, 1 + unlearn_steps + relearn_steps):
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
        if wandb.run:
            wandb.log(results, step=step)

save_file_and_stdout_close(path)

# %%
