# %% init things, which need to be run once
import time

import matplotlib.pyplot as plt
import torch as pt
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import *

pt.set_default_device("cuda")
set_seeds(42)


# load model
# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)
forget_iter = looping_iter(forget_set["train"])
retain_iter = looping_iter(retain_set["train"])
forget_eval_batch = get_batch(iter(forget_set["validation"]), 32)
retain_eval_batch = get_batch(iter(retain_set["validation"]), 32)

def only_grad_on(model, name_part):
    for name, param in model.named_parameters():
        param.requires_grad = name_part in name



# %% initialize LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=1,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],  # fmt: skip
)
model.add_adapter(lora_config, adapter_name="adversarial_lora")

# %% initialize optimizers
base_optimizer = pt.optim.Adam(
    # [p for n, p in model.named_parameters() if ".base_layer." in n],
    [p for n, p in model.named_parameters() if ".adversarial_lora." not in n],
    lr=0.000015,
    betas=(0.9, 0.999),
)
lora_optimizer = pt.optim.Adam(
    [p for n, p in model.named_parameters() if ".adversarial_lora." in n],
    lr=0.001,  # even 0.002 makes it unstable
    betas=(0.9, 0.999),
)

# %% config
c = SimpleNamespace(
    model_id=model_id,
    base_lr=base_optimizer.param_groups[0]["lr"],
    lora_lr=lora_optimizer.param_groups[0]["lr"],
    loudness_vs_retain=0.2,
    adv_forget_vs_adv_retain=0.66,
    train_base=True,
    train_lora=True,
    lora_config=lora_config.to_dict(),
)
wandb.init(
    project="adversarial_adaptation", group="unlearning", config=vars(c), name="no_lora"
)
# because of how Adam adapts itself, loss scale doesn't matter,
#     only the proportions of the loss components:
# loudness and retain
# adv_forget and adv_retain
assert 0 <= c.loudness_vs_retain <= 1
assert 0 <= c.adv_forget_vs_adv_retain <= 1
i = 0

# %%
print_perplexities(model, [forget_eval_batch, retain_eval_batch], -1)
# %% training loop
# c.train_base = False
# c.loudness_vs_retain = 0.3

start_time = time.time()
for _ in range(10000):
    i += 1
    # For base model updates
    model.train()
    loss = DefaultNamespace()

    # note: only_grad_on must always come after set_adapter,
    #     bc set_adapter changes requires_grad
    # also we need to finalize each 4-line chunk with backward()
    #     before setting new requires_grad in the new chunk
    # even the forward pass seems to depend on which requires_grad are set
    # ! TLDR: just don't mess with the order in these 4-line blocks

    # todo: we currently don't train layer norm params, maybe we should

    if c.train_base:
        base_optimizer.zero_grad(set_to_none=True)

        model.set_adapter(["adversarial_lora"])
        only_grad_on(model, ".base_layer.")
        f_input_ids = get_batch(forget_iter, 8)
        loss.adv_loudness = clipped_correct_logit_loss(model(f_input_ids), f_input_ids)
        (loss.adv_loudness * c.loudness_vs_retain).backward()

        model.set_adapter([])
        only_grad_on(model, ".base_layer.")
        r_input_ids = get_batch(retain_iter, 16)
        loss.retain = cross_entropy_loss(model(r_input_ids), r_input_ids)
        (loss.retain * (1 - c.loudness_vs_retain)).backward()

        base_optimizer.step()

    if c.train_lora:
        lora_optimizer.zero_grad(set_to_none=True)

        model.set_adapter(["adversarial_lora"])
        only_grad_on(model, ".adversarial_lora.")
        f_input_ids = get_batch(forget_iter, 16)
        loss.adv_forget = cross_entropy_loss(model(f_input_ids), f_input_ids)
        (loss.adv_forget * c.adv_forget_vs_adv_retain).backward()

        model.set_adapter(["adversarial_lora"])
        only_grad_on(model, ".adversarial_lora.")
        r_input_ids = get_batch(retain_iter, 8)
        loss.adv_retain = cross_entropy_loss(model(r_input_ids), r_input_ids)
        (loss.adv_retain * (1 - c.adv_forget_vs_adv_retain)).backward()

        lora_optimizer.step()

    # evaluate
    if i % 10 == 0:
        model.eval()
        with pt.no_grad():
            model.set_adapter([])
            loss.loudness = correct_logit_loss(model(forget_eval_batch), forget_eval_batch)
            loss.forget = cross_entropy_loss(model(forget_eval_batch), forget_eval_batch)
            loss.retain = cross_entropy_loss(model(retain_eval_batch), retain_eval_batch)
            model.set_adapter(["adversarial_lora"])
            loss.adv_loudness = correct_logit_loss(model(forget_eval_batch), forget_eval_batch)
            loss.adv_forget = cross_entropy_loss(model(forget_eval_batch), forget_eval_batch)
            loss.adv_retain = cross_entropy_loss(model(retain_eval_batch), retain_eval_batch)

    # calculate and print stats
    stats = dict(
        loudness=loss.loudness.item(),
        forget=loss.forget.exp(),
        retain=loss.retain.exp(),
        adv_loudness=loss.adv_loudness.item(),
        adv_forget=loss.adv_forget.exp(),
        adv_retain=loss.adv_retain.exp(),
    )

    if i % 10 == 0:
        # if current run is active, log stats
        if wandb.run:
            wandb.log(stats, step=i)

    if i % 10 == 1:
        print("\n      " + "   ".join(f"{k:>10}" for k in stats.keys()))
    print(f"{i:4d}  " + "   ".join(f"{v:10.2f}" for v in stats.values()))

    # training past the point of breaking LoRA doesn't improve final performance
    # so stop if we broke through the LoRA
    if stats["adv_forget"] > 500 and (i + 1) % 10 == 0:
        break

    # if i % 100 == 0:
    #     # save model
    #     model_path = get_repo_root() / "models" / f"{wandb.run.name}_{i}steps.pt"
    #     pt.save(model.state_dict(), model_path)

print(f"time: {time.time() - start_time:.2f}s")

# %%
if wandb.run:
    wandb.finish()

# # %% remove lora
# # (to use delete_adapter we'd need to use get_peft_model)
# state_dict = model.state_dict()
# remove_lora(state_dict)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
# model.load_state_dict(state_dict)

# # %% save model
# model_path = get_repo_root() / "models" / f"{wandb.run.name}_{i}steps.pt"
# pt.save(model.state_dict(), model_path)
