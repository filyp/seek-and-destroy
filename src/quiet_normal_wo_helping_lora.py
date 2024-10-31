# %% init things, which need to be run once
import time

import matplotlib.pyplot as plt
import torch as pt
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *

pt.set_default_device("cuda")
set_seeds(42)


def get_batch(iter, n):
    return pt.cat([next(iter)["input_ids"] for _ in range(n)])


def forward(model, batch):
    # forward pass
    logits = model(batch).logits
    # compute loss
    loss_fn = pt.nn.CrossEntropyLoss()
    return loss_fn(logits[:, :-1, :].flatten(end_dim=1), batch[:, 1:].flatten())


def forward_with_clipped_logit(model, batch):
    logits = model(batch).logits
    logits = logits[:, :-1, :].flatten(end_dim=1)
    ids = batch[:, 1:].flatten()
    true_logits = logits[pt.arange(len(ids)), ids]
    return true_logits.clip(0).mean()


def only_grad_on(model, name_part):
    for name, param in model.named_parameters():
        param.requires_grad = name_part in name


# load model
# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)
forget_iter = iter(forget_set["unlearn"])
retain_iter = iter(retain_set["unlearn"])
forget_eval_batch = get_batch(iter(forget_set["validation"]), 32)
retain_eval_batch = get_batch(iter(retain_set["validation"]), 32)


# initialize LoRAs
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],  # fmt: skip
)
model.add_adapter(lora_config, adapter_name="adversarial_lora")

# get initial perplexities
model.eval()
with pt.no_grad():
    initial_forget_ppl = forward(model, forget_eval_batch).exp()
    initial_retain_ppl = forward(model, retain_eval_batch).exp()

# Replace the single optimizer with two separate ones
base_optimizer = pt.optim.Adam(
    # [p for n, p in model.named_parameters() if ".base_layer." in n],
    [p for n, p in model.named_parameters() if ".adversarial_lora." not in n],
    lr=0.0001,
    betas=(0.9, 0.999),
)
lora_optimizer = pt.optim.Adam(
    [p for n, p in model.named_parameters() if ".adversarial_lora." in n],
    lr=0.001,
    betas=(0.9, 0.999),
)

# %%
# because of how Adam adapts itself, loss scale doesn't matter,
#     only the proportions of the loss components:
# loudness and retain
# adv_forget and adv_retain
loudness_vs_retain = 0.02
adv_forget_vs_adv_retain = 0.66
assert 0 <= loudness_vs_retain <= 1
assert 0 <= adv_forget_vs_adv_retain <= 1

train_base = True
train_lora = False

start_time = time.time()
for i in range(10):
    # For base model updates
    model.train()
    loss = DefaultNamespace()

    # loss = defaultdict(lambda: pt.tensor(pt.nan))
    # note: only_grad_on must always come after set_adapter,
    #     bc set_adapter changes requires_grad
    # also we need to finalize each 4-line chunk with backward()
    #     before setting new requires_grad in the new chunk
    # even the forward pass seems to depend on which requires_grad are set
    # ! TLDR: just don't mess with the order in these 4-line blocks

    if train_base:
        base_optimizer.zero_grad(set_to_none=True)

        model.set_adapter(["adversarial_lora"])
        only_grad_on(model, ".base_layer.")
        loss.loudness = forward_with_clipped_logit(model, get_batch(forget_iter, 4))
        (loss.loudness * loudness_vs_retain).backward()

        model.set_adapter([])
        only_grad_on(model, ".base_layer.")
        loss.retain = forward(model, get_batch(retain_iter, 32))
        (loss.retain * (1 - loudness_vs_retain)).backward()

        base_optimizer.step()

    if train_lora:
        lora_optimizer.zero_grad(set_to_none=True)

        model.set_adapter(["adversarial_lora"])
        only_grad_on(model, ".adversarial_lora.")
        loss.adv_forget = forward(model, get_batch(forget_iter, 16))
        (loss.adv_forget * adv_forget_vs_adv_retain).backward()

        model.set_adapter(["adversarial_lora"])
        only_grad_on(model, ".adversarial_lora.")
        loss.adv_retain = forward(model, get_batch(retain_iter, 8))
        (loss.adv_retain * (1 - adv_forget_vs_adv_retain)).backward()

        lora_optimizer.step()

    # evaluate
    if (i + 1) % 10 == 0:
        model.eval()
        with pt.no_grad():
            model.set_adapter(["adversarial_lora"])
            loss.loudness = forward_with_clipped_logit(model, forget_eval_batch)
            model.set_adapter([])
            loss.forget = forward(model, forget_eval_batch)
            model.set_adapter([])
            loss.retain = forward(model, retain_eval_batch)
            model.set_adapter(["adversarial_lora"])
            loss.adv_forget = forward(model, forget_eval_batch)
            model.set_adapter(["adversarial_lora"])
            loss.adv_retain = forward(model, retain_eval_batch)

    # calculate and print stats
    stats = dict(
        loudness=loss.loudness.item(),
        forget=loss.forget.exp() - initial_forget_ppl,
        retain=loss.retain.exp() - initial_retain_ppl,
        adv_forget=loss.adv_forget.exp() - initial_forget_ppl,
        adv_retain=loss.adv_retain.exp() - initial_retain_ppl,
    )
    if i % 10 == 0:
        print("\n      " + "   ".join(f"{k:>10}" for k in stats.keys()))
    print(f"{i + 1:4d}  " + "   ".join(f"{v:10.2f}" for v in stats.values()))

print(f"time: {time.time() - start_time:.2f}s")

# %% save model
model_path = get_repo_root() / "models" / "no_lora_200_steps.pt"
pt.save(model.state_dict(), model_path)

# %% load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
state_dict = pt.load(model_path, weights_only=True)
remove_lora(state_dict)
# assert set(model.state_dict().keys()) == set(state_dict.keys())
model.load_state_dict(state_dict)
