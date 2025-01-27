# %%
from IPython import get_ipython

# automatically reload all modules
ipython = get_ipython()
if ipython is not None:  # Only runs in IPython environment
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %%
# necessary for determinism:
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"  # less mem but slower

import logging
from copy import deepcopy
from types import SimpleNamespace

import torch as pt
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_methods.surgical_tar import surgical_tar
from unlearning_methods.surgical_tar_lora import surgical_tar_lora
from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.loss_fns import loss_fns
from utils.model_operations import relearn
from utils.training import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# %%
# load YAML configuration
# config_path = repo_root() / "configs" / "pythia_ablation.yaml"
config_path = repo_root() / "configs" / "smol_target_modules2.yaml"
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

hyperparam_ranges = full_config["hyperparams"]
config = full_config["general_config"]

config = SimpleNamespace(**config)
relearn_config = SimpleNamespace(**full_config["relearn_config"])

# custom
# config.train_adversary = False
config.target_modules = ["up_proj", "down_proj", "gate_proj", "o_proj"]

# %%

print(f"{hyperparam_ranges=}")

pt.set_default_device("cuda")

# load datasets
set_seeds(42)
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
retain_set = dataset_loaders[config.retain_set_name](tokenizer)
forget_set = dataset_loaders[config.forget_set_name](tokenizer)
retain_batches = CachedBatches(retain_set["train"], config.batch_size)
forget_batches = CachedBatches(forget_set["train"], config.batch_size)
retain_val_batches = CachedBatches(retain_set["validation"], config.batch_size)
forget_val_batches = CachedBatches(forget_set["validation"], config.batch_size)
r_eval = next(iter(retain_val_batches))
f_eval = next(iter(forget_val_batches))

allowed_f_loss = eval_(
    AutoModelForCausalLM.from_pretrained(config.model_id), f_eval, r_eval
)["retain_loss"]

unlearning_func = dict(
    surgical_tar_lora=surgical_tar_lora,
    surgical_tar=surgical_tar,
)[config.method_name]


# construct hyperparams
hyperparams = {
    hp_name: (low + high) / 2 for hp_name, (low, high, log) in hyperparam_ranges.items()
}
hyperparams = SimpleNamespace(**hyperparams)

set_seeds(42)

# %%
h = hyperparams
h.fork_every_n_loops = int(h.fork_every_n_loops)

model = AutoModelForCausalLM.from_pretrained(config.model_id)
adversary = AutoModelForCausalLM.from_pretrained(config.model_id)
model.config.use_cache = False
adversary.config.use_cache = False

clip_at = h.additional_param if config.additional_param_name == "clip_at" else 0

# get params to intervene on
interven_params = [
    p
    for name, p in model.named_parameters()
    if any(f"{m}.weight" in name for m in config.target_modules)
]
adv_interven_params = [
    p
    for name, p in adversary.named_parameters()
    if any(f"{m}.weight" in name for m in config.target_modules)
]

# require grads only on intervened params
for p in model.parameters():
    p.requires_grad = id(p) in [id(p) for p in interven_params]
for p in adversary.parameters():
    p.requires_grad = id(p) in [id(p) for p in adv_interven_params]

for n, p in model.named_parameters():
    p.param_name = n

for p in interven_params:
    p.retain_momentum = pt.zeros_like(p.data)
    if config.additional_param_name == "forget_momentum_decay":
        p.forget_momentum = pt.zeros_like(p.data)

# ! unlearning loop
logging.info("step      base_f      base_r")
retain_iter = iter(retain_batches)
forget_iter = iter(forget_batches)

# ! unlearning loop
passes_per_loop = 4 + int(config.train_adversary)
assert config.unlearn_steps % passes_per_loop == 0

# %%
total_interven_numel = sum(p.numel() for p in interven_params)
# %%
total_interven_numel ** 0.5
# %%

model.train()
adversary.train()

# ! retain pass
model.zero_grad(set_to_none=True)
r_input_ids = next(retain_iter)
output = model(r_input_ids)
loss = cross_entropy_loss(output, r_input_ids)
loss.backward()
for p in interven_params:
    # ! update disruption scores
    p.retain_momentum *= h.retain_momentum_decay
    p.retain_momentum += p.grad * (1 - h.retain_momentum_decay)
    # ! retain update
    p.data -= h.retaining_rate * p.retain_momentum
model.zero_grad(set_to_none=True)

# if (loop_num % h.fork_every_n_loops == 0) or (not config.train_adversary):
#     # if not training adversary, make sure it's always the same as base model
#     adversary.load_state_dict(model.state_dict())

# for _ in range(adv_per_orig_step):
# ! relearn the adversary
# todo actually could store only the intervened params for adversary
adversary.zero_grad(set_to_none=True)
f_input_ids = next(forget_iter)
output = adversary(f_input_ids)
if config.train_adversary:
    loss = cross_entropy_loss(output, f_input_ids)
    loss.backward(retain_graph=True)
    for p, adv_p in zip(interven_params, adv_interven_params):
        # apply adversary update
        adv_p.data -= h.adv_lr * adv_p.grad
        # decay adversary into base model
        adv_p.data *= h.adv_decay
        adv_p.data += p.data * (1 - h.adv_decay)
else:
    for p, adv_p in zip(interven_params, adv_interven_params):
        assert (p.data == adv_p.data).all()

# ! get unlearning grads loss from adversary
# reuse the computation graph from previous block
adversary.zero_grad(set_to_none=True)
loss_fn = loss_fns[config.unlearning_loss_fn]
loss = loss_fn(output, f_input_ids, clip_at)
loss.backward()

# %%
# ! unlearning step with masking
for p, adv_p in zip(interven_params, adv_interven_params):
    update = adv_p.grad

    if config.additional_param_name == "forget_momentum_decay":
        p.forget_momentum *= h.additional_param
        p.forget_momentum += update * (1 - h.additional_param)
        update = p.forget_momentum

    if config.use_masking:
        mask = p.retain_momentum.sign() == update.sign()
        update *= mask

    if config.additional_param_name == "discard_growing_weights":
        mask2 = p.data.sign() != update.sign()
        update[mask2] *= h.additional_param

    adv_p.grad = update
    
global_norm = sum(adv_p.grad.norm() ** 2 for adv_p in adv_interven_params) ** 0.5

for p, adv_p in zip(interven_params, adv_interven_params):
    update = adv_p.grad

    # normalize
    if config.global_normalization:
        update *= total_interven_numel ** 0.5 / global_norm

    p.data -= h.unlearning_rate * update

    if config.additional_param_name == "adv_update":
        adv_p.data -= h.unlearning_rate * update * h.additional_param

# %%

model = unlearning_func(
    hyperparams,
    config,
    retain_batches,
    forget_batches,
    f_eval,
    r_eval,
    allowed_f_loss,
)

# %%

set_seeds(42)

relearn_config.relearn_lr = 1e-4
# relearn_config.relearn_lr = 0.5e-1
relearn_config.relearn_steps = 50

forget_losses = relearn(
    deepcopy(model),
    relearn_config,
    retain_val_batches,
    forget_val_batches,
)

# %%


# %%

for n, p in model.named_parameters():
    print(n, p.requires_grad)
