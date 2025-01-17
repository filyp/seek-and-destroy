# %%
from study_runner import *

# %%
config.__dict__.update(
    # target_modules=["dense_4h_to_h"],
    target_modules=["dense_h_to_4h"],
    # ! Training constants
    unlearn_steps=1000,
)
relearn_config = SimpleNamespace(
    relearn_steps=100,
    relearn_lr=3e-4,
    relearn_lora_conf=dict(target_modules="all-linear"),
)

# %%

set_seeds(42)
trial = MockTrial(
    # **params,
    retaining_rate=0e-4,
    disruption_score_decay=0.98,
    unlearning_lr=1e-3,
    adv_lr=1e-3,
    fork_every_n_steps=48,
    adv_per_orig_step=2,
)
model = unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
)

# %%
forget_losses = relearn(model, relearn_config, retain_val_batches, forget_val_batches)
print(config.circuit_names)
