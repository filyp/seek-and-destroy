# %%
from IPython import get_ipython

# automatically reload all modules
ipython = get_ipython()
if ipython is not None:  # Only runs in IPython environment
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

from study_runner import *

# %%

config.__dict__.update(
    # target_modules=["dense_4h_to_h"],
    target_modules=["dense_h_to_4h"],
    # target_modules=["dense"],
    # target_modules=["dense_h_to_4h", "dense"],
    # target_modules=["dense_h_to_4h", "dense_4h_to_h"],
    # ! Training constants
    unlearn_steps=960,
)

Path("debug.txt").write_text("")  # clear data in debug.txt

set_seeds(42)
trial = MockTrial(
    # **params,
    adv_lr=1e-3,
    clip_at=0,
    retain_momentum_decay=0.9,
    forget_momentum_decay=0.9,
    fork_every_n_loops=12,
    retaining_rate=1e-3,
    unlearning_rate=1e-3 * 15,
    lora_rank=10,
    lora_amount=3,
    adv_decay=0.99,
)
model = unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
)

relearn_config.__dict__.update(relearn_steps=240)
forget_losses = relearn(model, relearn_config, retain_val_batches, forget_val_batches)

# %%
