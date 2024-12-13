"""usage: python .continue_old_run.py NUM_STEPS"""
import sys

import optuna

from adversarial_adapters import *
from utils.git_and_reproducibility import get_storage, is_repo_clean

storage = get_storage()

# make sure the study exists by trying to load it
try:
    optuna.load_study(study_name=study_name, storage=storage)
except KeyError:
    print(f"Study {study_name} not found in storage {storage}")
    sys.exit(1)

assert is_repo_clean()

num_trials = int(sys.argv[1])

study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction="maximize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=num_trials)
