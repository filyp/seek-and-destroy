# %%
import os
import re
from copy import deepcopy

import optuna
from plotly.subplots import make_subplots

from utils.git_and_reproducibility import *
from utils.model_operations import *
from utils.plots_and_stats import *

# get the latest study
storage = get_storage()
# storage = f"sqlite:///{repo_root() / 'win.sqlite3'}"

# # %%
# study_summaries = optuna.study.get_all_study_summaries(storage)
# sorted_studies = sorted(study_summaries, key=lambda s: s.datetime_start)

# # %%
# latest_study = sorted_studies[-1]
# study = optuna.load_study(study_name=latest_study.study_name, storage=storage)
# print(study.study_name)

# %%
study_names = [
    "480,180,python,f_momentum,unl_normalization,adv_decay,better_ranges3",
    "480,180,python,f_momentum,unl_normalization,adv_decay,better_ranges2",
    "480,180,python,f_momentum,unl_normalization,adv_decay,better_ranges",
]
studies = [
    optuna.load_study(study_name=study_name, storage=storage)
    for study_name in study_names
]
# %%

stacked_slice_plot(studies)

# %%
stacked_history_and_importance_plots(studies)
# %%
