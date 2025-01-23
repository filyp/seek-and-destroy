# %%
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

# %%
study_summaries = optuna.study.get_all_study_summaries(storage)
sorted_studies = sorted(study_summaries, key=lambda s: s.datetime_start)

# %%
latest_study = sorted_studies[-1]
study = optuna.load_study(study_name=latest_study.study_name, storage=storage)
# study_name = "python,200-50,more_steps_with_best_values"
# study = optuna.load_study(study_name=study_name, storage=storage)
print(study.study_name)

# %%

slice_fig = plot_slice_layout(study)
slice_fig

# %%

importances_fig = plot_opt_history_and_importances(study)
importances_fig

# %%
# # ! rerun the best trial, with more steps
# trials = study.get_trials()
# finished_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
# worst_to_best = sorted(finished_trials, key=lambda t: t.value)

# # config.unlearn_steps = 300
# best_params = deepcopy(worst_to_best[-1].params)
# # best_params["unlearn_lr"] *= 0.3
# # best_params["ret_lora_lr"] = 0.001
# # best_params["quantile"] = 0.007
# best_params
# worst_to_best[-1].value
# config.relearn_steps = 1000
# i = 0
# while i < 1:
#     try:
#         objective(MockTrial(**best_params))
#         i += 1
#     except optuna.TrialPruned:
#         pass
# # %%
# best_model_path = repo_root() / "models" / "best_model.pt"
# best_model = AutoModelForCausalLM.from_pretrained(config.model_id)
# best_model.load_state_dict(pt.load(best_model_path, weights_only=True))

# # %%
# config.relearn_steps = 1000
# forget_losses = relearn(
#     deepcopy(best_model),
#     config,
#     iter(retain_val_batches),
#     iter(forget_val_batches),
# )

# %%

