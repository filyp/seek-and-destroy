# %%
from copy import deepcopy

import optuna
import optuna.visualization as vis

from utils.git_and_reproducibility import *
from utils.model_operations import *

# get the latest study
# storage = get_storage()
storage = f"sqlite:///{repo_root() / 'win.sqlite3'}"

# %%
study_summaries = optuna.study.get_all_study_summaries(storage)
sorted_studies = sorted(study_summaries, key=lambda s: s.datetime_start)
latest_study = sorted_studies[-1]
study = optuna.load_study(study_name=latest_study.study_name, storage=storage)
print(study.study_name)

# study_name = "python,200-50,more_steps_with_best_values"
# study = optuna.load_study(study_name=study_name, storage=storage)

# %%
layout = dict(
    template="plotly_white",
    font=dict(family="Times Roman", size=20),
    title={"text": study.study_name, "xanchor": "center", "x": 0.5, "y": 0.95},
    title_font_size=30,
)

# Create SVG strings directly in memory
slice_fig = vis.plot_slice(study, target_name="Final forget loss")
slice_fig.update_layout(**layout)
# # save slice_fig
# path = repo_root() / "paper" / "Figures" / f"{study.study_name}_slice.svg"
# slice_fig.write_image(path)

# %%
opt_history_fig = vis.plot_optimization_history(study)
opt_history_fig.update_layout(**layout, width=slice_fig.layout.width)
# # save opt_history_fig
# path = repo_root() / "paper" / "Figures" / f"{study.study_name}_opt_history.svg"
# opt_history_fig.write_image(path)

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
#         objective(MockTrial(best_params))
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
#     retain_val_batches.fresh_iterator(),
#     forget_val_batches.fresh_iterator(),
# )
