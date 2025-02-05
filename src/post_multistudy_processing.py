# %%
from types import SimpleNamespace

import optuna
import yaml

from utils.git_and_reproducibility import *
from utils.model_operations import *
from utils.plots_and_stats import *
from utils.training import (
    get_stats_from_last_n_trials,
    make_sure_optimal_values_are_not_near_range_edges,
)

# %%

db_url = json.load(open(repo_root() / "secret.json"))["db_url"]
storage = get_storage(db_url)
# storage = get_storage()

# config_path = repo_root() / "configs" / "pythia_python.yaml"
# config_path = repo_root() / "configs" / "smol_cruelty.yaml"
# config_path = repo_root() / "configs" / "smol_cruelty3.yaml"
# config_path = repo_root() / "configs" / "smol_target_modules3.yaml"
# config_path = repo_root() / "configs" / "smol_target_modules_cruelty.yaml"
# config_path = repo_root() / "configs" / "pythia_normalization_test.yaml"
# config_path = repo_root() / "configs" / "pythia_target_modules.yaml"
# config_path = repo_root() / "configs" / "ablations_and_loss,smol,python.yaml"


# config_path = repo_root() / "configs" / "ablations_and_loss2,pythia,python.yaml"
# config_path = repo_root() / "configs" / "ablations_and_loss2,pythia,pile-bio.yaml"
# config_path = repo_root() / "configs" / "ablations_and_loss2,smol,python.yaml"
config_path = repo_root() / "configs" / "ablations_and_loss2,smol,pile-bio.yaml"
# config_path = repo_root() / "configs" / "ablations_and_loss2,llama32,python.yaml"
# config_path = repo_root() / "configs" / "ablations_and_loss2,llama32,pile-bio.yaml"

# study_summaries = optuna.study.get_all_study_summaries(storage)
# sorted_studies = sorted(study_summaries, key=lambda s: s.datetime_start)

# %% get the studies
# note: trials loading takes some time, and also DB usage, so we cache it
# load YAML configuration
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

multistudy_name = Path(config_path).stem

config = SimpleNamespace(**full_config["general_config"])
relearn_config = SimpleNamespace(**full_config["relearn_config"])

studies = []
all_trials = []
for variant_name in full_config["variants"]:
    study_name = (
        f"{config.unlearn_steps},{relearn_config.relearn_steps},"
        f"{config.forget_set_name}"
        f"|{multistudy_name}|{variant_name}"
    )
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        trials = study.get_trials()
        if any(t.state == optuna.trial.TrialState.COMPLETE for t in trials):
            print(study_name, len(trials))
            studies.append(study)
            all_trials.append(trials)
        else:
            print(f"Study {study_name} has no complete trials!")
    except KeyError:
        print(f"Study {study_name} not found")


# %% slice plot
plot = stacked_slice_plot(studies, all_trials)
save_img(plot, f"{multistudy_name}_slice")
plot

# %% history and importance plots
# plot param importances takes quite long
plot = stacked_history_and_importance_plots(studies, all_trials)
save_img(plot, f"{multistudy_name}_history_and_importance")
plot

# %% check if optimal values are near range edges
for study in studies:
    make_sure_optimal_values_are_not_near_range_edges(study)

# %% get stats for the last N trials
markdown_table = """\
| last n trials<br>mean±sem | study_name | notes |
| ------------------- | ---------- | ----- |"""
python_results = ""
for study, trials in zip(studies, all_trials):
    markdown_line, last_n_mean, last_n_sem = get_stats_from_last_n_trials(study, trials, n=20)
    markdown_table += f"\n{markdown_line}"

    pure_name = study.study_name.split("|")[-1]
    python_results += f'\t("{pure_name}", {last_n_mean}, {last_n_sem}),\n'
print(markdown_table)
print(python_results)

# # %%
# # trimming trials, by creating a new study
# new_study = optuna.create_study(
#     study_name=study.study_name + "new",
#     storage=storage,
#     load_if_exists=False,
#     direction="maximize",
# )
# new_study.set_metric_names(["forget_loss"])
# for k, v in study.user_attrs.items():
#     print(k, v)
#     new_study.set_user_attr(k, v)
# trials = sorted(study.trials, key=lambda t: t.number)[:250]
# print(trials)
# for trial in trials:
#     new_study.add_trial(trial)

# %%
