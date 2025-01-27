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

# config_path = repo_root() / "configs" / "pythia_ablation2.yaml"
config_path = repo_root() / "configs" / "smol_target_modules3.yaml"
# config_path = repo_root() / "configs" / "pythia_normalization_test.yaml"
# config_path = repo_root() / "configs" / "pythia_target_modules.yaml"

# study_summaries = optuna.study.get_all_study_summaries(storage)
# sorted_studies = sorted(study_summaries, key=lambda s: s.datetime_start)

# %% get the studies
# load YAML configuration
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

multistudy_name = Path(config_path).stem

config = SimpleNamespace(**full_config["general_config"])
relearn_config = SimpleNamespace(**full_config["relearn_config"])


studies = []
for variant_name in full_config["variants"]:
    study_name = (
        f"{config.unlearn_steps},{relearn_config.relearn_steps},"
        f"{config.forget_set_name}"
        f"|{multistudy_name}|{variant_name}"
    )
    print(study_name)
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            studies.append(study)
        else:
            print(f"Study {study_name} has no complete trials!")
    except KeyError:
        print(f"Study {study_name} not found")


# %% slice plot
plot = stacked_slice_plot(studies)
save_img(plot, f"{multistudy_name}_slice")
plot


# %% history and importance plots
plot = stacked_history_and_importance_plots(studies)
save_img(plot, f"{multistudy_name}_history_and_importance")
plot

# %% check if optimal values are near range edges
for study in studies:
    make_sure_optimal_values_are_not_near_range_edges(study)

# %% get stats for the last 100 trials
markdown_table = """\
| last n trials<br>mean±sem | max   | study_name | notes |
| ------------------- | ----- | ---------- | ----- |"""
for study in studies:
    markdown_line, _, _ = get_stats_from_last_n_trials(study, n=30)
    markdown_table += f"\n{markdown_line}"
print(markdown_table)

# %%
