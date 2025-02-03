# %%
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml

# add to python path __file__.parent.parent
sys.path.append(str(Path(__file__).parent.parent))
from utils.git_and_reproducibility import *
from utils.git_and_reproducibility import repo_root
from utils.model_operations import *
from utils.plots_and_stats import *
from utils.training import (
    get_stats_from_last_n_trials,
    make_sure_optimal_values_are_not_near_range_edges,
)

plt.style.use("default")  # Reset to default style

db_url = json.load(open(repo_root() / "secret.json"))["db_url"]
storage = get_storage(db_url)

# %% get the studies

# config_path = repo_root() / "configs" / "ablations_and_loss,smol,python.yaml"
# config_path = repo_root() / "configs" / "ablations_and_loss,smol,pile-bio.yaml"
config_path = repo_root() / "configs" / "ablations_and_loss,llama32,python.yaml"
# config_path = repo_root() / "configs" / "ablations_and_loss,llama32,pile-bio.yaml"
# config_path = repo_root() / "configs" / "ablations_and_loss,pythia,python.yaml"
# config_path = repo_root() / "configs" / "ablations_and_loss,pythia,pile-bio.yaml"

config_to_study_stats = dict()

# %%

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


# # %% check if optimal values are near range edges
# for study in studies:
#     make_sure_optimal_values_are_not_near_range_edges(study)

# # %% get stats for the last N trials
study_stats = []
for study, trials in zip(studies, all_trials):
    markdown_line, last_n_mean, last_n_sem = get_stats_from_last_n_trials(
        study, trials, n=20
    )
    pure_name = study.study_name.split("|")[-1]
    study_stats.append((pure_name, last_n_mean, last_n_sem))

# %%
config_to_study_stats[config_path.stem] = study_stats

# %%


data = [
    dict(
        study_stats=list(config_to_study_stats.values())[0],
        title=list(config_to_study_stats.keys())[0],
        baseline=0,
        y_min=0,
    ),
]


# name_mapping = dict(
#     SIU="SIU",
#     no_r_momentum="- retain momentum",
#     no_adv_decay="- adversary decay",
#     no_masking="- masking",
#     no_adversary="- adversary",
#     SIU_repE_retain="+ repr. eng. retain loss",
#     SIU_discard_growing_weights="+ only shrink weights",
#     SIU_f_momentum="+ forget momentum",
#     SIU_adv_update="+ adversary update",
#     neg_entropy="neg entropy loss",
#     neg_cross_entropy="neg cross entropy loss",
#     SIU_lora="+ adversary is LoRA",
#     circuit_breakers_no_lora2="circuit breakers",
#     circuit_breakers="circuit breakers w/ LoRA",
# )


def create_model_comparison_plot_horizontal(
    data_list: List[dict],
) -> tuple[plt.Figure, list[plt.Axes]]:

    # Create the plot with n subplots side by side
    n_plots = len(data_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(8, 4))

    # Convert to array of axes if single plot
    if n_plots == 1:
        axes = [axes]

    # Use default color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Function to create a single subplot
    def create_subplot(ax, data, title, baseline, y_min, show_labels=True):
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=["study_name", "mean", "sem"])
        # Index rows from n-1 to 0, into a "pos" column
        df["pos"] = np.arange(len(df))[::-1]
        # Add a "color" column
        df["color"] = colors[: len(df)]
        # Drop rows where study_name is empty
        df = df[df["study_name"] != ""]

        # insert a blank slot at index 3, by shifting pos
        df.loc[:2, "pos"] += 1

        # Add "no valid trials" text for zero-value entries with non-empty study names
        for idx, row in df.iterrows():
            if row["mean"] == 0 and row["sem"] == 0 and row["study_name"]:
                ax.text(
                    # (baseline + 3 * y_min) / 4,
                    y_min,
                    row["pos"],
                    "no valid trials",
                    va="center",
                    ha="left",
                    color="black",
                )

        # Plot bars
        ax.barh(
            df["pos"],
            df["mean"],
            xerr=df["sem"],
            height=1,
            capsize=3,
            color=df["color"],
        )

        # Update yticks for reversed order
        ax.set_yticks(df["pos"])
        if show_labels:
            # study_names = [name_mapping[name] for name in df["study_name"]]
            study_names = df["study_name"]
            ax.set_yticklabels(study_names)
        else:
            ax.set_yticklabels([])
            # Verify that study names match the first plot
            first_plot_names = [triple[0] for triple in data_list[0]["study_stats"]]
            current_names = [d[0] for d in data]
            for name1, name2 in zip(first_plot_names, current_names):
                assert name1 == name2
                # todo maybe allow later plots to be blank

        ax.set_xlabel("Forget loss")
        ax.set_title(title)

        # Start x-axis at 0
        ax.set_xlim(y_min, None)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add baseline
        ax.axvline(x=baseline, color="black", linestyle="--", alpha=0.3)

    # Create all subplots with their respective data and baselines
    for i, (ax, data) in enumerate(zip(axes, data_list)):
        dataset = data["study_stats"]
        title = data["title"]
        baseline = data["baseline"]
        y_min = data["y_min"]
        create_subplot(ax, dataset, title, baseline, y_min, show_labels=(i == 0))

    # Adjust layout
    plt.tight_layout()

    return fig, axes


# Create and show the plot
fig, axes = create_model_comparison_plot_horizontal(data)
plt.show()

# %%

# plot_path = repo_root() / "paper" / "plots" / "main_comparison.pdf"
# fig.savefig(plot_path)
# # %%
