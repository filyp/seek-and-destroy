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
multistudy_to_method_stats = dict()
multistudy_names = [
    "pythia,python",
    "pythia,pile-bio",
    "smol,python",
    "smol,pile-bio",
    "llama32,python",
    "llama32,pile-bio",
]
for multistudy_name in multistudy_names:
    multistudy_to_method_stats[multistudy_name] = dict()

    # load YAML configuration
    config_path = (
        repo_root() / "configs" / f"ablations_and_loss2,{multistudy_name}.yaml"
    )
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    config = SimpleNamespace(**full_config["general_config"])
    relearn_config = SimpleNamespace(**full_config["relearn_config"])

    for variant_name in full_config["variants"]:
        study_name = (
            f"{config.unlearn_steps},{relearn_config.relearn_steps},"
            f"{config.forget_set_name}"
            f"|{Path(config_path).stem}|{variant_name}"
        )

        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except KeyError:
            # print(f"Study {study_name} not found")
            continue

        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            # print(f"Study {study_name} has no complete trials!")
            continue

        # get stats for the last N trials
        trials = study.get_trials()
        markdown_line, last_n_mean, last_n_sem = get_stats_from_last_n_trials(
            study, trials, n=20
        )
        multistudy_to_method_stats[multistudy_name][variant_name] = (
            last_n_mean,
            last_n_sem,
        )

        # check if optimal values are near range edges
        make_sure_optimal_values_are_not_near_range_edges(study)


# %%
titles_dict = {
    "neg_cross_entropy_loss": "neg cross entropy loss",
    "neg_entropy_loss": "neg entropy loss",
    "logit_loss": "logit loss",
    "no_masking": "no masking",
    "no_r_momentum": "no r momentum",
    "no_adversary": "no meta learning",
    "no_adv_decay": "no adversary decay",
}
positions_dict = {
    "neg_cross_entropy_loss": 7,
    "neg_entropy_loss": 6,
    "logit_loss": 5,
    "no_masking": 3,
    "no_r_momentum": 2,
    "no_adversary": 1,
    "no_adv_decay": 0,
}

# Create the plot with n subplots side by side
fig, axes = plt.subplots(3, 2, figsize=(8, 12))


# Create a color mapping for methods
# Use default color cycle
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_method_names = list(titles_dict.keys())
method_to_color = {name: color for name, color in zip(_method_names, colors)}
method_to_color

# Define data for plotting
data = []  # Initialize data list

for n, (multistudy_name, method_stats) in enumerate(multistudy_to_method_stats.items()):
    ax = axes[n // 2, n % 2]

    ax.barh(
        [positions_dict[name] for name in method_stats.keys()],
        [mean for mean, sem in method_stats.values()],
        xerr=[sem for mean, sem in method_stats.values()],
        height=1,
        capsize=3,
        color=[method_to_color[name] for name in method_stats.keys()],
    )

    # Update yticks for reversed order
    ax.set_yticks([positions_dict[name] for name in method_stats.keys()])
    if n % 2 == 0:
        ax.set_yticklabels([titles_dict[name] for name in method_stats.keys()])
    else:
        ax.set_yticklabels([])

    ax.set_xlabel("Forget loss")
    ax.set_title(multistudy_name)

    # Calculate the minimum and maximum mean values for the xlim
    min_bar = min(mean for mean, sem in method_stats.values())
    max_bar = max(mean for mean, sem in method_stats.values())
    margin = (max_bar - min_bar) / 3  # Calculate the margin
    ax.set_xlim(min_bar - margin, max_bar + margin)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # # Add baseline
    # ax.axvline(x=baseline, color="black", linestyle="--", alpha=0.3)

plt.tight_layout()

# Create and show the plot
# plt.show()  # Ensure the plot is displayed

# %%

plot_path = repo_root() / "paper" / "plots" / "ablations_and_loss.pdf"
fig.savefig(plot_path)
