# %%
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# add to python path __file__.parent.parent
sys.path.append(str(Path(__file__).parent.parent))

from utils.git_and_reproducibility import repo_root

# from utils.plots_and_stats import *

plt.style.use("default")  # Reset to default style


# %%
def create_model_comparison_plot_horizontal(
    datasets: List[List[Tuple[str, float, float]]],
    model_names: List[str],
    baselines: List[float],
) -> tuple[plt.Figure, list[plt.Axes]]:
    # Ensure we have matching numbers of datasets, model names, and baselines
    assert (
        len(datasets) == len(model_names) == len(baselines)
    ), "Number of datasets, model names, and baselines must match"

    # Create the plot with n subplots side by side
    n_plots = len(datasets)
    fig, axes = plt.subplots(1, n_plots, figsize=(8, 4))

    # Convert to array of axes if single plot
    if n_plots == 1:
        axes = [axes]

    # Use tab20 colormap which has 20 distinct colors
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in np.linspace(0, 1, 20)]

    # Function to create a single subplot
    def create_subplot(ax, data, model_name, baseline):
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=["study_name", "mean", "sem"])
        # Index rows from n-1 to 0, into a "pos" column
        df["pos"] = np.arange(len(df))[::-1]
        # Add a "color" column
        df["color"] = colors[: len(df)]
        # Drop rows where study_name is empty
        df = df[df["study_name"] != ""]

        # Plot bars
        ax.barh(
            df["pos"],
            df["mean"],
            xerr=df["sem"],
            height=1,
            capsize=3,
            color=df["color"],
        )

        # Add "no valid trials" text for zero-value entries with non-empty study names
        for idx, row in df.iterrows():
            if row["mean"] == 0 and row["sem"] == 0 and row["study_name"]:
                ax.text(
                    1,
                    row["pos"],
                    "no valid trials",
                    va="center",
                    ha="left",
                    color="black",
                )

        # Update yticks for reversed order
        ax.set_yticks(df["pos"])
        ax.set_yticklabels(df["study_name"])
        ax.set_xlabel("Forget loss")
        ax.set_title(model_name)

        # Start x-axis at 0
        ax.set_xlim(0, None)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add baseline
        ax.axvline(x=baseline, color="black", linestyle="--", alpha=0.3)

    # Create all subplots with their respective data and baselines
    for ax, dataset, model_name, baseline in zip(
        axes, datasets, model_names, baselines
    ):
        create_subplot(ax, dataset, model_name, baseline)

    # Adjust layout
    plt.tight_layout()

    return fig, axes


def create_model_comparison_plot_vertical(
    datasets: List[List[Tuple[str, float, float]]],
    model_names: List[str],
    baselines: List[float],
) -> tuple[plt.Figure, list[plt.Axes]]:
    # Ensure we have matching numbers of datasets, model names, and baselines
    assert (
        len(datasets) == len(model_names) == len(baselines)
    ), "Number of datasets, model names, and baselines must match"

    # Create the plot with n subplots stacked vertically
    n_plots = len(datasets)
    fig, axes = plt.subplots(n_plots, 1, figsize=(4.5, 9))

    # Convert to array of axes if single plot
    if n_plots == 1:
        axes = [axes]

    # Use tab20 colormap which has 20 distinct colors
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in np.linspace(0, 1, 20)]

    # Function to create a single subplot
    def create_subplot(ax, data, model_name, baseline):
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=["study_name", "mean", "sem"])
        # Index rows from n-1 to 0, into a "pos" column
        df["pos"] = np.arange(len(df))
        # Add a "color" column
        df["color"] = colors[: len(df)]
        # Create positions for bars
        df = df[df["study_name"] != ""]  # Drop empty study names

        # Plot bars
        ax.bar(
            df["pos"],
            df["mean"],
            yerr=df["sem"],
            width=1,
            capsize=3,
            color=df["color"],
        )

        # Add "no valid trials" text for zero-value entries with non-empty study names
        for idx, row in df.iterrows():
            if row["mean"] == 0 and row["sem"] == 0 and row["study_name"]:
                ax.text(
                    row["pos"],
                    1,
                    "no valid trials",
                    va="bottom",
                    ha="center",
                    rotation=90,
                    color="black",
                )

        # Update xticks
        ax.set_xticks(df["pos"])
        ax.set_xticklabels(df["study_name"], rotation=45, ha="right")
        ax.set_ylabel("Forget loss")
        ax.set_title(model_name)

        # Start y-axis at 0
        ax.set_ylim(0, None)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add baseline
        ax.axhline(y=baseline, color="black", linestyle="--", alpha=0.3)

    # Create all subplots with their respective data and baselines
    for ax, dataset, model_name, baseline in zip(
        axes[::-1], datasets, model_names, baselines
    ):
        model_name = model_name.replace("\n", " - ")
        create_subplot(ax, dataset, model_name, baseline)

    # Adjust layout
    plt.tight_layout()

    return fig, axes


# - query_key_value had no complete trials!
# - retaining_rate near bottom edge
# - (stats made from last 30, where 500 trials total)
# - note: this uses local normalization
# configs/pythia_target_modules.yaml
pythia_python = [
    ("down_proj", 5.24, 0.08),  # dense_4h_to_h
    ("gate_proj", 10.51, 0.58),  # dense_h_to_4h
    ("", 0, 0),  # no up_proj
    ("o_proj", 5.86, 0.19),  # dense
    ("q_k_v_proj", 0, 0),  # query_key_value
    ("", 0, 0),
    ("", 0, 0),
    ("", 0, 0),
    ("gate+down", 13.10, 0.66),  # mlp
    ("gate+down+o", 14.20, 0.39),  # all_but_query_key_value
    ("", 0, 0),
    ("", 0, 0),
    # ("", 0, 0),
    ("all_linear", 4.21, 0.05),
]

# 240/120, 250 trials, smol, python, last 30 trials
# configs/smol_target_modules3.yaml
smol_python = [
    ("down_proj", 2.74, 0.03),
    ("gate_proj", 8.93, 0.26),
    ("up_proj", 4.83, 0.16),
    ("o_proj", 3.97, 0.20),
    ("q_proj", 2.71, 0.05),
    ("k_proj", 2.23, 0.01),
    ("v_proj", 7.93, 0.34),
    ("", 0, 0),
    ("gate+v", 10.07, 0.46),
    ("gate+v+up", 9.58, 0.37),
    ("gate+v+up+o", 9.30, 0.36),
    ("gate+v+up+o+q", 9.20, 0.34),
    # ("", 0, 0),
    ("all_linear", 7.06, 0.32),
]

# 240/120, 250 trials, smol, cruelty, 0.1 retain_loss_budget
# configs/smol_target_modules_cruelty.yaml
# ...

# Create and show the plot
fig, axes = create_model_comparison_plot_vertical(
# fig, axes = create_model_comparison_plot_horizontal(
    [pythia_python, smol_python, smol_python],  # Example with 3 plots using same data
    ["Pythia-14M\npython", "SmolLM-135M\npython", "SmolLM-135M\ncruelty"],
    [3.63, 2.11, 2.682],
)
plt.show()

plot_path = repo_root() / "paper" / "plots" / "target_modules.pdf"
fig.savefig(plot_path)
# %%
