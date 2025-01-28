# %%
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from utils.plots_and_stats import *

plt.style.use("default")  # Reset to default style

# %%


def create_model_comparison_plot(
    datasets: List[List[Tuple[str, float, float]]],
    model_names: List[str],
    baselines: List[float],
) -> tuple[plt.Figure, list[plt.Axes]]:
    # Ensure we have matching numbers of datasets, model names, and baselines
    assert len(datasets) == len(model_names) == len(baselines), \
        "Number of datasets, model names, and baselines must match"
    
    # Create the plot with n subplots side by side
    n_plots = len(datasets)
    fig, axes = plt.subplots(1, n_plots, figsize=(10, 5))
    
    # Convert to array of axes if single plot
    if n_plots == 1:
        axes = [axes]
    
    # Get default colors from matplotlib's color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Function to create a single subplot
    def create_subplot(ax, data, model_name, baseline):
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=["study_name", "mean", "sem"])
        # Index rows from n-1 to 0, into a "pos" column
        df["pos"] = np.arange(len(df))[::-1]
        # Drop rows where study_name is empty
        df = df[df["study_name"] != ""]
        
        ax.barh(
            df["pos"],
            df["mean"],
            xerr=df["sem"],
            height=1,
            capsize=3,
            color=colors[:len(df)],
        )

        # Update yticks for reversed order
        ax.set_yticks(df["pos"])
        ax.set_yticklabels(df["study_name"])
        ax.set_xlabel("Accuracy")
        ax.set_title(model_name)

        # Start x-axis at 0
        ax.set_xlim(0, None)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add baseline
        ax.axvline(x=baseline, color="black", linestyle="--", alpha=0.5)
    
    # Create all subplots with their respective data and baselines
    for ax, dataset, model_name, baseline in zip(axes, datasets, model_names, baselines):
        create_subplot(ax, dataset, model_name, baseline)

    # Adjust layout
    plt.tight_layout()

    return fig, axes


pythia_python = []

# Create data from the table
smol_python = [
    ("up_proj", 4.83, 0.16),
    ("gate_proj", 8.93, 0.26),
    ("down_proj", 2.74, 0.03),
    ("q_proj", 2.71, 0.05),
    ("k_proj", 2.23, 0.01),
    ("v_proj", 7.93, 0.34),
    ("o_proj", 3.97, 0.20),
    ("", 0, 0),
    ("gate_v", 10.07, 0.46),
    ("gate_v_up", 9.58, 0.37),
    ("gate_v_up_o", 9.30, 0.36),
    ("gate_v_up_o_q", 9.20, 0.34),
    ("", 0, 0),
    ("all_linear", 7.06, 0.32),
]

# Create and show the plot
fig, axes = create_model_comparison_plot(
    [smol_python, smol_python, smol_python],  # Example with 3 plots using same data
    ["SmolLM-135M\nPython", "SmolLM-135M\nJava", "SmolLM-135M\nRust"],
    [2.11, 2.11, 2.11]
)
plt.show()

# %%
