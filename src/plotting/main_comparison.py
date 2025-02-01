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
name_mapping = dict(
    SIU="SIU",
    no_r_momentum="- retain momentum",
    no_adv_decay="- adversary decay",
    no_masking="- masking",
    no_adversary="- adversary",
    SIU_repE_retain="+ repr. eng. retain loss",
    SIU_discard_growing_weights="+ only shrink weights",
    SIU_f_momentum="+ forget momentum",
    SIU_adv_update="+ adversary update",
    neg_entropy="neg entropy loss",
    neg_cross_entropy="neg cross entropy loss",
    SIU_lora="+ adversary is LoRA",
    circuit_breakers_no_lora2="circuit breakers",
)


def create_model_comparison_plot_horizontal(
    datasets: List[List[Tuple[str, float, float]]],
    model_names: List[str],
    baselines: List[float],
    y_min: List[float],
) -> tuple[plt.Figure, list[plt.Axes]]:
    # Ensure we have matching numbers of datasets, model names, and baselines
    assert (
        len(datasets) == len(model_names) == len(baselines) == len(y_min)
    ), "Number of datasets, model names, baselines, and y_min must match"

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
    def create_subplot(ax, data, model_name, baseline, y_min, show_labels=True):
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

        # Update yticks for reversed order
        ax.set_yticks(df["pos"])
        if show_labels:
            study_names = [name_mapping[name] for name in df["study_name"]]
            ax.set_yticklabels(study_names)
        else:
            ax.set_yticklabels([])
            # Verify that study names match the first plot
            first_plot_names = [d[0] for d in datasets[0]]
            current_names = [d[0] for d in data]
            for name1, name2 in zip(first_plot_names, current_names):
                assert name1 == name2
                # todo maybe allow later plots to be blank

        ax.set_xlabel("Forget loss")
        ax.set_title(model_name)

        # Start x-axis at 0
        ax.set_xlim(y_min, None)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add baseline
        ax.axvline(x=baseline, color="black", linestyle="--", alpha=0.3)

    # Create all subplots with their respective data and baselines
    for i, (ax, dataset, model_name, baseline, y_min) in enumerate(
        zip(axes, datasets, model_names, baselines, y_min)
    ):
        create_subplot(ax, dataset, model_name, baseline, y_min, show_labels=(i == 0))

    # Adjust layout
    plt.tight_layout()

    return fig, axes


# - circuit_breakers (with lora) had no complete trials!
# - retaining_rate near bottom edge for SIU and neg_entropy
# - adv_lr sometimes wants to be higher but it's already crazy high
# - (stats made from last 50, where 600 trials total)
# configs/pythia_python.yaml
pythia_python = [
    ("SIU", 12.029057579040527, 0.5307093972856548),
    ("", 0, 0),
    ("no_r_momentum", 8.196117925643923, 0.38244385056030655),
    ("no_adv_decay", 6.106322660446167, 0.05145514631107573),
    ("no_masking", 4.665606794357299, 0.08999768020566257),
    ("no_adversary", 4.820731468200684, 0.0900553940410563),
    ("", 0, 0),
    ("SIU_repE_retain", 9.868167924880984, 0.5643592509286416),
    ("SIU_discard_growing_weights", 10.06520764350891, 0.45754967130730717),
    ("SIU_f_momentum", 7.507715721130369, 0.3535885110755663),
    ("SIU_adv_update", 7.876785879135134, 0.35882352873382145),
    ("SIU_lora", 5.621265392303467, 0.031518234566582884),
    ("", 0, 0),
    ("neg_entropy", 5.755290613174439, 0.10519114993798162),
    ("neg_cross_entropy", 6.324386205673218, 0.054910998016978824),
    ("", 0, 0),
    ("circuit_breakers_no_lora2", 3.3626023578643798, 0.001576164304744805),
    # note that circuit breakers has no LoRA here, the one with lora has no complete trials
    # also note that CB is below initial loss line, because of relearning
]

# 240/120, 100 trials, smol, cruelty, last 50 trials
# configs/smol_cruelty.yaml
smol_cruelty = [
	("SIU", 2.8428026723861692, 0.010094514555167924),
    ("", 0, 0),
	("no_r_momentum", 2.7638086271286006, 0.003586928917786989),
	("no_adv_decay", 2.8605710840225225, 0.013383969747182952),
	("no_masking", 2.722558611317686, 0.0009049595295712889),
	("no_adversary", 2.8703684759140016, 0.013305645010619538),
    ("", 0, 0),
	("SIU_repE_retain", 2.8191870498657225, 0.010730855649932541),
	("SIU_discard_growing_weights", 2.842701416015625, 0.010646700957507917),
	("SIU_f_momentum", 2.8377113485336305, 0.010506982889495368),
	("SIU_adv_update", 2.8458503675460816, 0.010035388469802372),
	("SIU_lora", 2.8540265417099, 0.009953808890713182),
    ("", 0, 0),
	("neg_entropy", 2.845289521217347, 0.010058451789250644),
	("neg_cross_entropy", 3.256914553642273, 0.04405572860637923),
    ("", 0, 0),
	("circuit_breakers_no_lora2", 2.730552062988282, 0.004185927219279357),
    ("", 0, 0),
    # unknown yet
	# ("circuit_breakers", 2.665551424026489, 0.00012839538391034642),
]
#     ("down_proj", 2.74, 0.03),
#     ("gate_proj", 8.93, 0.26),
#     ("up_proj", 4.83, 0.16),
#     ("o_proj", 3.97, 0.20),
#     ("q_proj", 2.71, 0.05),
#     ("k_proj", 2.23, 0.01),
#     ("v_proj", 7.93, 0.34),
#     ("", 0, 0),
#     ("gate+v", 10.07, 0.46),
#     ("gate+v+up", 9.58, 0.37),
#     ("gate+v+up+o", 9.30, 0.36),
#     ("gate+v+up+o+q", 9.20, 0.34),
#     # ("", 0, 0),
#     ("all_linear", 7.06, 0.32),
# ]

# # 240/120, 250 trials, smol, cruelty, 0.1 retain_loss_budget, last 30 trials
# # configs/smol_target_modules_cruelty.yaml
# smol_cruelty = [
#     ("down_proj", 2.7491, 0.0003),
#     ("gate_proj", 2.9338, 0.0034),
#     ("up_proj", 2.8199, 0.0044),
#     ("o_proj", 2.9478, 0.0088),
#     ("q_proj", 2.7555, 0.0014),
#     ("k_proj", 2.7638, 0.0009),
#     ("v_proj", 2.8453, 0.0058),
#     ("", 0, 0),
#     ("gate_v", 2.8640, 0.0022),
#     ("gate_v_up", 2.8572, 0.0032),
#     ("gate_v_up_o", 2.8864, 0.0034),
#     ("gate_v_up_o_q", 2.8722, 0.0039),
#     ("all_linear", 2.8032, 0.0010),
# ]

# Create and show the plot
fig, axes = create_model_comparison_plot_horizontal(
    # fig, axes = create_model_comparison_plot_horizontal(
    [
        pythia_python,
        smol_cruelty,
        pythia_python,
    ],  # Example with 3 plots using same data
    ["Pythia-14M\npython", "SmolLM-135M\ncruelty", "todo"],
    baselines=[3.626, 2.682, 0],
    y_min=[0, 2.65, 0],
)
plt.show()

# %%

plot_path = repo_root() / "paper" / "plots" / "main_comparison.pdf"
fig.savefig(plot_path)
# %%
