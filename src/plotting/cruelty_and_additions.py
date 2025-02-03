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
    circuit_breakers="circuit breakers w/ LoRA",
    TAR="TAR",
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
        
        # if mean is 0, make it baseline
        df.loc[df["mean"] == 0, "mean"] = y_min

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
# - (stats made from last 20, where 600 trials total)
# configs/pythia_python.yaml
pythia_python = [
	("SIU", 11.302746677398682, 0.9497092536865647),
    ("", 0, 0),
	("no_masking", 4.568444263935089, 0.13803239785316707),
	("no_r_momentum", 9.303237581253063, 0.5641017899212439),
	("no_adversary", 4.960049247741699, 0.1585683651618381),
	("no_adv_decay", 6.130108523368835, 0.0807836960087648),
    ("", 0, 0),
	("SIU_repE_retain", 10.48015248775482, 0.8821893503325897),
	("SIU_discard_growing_weights", 9.624690055847173, 0.805131975150987),
	("SIU_f_momentum", 12.176710772514344, 0.966813458061461),
	("SIU_adv_update", 8.651150655746466, 0.5897299821533768),
	("SIU_lora", 5.689364814758301, 0.03925986673657358),
    ("", 0, 0),
	("neg_entropy", 5.7411521673202515, 0.16596601964204197),
	("neg_cross_entropy", 6.559857630729676, 0.039417393403911184),
    ("", 0, 0),
	("circuit_breakers_no_lora2", 3.362711751461029, 0.0024293986434816813),
    ("circuit_breakers", 0, 0),
	("TAR", 4.0887900471687315, 0.05728236559810787),
]

# 240/120, 100 trials, smol, cruelty, last 20 trials
# configs/smol_cruelty.yaml
smol_cruelty = [
	("SIU", 2.882917904853822, 0.014464900941069923),
    ("", 0, 0),
	("no_masking", 2.722558611317686, 0.0014308666518598864),
	("no_r_momentum", 2.7780295968055713, 0.003445853332054254),
	("no_adversary", 2.930646777153016, 0.015212119188045192),
	("no_adv_decay", 2.876294970512391, 0.017921953530745247),
    ("", 0, 0),
	("SIU_repE_retain", 2.8485352873802183, 0.01692174053410757),
	("SIU_discard_growing_weights", 2.839431333541869, 0.016896582883511726),
	("SIU_f_momentum", 2.8836338758468614, 0.01338708066582641),
	("SIU_adv_update", 2.873016190528869, 0.014488199160083039),
	("SIU_lora", 2.8853751540184027, 0.012510010430689557),
    ("", 0, 0),
	("neg_entropy", 2.8860881447792064, 0.013295418607136959),
	("neg_cross_entropy", 3.3219134926795943, 0.06611723283540369),
    ("", 0, 0),
	("circuit_breakers_no_lora2", 2.747757804393768, 0.003496623197401452),
	("circuit_breakers", 2.665582692623138, 7.621886663560849e-05),
	("TAR", 2.726053619384765, 0.002887879791405018),

]

# 240/240, 100 trials, smol, cruelty, last 20 trials
# configs/smol_cruelty3.yaml
# it has a much more aggressive relearning than smol_cruelty
smol_cruelty3 = [
	("SIU", 2.460569596290587, 0.002824004527368837),
    ("", 0, 0),
	("no_masking", 2.4219161152839654, 0.00012044505171894194),
	("no_r_momentum", 2.4281624913215625, 0.0005217204324291974),
	("no_adversary", 2.4600162625312807, 0.002734088986453469),
	("no_adv_decay", 2.4825207471847532, 0.0037945551420759797),
    ("", 0, 0),
	("SIU_repE_retain", 2.4557696938514715, 0.003714679692389006),
	("SIU_discard_growing_weights", 2.4590118646621706, 0.003190898526513529),
	("SIU_f_momentum", 2.46364803314209, 0.0011249812809410052),
	("SIU_adv_update", 2.455708491802216, 0.0031911694528545254),
	("SIU_lora", 2.4467007517814636, 0.0020760997079391757),
    ("", 0, 0),
	("neg_entropy", 2.4415155291557302, 0.0022896276624128284),
	("neg_cross_entropy", 2.6198157191276548, 0.014454618108392098),
    ("", 0, 0),
	("circuit_breakers_no_lora2", 2.4364689111709605, 0.0013721766556624578),
	("circuit_breakers", 2.4158615469932547, 2.063177649415145e-05),
	("TAR", 2.4225908398628233, 0.0002518591581188142),
]

# Create and show the plot
fig, axes = create_model_comparison_plot_horizontal(
    # fig, axes = create_model_comparison_plot_horizontal(
    [
        pythia_python,
        smol_cruelty,
        smol_cruelty3,
    ],  # Example with 3 plots using same data
    ["Pythia-14M\npython\n", "SmolLM-135M\ncruelty\nrelearning_rate=1e-4", "SmolLM-135M\ncruelty3\nrelearning_rate=5e-3"],
    baselines=[3.626, 2.682, 2.682],
    y_min=[0, 2.6, 2.4],
)
plt.show()

# %%

plot_path = repo_root() / "paper" / "plots" / "cruelty_and_additions.pdf"
fig.savefig(plot_path)
# %%
