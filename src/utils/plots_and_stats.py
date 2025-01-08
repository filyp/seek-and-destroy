import shutil

import matplotlib.pyplot as plt
import optuna.visualization as vis

from utils.git_and_reproducibility import repo_root


def plot_slice_layout(study):
    layout = dict(
        template="plotly_white",
        font=dict(family="Times Roman", size=20),
        title={"text": study.study_name, "xanchor": "center", "x": 0.5, "y": 0.95},
        title_font_size=30,
    )

    slice_fig = vis.plot_slice(study, target_name="Final forget loss")
    slice_fig.update_layout(**layout)

    file_name = repo_root() / "results" / f"slice_layout.png"
    slice_fig.write_image(file_name)
    return slice_fig


# opt_history_fig = vis.plot_optimization_history(study)
# opt_history_fig.update_layout(**layout, width=slice_fig.layout.width)
# # # save opt_history_fig
# # path = repo_root() / "paper" / "Figures" / f"{study.study_name}_opt_history.svg"
# # opt_history_fig.write_image(path)


def visualize_param(param, mask, param_name):
    x = param.to_forget
    y = param.disruption_score
    c = mask
    # plot a 2d scatter plot
    # first flatten x and y and c and convert to cpy numpy
    x = x.flatten().cpu().numpy()
    y = y.flatten().cpu().numpy()
    c = c.flatten().cpu().numpy()

    plt.clf()

    # draw grey axes lines
    plt.axhline(0, color="grey")
    plt.axvline(0, color="grey")

    # draw the points
    plt.scatter(x, y, c=c, s=1)

    # label
    plt.xlabel("to_forget")
    plt.ylabel("disruption_score")

    # center at 0, 0
    xmax = max(abs(x))
    ymax = max(abs(y))
    plt.xlim(-xmax, xmax)
    plt.ylim(-ymax, ymax)

    # plt.ylim(0, 0.001)
    # plt.show()
    dir_name = repo_root() / "results" / f"param_toforget_vs_disruption"
    dir_name.mkdir(parents=True, exist_ok=True)
    file_name = dir_name / f"{param_name}.png"
    plt.savefig(file_name)
