# %%
import re

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

    dir_name = repo_root() / "results" / f"slice_layout"
    dir_name.mkdir(parents=True, exist_ok=True)
    file_name = dir_name / f"{study.study_name}.png"
    slice_fig.write_image(file_name)
    return slice_fig


# opt_history_fig = vis.plot_optimization_history(study)
# opt_history_fig.update_layout(**layout, width=slice_fig.layout.width)
# # # save opt_history_fig
# # path = repo_root() / "paper" / "Figures" / f"{study.study_name}_opt_history.svg"
# # opt_history_fig.write_image(path)


# def visualize_param(p, y, mask):
#     x = p.to_forget
#     c = mask
#     # plot a 2d scatter plot
#     # first flatten x and y and c and convert to cpy numpy
#     x = x.flatten().cpu().numpy()
#     y = y.flatten().cpu().numpy()
#     c = c.flatten().cpu().numpy()

#     plt.clf()

#     # draw grey axes lines
#     plt.axhline(0, color="grey")
#     plt.axvline(0, color="grey")

#     # draw the points
#     plt.scatter(x, y, c=c, s=1)

#     # label
#     plt.xlabel("to_forget")
#     plt.ylabel("disruption_score")

#     # center at 0, 0
#     xmax = max(abs(x))
#     ymax = max(abs(y))
#     plt.xlim(-xmax, xmax)
#     plt.ylim(-ymax, ymax)

#     # plt.ylim(0, 0.001)
#     # plt.show()
#     dir_name = repo_root() / "results" / f"param_toforget_vs_disruption"
#     dir_name.mkdir(parents=True, exist_ok=True)
#     file_name = dir_name / f"{p.param_name}.png"
#     plt.savefig(file_name)


# # 6x2 plot of to_forget vs disruption_score_pos and disruption_score_neg
# def layer_vs_pos_neg_sum_plot():
#     dir_ = repo_root() / "results" / "param_toforget_vs_disruption"
#     paths = dir_.glob("*.png")

#     # Sort the paths by layer number
#     def get_layer_num(path):
#         match = re.search(r"layers\.(\d+)", str(path))
#         return int(match.group(1)) if match else -1

#     # Sort paths by layer number
#     paths = sorted(paths, key=get_layer_num)

#     # Create 2x3 subplot
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))

#     # Plot each image
#     for i, path in enumerate(paths):
#         row = i // 3
#         col = i % 3
#         img = plt.imread(path)
#         axes[row, col].imshow(img)

#         # Extract layer number for title
#         layer_num = get_layer_num(path)
#         axes[row, col].set_title(f"Layer {layer_num}")
#         axes[row, col].axis("off")

#     plt.tight_layout()
#     plt.show()
