# %%
import os
import re

import matplotlib.pyplot as plt
import optuna.visualization as vis
import plotly.graph_objects as go
from optuna.visualization._slice import _generate_slice_subplot, _get_slice_plot_info
from plotly.subplots import make_subplots

from utils.git_and_reproducibility import repo_root

common_layout = dict(
    # template="plotly_white",
    template="simple_white",
    # template="presentation",  # too big points
    # template="ggplot2",   # weird to look at
    font=dict(family="Times Roman", size=20),
    # title_font_size=30,
)


def save_img(fig, file_name):
    dir_name = repo_root() / "plots"
    dir_name.mkdir(parents=True, exist_ok=True)
    # fig.write_image(dir_name / f"{file_name}.svg")  # too big and sometimes distorted
    fig.write_image(dir_name / f"{file_name}.pdf")  # needed for paper
    fig.write_image(dir_name / f"{file_name}.png")  # needed for notes
    return dir_name / f"{file_name}.svg"


# def plot_slice_layout(study, dir_="plots/slice_layout"):
#     layout = common_layout | dict(
#         title={"text": study.study_name, "xanchor": "center", "x": 0.5, "y": 0.95},
#     )

#     slice_fig = vis.plot_slice(study, target_name="Final forget loss")
#     slice_fig.update_layout(**layout)

#     dir_name = repo_root() / dir_
#     dir_name.mkdir(parents=True, exist_ok=True)
#     slice_fig.write_image(dir_name / f"{study.study_name}.svg")
#     slice_fig.write_image(dir_name / f"{study.study_name}.pdf")
#     return slice_fig


def stacked_slice_plot(studies):
    study_names = [s.study_name for s in studies]

    param_set = set()
    for study in studies:
        param_set.update(study.best_params.keys())
    param_list = sorted(param_set)
    # move unlearning_rate to the beginning
    param_list.remove("unlearning_rate")
    param_list.insert(0, "unlearning_rate")


    # Calculate overall y-axis range across all studies
    _values = [t.values[0] for s in studies for t in s.trials if t.values is not None]
    y_min = min(_values)
    y_max = max(_values)

    layout = go.Layout(common_layout)
    figure = make_subplots(
        rows=len(study_names),
        cols=len(param_list),
        shared_yaxes=True,
        # shared_xaxes="all",
        vertical_spacing=0.4 / len(studies),
    )
    common_prefix = os.path.commonprefix(study_names)
    figure.update_layout(layout)
    figure.update_layout(title={"text": common_prefix, "xanchor": "center", "x": 0.5})
    
    showscale = True
    for i, study in enumerate(studies, start=1):
        info = _get_slice_plot_info(study, None, None, "Final forget loss")
        for subplot_info in info.subplots:
            trace = _generate_slice_subplot(subplot_info)
            trace[0].update(marker={"showscale": showscale})
            showscale = False  # only needs to be set once

            j = param_list.index(subplot_info.param_name) + 1
            for t in trace:
                figure.add_trace(t, row=i, col=j)

            # Share x axes only if not additional_param
            if "additional_param" in subplot_info.param_name:
                matches = None
                title_text = study.user_attrs["additional_param_name"]
            else:
                matches = "x" + str(j)
                title_text = subplot_info.param_name
            figure.update_xaxes(
                title_text=title_text,
                row=i,
                col=j,
                matches=matches,
                showticklabels=True,
            )
            # Set y-axis range for all subplots
            figure.update_yaxes(range=[y_min, y_max], row=i, col=j)

            if j == 1:
                subtitle = study.study_name[len(common_prefix) :]
                figure.update_yaxes(title_text=subtitle, row=i, col=j)
            if not subplot_info.is_numerical:
                figure.update_xaxes(
                    type="category",
                    categoryorder="array",
                    categoryarray=subplot_info.x_labels,
                    row=i,
                    col=j,
                )
            elif subplot_info.is_log:
                figure.update_xaxes(type="log", row=i, col=j)

    figure.update_layout(
        width=300 * len(param_list),
        height=300 * len(study_names),
    )
    return figure


def stacked_history_and_importance_plots(studies):
    study_names = [s.study_name for s in studies]
    common_prefix = os.path.commonprefix(study_names)

    # Calculate overall y-axis range
    _values = [t.values[0] for s in studies for t in s.trials if t.values is not None]
    y_min = min(_values)
    y_max = max(_values)

    figure = make_subplots(rows=len(studies), cols=2, horizontal_spacing=0.22)

    for i, study in enumerate(studies, start=1):
        for trace in vis.plot_optimization_history(study).data:
            figure.add_trace(trace, row=i, col=1)
            subtitle = study.study_name[len(common_prefix) :]
            figure.update_yaxes(title_text=subtitle, row=i, col=1, range=[y_min, y_max])
        for trace in vis.plot_param_importances(study).data:
            figure.add_trace(trace, row=i, col=2)

    figure.update_layout(**common_layout)
    figure.update_layout(
        title={"text": common_prefix, "xanchor": "center", "x": 0.5},
        width=1200,
        height=300 * len(studies),
        showlegend=False,
    )
    return figure


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

# %%
# %%
