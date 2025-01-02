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
