# %%
import optuna.visualization as vis

from adversarial_adapters import *

# get the latest study
storage = get_storage()

study_summaries = optuna.study.get_all_study_summaries(storage)
sorted_studies = sorted(study_summaries, key=lambda s: s.datetime_start)
latest_study = sorted_studies[-1]
study = optuna.load_study(study_name=latest_study.study_name, storage=storage)
print(study.study_name)

# study_name = "26.11,pl,dont_terminate_on_alora_break,better_range7"
# study = optuna.load_study(study_name=study_name, storage=storage)

# %%
# ! plot the slices for each hyperparam
fig = vis.plot_slice(study, target_name="Final forget loss")
# params=["quantile", "adv_lora_lr", "ret_lora_lr", "unlearn_lr", "retain_amp"],
fig.update_layout(
    title={"text": study.study_name, "xanchor": "center", "x": 0.5, "y": 0.95},
    template="plotly_white",
    font=dict(family="Times New Roman", size=20),
    title_font_size=30,
)
# save the figure
save_path = repo_root() / "paper" / "Figures" / f"{study.study_name}.png"
fig.write_image(str(save_path))
fig.show()
# fig = vis.plot_optimization_history(study)
# fig.show()

# %%
# ! rerun the best trial, with more steps
trials = study.get_trials()
finished_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
worst_to_best = sorted(finished_trials, key=lambda t: t.value)

config.unlearn_steps = 300
config.relearn_steps = 300

best_params = deepcopy(worst_to_best[-1].params)
# best_params["unlearn_lr"] *= 0.3
# best_params["ret_lora_lr"] = 0.001
# best_params["quantile"] = 0.007

for n, p in best_params.items():
    print(f"{n:15} {p}")
result = objective(MockTrial(best_params))
logging.info(f"Final result: {result}")
