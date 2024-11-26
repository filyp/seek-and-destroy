# %%
from adversarial_adapters import *

# ! plot the slices for each hyperparam
# get the latest study
db_path = f"sqlite:///{repo_root() / 'results' / 'db.sqlite3'}"
storage = optuna.storages.RDBStorage(url=db_path)
# url="sqlite:///../archive/old_results/aa_hyperparam_robustness.sqlite3"
study_summaries = optuna.study.get_all_study_summaries(storage)
latest_study = max(study_summaries, key=lambda s: s.datetime_start)
study = optuna.load_study(study_name=latest_study.study_name, storage=storage)

# %%
# plot slice plot for each parameter
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

# config.unlearn_steps = 250
# config.relearn_steps = 200

best_params = deepcopy(worst_to_best[-1].params)
# best_params["unlearn_lr"] *= 0.7
# best_params["unlearn_lr_mult"] **= 1.5
for n, p in best_params.items():
    print(f"{n:15} {p}")
result = objective(MockTrial(best_params))
logging.info(f"Final result: {result}")
