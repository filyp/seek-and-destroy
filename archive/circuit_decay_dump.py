

# %%
eval_vals(0.1, 0)

# %%

# %% grid search
# for percentile in [0.01, 0.03, 0.1, 0.3, 1]:
# for percentile in [0.1 ]:
for percentile in [0.08, 0.09, 0.10, 0.11, 0.12]:
    # for mult in [0.5, 0, -0.5, -1, -1.5]:
    # for mult in [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2]:
    for mult in [0.1]:
        eval_vals(percentile, mult)
    print()