
# %%
unsafe_log_probs = []
unsafe_lens = []
for example in islice(unsafe_examples, 50):
    with pt.no_grad():
        resp_log_prob, response_len = get_response_log_prob(example, model, tokenizer)
    unsafe_log_probs.append(resp_log_prob.cpu().detach().float())
    unsafe_lens.append(response_len)

safe_log_probs = []
safe_lens = []
for example in islice(safe_examples, 50):
    with pt.no_grad():
        resp_log_prob, response_len = get_response_log_prob(example, model, tokenizer)
    safe_log_probs.append(resp_log_prob.cpu().detach().float())
    safe_lens.append(response_len)

# %%
plt.plot(unsafe_lens, unsafe_log_probs, "ro", markersize=3)
plt.plot(safe_lens, safe_log_probs, "go", markersize=3)

# %%
safe_ratios = [lp / len_ for lp, len_ in zip(safe_log_probs, safe_lens)]
unsafe_ratios = [lp / len_ for lp, len_ in zip(unsafe_log_probs, unsafe_lens)]
safe_ratios = np.array(safe_ratios)
unsafe_ratios = np.array(unsafe_ratios)

# %%
safe_ratios.mean(), unsafe_ratios.mean()
# %%
# sem
safe_ratios.std() / np.sqrt(len(safe_ratios)), unsafe_ratios.std() / np.sqrt(len(unsafe_ratios))
# %%