# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "microsoft/Phi-3.5-mini-instruct"
# model_id = "microsoft/phi-1_5"  # bad multilingual perf

# dataset = load_dataset("open_subtitles", lang1="en", lang2="pl", trust_remote_code=True)
# dataset = load_dataset("oscar-corpus/oscar", "unshuffled_deduplicated_pl", trust_remote_code=True)

# load_dataset(
#     "oscar-corpus/OSCAR-2301",  # big multilingual corpus
#     "pl",
#     trust_remote_code=True,
#     streaming=True,  # stream because it's huge
#     split="train",  # train is the only split in OSCAR-2301
# )


    # .batch(batch_size=batch_size, drop_last_batch=True)
    # # consolidate the batches from a list into a tensor
    # .map(
    #     lambda batch: dict(
    #         input_ids=pt.cat(batch["input_ids"]),
    #         attention_mask=pt.cat(batch["attention_mask"]),
    #     )
    # )


#     # # ! for some reason I can't fathom, this results in each split having the same data !
#     # split into 4 quarters
#     half1, half2 = raw_dataset.train_test_split(test_size=0.5, seed=42).values()
#     q1, q2 = half1.train_test_split(test_size=0.5, seed=42).values()
#     q3, q4 = half2.train_test_split(test_size=0.5, seed=42).values()
#     dataset = IterableDatasetDict()
#     # define splits; make it iterable so that it can be processed on demand
#     for split_name, split in [
#         ("unlearn", q1),
#         ("relearn", q2),
#         ("validation", q3),
#         ("test", q4),
#     ]:
#         dataset[split_name] = (
#             IterableDataset.from_generator(lambda: (ex for ex in split))
#             # process the raw data, following OSCAR-2301.py
#             .map(lambda ex: {"text": json.loads(ex["text"])["content"]})
#             # tokenize
#             .map(
#                 lambda ex: tokenizer(
#                     ex["text"],
#                     return_tensors="pt",
#                     max_length=context_len,
#                     truncation=True,
#                 ).to(device),
#             )
#             # filter out the short ones
#             .filter(lambda ex: ex["input_ids"].shape[-1] >= context_len)
#         )
#     return dataset
# assert next(iter(dataset["test"]))["text"] != next(iter(dataset["unlearn"]))["text"]


        # # squeeze tensors
        # .map(
        #     lambda ex: dict(
        #         input_ids=ex["input_ids"].squeeze(),
        #         attention_mask=ex["attention_mask"].squeeze(),
        #     )
        # )
# for ex in DataLoader(dataset["test"].take(1024), batch_size=8):

# optimizer = pt.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)




    # # clean memory
    # del logits
    # pt.cuda.empty_cache()


# %% save mlp input activations
mlp_input_activations = dict()


def save_input_activation_hook(module, args, output):
    mlp_input_activations[module] = args[0]


for layer in model.model.layers:
    layer.mlp.down_proj._forward_hooks.clear()
    layer.mlp.down_proj.register_forward_hook(save_input_activation_hook)
    break

# 
act = mlp_input_activations[module]
assert pt.einsum("bti,btj->ij", grad, act).allclose(module.weight.grad)

# %%
# update = output_grad[0, 0].reshape(-1, 1) @ projection_amps[0, 0].reshape(1, -1)
# update

# %% trim the model to just two layers
model.model.layers = model.model.layers[:2]
pt.cuda.empty_cache()


# unlearn_and_relearn(
#     model,
#     forget_dataset,
#     retain_dataset,
#     num_unlearning_steps=30,
#     num_relearning_steps=30,
#     allowed_retain_ppl_multiplier=float("inf"),
# )
# # %%
# # get the latest wandb run
# import wandb

# runs = wandb.Api().runs("filyp/fading_backprop")
# latest_run = sorted(runs, key=lambda run: run.created_at, reverse=True)[0]
# hist = latest_run.history()
# plt.scatter(hist["w_delta"], hist["retain"])

# # %%
# plt.scatter(hist["w_delta"], hist["forget"])

    # if norm > norm_lim:
    #     scale_factor = norm_lim / norm
    #     print(f"scaling from {norm:.2f} to {norm_lim:.2f}")
    #     scale_perturbation(model, original_state_dict, scale_factor)
    #     lr *= scale_factor
    #     norm = get_norm_of_weights_change(model, original_state_dict)


# for g in optimizer.param_groups:
#     g["lr"] = 1000000

# %%
# inputs = tokenizer(["Litwo, czemu"], return_tensors="pt").to(device)
# # outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True, do_sample=True )
# outputs = original_model.generate(**inputs, max_new_tokens=60, return_dict_in_generate=True, output_scores=True)
# for seq in outputs.sequences:
#     print(tokenizer.decode(seq))


# %%
batch = b1
input_ids = pt.cat(batch["input_ids"])

with pt.no_grad():
    outputs = model(input_ids)

ids = input_ids[:, 1:].flatten()
logits = outputs.logits[:, :-1].flatten(end_dim=1)
probs = pt.nn.functional.softmax(logits, dim=-1)
chosen_probs = probs[pt.arange(len(ids)), ids]
# %%

# %%
m1 = pt.log(1 / chosen_probs).mean()
m1.exp()
# %%
((m1 + m2) / 2).exp()
