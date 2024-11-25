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


# training code
set_fade_factor(model, 0)
# activation_agnostic(model, next(forget_unlearn_iter), lr=0.4)
set_fade_factor(model, 1)

# normal_train_step(model, next(forget_unlearn_iter), 0.0003, loss_sign=-1)
normal_train_step(model, next(forget_relearn_iter), 0.0003)
normal_train_step(model, next(retain_relearn_iter), 0.0005)
# scale_perturbation(model, original_state_dict, 0.99)


# disassociation
effect_activations = original_model.model.layers[10].mlp.act_fn.last_post_activations
cause_activations = original_model.model.layers[8].mlp.act_fn.last_post_activations
# %%
cause_activations = cause_activations[0, 0]
effect_activations = effect_activations[0, 0]
# %%
cutoff = effect_activations.to(pt.float).quantile(0.99)
cutoff
# %%
effect_indexes = pt.where(effect_activations > cutoff)
effect_indexes = effect_indexes[0]
effect_indexes
# %%
effect_activations[effect_indexes]
# %%
# %%
from transformers import BitsAndBytesConfig

og_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=pt.float8_e5m2),
    low_cpu_mem_usage=True,
)

# %%
# !!
# plt.hist(pt.log10(rel_imps).cpu().float(), bins=100)
# plt.title("log10(relative importance on neurons)")


# og_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
# model.load_state_dict(og_model.state_dict())

# %% manual eval
# loss = DefaultNamespace()
# i = 999
# with pt.no_grad():
#     model.set_adapter(["adversarial_lora"])
#     loss.loudness = forward_with_clipped_logit(model, forget_eval_batch)
#     model.set_adapter([])
#     loss.forget = forward(model, forget_eval_batch)
#     model.set_adapter([])
#     loss.retain = forward(model, retain_eval_batch)
#     model.set_adapter(["adversarial_lora"])
#     loss.adv_forget = forward(model, forget_eval_batch)
#     model.set_adapter(["adversarial_lora"])
#     loss.adv_retain = forward(model, retain_eval_batch)
# # calculate and print stats
# stats = dict(
#     loudness=loss.loudness.item(),
#     forget=loss.forget.exp() - initial_forget_ppl,
#     retain=loss.retain.exp() - initial_retain_ppl,
#     adv_forget=loss.adv_forget.exp() - initial_forget_ppl,
#     adv_retain=loss.adv_retain.exp() - initial_retain_ppl,
# )
# print(f"{i + 1:4d}  " + "   ".join(f"{v:10.2f}" for v in stats.values()))


# assert set(model.state_dict().keys()) == set(state_dict.keys())


# %% L1 and L2
# # L2 revert
# for name, param in model.named_parameters():
#     if ".base_layer" in name:
#         initial_weights = initial_state_dict[name]
#         delta = param.data - initial_weights
#         param.data = initial_weights + delta * c.L2_revert_factor

# # L1 revert
# a = 0.00003
# for name, param in model.named_parameters():
#     initial_weights = initial_state_dict[name]
#     delta = param.data - initial_weights
#     new_delta = delta.clip(max=-a) + delta.clip(min=a)
#     param.data = initial_weights + new_delta


# %% adapting forget_lr used to be needed when circuit wasn't sparse and retaining was harder
# # adapt forget_lr
# if (
#     stats["retain"] < c.acceptable_retain_ppl
#     and step - last_lr_update > 5 * 10
#     and forget_ppl_hist[-5] > stats["forget"]
# ):
#     c.forget_lr *= c.forget_ppl_increment
#     last_lr_update = step
#     print(f"forget_lr updated to {c.forget_lr:.1e}")
# forget_ppl_hist.append(stats["forget"])


# save model
if step % 500 == 0:
    run_name = f"f={forget_lr:.0e} r={retain_lr:.0e}"
    model_path = repo_root() / "models" / f"{run_name}_{step}steps.pt"
    pt.save(model.state_dict(), model_path)


# !!!
# # code for calculating threshold per parameter
# # ! load circuit
# circuit = c(circuit_name)
# # sparsify circuit
# for param_name, scores in kinda_safe_eval(criterion).items():
#     k = int(scores.numel() * quantile)
#     threshold = scores.flatten().kthvalue(k).values
#     circuit[param_name][scores < threshold] = 0


# # stop if forget_ppl is going down
# if f_ppl < last_forget_ppl:
#     break
# last_forget_ppl = f_ppl

# stop if retain_ppl is too high
if r_ppl > _stop_unlearning_at_ppl:
    print(f"Stopping unlearning due to high retain perplexity {r_ppl:.2f}")
    break


# training past the point of breaking LoRA doesn't improve final performance
# so stop if we broke through the LoRA
if stats["adv_forget"] > 500 and (i + 1) % 10 == 0:
    break


# path = save_file_and_stdout_open(__file__)
# save_file_and_stdout_close(path)


# model_id="HuggingFaceTB/SmolLM-135M",
# target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],


# # Plot optimization history
# fig = vis.plot_optimization_history(study)
# fig.show()

# # Plot parameter importance
# fig = vis.plot_param_importances(study)
# fig.show()

# # Plot parameter relationships
# fig = vis.plot_parallel_coordinate(study)
# fig.show()
