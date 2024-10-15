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


# %% save residual stream activations
grads = dict()


def save_grad_hook(module, grad_input, grad_output):
    grads[module] = grad_output[0]


for layer in model.model.layers:
    # clear hooks
    layer._backward_hooks.clear()
    # register hook
    layer.register_full_backward_hook(save_grad_hook)


# %% single example forward and backward pass
f = 0

batch = next(iter(pl_dataset["unlearn"].batch(1)))
loss_fn = pt.nn.CrossEntropyLoss()
# create batched input_ids
input_ids = pt.cat(batch["input_ids"])
# forward pass
output = model(input_ids)
# compute loss
loss = loss_fn(
    output.logits[:, :-1, :].flatten(end_dim=1),
    input_ids[:, 1:].flatten(),
)
# backpropagate
loss.backward()
# clean memory
del output, loss
pt.cuda.empty_cache()

# %% plot grads
one_val_from_each_layer = [grads[layer][0, -2, 0].item() for layer in model.model.layers]
plt.plot(one_val_from_each_layer)

