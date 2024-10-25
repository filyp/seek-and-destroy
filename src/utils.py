import json
from itertools import islice

import torch as pt
from datasets import IterableDataset, IterableDatasetDict, load_dataset
from torcheval.metrics.text import Perplexity

context_len = 100
device = "cuda"


def load_one_oscar_shard(lang, tokenizer):
    # only use one ~600MB shard
    # also, streaming would make splitting too slow
    raw_dataset = load_dataset(
        "text",
        split="train",  # train is the only split in oscar
        data_files=f"hf://datasets/oscar-corpus/OSCAR-2301/{lang}_meta/{lang}_meta_part_1.jsonl.zst",
    )

    # split into 4 quarters
    half1, half2 = raw_dataset.train_test_split(test_size=0.5, seed=42).values()
    quarter1, quarter2 = half1.train_test_split(test_size=0.5, seed=42).values()
    quarter3, quarter4 = half2.train_test_split(test_size=0.5, seed=42).values()

    dataset = (
        # define splits; make it iterable so that it can be processed on demand
        IterableDatasetDict(
            unlearn=IterableDataset.from_generator(lambda: (ex for ex in quarter1)),
            relearn=IterableDataset.from_generator(lambda: (ex for ex in quarter2)),
            validation=IterableDataset.from_generator(lambda: (ex for ex in quarter3)),
            test=IterableDataset.from_generator(lambda: (ex for ex in quarter4)),
        )
        # process the raw data, following OSCAR-2301.py
        .map(lambda ex: {"text": json.loads(ex["text"])["content"]})
        # tokenize
        .map(
            lambda ex: tokenizer(
                ex["text"],
                return_tensors="pt",
                max_length=context_len,
                truncation=True,
            ).to(device),
        )
        # filter out the short ones
        .filter(lambda ex: ex["input_ids"].shape[-1] >= context_len)
    )
    assert next(iter(dataset["test"]))["text"] != next(iter(dataset["unlearn"]))["text"]
    return dataset


def get_perplexity(model, dataset, num_batches=1, batch_size=32):
    metric = Perplexity(device=device)
    for batch in islice(dataset["validation"].batch(batch_size), num_batches):
        input_ids = pt.cat(batch["input_ids"])
        with pt.no_grad():
            outputs = model(input_ids)
        metric.update(outputs.logits[:, :-1], input_ids[:, 1:])
    return metric.compute().item()


def forward(model, batch):
    loss_fn = pt.nn.CrossEntropyLoss()
    # create batched input_ids
    input_ids = pt.cat(batch["input_ids"])
    assert input_ids.shape[1] == context_len
    # forward pass
    logits = model(input_ids).logits
    # compute loss
    return loss_fn(logits[:, :-1, :].flatten(end_dim=1), input_ids[:, 1:].flatten())


def get_norm_of_weights_change(model, original_state_dict):
    partial_norms = []
    for module_name, current_weights in model.named_parameters():
        original_weights = original_state_dict[module_name]
        norm = (original_weights - current_weights).norm()
        partial_norms.append(norm)
    return pt.tensor(partial_norms).norm()


def scale_perturbation(model, original_state_dict, scaling_factor):
    for module_name, current_weights in model.state_dict().items():
        original_weights = original_state_dict[module_name]
        # modification need to be done in-place so it's a bit awkward:
        current_weights -= original_weights
        current_weights *= scaling_factor
        current_weights += original_weights


def normal_train_step(model, batch, lr, loss_sign=1):
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    loss = forward(model, batch)
    loss *= loss_sign
    loss.backward()

    optimizer.step()


def get_stats(model, og_model, forget_set, retain_set):
    return pt.tensor([
        get_perplexity(model, forget_set),
        get_perplexity(model, retain_set),
        get_norm_of_weights_change(model, og_model.state_dict()),
    ])


def print_stats(stats):
    print(f"forget: {stats[0]:4.0f}  retain: {stats[1]:5.2f}  norm: {stats[2]:5.2f}  ratio: {stats[0] / stats[1]:.0f}")  # fmt: skip


def retrain_and_eval(
    model, og_model, forget, retain, num_batches=10, b_size=32, lr=0.0003
):
    """takes a model after an intervention and evals it before and after retraining"""
    initial_stats = get_stats(og_model, og_model, forget, retain)
    pre_retrain_diff = get_stats(model, og_model, forget, retain) - initial_stats
    print_stats(pre_retrain_diff)

    f_r = zip(forget["relearn"].batch(b_size), retain["relearn"].batch(b_size))
    for forget_batch, retain_batch in islice(f_r, num_batches):
        normal_train_step(model, forget_batch, lr)
        normal_train_step(model, retain_batch, lr)

    post_retrain_diff = get_stats(model, og_model, forget, retain) - initial_stats
    print_stats(post_retrain_diff)
    return pre_retrain_diff, post_retrain_diff
