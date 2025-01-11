import json
import re

import torch as pt
from datasets import IterableDataset, IterableDatasetDict, load_dataset


def looping_iter(iterable):
    # like itertools.cycle, but will not eat memory by storing element copies
    while True:
        yield from iterable


def get_batch(iter, n):
    return pt.cat([next(iter)["input_ids"] for _ in range(n)])


def prepare_dataset(raw_dataset, tokenizer, preprocess_fn=lambda ex: {}):
    # preprocess_fn is used to add additional fields to the dataset before tokenization
    context_len = 100

    # split into 4 quarters
    half1, half2 = raw_dataset.train_test_split(test_size=0.5, seed=42).values()
    quarter3, quarter4 = half2.train_test_split(test_size=0.5, seed=42).values()

    dataset = (
        # define splits; make it iterable so that it can be processed on demand
        IterableDatasetDict(
            train=IterableDataset.from_generator(lambda: (ex for ex in half1)),
            validation=IterableDataset.from_generator(lambda: (ex for ex in quarter3)),
            test=IterableDataset.from_generator(lambda: (ex for ex in quarter4)),
        ).map(preprocess_fn)
        # tokenize
        .map(
            lambda ex: tokenizer(
                ex["text"],
                return_tensors="pt",
                max_length=context_len,
                truncation=True,
            ),
        )
        # filter out the short ones
        # this together with truncation ensures that each example has exactly 100 tokens
        .filter(lambda ex: ex["input_ids"].shape[-1] >= context_len)
    )
    assert next(iter(dataset["test"]))["text"] != next(iter(dataset["train"]))["text"]
    return dataset


def load_one_oscar_shard(lang, tokenizer):
    # only use one ~600MB shard
    # also, streaming would make splitting too slow
    return prepare_dataset(
        load_dataset(
            "text",
            split="train",  # train is the only split in oscar
            data_files=f"hf://datasets/oscar-corpus/OSCAR-2301/{lang}_meta/{lang}_meta_part_1.jsonl.zst",
        ),
        tokenizer,
        # process the raw data, following OSCAR-2301.py
        lambda ex: {"text": json.loads(ex["text"])["content"]},
    )


def load_wikitext(tokenizer):
    return prepare_dataset(
        # train split is big enough so just use it - it's simpler
        load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train"),
        tokenizer,
    )


def load_cruelty(tokenizer):
    beavertails = load_dataset("PKU-Alignment/BeaverTails")
    split = beavertails["330k_train"]
    category = "animal_abuse"
    # from 300k examples filters down to 3k
    beaver_category = split.filter(lambda ex: ex["category"][category])
    # prepare dataset further filters out short examples, down to 1.5k, or 90 batches of 16
    return prepare_dataset(
        beaver_category,
        tokenizer,
        # lambda ex: {"text": ex["response"]},
        lambda ex: {"text": ex["prompt"] + "\n" + ex["response"]},
    )


def load_beaver_safe(tokenizer):
    beavertails = load_dataset("PKU-Alignment/BeaverTails")
    split = beavertails["330k_train"]
    # from 300k examples filters down to 134k
    safe_examples = split.filter(lambda ex: ex["is_safe"])
    # prepare dataset further filters out short examples, down to 40k examples
    return prepare_dataset(
        safe_examples,
        tokenizer,
        # lambda ex: {"text": ex["response"]},
        lambda ex: {"text": ex["prompt"] + "\n" + ex["response"]},
    )


def _remove_comments_and_docstrings(code: str) -> str:
    # Remove docstrings
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
    # Remove single-line comments
    code = re.sub(r"#.*", "", code)
    # Remove blank lines
    code = re.sub(r"\n\s*\n", "\n", code)
    # Remove leading/trailing whitespace
    code = code.strip()
    return code


def load_python_dataset(tokenizer):
    return prepare_dataset(
        load_dataset("Nan-Do/code-search-net-python", split="train"),
        tokenizer,
        lambda ex: {"text": _remove_comments_and_docstrings(ex["code"])},
    )


dataset_loaders = dict(
    wikitext=load_wikitext,
    python=load_python_dataset,
    oscar_en=lambda tokenizer: load_one_oscar_shard("en", tokenizer),
    oscar_pl=lambda tokenizer: load_one_oscar_shard("pl", tokenizer),
    cruelty=load_cruelty,
    beaver_safe=load_beaver_safe,
)


class CachedBatches:
    def __init__(self, base_iter, batch_size):
        assert isinstance(base_iter, IterableDataset)
        self.base_iter = looping_iter(base_iter)
        self.batch_size = batch_size
        self.cache = []

    def __iter__(self):
        yield from self.cache
        while True:
            new_item = get_batch(self.base_iter, self.batch_size)
            self.cache.append(new_item)
            yield new_item