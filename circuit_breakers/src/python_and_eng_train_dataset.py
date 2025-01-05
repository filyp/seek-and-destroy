from _common_init import retain_set, forget_set
from torch.utils.data import IterableDataset
import datasets
from datasets import load_dataset
import transformers
from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
import json
import random
import csv
random.seed(0)


# class CircuitBreakerDataset(IterableDataset):

#     def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
#         super(CircuitBreakerDataset, self).__init__()
#         retain_train = retain_set['train']
#         forget_train = forget_set['train']
#         forget_val = forget_set['val']

#     def __len__(self):

#         return 1000

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:

#         return dict(
#             input_ids_circuit_breaker=combined_input_ids_circuit_breaker,
#             attention_mask_circuit_breaker=combined_attention_mask_circuit_breaker,
#             input_ids=tokenized_inputs_retain["input_ids"],
#             attention_mask=tokenized_inputs_retain["attention_mask"],
#             input_ids_val=tokenized_inputs_val["input_ids"],
#             attention_mask_val=tokenized_inputs_val["attention_mask"],
#         )


class IterableCircuitBreakerDataset(IterableDataset):

    def __init__(self):
        super(IterableCircuitBreakerDataset, self).__init__()
        self.retain_train_iter = iter(retain_set['train'])
        self.forget_train_iter = iter(forget_set['train'])
        self.forget_val_iter = iter(forget_set['validation'])

    def __iter__(self):
        for i in range(1000):
            cb = next(self.forget_train_iter)
            ret = next(self.retain_train_iter)
            val = next(self.forget_val_iter)

            yield dict(
                input_ids_circuit_breaker=cb['input_ids'],
                attention_mask_circuit_breaker=cb['attention_mask'],
                input_ids=ret['input_ids'],
                attention_mask=ret["attention_mask"],
                input_ids_val=val["input_ids"],
                attention_mask_val=val["attention_mask"],
            )

    def __len__(self):
        return 1000
