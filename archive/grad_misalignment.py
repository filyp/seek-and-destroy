# %%
import logging
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch as pt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.circuit_creation import *
from utils.data_loading import CachedBatches, dataset_loaders
from utils.training import *

config = SimpleNamespace(
    method_name="seek_and_destroy",
    # loss_fn_name="correct_logit",
    loss_fn_name="neg_cross_entropy",
    model_id="EleutherAI/pythia-14m",
    retain_set_name="wikitext",
    forget_set_name="python",
    batch_size=16,
)

pt.set_default_device("cuda")

# load datasets
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
forget_set = dataset_loaders[config.forget_set_name](tokenizer)
forget_batches = CachedBatches(forget_set["train"], config.batch_size)

# accumulate grads
loss_fn = loss_fns[config.loss_fn_name]
model = AutoModelForCausalLM.from_pretrained(config.model_id)

loss_fn = loss_fns[config.loss_fn_name]

input_ids = next(iter(forget_batches))

# %%
def save_grad(module, grad_input, grad_output):
    module.grad_out = grad_output[0]
    module.grad_in = grad_input[0]


for l in model.gpt_neox.layers:
    l._backward_hooks.clear()
    l.register_full_backward_hook(save_grad)

    l.post_attention_layernorm._backward_hooks.clear()
    l.post_attention_layernorm.register_full_backward_hook(save_grad)

    l.mlp.dense_h_to_4h._backward_hooks.clear()
    l.mlp.dense_h_to_4h.register_full_backward_hook(save_grad)
# %%
out = model(input_ids)
loss = cross_entropy_loss(out, input_ids)
loss.backward()
# assert pt.allclose(l.post_attention_dropout.grad_out, l.grad_out)
# assert pt.allclose(l.post_mlp_dropout.grad_out, l.grad_out)
# assert pt.allclose(
#     l.grad_out + l.post_attention_layernorm.grad_in + l.input_layernorm.grad_in,
#     l.grad_in,
# )

# %%
l.post_attention_layernorm.grad_in / l.post_attention_layernorm.grad_out
# %%
ln = l.post_attention_layernorm
# %%
in_ = l.grad_in * ln.grad_in / ln.grad_out
out = l.mlp.dense_h_to_4h.grad_out
# replace nan with 0
in_ = in_.nan_to_num()
# %%
out.shape
# %%
update = pt.einsum("bti,bto->oi", in_, out)
assert not pt.isnan(update).any()
# %%
l.mlp.dense_h_to_4h.weight
