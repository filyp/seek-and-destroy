from typing import Dict

import torch as pt
from transformers import AutoModelForCausalLM

from utils.loss_fns import *
from utils.training import *


def random_mapping(
    h, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False

    model.train()
    model.zero_grad(set_to_none=True)

    retain_iterator = iter(retain_batches)
    forget_iterator = iter(forget_batches)
    """
    retain_iterator, retain_dataloader = (
        iter(dataloaders["retain"]),
        dataloaders["retain"],
    )
    forget_iterator, forget_dataloader = (
        iter(dataloaders["forget_train"]),
        dataloaders["forget_train"],
    )
    """
    stream_hash_table = pt.randn(
        model.config.vocab_size, model.config.hidden_size, requires_grad=False
    )

    # todo later figure out the actual number of forward and backward passes
    passes_per_loop = 1
    for loop_num in range(config.unlearn_steps // passes_per_loop):
        """
        retain_batch, retain_iterator = get_next_batch(
            retain_iterator, retain_dataloader
        )
        forget_batch, forget_iterator = get_next_batch(
            forget_iterator, forget_dataloader
        )
        """
        retain_batch = next(retain_iterator)
        forget_batch = next(forget_iterator)

        lm_loss, cos_loss = random_vector_cosine_obj(
            model=model,
            x_r=retain_batch,
            x_f=forget_batch,
            # accelerator=accelerator,
            stream_hash_table=stream_hash_table,
            compute_lm_loss=True,
        )
        # total_lm_loss = lm_loss
        # total_cos_loss = cos_loss

        optimizer.step()

        model.zero_grad(set_to_none=True)

    return model


def random_vector_cosine_obj(
    model: pt.nn.Module = None,
    x_r: Dict[str, pt.Tensor] = None,
    x_f: Dict[str, pt.Tensor] = None,
    # accelerator: Accelerator = None,
    stream_hash_table: pt.Tensor = None,
    compute_lm_loss: bool = False,
) -> int:
    """
    Summary: Maximize cosine similarity between forget and random vectors while maximizing next-token likelihood for the retain set

    Args:
        model (pt.nn.Module): The model to be used for the computation
        x_r (Dict[str, pt.Tensor]): The retain data
        x_f (Dict[str, pt.Tensor]): The forget data
        accelerator (Accelerator): The accelerator to be used for the computation
        stream_hash_table (pt.Tensor): The hash table for random vectors

    Returns:
    """
    _x_r = _filter_inputs(x_r)
    _x_f = _filter_inputs(x_f)

    # (1) Compute LM loss for retain data
    x_r_lm_loss = pt.tensor(0.0)
    if compute_lm_loss:
        logits = model(**_x_r).logits
        x_r_lm_loss = (
            log_p_loss(logits, x_r.get("labels"), model.vocab_size)
            # * scale
        )
        # accelerator.backward(x_r_lm_loss)
        x_r_lm_loss.backward()

    # (2) (flatten sequence length and batch size)
    outputs = model(**_x_f, output_hidden_states=True)
    f_stream = [stream.view(-1, stream.size(-1)) for stream in outputs.hidden_states]

    f_input_ids = x_f.get("input_ids").view(-1)
    rand_stream = [stream_hash_table[f_input_ids] for _ in f_stream]

    # maximize cosine similarity between each unit-normalized random vector 
    # and each unit-normalized forget stream row (batch_size * sequence_length, hidden_size)
    cos_sim_loss = pt.stack([
        (1 - pt.abs(pt.nn.functional.cosine_similarity(f, r, dim=-1))).mean()
        for f, r in zip(f_stream, rand_stream)
    ]).mean()
    # accelerator.backward(cos_sim_loss)
    cos_sim_loss.backward()

    return x_r_lm_loss.item(), cos_sim_loss.item()
