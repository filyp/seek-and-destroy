import gc

import torch as pt


def cross_entropy_loss(output, input_ids, _dummy=None):
    return pt.nn.CrossEntropyLoss()(
        output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32),
        input_ids[:, 1:].flatten(),
    )


def neg_cross_entropy_loss(output, input_ids, _dummy=None):
    return -cross_entropy_loss(output, input_ids)


def stream_activation_loss(output, input_ids, _dummy=None):
    return sum(
        activation.norm(dim=-1).mean() ** 2
        # last activation is huge for some reason, so ignore it
        for activation in output.hidden_states[:-1]
    )


# adapted from https://github.com/rishub-tamirisa/tamper-resistance/blob/41b749ca4d9bcb7608c7ead2ca48b0508714af99/modules/objectives.py#L114
def neg_entropy_loss(output, input_ids, _dummy=None) -> pt.Tensor:
    """
    Compute the negative mean entropy loss for the given logits.

    This function calculates the entropy of the softmax distribution of the input logits
    and returns the negative mean entropy as a loss value. Minimizing this loss
    encourages the model to produce more uniform (higher entropy) probability distributions.

    Returns:
        pt.Tensor: The negative mean entropy loss.
    """
    logits = output.logits
    softmax = pt.nn.functional.softmax(logits, dim=-1)
    log_softmax = pt.nn.functional.log_softmax(logits, dim=-1)
    entropy = pt.sum(-softmax * log_softmax, dim=-1).mean()
    return entropy.mean() * -1


def biased_neg_entropy_loss(output, input_ids, correct_logit_bias) -> pt.Tensor:
    logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = input_ids[:, 1:].flatten()
    # shift up the correct logits, so that they'll need to be brought further down
    logits[pt.arange(len(ids)), ids] += correct_logit_bias
    # calculate entropy
    softmax = pt.nn.functional.softmax(logits, dim=-1)
    log_softmax = pt.nn.functional.log_softmax(logits, dim=-1)
    entropy = pt.sum(-softmax * log_softmax, dim=-1).mean()
    return entropy.mean() * -1


def correct_logit_minus_avg_loss(output, input_ids, clip_at=0):
    logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = input_ids[:, 1:].flatten()
    true_logits = logits[pt.arange(len(ids)), ids]
    true_logits -= logits.mean(dim=-1)
    true_logits = true_logits.clip(min=clip_at)
    return true_logits.mean()


loss_fns = dict(
    cross_entropy=cross_entropy_loss,
    neg_cross_entropy=neg_cross_entropy_loss,
    neg_entropy=neg_entropy_loss,
    correct_logit_minus_avg=correct_logit_minus_avg_loss,
    stream_activation=stream_activation_loss,
)


def circuit_breaker_forget_loss(
    model,
    forget_input_ids,
    target_layers,
    frozen_model=None,
    lora_model=None,
):

    forget_attention_mask = pt.ones_like(forget_input_ids)

    forget_inputs = dict(
        input_ids=forget_input_ids,
        attention_mask=forget_attention_mask,
        output_hidden_states=True,
    )

    # ===== loss components =====
    layers_forget_attention_mask = forget_attention_mask.repeat(
        len(target_layers), 1, 1
    ).unsqueeze(-1)

    if lora_model is not None and frozen_model is None:
        lora_model.disable_adapter_layers()
        frozen_model = model

    if lora_model is None and frozen_model is None:
        raise Exception("Function did not get frozen model and LoRA is disabled.")

    assert frozen_model is not None

    frozen_model.eval()
    with pt.no_grad():
        forget_outputs = frozen_model(**forget_inputs).hidden_states
        forget_hidden = pt.stack([forget_outputs[l].detach() for l in target_layers])
    del forget_outputs
    gc.collect()

    if lora_model is not None:
        lora_model.enable_adapter_layers()
    model.train()

    lora_forget_outputs = model(**forget_inputs).hidden_states
    lora_forget_hidden = pt.stack([lora_forget_outputs[l] for l in target_layers])

    normalized_lora_forget_outputs = lora_forget_hidden / (
        pt.norm(lora_forget_hidden, dim=-1, keepdim=True, dtype=pt.float)
    )
    normalized_forget_outputs = forget_hidden / (
        pt.norm(forget_hidden, dim=-1, keepdim=True, dtype=pt.float)
    )
    inner_product = (
        normalized_lora_forget_outputs * normalized_forget_outputs
    ) * layers_forget_attention_mask
    forget_loss = (
        pt.relu(inner_product.sum(dim=-1)).sum() / layers_forget_attention_mask.sum()
    )

    return forget_loss


def circuit_breaker_retain_loss(model, retain_input_ids, frozen_model=None, lora_model=None):

    retain_attention_mask = pt.ones_like(retain_input_ids)

    retain_inputs = dict(
        input_ids=retain_input_ids,
        attention_mask=retain_attention_mask,
        output_hidden_states=True,
    )

    if lora_model is not None:
        lora_model.disable_adapter_layers()
        frozen_model = model

    if lora_model is None and frozen_model is None:
        raise Exception("Function did not get frozen model and LoRA is disabled.")

    assert frozen_model is not None

    frozen_model.eval()
    with pt.no_grad():
        orig_retain_outputs = frozen_model(**retain_inputs).hidden_states
        orig_retain_hidden = pt.stack(orig_retain_outputs).detach()
        layers_retain_attention_mask = retain_attention_mask.repeat(
            len(orig_retain_outputs), 1, 1
        ).unsqueeze(-1)
        orig_retain_hidden *= layers_retain_attention_mask

    del orig_retain_outputs
    gc.collect()

    if lora_model is not None:
        lora_model.enable_adapter_layers()
    model.train()

    lora_retain_outputs = model(**retain_inputs).hidden_states
    lora_retain_hidden = pt.stack(lora_retain_outputs) * layers_retain_attention_mask
    diffs = lora_retain_hidden - orig_retain_hidden
    # the last hidden state is anomalously high (at least for pythia)
    diffs = diffs[:-1]
    retain_loss = pt.norm(diffs, dim=-1, p=2, dtype=pt.float).nanmean()

    return retain_loss


# def correct_logit_loss(output, input_ids):
#     logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     true_logits = logits[pt.arange(len(ids)), ids]
#     return true_logits.mean()


# def clipped_correct_logit_loss(output, input_ids):
#     logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     true_logits = logits[pt.arange(len(ids)), ids]
#     # note: clipping at 0 is actually pretty useless, because logits don't come that low!
#     return true_logits.clip(min=0).mean()


# def soft_clipped_correct_logit_loss(output, input_ids, atan_scale):
#     logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     true_logits = logits[pt.arange(len(ids)), ids]
#     soft_clipped = (true_logits / atan_scale).atan() * atan_scale
#     return soft_clipped.mean()


# def soft_clipped_cross_entropy_loss(output, input_ids, atan_scale):
#     logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     probs = pt.nn.functional.softmax(logits, dim=-1)
#     true_probs = probs[pt.arange(len(ids)), ids]
#     losses = -pt.log(true_probs)
#     soft_clipped = (losses / atan_scale).atan() * atan_scale
#     return soft_clipped.mean()


# def flipped_prob_loss(output, input_ids, correct_logit_bias=0, only_grad_correct=False):
#     """Compute loss that tries to reduce probability of correct tokens.

#     In contrast to neg_entropy_loss, it's monotonic in respect to the correct logit.
#     It has two asymptotes: on the right has derivative 1, on the left has derivative 0.

#     Args:
#         output: Model output containing logits
#         input_ids: Input token IDs
#         correct_logit_bias: Bias added to correct token logits
#         only_grad_correct:
#             If True, only compute gradients for correct token positions
#             If False, compute gradients for all tokens
#     """
#     orig_logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     target_indices = (pt.arange(len(ids)), ids)

#     if only_grad_correct:
#         # create a detached copy of logits
#         logits = orig_logits.detach().clone()
#         # only keep gradients for the target tokens
#         logits[target_indices] = orig_logits[target_indices]
#     else:
#         logits = orig_logits

#     # shift up the correct logits, so that they'll need to be brought further down
#     logits[target_indices] += correct_logit_bias
#     probs = pt.nn.functional.softmax(logits, dim=-1)

#     true_probs = probs[pt.arange(len(ids)), ids]
#     # ! invert the probability
#     losses = -pt.log(1 - true_probs)
#     return losses.mean()
