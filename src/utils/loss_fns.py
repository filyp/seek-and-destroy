import torch as pt


def cross_entropy_loss(output, input_ids):
    return pt.nn.CrossEntropyLoss()(
        output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32),
        input_ids[:, 1:].flatten(),
    )


def correct_logit_loss(output, input_ids):
    logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = input_ids[:, 1:].flatten()
    true_logits = logits[pt.arange(len(ids)), ids]
    return true_logits.mean()


def clipped_correct_logit_loss(output, input_ids):
    logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = input_ids[:, 1:].flatten()
    true_logits = logits[pt.arange(len(ids)), ids]
    # note: clipping at 0 is actually pretty useless, because logits don't come that low!
    return true_logits.clip(min=0).mean()


def soft_clipped_correct_logit_loss(output, input_ids, atan_scale):
    logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = input_ids[:, 1:].flatten()
    true_logits = logits[pt.arange(len(ids)), ids]
    soft_clipped = (true_logits / atan_scale).atan() * atan_scale
    return soft_clipped.mean()


def soft_clipped_cross_entropy_loss(output, input_ids, atan_scale):
    logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = input_ids[:, 1:].flatten()
    probs = pt.nn.functional.softmax(logits, dim=-1)
    true_probs = probs[pt.arange(len(ids)), ids]
    losses = -pt.log(true_probs)
    soft_clipped = (losses / atan_scale).atan() * atan_scale
    return soft_clipped.mean()


def neg_cross_entropy_loss(output, input_ids):
    return -cross_entropy_loss(output, input_ids)


def stream_activation_loss(output, input_ids):
    return sum(
        activation.norm(dim=-1).mean() ** 2
        # last activation is huge for some reason, so ignore it
        for activation in output.hidden_states[:-1]
    )


# adapted from https://github.com/rishub-tamirisa/tamper-resistance/blob/41b749ca4d9bcb7608c7ead2ca48b0508714af99/modules/objectives.py#L114
def neg_entropy_loss(output, input_ids) -> pt.Tensor:
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


loss_fns = dict(
    cross_entropy=cross_entropy_loss,
    clipped_correct_logit=clipped_correct_logit_loss,
    correct_logit=correct_logit_loss,
    neg_cross_entropy=neg_cross_entropy_loss,
    stream_activation=stream_activation_loss,
    neg_entropy=neg_entropy_loss,
)
