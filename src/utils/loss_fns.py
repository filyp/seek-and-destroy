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


loss_fns = dict(
    cross_entropy=cross_entropy_loss,
    clipped_correct_logit=clipped_correct_logit_loss,
    correct_logit=correct_logit_loss,
    neg_cross_entropy=neg_cross_entropy_loss,
    stream_activation=stream_activation_loss,
    neg_entropy=neg_entropy_loss,
)


def flipped_prob_loss(output, input_ids, correct_logit_bias=0, only_grad_correct=False):
    """Compute loss that tries to reduce probability of correct tokens.

    In contrast to neg_entropy_loss, it's monotonic in respect to the correct logit.
    It has two asymptotes: on the right has derivative 1, on the left has derivative 0.

    Args:
        output: Model output containing logits
        input_ids: Input token IDs
        correct_logit_bias: Bias added to correct token logits
        only_grad_correct:
            If True, only compute gradients for correct token positions
            If False, compute gradients for all tokens
    """
    orig_logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = input_ids[:, 1:].flatten()
    target_indices = (pt.arange(len(ids)), ids)

    if only_grad_correct:
        # create a detached copy of logits
        logits = orig_logits.detach().clone()
        # only keep gradients for the target tokens
        logits[target_indices] = orig_logits[target_indices]
    else:
        logits = orig_logits

    # shift up the correct logits, so that they'll need to be brought further down
    logits[target_indices] += correct_logit_bias

    # # for numerical reasons shift logits (this does not change probabilities)
    # logits = logits - logits.max(dim=-1, keepdims=True).values
    # # softmax
    # exp_logits = logits.exp()
    # probs = exp_logits / exp_logits.sum(dim=-1, keepdims=True)
    probs = pt.nn.functional.softmax(logits, dim=-1)

    true_probs = probs[pt.arange(len(ids)), ids]
    # ! invert the probability
    losses = -pt.log(1 - true_probs)
    return losses.mean()


def correct_logit_minus_avg_loss(output, input_ids, clip_at):
    logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = input_ids[:, 1:].flatten()
    true_logits = logits[pt.arange(len(ids)), ids]
    true_logits -= logits.mean(dim=-1)
    # # append true_logits to debug.txt
    # with open("debug.txt", "a") as f:
    #     f.write(f"{true_logits.mean():.0f}, {min(true_logits):.0f}\n")
    true_logits = true_logits.clip(min=clip_at)
    return true_logits.mean()
