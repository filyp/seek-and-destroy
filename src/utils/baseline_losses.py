import torch as pt


def neg_cross_entropy_loss(output, input_ids):
    return -pt.nn.CrossEntropyLoss()(
        output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32),
        input_ids[:, 1:].flatten(),
    )


baseline_loss_fns = dict(
    neg_cross_entropy=neg_cross_entropy_loss,
)
