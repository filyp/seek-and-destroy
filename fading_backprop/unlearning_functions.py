# %%
import torch as pt
from utils import forward


def dummy(model, batch, lr):
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    # COPY THIS FUNCTION AND WRITE YOUR CODE HERE

    optimizer.step()


def activation_agnostic(model, batch, lr):
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)

    loss = forward(model, batch)
    loss.backward()
    # get rid of the normal grad
    optimizer.zero_grad(set_to_none=True)

    # calculate custom grads
    for layer in model.model.layers:
        module = layer.mlp.down_proj
        # get the downstream gradient saved by the hook
        output_grad = module.output_grad
        output_grad = output_grad[:, :-1, :]  # last token position has no gradient
        assert output_grad.norm(dim=-1).all()

        # calculate the projection
        projection_amps = pt.einsum("oi,bto->bti", module.weight, output_grad)
        projection_amps /= output_grad.norm(dim=-1, keepdim=True)
        update = pt.einsum("bto,bti->oi", output_grad, projection_amps)
        module.weight.grad = update

    optimizer.step()


name_to_function = dict(
    activation_agnostic=activation_agnostic,
)
