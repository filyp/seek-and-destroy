Flipping top neurons' value vectors:
- Run a model on the forget and retain set, 3k tokens each.
- For each MLP neuron, calculate its `forget_importance` as `sum(activations ** 2)` and analogously for `retain_importance`.
- Choose 0.2% neurons with the highest `forget_importance / retain_importance` ratio.
- Invert their value vectors (their columns in the second MLP layer).

Fading backprop:
- note: can be done with or without activation_agnostic trick



# 1. Accumulate gradients on the whole forget set

```python
model.zero_grad()
for batch in forget_set:
    loss = forward(model, batch)
    loss.backward()
    # do not optimizer.step() !

unwanted_circuit = {
    name: -param.grad / len(forget_set)
    for name, param in model.named_parameters()
}
```

# 2. Apply the gradient from step 1, while training on the retain set

```python
optimizer = pt.optim.Adam(model.parameters(), lr=retain_lr, betas=(0.9, 0.999))
model.train()
for batch in retain_set:
    # break the unwanted circuit a bit
    for name, param in model.named_parameters():
        param.data -= unwanted_circuit[name] * forget_lr

    # train on the retain set
    optimizer.zero_grad(set_to_none=True)
    loss = forward(model, batch)
    loss.backward()
    optimizer.step()
```
