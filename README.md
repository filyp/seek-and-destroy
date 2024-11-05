
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
