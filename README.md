
# 1. Accumulate gradients on the whole forget set

```python
model.zero_grad()
for batch in forget_set:
    loss = forward(model, batch)
    loss.backward()
    # do not optimizer.step() !
grads = {name: param.grad / len(forget_set) for name, param in model.named_parameters()}
```
