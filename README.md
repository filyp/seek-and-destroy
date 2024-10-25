todo: top from all neurons, not from each layer

Flipping top neurons' value vectors:
- Run a model on the forget and retain set, 3k tokens each.
- For each MLP neuron, calculate its `forget_importance` as `sum(activations ** 2)` and analogously for `retain_importance`.
- On each layer, choose 0.1% neurons with the highest `forget_importance / retain_importance` ratio.
- Invert their value vectors (their columns in the second MLP layer).