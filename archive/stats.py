# %%
from common_startup_code import *

model = og_model

# %%

acts = []


def act_stat(module, args, output):
    global acts
    acts.append(args[0].detach().clone())


model.model.layers[0].mlp.act_fn._forward_hooks.clear()
model.model.layers[0].mlp.act_fn.register_forward_hook(act_stat)

for batch in islice(forget_set["unlearn"].batch(32), 1):
    with pt.no_grad():
        forward(model, batch)
# %%
acts = pt.stack(acts)

# %%
# histogram of activations
import matplotlib.pyplot as plt

acts = acts.to("cpu").float()

plt.hist(acts.view(-1), bins=100)
plt.show()
# %%
w1 = model.model.layers[1].mlp.down_proj.weight
w2 = model.model.layers[21].mlp.up_proj.weight

w1 = w1 / w1.norm(dim=0).reshape(1, -1)
w2 = w2 / w2.norm(dim=1).reshape(-1, 1)
cs = (w2 @ w1).flatten()
cs = cs.to("cpu").float().detach()
plt.hist(cs, bins=100)
cs[:1000000].quantile(0.99)

# %%
# sad: there won't be any top super neurons :c
