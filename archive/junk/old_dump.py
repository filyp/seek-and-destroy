
# %%
# t = model.model.layers[5].mlp.down_proj._parameters["weight"]

# # plot 2D

# import matplotlib.pyplot as plt

# # bfloat to float
# t = t.float()
# t = t[:100, :100]

# plt.imshow(t.cpu().detach().numpy(), cmap="hot", interpolation="nearest")


# %%
# input_text = "Lol"
# inputs = tokenizer(input_text, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=1)
# text = tokenizer.decode(outputs[0])
# print(text)

# # outputs = model(**inputs)
# # outputs.logits.shape


# %%
for name, layer in model.model.layers.named_children():
    print(name, layer)
    layer.__name__ = name
    layer.register_forward_hook(
        lambda self, input, output: print(f"{self.__name__} forward, output shape: {len(output)}")
    )


# %%
# hits = pt.argmax(outputs.logits, dim=-1)[0,:-1] == inputs["input_ids"][0,1:]
# pt.mean(hits.float())


# %%
probs = pt.nn.functional.softmax(outputs.logits, dim=-1)
probs = probs[:, :, :]
input_ids = inputs["input_ids"][:, :]

valid_probs = pt.gather(probs, 2, input_ids.unsqueeze(-1))
# flatten the last two dimensions
valid_probs = valid_probs.squeeze(-1)
# %%
valid_probs

# %%
perplexity = pt.exp(pt.sum(pt.log(1 / valid_probs)) / valid_probs.numel())
perplexity


# %%
# index = 10
# output = model(input_ids=chunks[index:index+1].to(device))

# # %%
# out_tokens = output.logits.argmax(dim=-1)
# out_text = tokenizer.decode(out_tokens[0])
# print(tokenizer.decode(chunks[index]))
# print()
# print(out_text)
# # %%
# infernce_tokens = model.generate(chunks[index:index+1].to(device), max_new_tokens=100)
# inference_text = tokenizer.decode(infernce_tokens[0])
# print(inference_text)
# # %%
# %%

# # %%
# # prepare
# offset = 2
# batch_size = 1
# batch = train_chunks[offset : offset + batch_size].to(device)
# optimizer.zero_grad()

# # %%

# # forward pass
# output = model(batch)
# # compute loss
# pred = output.logits[:, :-1, :]
# true = batch[:, 1:]
# pred_flat = pred.reshape(-1, pred.shape[-1])
# loss = loss_fn(pred_flat, true.flatten())

# # backprop
# loss.backward()

# # %%
# grad2 = model.model.layers[15].mlp.down_proj.weight.grad
# # print(f"grad: {pt.norm(grad)}")
# # %%
# pt.cosine_similarity(grad.reshape(1, -1), grad2.reshape(1, -1))
# # %%

# # %%
# del output, pred, true, pred_flat, loss
# pt.cuda.empty_cache()

# %% find the best slope
# for slope in np.linspace(0, 0.02, 21):
#     for layer in model.model.layers:
#         layer.mlp.activation_fn = pt.nn.LeakyReLU(negative_slope=slope)
#     ppl = eval_perplexity(model, eval_chunks[:32])
#     print(f"{slope=:.3f}: {ppl=:.2f}")
# # best slope for phi-1.5: 0.01


# %%
plt.scatter(
    output_grad.flatten().cpu().float(),
    input_grad.flatten().cpu().float(),
    c=activation.flatten().cpu().float().detach(),
    s=0.01,
    # alpha=0.1,
)
plt.gca().set_aspect("equal")
# plot colorscheme scale
plt.colorbar()

# forward
text = example["prompt"] + "\n" + example["response"]
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
output = model(input_ids)
# backward
loss = cross_entropy(
    output.logits[:, :-1].flatten(end_dim=1),
    input_ids[:, 1:].reshape(-1),
)
loss.backward()
