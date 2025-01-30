# def surgical_tar(
#     h, config, retain_batches, forget_batches, f_eval, r_eval, allowed_r_loss
# ):
#     fork_every_n_loops = int(fork_every_n_loops)

#     model = AutoModelForCausalLM.from_pretrained(config.model_id)
#     adversary = AutoModelForCausalLM.from_pretrained(config.model_id)
#     model.config.use_cache = False
#     adversary.config.use_cache = False

#     clip_at = additional_param if config.additional_param_name == "clip_at" else 0

#     # get params to intervene on
#     interven_params = [
#         p
#         for name, p in model.named_parameters()
#         if any(f"{m}.weight" in name for m in config.target_modules)
#     ]
#     adv_interven_params = [
#         p
#         for name, p in adversary.named_parameters()
#         if any(f"{m}.weight" in name for m in config.target_modules)
#     ]
#     total_interven_numel = sum(p.numel() for p in interven_params)

#     # require grads only on intervened params
#     for p in model.parameters():
#         p.requires_grad = id(p) in [id(p) for p in interven_params]
#     for p in adversary.parameters():
#         p.requires_grad = id(p) in [id(p) for p in adv_interven_params]

#     for p in interven_params:
#         p.retain_acc = pt.zeros_like(p.data)
#         if config.additional_param_name == "forget_momentum_decay":
#             p.forget_momentum = pt.zeros_like(p.data)

#     # ! unlearning loop
#     logging.info("step      base_f      base_r")
#     retain_iter = iter(retain_batches)
#     forget_iter = iter(forget_batches)

#     # ! unlearning loop
#     passes_per_loop = 4 + int(config.train_adversary)
#     assert config.unlearn_steps % passes_per_loop == 0


for loop_num, (retain_batch, forget_batch) in enumerate(batch_pairs):
    if (loop_num % fork_every_n_loops == 0):
        # ! fork adversary
        adversary.load_state_dict(model.state_dict())

    # ! retain pass
    model.zero_grad()
    output = model(retain_batch)
    cross_entropy_loss(output, retain_batch).backward()
    for p in interven_params:
        # update disruption scores
        p.retain_acc = p.retain_acc * retain_momentum + p.grad * (1 - retain_momentum)
        p.data -= retaining_rate * p.retain_acc  # update
    model.zero_grad(set_to_none=True) # unneeded?

    # ! relearn the adversary
    adversary.zero_grad()
    output = adversary(forget_batch)
    cross_entropy_loss(output, forget_batch).backward(retain_graph=True)
    for p, adv_p in zip(interven_params, adv_interven_params):
        # apply adversary update
        adv_p.data -= adv_lr * adv_p.grad
        # decay adversary into base model
        adv_p.data = adv_p.data * adv_decay + p.data * (1 - adv_decay)

    # ! get unlearning grads loss from adversary
    # reuse the computation graph from previous block
    adversary.zero_grad()
    correct_logit_minus_avg_loss(output, forget_batch).backward()
    # ! unlearning step with masking
    grad_norm = sum(adv_p.grad.norm() ** 2 for adv_p in adv_interven_params) ** 0.5
    for p, adv_p in zip(interven_params, adv_interven_params):
        adv_p.grad *= p.retain_acc.sign() == adv_p.grad.sign()  # mask
        adv_p.grad *= total_interven_numel**0.5 / grad_norm  # normalize
        p.data -= unlearning_rate * adv_p.grad  # update
