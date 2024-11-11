# %% init things, which need to be run once
import torch as pt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *

model_id = "Qwen/Qwen2.5-0.5B"
pt.set_default_device("cuda")

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
forget_set = load_one_oscar_shard("pl", tokenizer)
retain_set = load_one_oscar_shard("en", tokenizer)
forget_eval = get_batch(iter(forget_set["validation"]), 32)
retain_eval = get_batch(iter(retain_set["validation"]), 32)


# %%
def unlearn_and_relearn(
    quantile=0.99,  # between 0 and 1
    circuit_name="forget_linear_correct_logit",
    criterion='c("retain_absolu_correct_logit").abs() * (-1)',
    # note: too small forget_lr with bfloat16, can cause updates to become 0 due to numerical errors
    forget_lr=1.0e-2,
    retain_lr=4e-4,
    relearn_lr=1.2e-3,
    unlearning_steps=600000,
    relearning_steps=30,
    _stop_unlearning_at_ppl=28.4,
):
    set_seeds(42)

    # code for calculating threshold per parameter
    # # ! load circuit
    # circuit = c(circuit_name)
    # # sparsify circuit
    # for param_name, scores in kinda_safe_eval(criterion).items():
    #     k = int(scores.numel() * quantile)
    #     threshold = scores.flatten().kthvalue(k).values
    #     circuit[param_name][scores < threshold] = 0

    scores_dict = kinda_safe_eval(criterion)
    scores_flat = pt.cat([scores.flatten() for scores in scores_dict.values()])
    k = int(scores_flat.numel() * quantile)
    threshold = scores_flat.kthvalue(k).values
    print(f"{threshold=:.5f}")
    del scores_flat

    circuit = c(circuit_name)
    for param_name, scores in scores_dict.items():
        circuit[param_name][scores < threshold] = 0
    del scores_dict


    # load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
    # add lora
    lora_config = LoraConfig(r=8, target_modules="all-linear")
    peft_model = get_peft_model(model, lora_config, adapter_name="retain_lora")
    model = peft_model.model

    # prepare data iterators
    forget_iter = looping_iter(forget_set["train"])
    retain_iter = looping_iter(retain_set["train"])

    optimizer = pt.optim.Adam(model.parameters(), lr=retain_lr, betas=(0.9, 0.999))

    # ! unlearning loop
    print("step      forget       retain")
    print("unlearning...")
    for step in range(1, 1 + unlearning_steps):
        # break the circuit a bit
        for name, param in model.named_parameters():
            name = name.replace(".base_layer", "")
            if "lora" in name or name == "model.embed_tokens.weight":
                continue
            # if "_proj" in name:
            param.data -= circuit[name] * forget_lr

        # standard forward, backward, and update
        if retain_lr > 0:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            input_ids = get_batch(retain_iter, 8)
            loss = cross_entropy_loss(model(input_ids), input_ids)
            loss.backward()
            optimizer.step()

        if step % 10 == 0:
            # evaluate
            f_ppl, r_ppl = get_perplexities(model, [forget_eval, retain_eval])
            stats = dict(forget=f_ppl, retain=r_ppl)
            print(f"{step:4d}  " + "   ".join(f"{v:10.2f}" for v in stats.values()))

            if r_ppl > _stop_unlearning_at_ppl:
                break

    # for name, param in circuit.items():
    #     print((param != 0).sum().item(), name)
    del circuit

    # ! prepare for relearning
    # merge retain_lora weights
    model = peft_model.merge_and_unload()

    # add relearning lora
    lora_config = LoraConfig(r=8, target_modules="all-linear")
    peft_model = get_peft_model(model, lora_config, adapter_name="relearning_lora")
    model = peft_model.model

    optimizer = pt.optim.Adam(model.parameters(), lr=relearn_lr, betas=(0.9, 0.999))

    # ! relearning loop
    print("relearning...")
    for step in range(1, 1 + relearning_steps):
        # standard forward, backward, and update
        model.train()
        optimizer.zero_grad(set_to_none=True)
        forget_input_ids = get_batch(forget_iter, 16)
        retain_input_ids = get_batch(retain_iter, 16)
        loss_forget = cross_entropy_loss(model(forget_input_ids), forget_input_ids)
        loss_retain = cross_entropy_loss(model(retain_input_ids), retain_input_ids)
        (loss_forget + loss_retain).backward()
        optimizer.step()

        if step % 10 == 0:
            # evaluate
            f_ppl, r_ppl = get_perplexities(model, [forget_eval, retain_eval])
            stats = dict(forget=f_ppl, retain=r_ppl)
            print(f"{step:4d}  " + "   ".join(f"{v:10.2f}" for v in stats.values()))

# %%
unlearn_and_relearn(retain_lr=0, quantile=0.95)

# %% takeaways: (for quantile=0.95, may want to verify for higher)
# circuit_name:
# forget_linear_correct_logit > forget_linear_cross_entropy

# criterion:
# retain_*_correct_logit >> retain_*_cross_entropy
# retain_absolu_cross_entropy > retain_square_cross_entropy > retain_linear_cross_entropy
# 1/R >> F/R  ? (but this needs more thorough verification)

# !
# wait! the thresh is just broken! it's zero! so maybe the previous results are bad too!
# ! no, it's not numerical, it's the fuckin embed_tokens!!!
# calculating threshold:
# per model is kinda the same as per parameter, but per model allows more extreme quantiles
# %%

criterion = "c('retain_absolu_correct_logit').abs() * (-1)"
scores_dict = kinda_safe_eval(criterion)
# %%
for n, scores in scores_dict.items():
    print(n, (scores == 0).sum().item())

# %%
