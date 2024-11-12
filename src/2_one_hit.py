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
    quantile=0.9,  # between 0 and 1
    circuit_name="forget_linear_correct_logit",
    criterion='c("retain_absolu_correct_logit").abs() * (-1)',
    module_type="up_proj",
    # note: too small forget_lr with bfloat16, can cause updates to become 0 probably due to numerical errors
    forget_lr=1e-1,
    retain_lr=0,  # 4e-4,
    relearn_lr=1e-3,
    unlearn_steps=50,
    relearn_steps=30,
):
    set_seeds(42)
    # prepare data iterators
    forget_iter = looping_iter(forget_set["train"])
    retain_iter = looping_iter(retain_set["train"])

    # ! calculate one threshold for the whole model
    scores_dict = kinda_safe_eval(criterion)
    scores_dict = {k: v for k, v in scores_dict.items() if module_type in k}
    scores_flat = pt.cat([scores.flatten() for scores in scores_dict.values()])
    k = int(scores_flat.numel() * quantile)
    threshold = scores_flat.kthvalue(k).values
    # print(f"{threshold=:.2e}")
    del scores_flat

    # ! load circuit and sparsify
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

    # Adam seems to need to warmup period - although theoretically at start it should be stronger
    optimizer = pt.optim.SGD(model.parameters(), lr=retain_lr)

    # ! unlearning loop
    # print("step      forget       retain")
    for step in range(1, 1 + unlearn_steps):
        # break the circuit a bit
        for name, param in model.named_parameters():
            name = name.replace(".base_layer", "")
            if "lora" in name or name == "model.embed_tokens.weight":
                continue
            if module_type in name:
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

    del circuit

    # ! prepare for relearning
    # merge retain_lora weights
    model = peft_model.merge_and_unload()

    # add relearning lora
    lora_config = LoraConfig(r=1, target_modules="all-linear")
    peft_model = get_peft_model(model, lora_config, adapter_name="relearning_lora")
    model = peft_model.model

    optimizer = pt.optim.Adam(model.parameters(), lr=relearn_lr, betas=(0.9, 0.999))

    # ! relearning loop
    print("relearning...")
    for step in range(1, 1 + relearn_steps):
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


# fmt: off
# %%
# unlearn_and_relearn(quantile=0.93, forget_lr=6e-1, retain_lr=1e-2, criterion='(c("forget_absolu_correct_logit").abs() + 0.1) / c("retain_absolu_correct_logit").abs()')
# 27.21 / 575 (at 30) / 32.66 (at 200)
# unlearn_and_relearn(quantile=0.8, forget_lr=4e-2, retain_lr=1e-2, unlearn_steps=400, relearn_steps=200, criterion='(c("forget_absolu_correct_logit").abs() + 0.1) / c("retain_absolu_correct_logit").abs()')
# 27.89 / 660 (at 30 r1)
unlearn_and_relearn(quantile=0.9, forget_lr=3e-1, retain_lr=0, unlearn_steps=50, relearn_steps=200, criterion='(c("forget_absolu_correct_logit").abs() + 0.1) / c("retain_absolu_correct_logit").abs()')

# %% experiments
# (next 4 blocks are done with module_type="")
# % compare circuit_name
# * circuit_name: forget_linear_correct_logit > forget_linear_cross_entropy
# unlearn_and_relearn(circuit_name="forget_linear_correct_logit", quantile=0.9)
# unlearn_and_relearn(circuit_name="forget_linear_cross_entropy", quantile=0.9)

# % compare criterion
# * absolute > square > linear (for correct_logit)
# unlearn_and_relearn(criterion='c("retain_absolu_correct_logit").abs() * (-1)')
# unlearn_and_relearn(criterion='c("retain_square_correct_logit").abs() * (-1)')
# unlearn_and_relearn(criterion='c("retain_linear_correct_logit").abs() * (-1)')

# % compare criterion
# * correct_logit > cross_entropy (but it's more nuanced)
# unlearn_and_relearn(criterion='c("retain_absolu_correct_logit").abs() * (-1)')
# unlearn_and_relearn(criterion='c("retain_absolu_cross_entropy").abs() * (-1)', quantile=0.92)

# % comparre criterion
# * (F+a)/R >> 1/R >> F/R
# a needs to be tuned well - it has a huuge impact
# unlearn_and_relearn(criterion='c("retain_absolu_correct_logit").abs() ** (-1)')
# unlearn_and_relearn(criterion='c("forget_absolu_correct_logit").abs() / c("retain_absolu_correct_logit").abs()')  # broken
# unlearn_and_relearn(criterion='(c("forget_absolu_correct_logit").abs() + 0.1) / c("retain_absolu_correct_logit").abs()')

# % compare module_type
# * up_proj is the best, even better than intervening on all!
#     maybe only down_proj could be used, but it's much worse
#     layernorms and attention do nothing
#     apart from v_proj.weight and o_proj.weight, but they are meh
#     just intervening on up_proj looks best
# # common = dict(forget_lr=3e-1, criterion='(c("forget_absolu_correct_logit").abs() + 0.1) / c("retain_absolu_correct_logit").abs()')
# common = dict(forget_lr=3e-1, criterion='c("retain_absolu_correct_logit").abs() ** (-1)')
# unlearn_and_relearn(**common, module_type="")
# unlearn_and_relearn(**common, module_type="down_proj")
# unlearn_and_relearn(**common, module_type="up_proj")
# unlearn_and_relearn(**common, module_type="gate_proj")
# unlearn_and_relearn(**common, module_type="input_layernorm")
# unlearn_and_relearn(**common, module_type="post_attention_layernorm")
# unlearn_and_relearn(**common, module_type="q_proj.weight")
# unlearn_and_relearn(**common, module_type="k_proj.weight")
# unlearn_and_relearn(**common, module_type="v_proj.weight")
# unlearn_and_relearn(**common, module_type="o_proj.weight")
# unlearn_and_relearn(**common, module_type="q_proj.bias")
# unlearn_and_relearn(**common, module_type="k_proj.bias")
# unlearn_and_relearn(**common, module_type="v_proj.bias")
# unlearn_and_relearn(**common, module_type="o_proj.bias")

# * practical tip: first experiment with values without retaining

# %% other notes

# * calculating threshold: per model >> per parameter
# also, attacking wider population (lower quantile) doesn't help that much
#     high q vs low q + retaining is very similar

# calibrating the right quantile seems important

# training for too long makes it come back!
#     also even with short training there's a spike and then a comedown
#     maybe because of Adam? - yeah I've never seen this with SGD so far


# ? there must be some precition error interacting
#     forget_lr=1e-1 is different than 2e-1 with twice less steps!
#     after understanding, try to recreate the benefit but intentionally
#         - but is there any benefit?
#     or for now just use high forget_lr to omit this


# %%
