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
    # note: too small forget_lr with bfloat16, can cause updates to become 0 probably due to numerical errors
    forget_lr=1e-1,
    retain_lr=0,  # 4e-4,
    relearn_lr=1e-3,
    unlearning_steps=50,
    relearning_steps=30,
    _stop_unlearning_at_ppl=28,
):
    set_seeds(42)

    # ! calculate one threshold for the whole model
    scores_dict = kinda_safe_eval(criterion)
    scores_flat = pt.cat([scores.flatten() for scores in scores_dict.values()])
    k = int(scores_flat.numel() * quantile)
    threshold = scores_flat.kthvalue(k).values
    print(f"{threshold=:.2e}")
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

    # prepare data iterators
    forget_iter = looping_iter(forget_set["train"])
    retain_iter = looping_iter(retain_set["train"])

    # Adam seems to need to warmup period - although theoretically at start it should be stronger
    # optimizer = pt.optim.Adam(model.parameters(), lr=retain_lr, betas=(0.9, 0.999))
    optimizer = pt.optim.SGD(model.parameters(), lr=retain_lr)

    # ! unlearning loop
    print("step      forget       retain")
    print("unlearning...")
    last_forget_ppl = 0
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

            # # stop if forget_ppl is going down
            # if f_ppl < last_forget_ppl:
            #     break
            # last_forget_ppl = f_ppl

            # stop if retain_ppl is too high
            if r_ppl > _stop_unlearning_at_ppl:
                print(f"Stopping unlearning due to high retain perplexity {r_ppl:.2f}")
                break

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

# %% experiments
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
unlearn_and_relearn(criterion='c("retain_absolu_correct_logit").abs() ** (-1)')
unlearn_and_relearn(criterion='c("forget_absolu_correct_logit").abs() / c("retain_absolu_correct_logit").abs()') # broken
unlearn_and_relearn(criterion='(c("forget_absolu_correct_logit").abs() + 0.1) / c("retain_absolu_correct_logit").abs()') 

# todo look at module types separately

# %% best so far
# todo also could try higher quantile rather than retraining
unlearn_and_relearn(unlearning_steps=600, retain_lr=1e-2, forget_lr=3e-2, criterion='(c("forget_absolu_correct_logit").abs() + 0.1) / c("retain_absolu_correct_logit").abs()')

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
