# https://arxiv.org/pdf/2211.14946
# https://github.com/Breakend/SelfDestructingModels/blob/main/train.py

# https://arxiv.org/pdf/2408.00761
# > MLAC-AR. Meta-Learned Adversarial Censoring (MLAC) [16] was originally proposed to prevent BERT-style models from learning binary classification for gender bias data. Since the approach is not immediately applicable to LLMs, we extend MLAC in a variant we call autoregressive MLAC (MLAC-AR). Since MLAC in its original formulation calls for “task-blocking” via negating the adversary’s loss during the inner loop of meta-learning, we implement this by negating the crossentropy loss of an LLM fine-tuning adversary. However, we found that this approach diverges in performance across a variety of hyperparameters, and opted to further improve performance of the MLAC-AR baseline by clamping the maximum cross-entropy loss at the value of the maximum entropy of the output vocabulary distribution, log(vocab_size). We show results in Table 12, finding that MLAC-AR does not maintain sufficient benign capabilities performance nor uniform tamper-resistance across weaponization domains.

import logging

import torch as pt
from transformers import AutoModelForCausalLM

from utils.loss_fns import *
from utils.training import eval_


def unlearning_func(
    trial, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    # ! parameters
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.0001, 0.01, log=True)
    retaining_rate = trial.suggest_float("retaining_rate", 0.00001, 0.001, log=True)
    logging.info(f"trial {trial.number} - {trial.params}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    unl_optimizer = pt.optim.SGD(model.parameters(), lr=unlearning_rate)
    ret_optimizer = pt.optim.SGD(model.parameters(), lr=retaining_rate)

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)
    for step in range(1, 1 + config.unlearn_steps):
        model.train()
        r_input_ids = next(retain_iter)
        f_input_ids = next(forget_iter)

        # ! unlearn
        model.zero_grad(set_to_none=True)
        output = model(f_input_ids)
        loss = neg_entropy_loss(output, f_input_ids)
        loss.backward()
        unl_optimizer.step()

        # ! retain
        model.zero_grad(set_to_none=True)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()
        ret_optimizer.step()

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, step)

    return model
