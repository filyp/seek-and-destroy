# %%
import logging
from types import SimpleNamespace

import optuna
import torch as pt
import wandb
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.dataloading import dataset_loaders
from utils.git import add_tag_to_current_commit, commit_hash, is_repo_clean
from utils.model_operations import *
from utils.training import loss_fns, set_seeds

config = SimpleNamespace(
    # Model/data configs
    # model_id="EleutherAI/pythia-14m",
    model_id="EleutherAI/pythia-70m",
    forget_set_name="python",
    lora_config=dict(
        r=4,
        lora_dropout=0.1,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    ),
    # Training constants
    unlearn_steps=100,
    batch_size=16,
    eval_batch_size=32,
    # Relearning params
    relearn_steps=50,
    relearn_lr=3e-4,
    relearn_batch_size=16,
    relearn_lora_conf=dict(r=1, target_modules=["dense_h_to_4h"]),
    # Default tunable params
    disruption_score_decay=0.95,
    disruption_score_warmup=20,
)

pt.set_default_device("cuda")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# load datasets
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
retain_set = dataset_loaders["wikitext"](tokenizer)
forget_set = dataset_loaders[config.forget_set_name](tokenizer)

f_eval_batch = get_batch(iter(forget_set["validation"]), config.eval_batch_size)
r_eval_batch = get_batch(iter(retain_set["validation"]), config.eval_batch_size)

model = AutoModelForCausalLM.from_pretrained(config.model_id)
init_forget = eval_loss(model, f_eval_batch)
init_retain = eval_loss(model, r_eval_batch)
logging.info(f"init forget: {init_forget:6.2f}    init retain: {init_retain:6.2f}")


# %%
def objective(trial):
    # ! parameters
    quantile = trial.suggest_float("quantile", 100e-6, 50e-3, log=True)
    unlearn_lr = trial.suggest_float("unlearn_lr", 5e-6, 200e-6, log=True)
    adv_lora_lr = trial.suggest_float("adv_lora_lr", 10e-6, 1e-3, log=True)
    ret_lora_lr = trial.suggest_float("ret_lora_lr", 10e-6, 1e-3, log=True)
    unl_loss_fn = loss_fns[trial.suggest_categorical("unl_loss_fn", loss_fns.keys())]
    ret_loss_fn = loss_fns[trial.suggest_categorical("ret_loss_fn", loss_fns.keys())]
    disrupt_loss_fn = loss_fns[trial.suggest_categorical("disrupt_loss_fn", loss_fns.keys())]  # fmt: skip
    forget_amp = trial.suggest_float("forget_amp", 0.5, 1.5)
    retain_amp = trial.suggest_float("retain_amp", 1, 2)
    mask_fn = lambda param: param.disruption_score / param.grad.abs() ** forget_amp

    trial.set_user_attr("lora_defeaten", False)
    trial.set_user_attr("retain_broken", False)

    # save_script_and_attach_logger(__file__)
    set_seeds(42)
    # note: smth is still undeterministic!
    # prepare data iterators
    forget_iter = looping_iter(forget_set["train"])
    retain_iter = looping_iter(retain_set["train"])

    # load model
    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    # add loras
    lora_config = LoraConfig(**config.lora_config)
    peft_model = get_peft_model(model, lora_config, adapter_name="adv_lora", mixed=True)
    model = peft_model.model
    peft_model.add_adapter("ret_lora", lora_config)

    interven_params = [p for n, p in model.named_parameters() if ".base_layer.weight" in n]  # fmt: skip
    adv_lora_params = [p for n, p in model.named_parameters() if ".adv_lora." in n]
    ret_lora_params = [p for n, p in model.named_parameters() if ".ret_lora." in n]

    # initialize optimizers
    base_optimizer = pt.optim.SGD(interven_params, lr=unlearn_lr)
    adv_optimizer = pt.optim.Adam(adv_lora_params, lr=adv_lora_lr)
    ret_optimizer = pt.optim.Adam(ret_lora_params, lr=ret_lora_lr)

    # initialize disruption scores
    for param in interven_params:
        param.disruption_score = pt.zeros_like(param)

    # %
    # ! unlearning loop
    logging.info("step      base_f      base_r       adv_f      adv_r")
    for step in range(1, 1 + config.unlearn_steps):
        model.train()
        f_input_ids = get_batch(forget_iter, config.batch_size)
        r_input_ids = get_batch(retain_iter, config.batch_size)

        # ! retain with helper lora
        peft_model.set_adapter(["ret_lora"])
        only_grad_on(model, interven_params + ret_lora_params)
        model.zero_grad(set_to_none=True)
        loss = disrupt_loss_fn(model(r_input_ids), r_input_ids)
        loss.backward()
        # ! update disruption scores
        for param in interven_params:
            param.disruption_score *= config.disruption_score_decay
            param.disruption_score += param.grad.abs() ** retain_amp
        if step <= config.disruption_score_warmup:
            continue
        # ! actual relearning step
        # todo: after some evals, definitely remove this complication
        model.zero_grad(set_to_none=True)
        loss = disrupt_loss_fn(model(r_input_ids), r_input_ids)
        loss.backward()
        ret_optimizer.step()

        # ! unlearn on the base model
        peft_model.set_adapter(["ret_lora", "adv_lora"])
        only_grad_on(model, interven_params)
        model.zero_grad(set_to_none=True)
        loss = unl_loss_fn(model(f_input_ids), f_input_ids)
        loss.backward()
        # ! get threshold
        final_scores = [mask_fn(p) for p in interven_params]
        threshold = get_threshold(quantile, final_scores)
        # ! apply mask
        for param in interven_params:
            mask = mask_fn(param) < threshold
            param.grad *= mask
        base_optimizer.step()

        # ! relearn with adversarial lora
        peft_model.set_adapter(["ret_lora", "adv_lora"])
        only_grad_on(model, adv_lora_params)
        model.zero_grad(set_to_none=True)
        loss = cross_entropy_loss(model(f_input_ids), f_input_ids)
        loss.backward()
        adv_optimizer.step()

        # ! eval
        if step % 10 == 0:
            res = {}
            peft_model.set_adapter(["ret_lora"])
            res["base_forget"] = eval_loss(model, f_eval_batch)
            res["base_retain"] = eval_loss(model, r_eval_batch)
            peft_model.set_adapter(["ret_lora", "adv_lora"])
            res["adv_forget"] = eval_loss(model, f_eval_batch)
            res["adv_retain"] = eval_loss(model, r_eval_batch)

            logging.info(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))
            if wandb.run:
                wandb.log(res, step=step)

            if res["base_retain"] > init_retain + 0.1:
                logging.error("Retain performance broken")
                trial.set_user_attr("retain_broken", True)
                # raise optuna.TrialPruned()
                return init_forget - 0.3
            if res["adv_forget"] > 10:
                logging.error("Adversarial LoRA defeaten")
                trial.set_user_attr("lora_defeaten", True)
                break
            # todo early stopping on bad performance?

        # # ! eval relearning
        # if step % 50 == 0:
        #     collapsed_model = copy_model_and_collapse_loras(peft_model)
        #     relearn(collapsed_model, config, forget_set, retain_set)

    # %
    # ! final bigger eval relearning
    collapsed_model = copy_model_and_collapse_loras(peft_model)
    forget_loss = relearn(collapsed_model, config, forget_set, retain_set)
    return forget_loss


# %%
study = optuna.create_study(
    study_name="9param_100/50s_noprune_Smol135_wikitext",
    storage="sqlite:///../results/aa_hyperparam_robustness.sqlite3",
    direction="maximize",
    # load_if_exists=True,  # This allows resuming existing studies
)
assert is_repo_clean()
add_tag_to_current_commit(study.study_name)
study.set_metric_names(["forget_loss"])
study.set_user_attr("commit_hash", commit_hash())
for k, v in config.__dict__.items():
    study.set_user_attr(k, v)
study.optimize(objective, n_trials=10000)

# %%
# test_params = {
#     "quantile": 0.005,
#     "unlearn_lr": 30e-6,
#     "adv_lora_lr": 3e-4,
#     "ret_lora_lr": 3e-4,
#     "unl_loss_fn": "clipped_correct_logit",
#     "ret_loss_fn": "cross_entropy",
#     "forget_amp": 1.0,
#     "retain_amp": 1.7,
# }
# config.unlearn_steps = 300

# result = objective(MockTrial(test_params))
# logging.info(f"Final result: {result}")
