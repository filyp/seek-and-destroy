# %%
import logging

import optuna
import torch as pt
import wandb
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *
from utils_dataloading import *
from utils_important import *

forget_set_name = "python"
model_id = "EleutherAI/pythia-14m"
target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
# model_id = "HuggingFaceTB/SmolLM-135M"
# target_modules = ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"]  # fmt: skip

pt.set_default_device("cuda")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
retain_set = dataset_loaders["en"](tokenizer)
forget_set = dataset_loaders[forget_set_name](tokenizer)

f_eval = get_batch(iter(forget_set["validation"]), 32)
r_eval = get_batch(iter(retain_set["validation"]), 32)

model = AutoModelForCausalLM.from_pretrained(model_id)
init_forget = eval_loss(model, f_eval)
init_retain = eval_loss(model, r_eval)
logging.info(f"init forget: {init_forget:6.2f}    init retain: {init_retain:6.2f}")


# %%
def objective(trial):
    # ! parameters
    quantile = trial.suggest_float("quantile", 0.0001, 1, log=True)
    unlearn_lr = trial.suggest_float("unlearn_lr", 1e-5, 1e-2, log=True)
    adv_lora_lr = 3e-4
    ret_lora_lr = 3e-4  # tip: first look for good params with this set to 0
    disruption_score_decay = 0.95
    unlearn_steps = 300
    relearn_lr = 3e-4
    forget_amp = trial.suggest_float("forget_amp", 0, 2)
    mask_fn = lambda param: param.disruption_score / param.grad.abs() ** forget_amp
    disruption_score_warmup = 20

    trial.set_user_attr("lora_defeated", False)
    trial.set_user_attr("retain_broken", False)

    save_script_and_attach_logger(__file__)
    set_seeds(42)
    # todo: smth is still undeterministic!
    # prepare data iterators
    forget_iter = looping_iter(forget_set["train"])
    retain_iter = looping_iter(retain_set["train"])

    # load model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # add loras
    adv_lora_conf = LoraConfig(r=1, target_modules=target_modules, lora_dropout=0.1)
    peft_model = get_peft_model(model, adv_lora_conf, adapter_name="adv_lora", mixed=True)  # fmt: skip
    model = peft_model.model
    ret_lora_conf = LoraConfig(r=1, target_modules=target_modules, lora_dropout=0.1)
    peft_model.add_adapter("ret_lora", ret_lora_conf)

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
    for step in range(1, 1 + unlearn_steps):
        model.train()
        f_input_ids = get_batch(forget_iter, 16)
        r_input_ids = get_batch(retain_iter, 16)

        # ! retain with helper lora
        peft_model.set_adapter(["ret_lora"])
        only_grad_on(model, interven_params + ret_lora_params)
        model.zero_grad(set_to_none=True)
        loss = cross_entropy_loss(model(r_input_ids), r_input_ids)
        loss.backward()
        # ! update disruption scores
        for param in interven_params:
            param.disruption_score *= disruption_score_decay
            param.disruption_score += param.grad**2
        if step <= disruption_score_warmup:
            continue
        ret_optimizer.step()

        # ! unlearn on the base model
        peft_model.set_adapter(["ret_lora", "adv_lora"])
        only_grad_on(model, interven_params)
        model.zero_grad(set_to_none=True)
        # loss_fn = loss_fns[trial.suggest_categorical("loss_fn", loss_fns.keys()])
        loss_fn = clipped_correct_logit_loss
        loss = loss_fn(model(f_input_ids), f_input_ids)
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
            res["base_forget"] = eval_loss(model, f_eval)
            res["base_retain"] = eval_loss(model, r_eval)
            peft_model.set_adapter(["ret_lora", "adv_lora"])
            res["adv_forget"] = eval_loss(model, f_eval)
            res["adv_retain"] = eval_loss(model, r_eval)

            logging.info(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))
            if wandb.run:
                wandb.log(res, step=step)

            if res["base_retain"] > init_retain + 0.1:
                logging.error("Retain performance broken")
                # raise optuna.TrialPruned()
                trial.set_user_attr("retain_broken", True)
                return init_forget - 0.1
            if res["adv_forget"] > 10:
                logging.error("Adversarial LoRA defeated")
                trial.set_user_attr("lora_defeated", True)
                break

        # # ! eval relearning
        # if step % 50 == 0:
        #     collapsed_model = copy_model_and_collapse_loras(peft_model)
        #     relearn(collapsed_model, relearn_lr, 20, forget_set, retain_set)

    # %
    # ! final bigger eval relearning
    collapsed_model = copy_model_and_collapse_loras(peft_model)
    final_forget_loss = relearn(collapsed_model, relearn_lr, 100, forget_set, retain_set)
    return final_forget_loss


# %%
study = optuna.create_study(
    study_name="clippedlogit_noprune",
    storage="sqlite:///db.sqlite3",
    direction="maximize",
    # load_if_exists=True  # This allows resuming existing studies
)
study.set_metric_names(["forget_loss"])
study.set_user_attr("forget_set", forget_set_name)
study.set_user_attr("model_id", model_id)
study.set_user_attr("target_modules", target_modules)
study.optimize(objective, n_trials=1000)

# %%
