# %%
import logging
from types import SimpleNamespace

import optuna
import torch as pt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import dataset_loaders
from utils.git import add_tag_to_current_commit, commit_hash, is_repo_clean
from utils.model_operations import *
from utils.training import MockTrial, loss_fns, save_script_and_attach_logger, set_seeds

config = SimpleNamespace(
    # Model/data configs
    model_id="EleutherAI/pythia-14m",
    # model_id="EleutherAI/pythia-70m",
    # model_id="HuggingFaceTB/SmolLM-135M",
    # forget_set_name="python",
    forget_set_name="oscar_pl",
    lora_config=dict(
        r=4,
        lora_dropout=0.1,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],        # target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],  # fmt: skip
    ),
    # Training constants
    # unlearn_steps=100,
    batch_size=16,
    eval_batch_size=32,
    # Relearning params
    relearn_steps=100,
    relearn_lr=3e-4,
    relearn_batch_size=16,
    relearn_lora_conf=dict(r=1, target_modules=["dense_h_to_4h"], lora_dropout=0.1),
    # relearn_lora_conf=dict(r=1, target_modules=["up_proj"], lora_dropout=0.1),
    # Default tunable params
    disruption_score_decay=0.95,
    disruption_score_warmup=10,
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
    quantile = trial.suggest_float("quantile", 1e-4, 1e-1, log=True)
    adv_lora_lr = trial.suggest_float("adv_lora_lr", 3e-4, 3e-3, log=True)
    ret_lora_lr = trial.suggest_float("ret_lora_lr", 1e-5, 1e-3, log=True)
    unlearn_lr = trial.suggest_float("unlearn_lr", 1e-2, 1e2, log=True)
    forget_amp = 1  # trial.suggest_float("forget_amp", 0.5, 1.5)
    retain_amp = 1.5  # trial.suggest_float("retain_amp", 1, 2)
    # unl_loss_fn = loss_fns[trial.suggest_categorical("unl_loss_fn", loss_fns.keys())]
    unl_loss_fn = loss_fns["clipped_correct_logit"]
    ret_loss_fn = loss_fns["cross_entropy"]
    # disrupt_loss_fn = loss_fns[trial.suggest_categorical("disrupt_loss_fn", loss_fns.keys())]  # fmt: skip
    unlearn_steps = trial.suggest_int("unlearn_steps", 80, 100, step=10)

    mask_fn = lambda param: param.disruption_score / param.grad.abs() ** forget_amp
    trial.set_user_attr("lora_defeaten", False)
    trial.set_user_attr("retain_broken", False)

    set_seeds(42)  # note: something is still undeterministic!
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
    for step in range(1, 1 + unlearn_steps):
        model.train()
        f_input_ids = get_batch(forget_iter, config.batch_size)
        r_input_ids = get_batch(retain_iter, config.batch_size)

        # ! retain with helper lora
        peft_model.set_adapter(["ret_lora"])
        only_grad_on(model, interven_params + ret_lora_params)
        model.zero_grad(set_to_none=True)
        loss = ret_loss_fn(model(r_input_ids), r_input_ids)
        loss.backward()
        # ! update disruption scores
        for param in interven_params:
            param.disruption_score *= config.disruption_score_decay
            param.disruption_score += param.grad.abs() ** retain_amp
        if step <= config.disruption_score_warmup:
            continue
        # model.zero_grad(set_to_none=True)
        # loss = ret_loss_fn(model(r_input_ids), r_input_ids)
        # loss.backward()
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
        # ! normalize gradients
        grad_norm = sum(p.grad.norm() ** 2 for p in interven_params) ** 0.5
        for p in interven_params:
            p.grad /= grad_norm
        # logging.info(f"gnorm: {grad_norm:6.2f}")
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

            # prune if retain performance broken
            if res["base_retain"] > init_retain + 0.1:
                logging.error("Retain performance broken")
                trial.set_user_attr("retain_broken", True)
                raise optuna.TrialPruned()
            # prune if base forget loss doesn't improve
            if step >= 30 and res["base_forget"] < init_forget + 0.05:
                logging.info("Forget loss stalled")
                raise optuna.TrialPruned()
            # prune if adversarial lora is defeaten
            if res["adv_forget"] > 10:
                logging.error("Adversarial LoRA defeaten")
                trial.set_user_attr("lora_defeaten", True)
                raise optuna.TrialPruned()
            # prune if nan
            if any(pt.isnan(v) for v in res.values()):
                logging.error("NaN in eval results")
                raise optuna.TrialPruned()

    # %
    # ! final bigger eval relearning
    collapsed_model = copy_model_and_collapse_loras(peft_model)
    forget_loss = relearn(collapsed_model, config, forget_set, retain_set)
    return forget_loss


# %%
assert is_repo_clean()
study = optuna.create_study(
    study_name="pythia-14m,oscar_pl,normalize_grads",
    storage="sqlite:///../results/aa_hyperparam_robustness.sqlite3",
    direction="maximize",
    # load_if_exists=True,  # This allows resuming existing studies
)
# add_tag_to_current_commit(study.study_name)
save_script_and_attach_logger(__file__, study.study_name)
study.set_metric_names(["forget_loss"])
study.set_user_attr("commit_hash", commit_hash())
for k, v in config.__dict__.items():
    study.set_user_attr(k, v)
study.optimize(objective, n_trials=1000)

# # %%
# study = optuna.load_study(
#     study_name="pythia-14m/oscar_pl/wikitext",
#     storage="sqlite:///../results/aa_hyperparam_robustness.sqlite3",
# )
# trials = study.get_trials()
# # uninterrupted_trials = [
# #     t for t in trials if t.user_attrs.get("steps", 0) == config.unlearn_steps
# # ]
# finished_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

# # worst_to_best = sorted(uninterrupted_trials, key=lambda t: t.value)
# worst_to_best = sorted(finished_trials, key=lambda t: t.value)

# # %%
# # rerun the best trial, with more steps
# config.unlearn_steps = 30
# config.relearn_steps = 100

# best = worst_to_best[-1]

# best_params = deepcopy(best.params)
# # config.lora_config["target_modules"] = ["dense"]
# # config.lora_config["target_modules"] = ["query_key_value"]
# best_params["unlearn_lr"] *= 3000
# # best_params["adv_lora_lr"] = 0
# # best_params["unlearn_lr"] /= 1.5
# # best_params["ret_lora_lr"] /= 2

# result = objective(MockTrial(best_params))
# logging.info(f"Final result: {result}")

# # %%
# best.params
