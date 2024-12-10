# %%
import logging
from datetime import datetime
from types import SimpleNamespace

import optuna
import torch as pt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.model_operations import *
from utils.training import MockTrial, cross_entropy_loss, eval_, loss_fns, set_seeds

config = SimpleNamespace(
    # Model/data configs
    model_id="EleutherAI/pythia-14m",
    forget_set_name="python",
    # Training constants
    unlearn_steps=100,
    batch_size=16,
    # Relearning params
    relearn_steps=50,
    eval_batch_size=16,
    relearn_lr=1e-4,
    relearn_lora_conf=dict(target_modules="all-linear"),
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
retain_batches = CachedBatches(retain_set["train"], config.batch_size)
forget_batches = CachedBatches(forget_set["train"], config.batch_size)
retain_val_batches = CachedBatches(retain_set["validation"], config.eval_batch_size)
forget_val_batches = CachedBatches(forget_set["validation"], config.eval_batch_size)
r_eval_batch = next(retain_val_batches.fresh_iterator())
f_eval_batch = next(forget_val_batches.fresh_iterator())

base_model = AutoModelForCausalLM.from_pretrained(config.model_id)
init_forget = eval_loss(base_model, f_eval_batch)
init_retain = eval_loss(base_model, r_eval_batch)
logging.info(f"init forget: {init_forget:6.2f}    init retain: {init_retain:6.2f}")
del base_model

_circuit_dir = repo_root() / "circuits" / config.model_id.replace("/", "_")
_circuit_name = f"{config.forget_set_name}_correct_logit.pt"
circuit = pt.load(_circuit_dir / _circuit_name, weights_only=True)


# %%
def objective(trial):
    # ! parameters
    forget_thresh = trial.suggest_float("forget_thresh", 0.0001, 0.1, log=True)
    retain_thresh = trial.suggest_float("retain_thresh", 0.0001, 0.1, log=True)
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.00001, 0.01, log=True)

    # prepare data iterators
    retain_iter = retain_batches.fresh_iterator()
    # load model - for speed we could also do: model = deepcopy(base_model)
    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    # get params to intervene on and initialize disruption scores
    target_modules = ["dense_4h_to_h", "dense"]
    for p in model.parameters():
        p.requires_grad = False
    interven_params = []
    for name, p in model.named_parameters():
        if any(f"{m}.weight" in name for m in target_modules):
            interven_params.append(p)
            # initialize to_forget
            p.to_forget = circuit[name]
            # require grad
            p.requires_grad = True

    # ! unlearning loop
    model.zero_grad(set_to_none=True)
    logging.info("step      base_f      base_r")
    for step in range(1, 1 + config.unlearn_steps):
        model.train()
        r_input_ids = next(retain_iter)

        # ! unlearn on the base model
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()
        for p in interven_params:
            _same_sign = p.grad.sign() == p.to_forget.sign()
            _retain_big = p.grad.abs() > retain_thresh
            _forget_big = p.to_forget.abs() > forget_thresh
            mask = _same_sign.logical_and(_retain_big).logical_and(_forget_big)
            p.data -= mask * unlearning_rate * p.to_forget

        # ! eval current loss
        if step % 10 == 0:
            eval_(model, f_eval_batch, r_eval_batch, init_retain, step)

    # ! eval relearning
    model_copy = deepcopy(model)
    forget_losses = relearn(model_copy, config, retain_val_batches, forget_val_batches)
    # use min rather than last, in case it anomalously increases
    forget_loss = min(forget_losses)

    return forget_loss


# %%
# objective(
#     MockTrial(
#         dict(
#             unlearning_rate=0.00005,
#             retain_thresh=0.001,
#             forget_thresh=0.1,
#         )
#     )
# )

# %%

study_name = f"small,{config.forget_set_name},no_abs"
if __name__ == "__main__":
    assert is_repo_clean()
    study = optuna.create_study(
        study_name=study_name,
        storage=get_storage(),
        direction="maximize",
        # load_if_exists=True,
    )
    save_script_and_attach_logger(__file__, study.study_name)
    study.set_metric_names(["forget_loss"])
    study.set_user_attr("commit_hash", commit_hash())
    for k, v in config.__dict__.items():
        study.set_user_attr(k, v)
    study.optimize(objective, n_trials=300)