# %%
from _common_init import *


# %%
def objective(trial):
    # ! parameters
    unlearning_rate = trial.suggest_float("unlearning_rate", 0.0001, 0.01, log=True)
    retaining_rate = trial.suggest_float("retaining_rate", 0.00001, 0.001, log=True)
    # ... any other params here

    # prepare data iterators
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)
    # load model
    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    # ... any custom setup goes here
    unl_optimizer = pt.optim.SGD(model.parameters(), lr=unlearning_rate)
    ret_optimizer = pt.optim.SGD(model.parameters(), lr=retaining_rate)

    # ! unlearning loop
    res = {}
    logging.info("step      base_f      base_r")
    for step in range(1, 1 + config.unlearn_steps):
        model.train()
        r_input_ids = next(retain_iter)
        f_input_ids = next(forget_iter)

        if res.get("retain_loss_ok", True):
            # ... change this block to the actual unlearning method
            # ! unlearn
            model.zero_grad(set_to_none=True)
            output = model(f_input_ids)
            loss = -cross_entropy_loss(output, f_input_ids)
            loss.backward()
            unl_optimizer.step()

        # ! retain
        # ... change this block to the actual retaining method
        model.zero_grad(set_to_none=True)
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        loss.backward()
        ret_optimizer.step()

        # ... keep the rest the same for consistency!
        # ! eval current loss
        if step % 10 == 0:
            res = eval_(model, f_eval_batch, r_eval_batch, init_retain, step)

    # ! eval relearning
    model_copy = deepcopy(model)
    forget_losses = relearn(model_copy, config, retain_val_batches, forget_val_batches)
    # use min rather than last, in case it anomalously increases
    forget_loss = min(forget_losses)
    return forget_loss


# %%
run_study(objective, config, __file__, "study_name")
