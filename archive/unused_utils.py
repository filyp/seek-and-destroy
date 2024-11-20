from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
from tensordict import TensorDict


def retrain_and_eval(
    model, initial_stats, forget, retain, num_batches=10, b_size=16, lr=0.0003
):
    """takes a model after an intervention and evals it before and after retraining"""
    pre_retrain_diff = get_stats(model, forget, retain) - initial_stats
    print_stats(pre_retrain_diff)

    f_r = zip(forget["relearn"].batch(b_size), retain["relearn"].batch(b_size))
    for forget_batch, retain_batch in islice(f_r, num_batches):
        normal_train_step(model, forget_batch, lr)
        normal_train_step(model, retain_batch, lr)

    post_retrain_diff = get_stats(model, forget, retain) - initial_stats
    print_stats(post_retrain_diff)
    return pre_retrain_diff, post_retrain_diff


def scale_perturbation(model, original_state_dict, scaling_factor):
    for module_name, current_weights in model.state_dict().items():
        original_weights = original_state_dict[module_name]
        # modification need to be done in-place so it's a bit awkward:
        # or I could just use param.data instead
        current_weights -= original_weights
        current_weights *= scaling_factor
        current_weights += original_weights


def normal_train_step(model, batch, lr, loss_sign=1):
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    loss = forward(model, batch)
    loss *= loss_sign
    loss.backward()

    optimizer.step()


def calculate_cutoff(modules, percentile):
    rel_imps = [m.imp["forget"] / m.imp["retain"] for m in modules]
    rel_imps = pt.cat(rel_imps).flatten()
    k = int(len(rel_imps) * (1 - percentile / 100))
    return rel_imps.kthvalue(k).values.item()


def intervene_down_proj(model, mult, cutoff=None):
    for m in [getattr(l.mlp, "down_proj") for l in model.model.layers]:
        rel_imp = m.imp["forget"] / m.imp["retain"]
        m.weight.data[:, rel_imp > cutoff] *= mult


# intervene, based on relative importance
def intervene(model, module_name, mult, cutoff=None):
    for m in [getattr(l.mlp, module_name) for l in model.model.layers]:
        rel_imp = m.imp["forget"] / m.imp["retain"]
        mask = rel_imp > cutoff
        m.weight.data[mask] *= mult


def get_perplexity(model, dataset, num_batches=1, batch_size=32):
    total_loss = 0
    for batch in islice(dataset["validation"].batch(batch_size), num_batches):
        with pt.no_grad():
            total_loss += forward(model, batch)
    return (total_loss / num_batches).exp()


def get_norm_of_weights_change(model, original_state_dict):
    partial_norms = []
    for module_name, current_weights in model.named_parameters():
        original_weights = original_state_dict[module_name]
        norm = (original_weights - current_weights).norm()
        partial_norms.append(norm)
    return pt.tensor(partial_norms).norm()


def get_stats(model, forget_set, retain_set):
    return pt.tensor([
        get_perplexity(model, forget_set),
        get_perplexity(model, retain_set),
        # get_norm_of_weights_change(model, og_model.state_dict()),
    ])


def print_stats(stats):
    print(f"forget: {stats[0]:4.0f}  retain: {stats[1]:5.2f}  ratio: {stats[0] / stats[1]:.0f}")  # fmt: skip


# def forward_and_get_quietness_loss(model, batch):
#     input_ids = pt.cat(batch["input_ids"])
#     out = model(input_ids, output_hidden_states=True)
#     return out.hidden_states[-1].norm(dim=-1).mean()


# def remove_lora(state_dict):
#     keys = list(state_dict.keys())
#     for key in keys:
#         if "lora" in key:
#             del state_dict[key]
#             continue
#         value = state_dict[key]
#         del state_dict[key]
#         new_key = key.replace("base_layer.", "")
#         state_dict[new_key] = value


class DefaultNamespace(SimpleNamespace):
    def __getattr__(self, name):
        # This is called when an attribute doesn't exist
        return pt.tensor(pt.nan)


def load_circuit(circuit_name):
    circ = pt.load(repo_root() / "circuits" / f"{circuit_name}.pt", weights_only=True)
    # this is filled with zero imps, at least for polish
    for name in list(circ.keys()):
        if "embed" in name:
            del circ[name]
    return TensorDict(circ)


def kinda_safe_eval(expr):
    # Create a custom globals dictionary with necessary components
    restricted_globals = dict(safe_globals)
    restricted_globals.update({
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_getattr_": safer_getattr,
        # Add any other necessary functions/variables that your expression needs
        "load_circuit": load_circuit,
    })

    byte_code = compile_restricted(expr, filename="<inline code>", mode="eval")
    return eval(byte_code, restricted_globals)
