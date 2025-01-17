import logging

import torch as pt
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from utils.git_and_reproducibility import repo_root
from utils.loss_fns import loss_fns


def filter_and_normalize_circuit(circuit, target_modules):
    # first filter to keep only the target modules
    circuit = {
        name: param
        for name, param in circuit.items()
        if any(f"{m}.weight" in name for m in target_modules)
    }

    # normalize so that the total norm is the square root of the number of elements
    total_numel = sum(p.numel() for p in circuit.values())
    total_norm = sum(p.norm() ** 2 for p in circuit.values()) ** 0.5
    wanted_total_norm = total_numel**0.5
    for param in circuit.values():
        param *= wanted_total_norm / total_norm
    return circuit


def _get_circuit_dir(config):
    _model_name = config.model_id.replace("/", "_")
    circuit_dir = repo_root() / "circuits" / _model_name / config.forget_set_name
    circuit_dir.mkdir(parents=True, exist_ok=True)
    return circuit_dir


def get_circuit(config, batches, circuit_name):
    circuit_path = _get_circuit_dir(config) / f"{circuit_name}.pt"
    if circuit_path.exists():
        return pt.load(circuit_path, weights_only=True)
    logging.info(f"circuit {circuit_name} not found, creating")

    circuit_type, info = circuit_name.split(",", 1)
    match circuit_type:
        case "normal":
            loss_fn_name = info
            circuit = get_normal_circuit(config, batches, loss_fn_name)
        case "k_dampens_grad":
            circuit = get_circuit_k_dampens_grad(config, batches)
        case "k_dampens_grad_mlp_local":
            circuit = get_circuit_k_dampens_grad_mlp_local(config, batches)
        case "k_dampens_grad_neuron_local":
            circuit = get_circuit_k_dampens_grad_neuron_local(config, batches)
        case "fading_backprop":
            loss_fn_name, scale = info.split(",")
            scale = float(scale)
            circuit = get_circuit_with_fading_backprop(
                config, batches, loss_fn_name, scale
            )
        case "grad_misalign":
            circuit = get_grad_misaligning(config, batches, info)
        case _:
            raise ValueError(f"unknown circuit type {circuit_type}")

    # save circuit
    pt.save(circuit, circuit_path)
    return circuit


def get_normal_circuit(config, batches, loss_fn_name):
    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    loss_fn = loss_fns[loss_fn_name]

    # accumulate grads
    model.zero_grad(set_to_none=True)
    batch_iter = iter(batches)
    for _ in tqdm(range(config.circuit_num_steps)):
        input_ids = next(batch_iter)
        output = model(input_ids, output_hidden_states=True)
        loss = loss_fn(output, input_ids)
        loss.backward()

    return {
        name: param.grad
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def get_circuit_with_fading_backprop(config, batches, loss_fn_name, scale=0.9):
    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    loss_fn = loss_fns[loss_fn_name]

    def scale_grad(module, grad_input, grad_output):
        return (grad_input[0] * scale,)

    for name, module in model.named_modules():
        if name.endswith(".mlp") or name.endswith(".attention"):
            # module._backward_hooks.clear()
            module.register_full_backward_hook(scale_grad)

    # accumulate grads
    model.zero_grad(set_to_none=True)
    batch_iter = iter(batches)
    for _ in tqdm(range(config.circuit_num_steps)):
        input_ids = next(batch_iter)
        loss = loss_fn(model(input_ids), input_ids)
        loss.backward()
    return {name: param.grad for name, param in model.named_parameters()}


def get_grad_misaligning(config, batches, info):
    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    loss_fn = loss_fns["cross_entropy"]
    # don't require grads for the model
    model.requires_grad_(False)
    # this is needed to backpropagate despite not requiring grads
    model.gpt_neox.embed_in.requires_grad_(True)

    def save_misaligning_grad(module, grad_input, grad_output):
        alignment = grad_input[0].clone()
        activations = module.input_activations
        assert alignment.shape == activations.shape
        if info == "only_pos":
            alignment[alignment < 0] = 0
            alignment[activations < 0] = 0

        # normalize by grad norm, so that we depend on it linearly, not quadratically
        grad_norm = pt.norm(grad_output[0], dim=-1, keepdim=True)
        alignment = alignment / (grad_norm + 1e-10)
        misaligning = pt.einsum("bth,btr->rh", alignment, grad_output[0])
        module.weight.misaligning += misaligning

    def save_input_activation_hook(module, args, output):
        module.input_activations = args[0]

    for name, module in model.named_modules():
        if "mlp.dense_4h_to_h" in name:
            module.register_full_backward_hook(save_misaligning_grad)
            module.weight.misaligning = pt.zeros_like(module.weight)
            module.register_forward_hook(save_input_activation_hook)

    batch_iter = iter(batches)
    for _ in tqdm(range(config.circuit_num_steps)):
        input_ids = next(batch_iter)
        loss = loss_fn(model(input_ids), input_ids)
        loss.backward()

    return {
        name: param.misaligning
        for name, param in model.named_parameters()
        if hasattr(param, "misaligning")
    }


def get_circuit_k_dampens_grad(config, batches):
    assert "pythia" in config.model_id, "only pythia supported"

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    loss_fn = loss_fns["cross_entropy"]
    # don't require grads for the model
    model.requires_grad_(False)
    # this is needed to backpropagate despite not requiring grads
    model.gpt_neox.embed_in.requires_grad_(True)

    def save_grad(module, grad_input, grad_output):
        module.grad_out = grad_output[0]
        module.grad_in = grad_input[0]

    for l in model.gpt_neox.layers:
        l._backward_hooks.clear()
        l.register_full_backward_hook(save_grad)
        l.post_attention_layernorm._backward_hooks.clear()
        l.post_attention_layernorm.register_full_backward_hook(save_grad)
        l.mlp.dense_h_to_4h._backward_hooks.clear()
        l.mlp.dense_h_to_4h.register_full_backward_hook(save_grad)
        l.mlp.dense_h_to_4h.weight.circuit = pt.zeros_like(l.mlp.dense_h_to_4h.weight)

    # accumulate
    batch_iter = iter(batches)
    for _ in tqdm(range(config.circuit_num_steps)):
        input_ids = next(batch_iter)
        loss = loss_fn(model(input_ids), input_ids)
        loss.backward()
        for l in model.gpt_neox.layers:
            # calculate update
            ln = l.post_attention_layernorm
            in_ = l.grad_in
            # this step may be wrong:
            in_ *= ln.grad_in / ln.grad_out
            out = l.mlp.dense_h_to_4h.grad_out
            # replace nan with 0
            in_ = in_.nan_to_num()
            update = pt.einsum("bti,bto->oi", in_, out)
            assert not pt.isnan(update).any()
            l.mlp.dense_h_to_4h.weight.circuit += update

    return {
        name: param.circuit
        for name, param in model.named_parameters()
        if "mlp.dense_h_to_4h.weight" in name
    }


def get_circuit_k_dampens_grad_mlp_local(config, batches):
    assert "pythia" in config.model_id, "only pythia supported"

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    loss_fn = loss_fns["cross_entropy"]
    # don't require grads for the model
    model.requires_grad_(False)
    # this is needed to backpropagate despite not requiring grads
    model.gpt_neox.embed_in.requires_grad_(True)

    def save_grad(module, grad_input, grad_output):
        module.grad_out = grad_output[0]
        module.grad_in = grad_input[0]

    for l in model.gpt_neox.layers:
        l.mlp.dense_h_to_4h._backward_hooks.clear()
        l.mlp.dense_h_to_4h.register_full_backward_hook(save_grad)
        l.mlp.dense_h_to_4h.weight.circuit = pt.zeros_like(l.mlp.dense_h_to_4h.weight)

    # accumulate
    batch_iter = iter(batches)
    for _ in tqdm(range(config.circuit_num_steps)):
        input_ids = next(batch_iter)
        loss = loss_fn(model(input_ids), input_ids)
        loss.backward()
        for l in model.gpt_neox.layers:
            # calculate update
            in_ = l.mlp.dense_h_to_4h.grad_in
            out = l.mlp.dense_h_to_4h.grad_out
            update = pt.einsum("bti,bto->oi", in_, out)
            assert not pt.isnan(update).any()
            l.mlp.dense_h_to_4h.weight.circuit += update

    return {
        name: param.circuit
        for name, param in model.named_parameters()
        if "mlp.dense_h_to_4h.weight" in name
    }


def get_circuit_k_dampens_grad_neuron_local(config, batches):
    assert "pythia" in config.model_id, "only pythia supported"

    model = AutoModelForCausalLM.from_pretrained(config.model_id)
    loss_fn = loss_fns["cross_entropy"]
    # don't require grads for the model
    model.requires_grad_(False)
    # this is needed to backpropagate despite not requiring grads
    model.gpt_neox.embed_in.requires_grad_(True)

    def save_grad(module, grad_input, grad_output):
        module.grad_out = grad_output[0]
        module.grad_in = grad_input[0]

    for l in model.gpt_neox.layers:
        l.mlp.dense_h_to_4h._backward_hooks.clear()
        l.mlp.dense_h_to_4h.register_full_backward_hook(save_grad)
        l.mlp.dense_h_to_4h.weight.circuit = pt.zeros_like(l.mlp.dense_h_to_4h.weight)

    # accumulate
    batch_iter = iter(batches)
    for _ in tqdm(range(config.circuit_num_steps)):
        input_ids = next(batch_iter)
        loss = loss_fn(model(input_ids), input_ids)
        loss.backward()
        for l in model.gpt_neox.layers:
            # calculate update
            out = l.mlp.dense_h_to_4h.grad_out
            out_avg = out.mean(dim=0).mean(dim=0).reshape(-1, 1)
            weights = l.mlp.dense_h_to_4h.weight.data
            assert out_avg.shape[0] == weights.shape[0]
            update = weights * out_avg
            l.mlp.dense_h_to_4h.weight.circuit += update

    return {
        name: param.circuit
        for name, param in model.named_parameters()
        if "mlp.dense_h_to_4h.weight" in name
    }
