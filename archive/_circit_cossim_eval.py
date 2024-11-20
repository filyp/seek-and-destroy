# %%
import torch as pt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataloading_utils import *
from utils import *

# model_id = "Qwen/Qwen2.5-0.5B"
model_id = "EleutherAI/pythia-70m-deduped"
# model_id = "EleutherAI/pythia-160m-deduped"
# model_id = "HuggingFaceTB/SmolLM-135M"
pt.set_default_device("cuda")


# %%
def get_circuit(data_iter, grad_acc_fn, loss_fn, num_steps=1000):
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_id)

    state_dict = pt.load(repo_root() / "models" / "autosave.pt", weights_only=True)
    model.load_state_dict(state_dict)

    # note: using hooks is more complicated than processing grads after backward()
    #       but it prevents a memory spike - no need to store many param.grad
    def custom_grad_acc(param):
        # * note that we apply grad_acc_fn AFTER we've already summed over batch and token pos
        # * applying it before, would be really complicated
        param.custom_grad += grad_acc_fn(param.grad)
        param.grad = None

    for param in model.parameters():
        param.register_post_accumulate_grad_hook(custom_grad_acc)
        param.custom_grad = pt.zeros_like(param)

    for _ in tqdm(range(num_steps)):
        input_ids = get_batch(data_iter, 32)
        loss = loss_fn(model(input_ids), input_ids)
        loss.backward()

    # gather
    return {
        name: param.custom_grad / num_steps for name, param in model.named_parameters()
    }


# %%
# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
python_set = load_python_dataset(tokenizer)
en_set = load_one_oscar_shard("en", tokenizer)
pl_set = load_one_oscar_shard("pl", tokenizer)
(repo_root() / "circuits" / model_id).mkdir(parents=True, exist_ok=True)

for data_name, data_iter in [
    # ("python", looping_iter(python_set["train"])),
    ("en", looping_iter(en_set["train"])),
    # ("pl", looping_iter(pl_set["train"])),
]:
    for grad_acc_name, grad_acc_fn in [
        ("abs", lambda x: x.abs()),
        # ("linear", lambda x: x),
        # ("square", lambda x: x**2),
        # * actually these could be computed in parallel, but that would 3x mem usage
    ]:
        for loss_name, loss_fn in [
            ("logit", correct_logit_loss),
            # ("crossent", cross_entropy_loss),
        ]:
            circuit_name = f"{data_name}_{grad_acc_name}_{loss_name}.pt"
            print("calculating:", circuit_name)
            circuit = get_circuit(data_iter, grad_acc_fn, loss_fn)
            break
            pt.save(
                get_circuit(data_iter, grad_acc_fn, loss_fn),
                repo_root() / "circuits" / model_id / circuit_name,
            )


# %%
old_circuit = pt.load(repo_root() / "circuits" / model_id / "python_abs_logit.pt", weights_only=True)
for name, param in circuit.items():
    old_param = old_circuit[name]
    # cossim
    print(pt.cosine_similarity(param.flatten(), old_param.flatten(), dim=0).item(), name)
