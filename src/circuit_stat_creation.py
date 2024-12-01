# %%
import torch as pt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.model_operations import *
from utils.training import MockTrial, loss_fns, set_seeds

model_id="EleutherAI/pythia-14m"
pt.set_default_device("cuda")


# %%
# note: actually if we're not transforming the grads anymore, hooks are not needed
# simple gradient accumulation would do
def get_circuit(batch_iter, loss_fn, num_steps=1000):
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # note: using hooks is more complicated than processing grads after backward()
    #       but it prevents a memory spike - no need to store many param.grad
    def custom_grad_acc(param):
        # * note that we apply grad_acc_fn AFTER we've already summed over batch and token pos
        # * applying it before, would be really complicated
        # param.custom_grad += grad_acc_fn(param.grad)
        param.custom_grad += param.grad
        param.grad = None

    for param in model.parameters():
        param.register_post_accumulate_grad_hook(custom_grad_acc)
        param.custom_grad = pt.zeros_like(param)

    for _ in tqdm(range(num_steps)):
        input_ids = next(batch_iter)
        loss = loss_fn(model(input_ids), input_ids)
        loss.backward()

    # gather
    return {
        name: param.custom_grad / num_steps for name, param in model.named_parameters()
    }


# %%
dataset_name = "python"
loss_fn_name = "correct_logit"

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = dataset_loaders[dataset_name](tokenizer)
cached_batches = CachedBatches(dataset["train"], batch_size=32)
batch_iter = cached_batches.fresh_iterator()

loss_fn = loss_fns[loss_fn_name]
circuit = get_circuit(batch_iter, loss_fn)

circuit_name = f"{dataset_name}_{loss_fn_name}.pt"
(repo_root() / "circuits" / model_id).mkdir(parents=True, exist_ok=True)
pt.save(
    circuit,
    repo_root() / "circuits" / model_id / circuit_name,
)
