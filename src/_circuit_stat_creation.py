# %%
import torch as pt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import repo_root
from utils.training import loss_fns

model_id = "EleutherAI/pythia-14m"

pt.set_default_device("cuda")

# %%
dataset_name = "python"
loss_fn_name = "correct_logit"
num_steps = 1000

# load datasets
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = dataset_loaders[dataset_name](tokenizer)
cached_batches = CachedBatches(dataset["train"], batch_size=32)
batch_iter = iter(cached_batches)

loss_fn = loss_fns[loss_fn_name]
model = AutoModelForCausalLM.from_pretrained(model_id)

# accumulate grads
model.zero_grad(set_to_none=True)
for _ in tqdm(range(num_steps)):
    input_ids = next(batch_iter)
    loss = loss_fn(model(input_ids), input_ids)
    loss.backward()

circuit = {name: param.grad / num_steps for name, param in model.named_parameters()}

circuit_dir = repo_root() / "circuits" / model_id.replace("/", "_")
circuit_dir.mkdir(parents=True, exist_ok=True)
circuit_name = f"{dataset_name}_{loss_fn_name}.pt"
pt.save(circuit, circuit_dir / circuit_name)
