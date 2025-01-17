# run this file in repo root
import json
import os
import subprocess

import modal

repo = "https://github.com/filyp/seek-and-destroy.git"
branch = "main"

image = (
    modal.Image.debian_slim()
    .run_commands("apt-get install -y git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("psutil")
    .env({"PYTHONPATH": "/root/code/src:${PYTHONPATH:-}"})
)
app = modal.App("example-get-started", image=image)



@app.function(gpu="L4", timeout=3600)
def remote_func(db_url):
    # clone repo
    subprocess.run(["git", "clone", repo, "/root/code"], check=True)
    os.chdir("/root/code")
    subprocess.run(["git", "checkout", branch], check=True)

    # Check system resources using psutil
    import psutil
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Memory available: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.1f} GB")
    print(f"Memory total: {psutil.virtual_memory().total / (1024 * 1024 * 1024):.1f} GB")

    import torch as pt
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pt.set_default_device("cuda")
    # gpu
    if pt.cuda.is_available():
        print(f"GPU: {pt.cuda.get_device_name()}")
        print(f"GPU Memory: {pt.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU available")

    from utils.git_and_reproducibility import get_storage
    from utils.loss_fns import neg_cross_entropy_loss

    storage = get_storage(db_url)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
    print(neg_cross_entropy_loss)


@app.local_entrypoint()
def main():
    db_url = json.load(open("secret.json"))["db_url"]
    remote_func.remote(db_url)
