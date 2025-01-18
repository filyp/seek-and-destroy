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
    .env({"PYTHONPATH": "/root/code/src:${PYTHONPATH:-}"})
)
app = modal.App("example-get-started", image=image)


@app.function(gpu="L4", cpu=(1, 1), timeout=3600)
def remote_func(db_url):
    # clone repo
    subprocess.run(["git", "clone", repo, "/root/code"], check=True)
    os.chdir("/root/code")
    subprocess.run(["git", "checkout", branch], check=True)

    import torch as pt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pt.set_default_device("cuda")

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
