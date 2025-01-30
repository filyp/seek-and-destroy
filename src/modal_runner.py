# run this file in repo root
import json
import os
import subprocess
import sys

import modal
import yaml

from study_runner import run_study

repo = "https://github.com/filyp/seek-and-destroy.git"
branch = "main"

image = (
    modal.Image.debian_slim()
    .run_commands("apt-get install -y git")
    .pip_install_from_requirements("requirements.txt")
    .env({"PYTHONPATH": "/root/code/src:${PYTHONPATH:-}"})
)
app = modal.App("example-get-started", image=image)


# no timeout
@app.function(gpu="L4", cpu=(1, 1), timeout=24 * 3600)
def remote_func(db_url, config_path, variant_num, if_study_exists, n_trials, allow_dirty_repo):
    # clone repo
    subprocess.run(["git", "clone", repo, "/root/code"], check=True)
    os.chdir("/root/code")
    subprocess.run(["git", "checkout", branch], check=True)

    import torch as pt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pt.set_default_device("cuda")

    from utils.git_and_reproducibility import get_storage
    from utils.loss_fns import neg_cross_entropy_loss
    from utils.git_and_reproducibility import is_repo_clean

    storage = get_storage(db_url)

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if variant_num is None:
        # ! run all variants one after another
        for variant_num in range(len(full_config["variants"])):
            if not allow_dirty_repo:
                assert is_repo_clean()
            run_study(storage, config_path, variant_num, if_study_exists, n_trials)
    else:
        # ! run single variant
        if not allow_dirty_repo:
            assert is_repo_clean()
        run_study(storage, config_path, variant_num, if_study_exists, n_trials)


@app.local_entrypoint()
def main(
    config_path: str,
    variant_num: int | None = None,
    if_study_exists: str = "fail",
    n_trials: int | None = None,
    allow_dirty_repo: bool = False,
):
    db_url = json.load(open("secret.json"))["db_url"]
    remote_func.remote(db_url, config_path, variant_num, if_study_exists, n_trials, allow_dirty_repo)
