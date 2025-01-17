import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import optuna


def repo_root() -> Path:
    cmd = ["git", "rev-parse", "--show-toplevel"]
    root = subprocess.check_output(cmd, text=True).strip()
    return Path(root)


def commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def is_repo_clean() -> bool:
    """Check that git repository has no uncommitted changes."""
    staged = subprocess.run(["git", "diff", "--staged", "--quiet"]).returncode == 0
    unstaged = subprocess.run(["git", "diff", "--quiet"]).returncode == 0
    return staged and unstaged


def add_tag_to_current_commit(tag: str) -> None:
    """Add a git tag to the current commit. Fails if the tag already exists."""
    subprocess.run(["git", "tag", tag], check=True)


def save_script_and_attach_logger(file_name, study_name):
    # for reproducibility save the file state and append output into it
    # save script
    # folder = repo_root() / "results" / datetime.now().strftime("%Y-%m-%d")
    folder = repo_root() / "results" / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = folder / f"{timestamp} {study_name}.log"
    shutil.copy(file_name, path)
    # attach logger
    for h in logging.getLogger().handlers[1:]:
        logging.root.removeHandler(h)
    file_handler = logging.FileHandler(path)
    formatter = logging.Formatter("# %(asctime)s %(levelname)s  %(message)s")
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
    logging.info(f"commit hash: {commit_hash()}")


def get_storage(db_url=None):
    if db_url is None:
        # on default use local sqlite db
        path = repo_root() / "db.sqlite3"
        path = os.path.relpath(path, Path.cwd())
        return f"sqlite:///{path}"
    else:
        # secrets_file = repo_root() / "secret.json"
        # assert secrets_file.exists(), "secret.json not found"
        # db_url = json.load(open(secrets_file))["db_url"]
        return optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={
                "pool_size": 20,
                "max_overflow": 0,
                "pool_pre_ping": True,
                "connect_args": {"sslmode": "require"},
            },
        )


def get_last_study(num=-1):
    storage = get_storage()
    # redisstorage may be faster https://github.com/optuna/optuna/pull/974
    study_summaries = optuna.study.get_all_study_summaries(storage)
    sorted_studies = sorted(study_summaries, key=lambda s: s.datetime_start)
    latest_study = sorted_studies[num]
    return optuna.load_study(study_name=latest_study.study_name, storage=storage)


def get_first_line_of_last_commit():
    cmd = ["git", "log", "-1", "--pretty=%B"]
    return subprocess.check_output(cmd, text=True).splitlines()[0]


def get_dirty_files():
    return subprocess.check_output(["git", "diff", "--name-only"], text=True)
