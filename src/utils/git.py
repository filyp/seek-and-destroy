import subprocess
from pathlib import Path


def repo_root() -> Path:
    raw_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    return Path(raw_root.decode("utf-8").strip())


def commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def is_repo_clean() -> bool:
    """check that git repository has no uncommitted changes"""
    staged = subprocess.run(["git", "diff", "--staged", "--quiet"]).returncode == 0
    unstaged = subprocess.run(["git", "diff", "--quiet"]).returncode == 0
    return staged and unstaged


def add_tag_to_current_commit(tag: str) -> None:
    cmd = ["git", "tag", tag]
    subprocess.run(cmd, check=True)
