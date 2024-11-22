import subprocess
from pathlib import Path


def repo_root():
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


def commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def is_repo_clean():
    """Check that git repository has no uncommitted changes."""
    staged = subprocess.run(["git", "diff", "--staged", "--quiet"]).returncode == 0
    unstaged = subprocess.run(["git", "diff", "--quiet"]).returncode == 0
    return staged and unstaged


def add_tag_to_current_commit(tag):
    cmd = ["git", "tag", tag]
    subprocess.run(cmd, check=True)
