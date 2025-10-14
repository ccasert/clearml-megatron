import os
import subprocess


def clone_megatron_repo() -> str:
    """Clone Megatron-LM repository if it doesn't already exist"""

    repo_path = f"/mscratch/sd/c/ccasert/Megatron-LM"

    # Only clone if directory doesn't exist
    if os.path.exists(repo_path):
        print(f"Megatron-LM already exists at {repo_path}, skipping clone")
    else:
        print(f"Cloning Megatron-LM to {repo_path}...")
        subprocess.run(
            f"git clone git@github.com:NVIDIA/Megatron-LM.git {repo_path}",
            shell=True,
            check=True
        )

    os.environ["HOST_MEGATRON_LM_DIR"] = repo_path
    return repo_path

if __name__ == "__main__":
    clone_megatron_repo()