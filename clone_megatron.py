import os
import subprocess
from clearml import Task


# Initialize ClearML task
task = Task.init(project_name="Megatron", task_name="clone-megatron-step")

args = {
    'repo_url': 'https://github.com/NVIDIA/Megatron-LM.git',
}

# Connect arguments (allows changing from pipeline)
task.connect(args)
print('Arguments: {}'.format(args))

# Execute remotely when run by agent
task.execute_remotely()

# Actual work

repo_path ="$SCRATCH/Megatron-LM"

if os.path.exists(repo_path):
    print(f"Megatron-LM already exists at {repo_path}, skipping clone")
else:
    print(f"Cloning Megatron-LM to {repo_path}...")
    subprocess.run(
        f"git clone {args['repo_url']} {repo_path}",
        shell=True,
        check=True
    )
    print("Clone completed successfully")

print(f"Megatron-LM available at: {repo_path}")

# Upload path as artifact for next steps
task.upload_artifact('megatron_dir', repo_path)

print('Done')