from clearml import Task

task = Task.create(
    task_name="download-data-step",
    project_name="Megatron",
    task_type="data_processing",
    repo="https://github.com/ccasert/clearml-megatron.git",
    branch="main",
    binary="/bin/bash",  # Run bash script
    script="download_wikitext.sh",  # The bash wrapper
)

print(f"Created download task: {task.id}")
