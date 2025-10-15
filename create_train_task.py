from clearml import Task

task = Task.create(
    task_name="train-step",
    project_name="Megatron",
    task_type="training",
    repo="https://github.com/ccasert/clearml-megatron.git",
    branch="main",
    binary="/bin/bash",
    script="train.sh",
)

task.set_user_properties(
    num_nodes=4,
)


print(f"Created preprocess task: {task.id}")
