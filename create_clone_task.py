from clearml import Task

# Create the task
task = Task.create(
    task_name="clone-megatron-step",
    project_name="Megatron",
    task_type="data_processing",
    repo="https://github.com/ccasert/clearml-megatron.git",
    branch="main",
    script="clone_megatron.py",
)

# Set parameters
task.set_parameters({
    'repo_url': 'https://github.com/NVIDIA/Megatron-LM.git',
})

print(f"Created task: {task.id}")

# Enqueue it to run on the agent
Task.enqueue(task=task, queue_name="muller")

print("Task enqueued to muller queue - agent will execute and create it")
