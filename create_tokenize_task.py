from clearml import Task

task = Task.create(
    task_name="tokenize-data-step",
    project_name="Megatron",
    task_type="data_processing",
    repo="https://github.com/ccasert/clearml-megatron.git",
    branch="main",
    binary="/bin/bash",
    script="tokenize_data.sh",
)

# Set parameters that will be overridden by pipeline
task.set_parameters({
    'MEGATRON_DIR': '',
    'DATA_DIR': '',
    'TOKENIZER_DIR': '',
})

print(f"Created preprocess task: {task.id}")
