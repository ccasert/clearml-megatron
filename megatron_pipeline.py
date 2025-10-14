from clearml import Task
from clearml.automation import PipelineController


def pre_execute_callback(a_pipeline, a_node, current_param_override):
    """Callback before step execution"""
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    return True


def post_execute_callback(a_pipeline, a_node):
    """Callback after step execution"""
    print("Completed Task id={}".format(a_node.executed))
    return


# Create pipeline
pipe = PipelineController(
    name="Megatron-LM Pipeline",
    project="Megatron",
    version="1.0.0",
    add_pipeline_tags=False
)

# Set default execution queue
pipe.set_default_execution_queue("muller")

# Step 1: Clone Megatron-LM
pipe.add_step(
    name='clone_megatron',
    base_task_project='Megatron',
    base_task_name='clone-megatron-step',
    cache_executed_step=True,
    pre_execute_callback=pre_execute_callback,
    post_execute_callback=post_execute_callback,
)

# Start the pipeline
pipe.start(queue = "muller")

print("Pipeline started!")
