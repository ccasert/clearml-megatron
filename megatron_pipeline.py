from clearml import PipelineController
from clone_megatron import clone_megatron_repo


def main():
    # Create pipeline controller
    pipe = PipelineController(
        name="Megatron-LM Pipeline",
        project="Megatron",
        version="1.0.0",
        target_project = "Megatron"
    )

    # # Set default execution queue
    # pipe.set_default_execution_queue('muller')

    # Step 1: Clone Megatron-LM
    pipe.add_function_step(
        name='clone_megatron',
        function=clone_megatron_repo,
        function_return=['megatron_dir'],
        cache_executed_step=True,
        execution_queue='muller',
    )

    # Start the pipeline
    print("Starting pipeline...")
    pipe.start()

    print("Pipeline enqueued!")


if __name__ == "__main__":
    main()