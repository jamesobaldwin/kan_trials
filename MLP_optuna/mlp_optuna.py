import logging
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch,
    UniformIntegerParameterRange
)

# Load Optuna optimizer, if available
def load_optimizer():
    try:
        from clearml.automation.optuna import OptimizerOptuna  # noqa
        aSearchStrategy = OptimizerOptuna
    except ImportError:
        logging.warning(
            'Optuna is not installed, falling back to RandomSearch strategy.')
        aSearchStrategy = RandomSearch

    return aSearchStrategy


def job_complete_callback(
        job_id,                 # type: str
        objective_value,        # type: float
        objective_iteration,    # type: int
        job_parameters,         # type: dict
        top_performance_job_id  # type: str
    ):
        """Callback function triggered when a job is completed."""
        print(f'Job completed! {job_id} -> Loss: {objective_value}')
        if job_id == top_performance_job_id:
            print(f'🎉 New best model found! Loss: {objective_value}')
    
def main():   

    aSearchStrategy = load_optimizer()
    
    # 🔹 Initialize ClearML Task
    task = Task.init(
        project_name="MLP Optimization",
        task_name="Optuna Controller",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )
    
    # 🔹 Define Arguments
    args = {
        'template_task_id': '5866ebbcb7c54a008f30aaf218599540',  # Base experiment to optimize
        'run_as_service': False,
    }
    args = task.connect(args)
    
    # Get template task ID
    if not args['template_task_id']:
        args['template_task_id'] = Task.get_task(
            project_name="MLP Optimization",
            task_name="Optuna Base Trainer"  # Replace with actual task name in ClearML UI
        ).id
    
    # Set the execution queue for the training jobs
    execution_queue = 'gpu'  # Change if needed
    
    # 🔹 Define Hyperparameter Optimization
    optimizer = HyperParameterOptimizer(
        base_task_id=args['template_task_id'],
        hyper_parameters=[
            DiscreteParameterRange("General/init_size", values=[128, 256, 512, 1024]),
            UniformIntegerParameterRange("General/phi_depth", min_value=1, max_value=4),
            UniformIntegerParameterRange("General/rho_depth", min_value=1, max_value=4),
            DiscreteParameterRange("General/optimizer", values=["Adam", "AdamW", "SGD"]),
            DiscreteParameterRange("General/lr", values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
            DiscreteParameterRange("General/weight_decay", values=[1e-6, 1e-5, 1e-4, 1e-3]),
            DiscreteParameterRange("General/momentum", values=[0.5, 0.7, 0.9])  # Only applies to SGD
        ],
        objective_metric_title= "test_mse",
        objective_metric_series= "test_mse",
        objective_metric_sign= "min",  # We want to minimize Test MSE
        max_number_of_concurrent_tasks=100,
        optimizer_class=aSearchStrategy,
        execution_queue=execution_queue,
        total_max_jobs=300,  # Limit the number of trials
        max_iteration_per_job=30,
        time_limit_per_job=30.0,  # Each job runs for max 10 minutes
        pool_period_min=1,  # Check every 1 minute (can be increased)
    )
    
    # if running as a service, execute remotely
    if args['run_as_service']:
        task.execute_remotely(queue_name="services", exit_process=True)
    
    # start Hyperparameter Optimization
    optimizer.set_report_period(0.2)
    optimizer.start(job_complete_callback=job_complete_callback)
    
    # Set time limit for full optimization process
    optimizer.set_time_limit(in_minutes=120.0)
    optimizer.wait()  # Wait for process to complete
    
    # Retrieve top experiments
    top_exp = optimizer.get_top_experiments_details(top_k=1, all_hyper_parameters=True)
    print(top_exp[0])
    task.upload_artifact("top_exp", top_exp[0])
    
    # Ensure optimization stops
    optimizer.stop()
    
    print("Hyperparameter Optimization Completed.")

if __name__=="__main__":
    main()
