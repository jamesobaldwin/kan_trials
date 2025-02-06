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
        project_name="KAN-DS optimization",
        task_name="Optuna Controller",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )
    
    # 🔹 Define Arguments
    args = {
        'template_task_id': '898badb019f04991a5068d40353397a1',  # Base experiment to optimize
        'run_as_service': False,
    }
    args = task.connect(args)
    
    # Get template task ID
    if not args['template_task_id']:
        args['template_task_id'] = Task.get_task(
            project_name="KAN-DS optimization",
            task_name="V3 Template Trainer"  # Replace with actual task name in ClearML UI
        ).id
    
    # Set the execution queue for the training jobs
    execution_queue = 'default'  # Change if needed
    
    # 🔹 Define Hyperparameter Optimization
    optimizer = HyperParameterOptimizer(
        base_task_id=args['template_task_id'],
        hyper_parameters=[
            UniformIntegerParameterRange("General/num_layers_1", min_value=1, max_value=3),
            UniformIntegerParameterRange("General/num_layers_2", min_value=1, max_value=3),
            UniformIntegerParameterRange('General/a1_0', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/m1_0', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/a1_1', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/m1_1', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/a1_2', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/m1_2', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/a2_0', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/m2_0', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/a2_1', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/m2_1', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/a2_2', min_value=0, max_value=4),
            UniformIntegerParameterRange('General/m2_2', min_value=0, max_value=4),
            UniformIntegerParameterRange("General/transition_dim", min_value=1, max_value=10),
            UniformIntegerParameterRange('General/grid1', min_value=3, max_value=10),
            UniformIntegerParameterRange('General/grid2', min_value=3, max_value=10),
            DiscreteParameterRange("General/optimizer", values=["Adam", "AdamW", "SGD", "LBFGS"]),
            DiscreteParameterRange("General/lr", values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
            DiscreteParameterRange("General/weight_decay", values=[1e-6, 1e-5, 1e-4, 1e-3]),
            DiscreteParameterRange("General/momentum", values=[0.5, 0.7, 0.9]),  # Only applies to SGD
            DiscreteParameterRange("General/agg_function", values=["mean", "sum", "std"]),
            
        ],
        objective_metric_title= ["mse_train_loss", "mse_test_loss"],
        objective_metric_series= ["train", "test"],
        objective_metric_sign= ["min", "min"],  # We want to minimize Test MSE
        max_number_of_concurrent_tasks=4,
        optimizer_class=aSearchStrategy,
        execution_queue=execution_queue,
        total_max_jobs=10,  # Limit the number of trials
        max_iteration_per_job=30,
        time_limit_per_job=30.0,  # Each job runs for max 10 minutes
        pool_period_min=1,  # Check every 1 minute (can be increased)
    )
    
    # if running as a service, execute remotely
    if args['run_as_service']:
        task.execute_remotely(queue_name="services", exit_process=True)

    # start Hyperparameter Optimization; progress report every 5 minutes
    optimizer.set_report_period(5)
    optimizer.start(job_complete_callback=job_complete_callback)
    
    # Set time limit for full optimization process
    optimizer.set_time_limit(in_minutes=120.0)
    optimizer.wait()  # Wait for process to complete
    
    # Retrieve top experiments
    top_exp = optimizer.get_top_experiments_details(top_k=3, all_hyper_parameters=True)
    print(top_exp[0])
    task.upload_artifact("top_exp", top_exp[0])
    
    # Ensure optimization stops
    optimizer.stop()
    
    print("Hyperparameter Optimization Completed.")

if __name__=="__main__":
    main()