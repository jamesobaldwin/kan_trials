from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, UniformIntegerParameterRange
)
from clearml.automation.optuna import OptimizerOptuna

def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))

# Connecting CLEARML
task = Task.init(
   project_name='kan_tuning',
   task_name='test_tune',
   task_type=Task.TaskTypes.optimizer,
   reuse_last_task_id=False
)

# experiment template to optimize in the hyperparameter optimization
args = {
    'template_task_id': None,
    'run_as_service': False,
}
args = task.connect(args)
    
# Get the template task experiment that we want to optimize
if not args['template_task_id']:
    args['template_task_id'] = Task.get_task(
        project_name='kan_tuning', task_name='test_task').id

hidden_layer_vals = [[3,0], [3,3], [1,3]]

an_optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id=args['template_task_id'],
    hyper_parameters=[
        # UniformIntegerParameterRange('layer_1', min_value=128, max_value=512, step_size=128),
        # DiscreteParameterRange('batch_size', values=[96, 128, 160]),
        DiscreteParameterRange('hidden_layers', values=hidden_layer_vals),
        ],
    objective_metric_title='debug',
    objective_metric_series='my_metric',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=3,
    optimizer_class=OptimizerOptuna, # input optuna as search strategy
    execution_queue='default',
    total_max_jobs=10,
    max_iteration_per_job=100,
)

# if we are running as a service, just enqueue ourselves into the services queue and let it run the optimization
if args['run_as_service']:
    # if this code is executed by `clearml-agent` the function call does nothing.
    # if executed locally, the local process will be terminated, and a remote copy will be executed instead
    task.execute_remotely(queue_name='services', exit_process=True)

# report every 12 seconds, this is way too often, but we are testing here J
an_optimizer.set_report_period(0.2)
# start the optimization process, callback function to be called every time an experiment is completed
# this function returns immediately
an_optimizer.start(job_complete_callback=job_complete_callback)
# set the time limit for the optimization process (2 hours)

# set the time limit for the optimization process (2 hours)
an_optimizer.set_time_limit(in_minutes=10.0)
# wait until process is done (notice we are controlling the optimization process in the background)
an_optimizer.wait()
# optimization is completed, print the top performing experiments id
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
# make sure background optimization stopped
an_optimizer.stop()

print('We are done, good bye')