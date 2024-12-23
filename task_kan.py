from clearml import Task
from KAN_tuning import run_trial


task = Task.init(
    project_name='kan_tuning',
    task_name='test_task',
)

params = {
    'hidden_layers': [3,0],
}
params = task.connect(params)
hidden_layers = params.get('hidden_layers')
o = run_trial(hidden_layers=hidden_layers)
print(o)

logger = task.get_logger()
logger.report_scalar(title='debug', series='my_metric', value=0, iteration=0)