import json

from clearml import Task
from KAN_tuning import run_trial


task = Task.init(
    project_name='kan_tuning',
    task_name='test_task',
)

params = {
    'hidden_layers1': '[[3,0]]',
    'hidden_layers2': '[[0,3]]',
}
params = task.connect(params)
hidden_layers = params.get('hidden_layers')
if type(hidden_layers)==str:
    hidden_layers=json.loads(hidden_layers)
o = run_trial(hidden_layers=hidden_layers)
print(o)

logger = task.get_logger()
logger.report_scalar(title='debug', series='my_metric', value=0, iteration=0)