import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from clearml import Task
import joblib


def init_task(project_name: str, task_name: str) -> Task:
    """Initialize ClearML task"""
    task = Task.init(
        project_name=project_name, task_name=task_name
    )

    params = {
        "num_point_sets": 1000,   # number of point sets n
        "lower": 200,             # lower limit on number N points in point set
        "upper": 1000,            # upper " "
    }

    params = task.connect(params)
    
    return task, params

# helper function for creating datasets
def CoM(data):
    '''
    Function to calculate the true center of gravity of
    a set of points
    '''
    masses = data[:, 3]
    x_cog = np.sum(data[:, 0] * masses) / np.sum(masses)
    y_cog = np.sum(data[:, 1] * masses) / np.sum(masses)
    z_cog = np.sum(data[:, 2] * masses) / np.sum(masses)
    
    return np.array([x_cog, y_cog, z_cog])

def generate_train_test_sets(lower: int = 200, upper: int = 1000, n: int = 1000, random_state: int = 42):
    """
    Generate train and test sets (lists of arrays) where each array has a variable 
    number of points (between `lower` and `upper`) and each point has 4 coordinates.
    
    Args:
        lower (int): lower bound for the number of points in a set.
        upper (int): upper bound for the number of points in a set.
        n (int): number of point sets.
        random_state (int): seed for reproducibility.
    
    Returns:
        tuple: (train_sets, test_sets) where each is a list of n arrays with shape (N, 4)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random sizes for each set.
    sizes = np.random.randint(lower, upper, size=n)
    total_points = sizes.sum()
    
    # Generate all train and test points at once.
    train_all_points = np.random.rand(total_points, 4)
    test_all_points = np.random.rand(total_points, 4)
    
    # Get the indices to split.
    split_indices = np.cumsum(sizes)[:-1]
    
    # Split into individual sets.
    train_sets = np.split(train_all_points, split_indices)
    test_sets = np.split(test_all_points, split_indices)
    
    return train_sets, test_sets  # a list of numpy arrays, length n, where each array in the list is variable size [N, 4]

def create_targets(point_sets, scale: bool=True):
    center = np.array([0.5,0.5,0.5])
    ps_targets = np.array([(CoM(point_sets[i]) - center) for i in range(len(point_sets))])
    if scale:
        scaler = MinMaxScaler(feature_range=(-1,1))
        targets = scaler.fit_transform(ps_targets)
    else:
        return ps_targets
    return scaler, targets

def generate_train_test_data(params: dict) -> dict:
    
    X_train, X_test = generate_train_test_sets(lower = params['lower'], upper = params['upper'], n = params['num_point_sets'])
    scaler, targets = y_train(X_train)
    y_test = create_targets(X_test, scale=False)

    # create the artifacts dictionary
    train_test_data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }

    return train_test_data

def save_artifacts(task: Task, artifacts: dict):
    task.upload_artifact("train_test_data", artifacts)

def main():

    # create the ClearML task 
    task, params = init_task(project_name="KAN-DS optimization", task_name="Data Generation")

    # store the training, test, and scalar object data in a dictionary
    train_test_data = generate_train_test_data(params)

    # upload artifacts to ClearML servers
    save_artifacts(task, train_test_data)

    # close the task
    task.close()

if __name__ == "__main__":
    main()