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
    
    return task

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

def create_sets(lower: int=200, upper: int=1000, N: int=1000, random_state: int=42):
    '''
    Args:
        low (int): lower bound for number of points in point set
        upper (int): upper bound for number of points in point set
        N (int): number of point sets
        random_state (int): random seed used for reproducibility 
    
    Return:
        point_sets (ndarray): array of point sets of size (N, n, 4) where
        n is a variable number of points in the point set
    '''
    if random_state is not None:
        np.random.seed(random_state)
    
    point_sets = [
        np.column_stack((
            np.random.rand(n := np.random.randint(lower, upper)),  # x
            np.random.rand(n),                                     # y
            np.random.rand(n),                                     # z
            np.random.rand(n)                                      # masses
        ))
        for _ in range(N)
    ]
    
    return point_sets  # a list of numpy arrays, length N, where each array in the list is variable size [n, 4]

def create_targets(point_sets, scale: bool=True):
    center = np.array([0.5,0.5,0.5])
    ps_targets = np.array([(CoM(point_sets[i]) - center) for i in range(len(point_sets))])
    if scale:
        scaler = MinMaxScaler(feature_range=(-1,1))
        targets = scaler.fit_transform(ps_targets)
    else:
        return ps_targets
    return scaler, targets

def generate_train_test_data() -> dict:
    
    point_sets = create_sets()
    scaler, targets = create_targets(point_sets)
    test_sets = create_sets(lower=1000, upper=1001, N=1000)
    test_targets = create_targets(test_sets, scale=False)

    # create the artifacts dictionary
    train_test_data = {
        "X_train": point_sets,
        "y_train": targets,
        "X_test": test_sets,
        "y_test": test_targets,
        "scaler": scaler,
    }

    return train_test_data

def save_artifacts(task: Task, artifacts: dict):
    task.upload_artifacts("train_test_data", artifacts)

def main():

    # create the ClearML task 
    task = init_task(project_name="KAN deep set optimization", task_name="data generation")

    # store the training, test, and scalar object data in a dictionary
    train_test_data = generate_train_test_data()

    # upload artifacts to ClearML servers
    save_artifacts(task, train_test_data)

    # close the task
    task.close()

if __name__ == "__main__":
    main()