import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from clearml import Task
import joblib


def init_task(project_name: str, task_name: str) -> tuple[Task, dict['str', any]]:
    """Initialize ClearML task"""
    task = Task.init(
        project_name=project_name, task_name=task_name
    )

    params = {
        "num_sets": 50,
        "num_points": 250,
    }

    params = task.connect(params)
    
    return task, params

# helper function for creating datasets
def CoM(data) -> np.ndarray:
    '''
    Function to calculate the true center of gravity of
    a set of points
    '''
    masses = data[:, 3]
    x_cog = np.sum(data[:, 0] * masses) / np.sum(masses)
    y_cog = np.sum(data[:, 1] * masses) / np.sum(masses)
    z_cog = np.sum(data[:, 2] * masses) / np.sum(masses)
    
    return np.array([x_cog, y_cog, z_cog])

def generate_train_test_sets(n: int=1000, N: int=1000, random_state: int=42) -> tuple[list[np.ndarray], list[np.ndarray]]:
    '''
    n (int): number of point sets
    N (int): number of points in each set
    '''
    train_set = np.random.rand(n, N, 4)
    test_set = np.random.rand(n, N, 4)
    print(f"DEBUG: shape of train_set: {np.shape(train_set)}")

    # split along second axis to create list of n (N, 1, 4) arrays
    train_set_list = np.split(train_set, n, axis=0)
    test_set_list = np.split(test_set, n, axis=0)
    print(f"DEBUG: shape of train_set_list after split: {np.shape(train_set_list)}")

    # collapse the second dimension; left with list of n (N, 4) arrays
    train_set_list = [arr.reshape(N,4) for arr in train_set_list]
    test_set_list = [arr.reshape(N,4) for arr in test_set_list]
    print(f"DEBUG: shape of train_set_list after collapse: {np.shape(train_set_list)}")
    
    return train_set_list, test_set_list
    
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
    
    X_train, X_test = generate_train_test_sets(n=params['num_sets'], N=params['num_points'])
    scaler, y_train = create_targets(X_train)
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
    task, params = init_task(project_name="MLP Optimization", task_name="Data Generation")

    # store the training, test, and scalar object data in a dictionary
    train_test_data = generate_train_test_data(params)
    print(f"DEBUG: shape of X_train: {np.shape(train_test_data['X_train'])}")

    # upload artifacts to ClearML servers
    save_artifacts(task, train_test_data)

    # close the task
    task.close()

if __name__ == "__main__":
    main()