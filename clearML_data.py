import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from clearml import Task, Dataset
import joblib
import pickle

task = Task.init(project_name="KAN deep set optimization", task_name="data generation")
dataset = Dataset.create(dataset_name="train_test_data", dataset_project="KAN deep set optimization")

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
    
    return point_sets

def create_targets(point_sets, scale: bool=True):
    center = np.array([0.5,0.5,0.5])
    ps_targets = np.array([(CoM(point_sets[i]) - center) for i in range(len(point_sets))])
    if scale:
        scaler = MinMaxScaler(feature_range=(-1,1))
        targets = scaler.fit_transform(ps_targets)
    else:
        return ps_targets
    return scaler, targets

# generate and save training data
point_sets = create_sets(N=1000)
scaler, targets = create_targets(point_sets)

ps_tensor = [torch.tensor(point_sets[i], dtype=torch.float32) for i in np.arange(len(point_sets))]
target_tensor = [torch.tensor(targets[i], dtype=torch.float32) for i in np.arange(len(targets))]

# Save training data using pickle
with open("X_train.pkl", "wb") as f:
    pickle.dump(ps_tensor, f)
np.save("y_train.npy", target_tensor)
joblib.dump(scaler, "scaler.pkl")

# generate and save test data
test_sets = create_sets(lower=1000, upper=1001, N=1000)
targets = create_targets(test_sets, scale=False)

test_tensor = [torch.tensor(test_sets[i], dtype=torch.float32) for i in np.arange(len(test_sets))]

np.save("X_test.npy", test_tensor)
np.save("y_test.npy", targets)

# upload to clearML
dataset.add_files("X_train.pkl")
dataset.add_files("X_test.npy")
dataset.add_files("y_train.npy")
dataset.add_files("y_test.npy")
dataset.add_files("scaler.pkl")
dataset.upload()
dataset.finalize()
task.close()