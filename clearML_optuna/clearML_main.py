import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from clearml import Task
from kan import KAN
from sklearn.preprocessing import MinMaxScaler


def init_task(project_name: str, task_name: str) -> Task:
    """Initialize ClearML task"""
    task = Task.init(
        project_name=project_name, task_name=task_name
    )

    params = {
        "n_trials": 50, # num optuna trials
        "optimizer": "Adam",
        "loss_fn": "MSELoss",
    }
    
    task.connect(params)
    
    return task

def retrieve_train_test_data() -> dict:
    
    # Find the task that generated the artifact
    source_task = Task.get_task(project_name="KAN deep set optimization", task_name="data generation")
    
    # Retrieve the dictionary directly from the artifact
    train_test_data = source_task.artifacts["train_test_data"].get()

    return train_test_data

def unpack_and_convert(train_test_data: dict):
    '''
    Return the object of train test data, with X_train, y_train, and X_test converted to tensors
    '''
    X_train = [torch.tensor(train_test_data['X_train'][i], dtype=torch.float32) for i in np.arange(len(train_test_data['X_train']))]
    y_train = [torch.tensor(train_test_data['y_train'][i], dtype=torch.float32) for i in np.arange(len(train_test_data['y_train']))]
    X_test = [torch.tensor(train_test_data['X_test'][i], dtype=torch.float32) for i in np.arange(len(train_test_data['X_test']))]

    return X_train, y_train, X_test, train_test_data['y_test'], train_test_data['scaler']

def save_artifacts(task: Task, preds: np.ndarray):
    task.upload_artifacts("preds", preds)

def objective(trial: optuna.Trial, 
              task: Task, 
              X_train: List[torch.Tensor], 
              y_train: torch.Tensor, 
              X_test: List[torch.Tensor], 
              y_test: np.ndarray, 
              scaler: MinMaxScaler) -> float:
    # optimize number of hidden layers (1, 2, or 3)
    num_layers1 = trial.suggest_int("num_layers1", 1, 3)
    num_layers2 = trial.suggest_int("num_layers2", 1, 3)

    # optimize hidden layers for layer 1
    hidden_layers_1 = []
    for i in range(num_layers1):
        a = trial.suggest_int(f"a1_{i}", 0, 4)  # addition nodes, from 0 to 4
        m = trial.suggest_int(f"m1_{i}", 1, 4)  # addition nodes, from 1 to 4
        hidden_layers_1.append([a, m])

    # optimize output size of layer 1 (input size of layer 2)
    out_size_layer1 = trial.suggest_int("out_size_layer1", 3, 10)

    # optimize hidden layers for layer 2
    hidden_layers_2 = []
    for i in range(num_layers2):
        a = trial.suggest_int(f"a2_{i}", 0, 4)  # addition nodes, from 0 to 4
        m = trial.suggest_int(f"m2_{i}", 1, 4)  # addition nodes, from 1 to 4
        hidden_layers_2.append([a, m])

    # optimize grid sizes
    grid1 = trial.suggest_int("grid1", 3, 10)
    grid2 = trial.suggest_int("grid2", 3, 10)

    # Convert hidden layers into a dictionary-friendly format
    hidden_layers_dict = {
        "hidden_layers_1": {f"layer_{i}": {"a": a, "m": m} for i, (a, m) in enumerate(hidden_layers_1)},
        "hidden_layers_2": {f"layer_{i}": {"a": a, "m": m} for i, (a, m) in enumerate(hidden_layers_2)}
    }

    # Store hyperparameters for this trial
    trial_params = {
        "num_layers1": num_layers1,
        "num_layers2": num_layers2,
        "out_size_layer1": out_size_layer1,
        "grid1": grid1,
        "grid2": grid2,
        **hidden_layers_dict  # Merge the hidden layer dictionary
    }

    # Log trial hyperparameters in ClearML
    task.connect(trial_params)  # Logs per-trial hyperparameters

    # define the KAN model
    class KANModel(nn.Module):
        def __init__(
            self, hidden_layers_1, hidden_layers_2, out_size_layer1, grid1, grid2
        ):
            super(KANModel, self).__init__()
            self.layer1 = KAN(
                width=[4] + hidden_layers_1 + [out_size_layer1], grid=grid1, k=3
            )
            self.layer2 = KAN(
                width=[out_size_layer1] + hidden_layers_2 + [3], grid=grid2, k=3
            )

        def forward(self, point_set):
            layer1_out = self.layer1(point_set)  # Output shape (n, out_size_layer1)
            agg_out = torch.mean(layer1_out, dim=0).unsqueeze(
                0
            )  # Shape (1, out_size_layer1)
            output = self.layer2(agg_out).squeeze(0)  # Shape (3) representing (x,y,z)
            return output

    model = KAN(hidden_layers_1, hidden_layers_2, out_size_layer1, grid1, grid2)

    # train the model
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        model.train()

        for i in range(len(X_train)):
            optimizer.zero_grad()
            y_pred = model(X_train[i])
            loss = loss_fn(y_pred, y_train[i])
            loss.backward()
            optimizer.step()
            task.get_logger().report_scalar("Loss", "train", value=loss.item(), iteration=epoch)

    # test the model
    model.eval()
    preds = []
    with torch.no_grad():
        for point_set in X_test:
            pred = model(point_set)
            # remove batch dimension and store the prediction
            pred = scaler.inverse_transform(pred.reshape(1,-1)).squeeze()
            preds.append(pred)
            
    # convert to numpy array and save dataset
    preds = np.array(preds)    
    save_artifacts(task, preds)
    # calculate and store the MSE
    mse = np.mean((preds - y_train)**2)

    task.get_logger().report_scalar("MSE", "test", value=mse, iteration=1)

    return mse

def run_optuna(task, X_train, y_train, X_test, y_test, scaler):
    # === Run Optuna Optimization ===
    study = optuna.create_study(direction="minimize")
    
    study.optimize(lambda trial: objective(trial, task, X_train, y_train, X_test, y_test, scaler), n_trials=50)

def main():

    task, params = init_task(project_name="KAN deep set optimization", task_name="optuna controller")

    train_test_data = retrieve_train_test_data()

    # unpack and save data objects
    X_train, y_train, X_test, y_test, scaler = unpack_and_convert(train_test_data)

    # run the optuna optimization
    run_optuna(task, X_train, y_train, X_test, y_test, scaler)

    task.close()

if __name__ == "__main__":
    main()