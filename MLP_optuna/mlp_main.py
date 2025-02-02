import optuna

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Task
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def init_task(project_name: str, task_name: str) -> tuple[Task, dict[str, any]]:
    task = Task.init(project_name=project_name, task_name=task_name)

    params = {
        "num_point_sets": 1000,
        "num_points_per_set": 1000,
        "optuna_trials": 300,
        "num_epochs": 1000,
        "use_mps_gpu": True,
        "data_task_id": "aea0e01e9762410fb9acefccc549d893",
    }

    return task, params


def retrieve_data(task_id) -> dict[str, any]:
    """
    grab the training and test data created in mlp_data.py
    """
    data_task = Task.get_task(
        task_id=task_id
    )
    train_test_data = data_task.artifacts["train_test_data"].get()

    return train_test_data


def unpack_and_convert(
    train_test_data: dict,
):
    """
    Return the object of train test data, with X_train, y_train, and X_test converted to tensors
    """
    X_train = [
        torch.tensor(train_test_data["X_train"][i], dtype=torch.float32)
        for i in np.arange(len(train_test_data["X_train"]))
    ]
    y_train = [
        torch.tensor(train_test_data["y_train"][i], dtype=torch.float32)
        for i in np.arange(len(train_test_data["y_train"]))
    ]
    X_test = [
        torch.tensor(train_test_data["X_test"][i], dtype=torch.float32)
        for i in np.arange(len(train_test_data["X_test"]))
    ]

    return (
        X_train,
        y_train,
        X_test,
        train_test_data["y_test"],
        train_test_data["scaler"],
    )

# save artifacts for use in plotting final model
def save_artifacts(task: Task, preds: np.ndarray, y_test: np.ndarray, lr: float):
    task.upload_artifact("preds", preds)
    task.upload_artifact("y_test", y_test)
    task.upload_artifact("lr", lr)

def create_layers(input_size: int, init_size: int, phi_depth: int, rho_depth: int):

    assert isinstance(input_size, int), f"Error: input_size is {input_size}, expected int"
    assert isinstance(init_size, int), f"Error: init_size is {init_size}, expected int"
    
    layers = [nn.Linear(input_size, init_size), nn.ReLU()]
    
    # Construct phi hidden layers with increasing size
    for i in range(phi_depth):
        input_size = 2**(i) * init_size
        phi_out = 2*input_size
        layers.append(nn.Linear(input_size, phi_out))
        layers.append(nn.ReLU())
            
    # construct rho hidden layers with decreasing size
    input_size = phi_out if phi_depth != 0 else init_size
        
    for i in range(rho_depth):
        rho_out = input_size // 2
        layers.append(nn.Linear(input_size, rho_out))
        layers.append(nn.ReLU())
        input_size = rho_out
    
    # Add the final output layer
    rho_out = input_size // 2 if rho_depth == 0 else rho_out
    layers.append(nn.Linear(rho_out, 3))
    
    # Wrap layers in nn.Sequential
    return nn.Sequential(*layers)

class MLPModel(nn.Module):
    def __init__(self, input_size, init_size, phi_depth, rho_depth):
        super(MLPModel, self).__init__()
        self.mlp_relu_stack = create_layers(input_size=input_size, init_size=init_size, phi_depth=phi_depth, rho_depth=rho_depth)

    def forward(self, x):
        x = x.view(1,-1)  # flatten input to shape [1,4000]
        out = self.mlp_relu_stack(x)
        return out.squeeze(0)   # output shape [3,]

def trainMLP(trial, config, verbose):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = MLPModel(
        input_size=config["input_size"], 
        init_size=config["init_size"], 
        phi_depth=config["phi_depth"], 
        rho_depth=config["rho_depth"]
    ).to(device)

    criterion = nn.MSELoss()

    # Select optimizer
    if config["optimizer_type"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer_type"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer_type"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    elif config["optimizer_type"] == "Ranger":
        optimizer = Ranger(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    mlp_losses = []
    
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        
        for i in range(len(config["X_train"])):
            optimizer.zero_grad()
            outputs = model(config["X_train"][i].to(device))
            loss = criterion(outputs, config["y_train"][i].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_epoch_loss = total_loss / len(config["X_train"])
        mlp_losses.append(avg_epoch_loss)

        trial.report(avg_epoch_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if verbose and (epoch + 1) % (0.1 * config["num_epochs"]) == 0:
            print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {avg_epoch_loss:.4e}")
    
    return model, mlp_losses


def test(model, test_tensor, scaler):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    predictions = []
    with torch.no_grad():
        for point_set in test_tensor:
            output = model(point_set.to(device))
            output = scaler.inverse_transform(output.cpu().numpy().reshape(1, -1)).squeeze()
            predictions.append(output)

    return np.array(predictions)
    

def objective(trial, task, X_train, y_train, X_test, y_test, scaler) -> float:
    
    # training hyperparams
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "Ranger"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    momentum = trial.suggest_float("momentum", 0.5, 0.99) if optimizer_type == "SGD" else None
    
    # architecture hyperparams
    init_size = trial.suggest_categorical("init_size", [128, 256, 512, 1024])
    print(f"DEBUG: Optuna suggested init_size = {init_size}")
    assert isinstance(init_size, int), f"Error: init_size is {init_size}, expected int"

    phi_depth = trial.suggest_int("phi_depth", 0, 4)
    rho_depth = trial.suggest_int("rho_depth", 0, 4)

    config = {
        "input_size": X_train[0].numel(),
        "init_size": init_size,
        "phi_depth": phi_depth,
        "rho_depth": rho_depth,
        "num_epochs": 300,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "optimizer_type": optimizer_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum
    }
    

    try:
        model, losses = trainMLP(
            trial=trial,
            config=config,  # Pass dictionary instead of many arguments
            verbose=False
        )
    except optuna.TrialPruned:
        raise  # Stop trial early if necessary

    # Evaluate the model on the test set
    predictions = test(model, config["X_test"], config["scaler"])
    save_artifacts(task, predictions, config['y_test'], lr)
    
    # Calculate the MSE for optimization
    test_mse = mean_squared_error(config["y_test"], predictions)


    # Return the test MAE as the objective value
    return test_mse

def run(params: dict):
    # Initialize ClearML task
    task = Task.init(project_name="MLP Optimization", task_name="Optuna Optimization")

    # Load or generate data
    train_test_data = retrieve_data()
    X_train, y_train, X_test, y_test, scaler = unpack_and_convert(train_test_data)

    print(f"DEBUG: X_train[0].shape = {X_train[0].shape}")
    print(f"DEBUG: X_train[0].numel() = {X_train[0].numel()}")

    # Create Optuna study
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())

    # Optimize the objective function
    study.optimize(lambda trial: objective(trial, task, X_train, y_train, X_test, y_test, scaler), n_trials=params["optuna_trials"])

    # Log best trial results
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best hyperparameters: {best_trial.params}")
    print(f"Best validation loss: {best_trial.value}")

    # Finalize ClearML task
    task.close()


def main():

    # initialize the task
    task, params = init_task(project_name='MLP Optimization', task_name='optuna controller')

    # retrieve train and test data from mlp_data.py task
    train_test_data = retrieve_data(params["data_task_id"])

    # upack and store data, converting necessary data to tensors
    X_train, y_train, X_test, y_test, scaler = unpack_and_convert(train_test_data)

    # run optuna
    run(params)
        
if __name__ == "__main__":
    main()