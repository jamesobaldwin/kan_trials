import optuna

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Task
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle

def init_task(project_name: str, task_name: str) -> tuple[Task, dict[str, any]]:
    task = Task.init(project_name=project_name, task_name=task_name)

    params = {
        "num_point_sets": 50,
        "num_points_per_set": 250,
        "num_epochs": 10,
        "init_size": 512,
        "phi_depth": 2,
        "rho_depth": 3,
        "optimizer": 'Adam',
        "lr": 1e-5,
        "weight_decay": 1e-2,
        "momentum": None,
        "data_task_id": "b666a8559ac14c5e8ad142d571c0c9ba",
    }

    params = task.connect(params)
    
    return task, params

def retrieve_data(task_id) -> dict[str, any]:
    """
    Grab the training and test data created in mlp_data.py.
    """
    data_task = Task.get_task(task_id=task_id)
    
    # Get the artifact object
    artifact = data_task.artifacts["train_test_data"]
    
    # load data into memory
    train_test_data = artifact.get()
    
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

    # assert isinstance(input_size, int), f"Error: input_size is {input_size}, expected int"
    # assert isinstance(init_size, int), f"Error: init_size is {init_size}, expected int"
    
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
        x = x.view(1,-1)  # flatten input to shape [1,input_size]
        out = self.mlp_relu_stack(x)
        return out.squeeze(0)   # output shape [3,]

def trainMLP(config, logger, verbose):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLPModel(
        input_size=config["input_size"], 
        init_size=config["init_size"], 
        phi_depth=config["phi_depth"], 
        rho_depth=config["rho_depth"]
    ).to(device)

    lr = config["lr"]
    weight_decay = config["weight_decay"]
    momentum = config.get("momentum", 0.0)
    
    if config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config['optimizer'] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config['optimizer'] == "SGD":
        momentum = config.get("momentum", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    criterion = nn.MSELoss()
    scaler = config['scaler']
    
    for epoch in range(config["num_epochs"]):
        
        model.train()
        total_train_loss = 0
        total_test_loss = 0
        
        for i, (point_set, target) in enumerate(zip(config["X_train"], config["y_train"])):
            optimizer.zero_grad()
            # print(f"Before sending to model, point_set shape: {point_set.shape}")
            outputs = model(point_set.to(device))
            loss = criterion(outputs, target.to(device))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # log training loss per epoch
        avg_train_loss = total_train_loss / len(config["X_train"])
        logger.report_scalar(title='mse_train_loss', series='train', iteration=epoch, value=avg_train_loss)

        # validate
        model.eval()
        preds = []
        with torch.no_grad():
            for i,(point_set, target) in enumerate(zip(config['X_test'], config['y_test'])):
                pred = model(point_set.to(device))
                pred = scaler.inverse_transform(pred.cpu().numpy().reshape(1, -1)).squeeze()
                preds.append(pred)
                total_test_loss += mean_squared_error(pred, target)
                
        # log test loss per epoch
        avg_test_loss = total_test_loss / len(config["X_test"])
        logger.report_scalar(title='mse_test_loss', series='test', iteration=epoch, value=avg_test_loss)

        if verbose and (epoch + 1) % (0.1 * config["num_epochs"]) == 0:
            print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {avg_epoch_loss:.4e}")
    
    return model, preds


# def test(model, test_tensor, test_targets, scaler):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     predictions = []

#     with torch.no_grad():
#         for point_set in test_tensor:
       
#             output = model(point_set.to(device))
#             output = scaler.inverse_transform(output.cpu().numpy().reshape(1, -1)).squeeze()
#             predictions.append(output)

#     return np.array(predictions)

def upload_artifacts(task, preds, lr, y_test):
    task.upload_artifact("preds", preds)
    task.upload_artifact("lr", lr)
    task.upload_artifact("y_test", y_test)

def main():

    # initialize the task
    task, params = init_task(project_name='MLP Optimization', task_name='Base Trainer')

    logger = task.get_logger()

    # retrieve training and test data
    train_test_data = retrieve_data(params["data_task_id"])
    X_train, y_train, X_test, y_test, scaler = unpack_and_convert(train_test_data)

    # print(f"DEBUG: shape of X_train[0]: {np.shape(X_train[0])}") 

    config = {
        "input_size": X_train[0].numel(),
        "init_size": params['init_size'],
        "phi_depth": params['phi_depth'],
        "rho_depth": params['rho_depth'],
        "num_epochs": params['num_epochs'],
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "lr": params['lr'],
        "weight_decay": params['weight_decay'], 
        "momentum": params.get('momentum', 0.),
        "optimizer": params['optimizer'],
    }

    # perform the training and log the results
    model, preds = trainMLP(
        config=config,  # Pass dictionary instead of many arguments
        logger=logger,
        verbose=True
    )
    
    upload_artifacts(task, preds, config['lr'], config['y_test'])

    logger.flush()
        
if __name__ == "__main__":
    main()