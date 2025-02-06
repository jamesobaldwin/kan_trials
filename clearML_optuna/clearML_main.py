import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from clearml import Task
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# import warnings
# warnings.simplefilter("ignore")

def init_task(project_name: str, task_name: str) -> tuple[Task, dict[str, any]]:
    task = Task.init(project_name=project_name, task_name=task_name)

    params = {
        "num_point_sets": 50,                
        "num_points_per_set": 250,
        "num_epochs": 10,
        "num_layers_1": 1,        # number of hidden layers in the first KAN model
        "num_layers_2": 1,        # number of hidden layers in the second KAN model
        "a1_0": 0,                # a1_{i}: addition nodes in ith hidden layer of first KAN model
        "m1_0": 3,                # m1_{i}: multiplication nodes in ith hidden layer of first KAN model
        "a1_1": 2,
        "m1_1": 1,
        "a1_2": 1,
        "m1_2": 3,
        "a2_0": 0,                # a2_{i}: addition nodes in ith hidden layer of second KAN model
        "m2_0": 1,                # m2_{i}: multiplication nodes in ith hidden layer of second KAN model
        "a2_1": 3,
        "m2_1": 1,
        "a2_2": 2,
        "m2_2": 3,
        "transition_dim": 5,      # transition dimension between the two KAN models; output dimension of first model, input dimension of second model
        "grid1": 5,               # grid size for first KAN model
        "grid2": 10,
        "optimizer": 'Adam',
        "lr": 1e-5,
        "weight_decay": 1e-2,
        "momentum": None,
        "agg_function": 'mean',
        "data_task_id": "4688cfe53e1946de98401a33c5ec39c0",    # Data Generator 
    }

    params = task.connect(params)

    return task, params

def retrieve_data(data_task_id: str) -> dict:
    
    # Find the task that generated the artifact
    source_task = Task.get_task(task_id=data_task_id)
    
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

    return (
        X_train, 
        y_train, 
        X_test, 
        train_test_data['y_test'], 
        train_test_data['scaler'],
       )

# save artifacts for use in plotting final model
def save_artifacts(task: Task, preds: np.ndarray, y_test: np.ndarray, lr: float):
    task.upload_artifact("preds", preds)
    task.upload_artifact("y_test", y_test)
    task.upload_artifact("lr", lr)

def generate_layers(params):
    
    # build the list of layer parameters based on num_layers
    hidden_layers_1 = []
    for i in range(params['num_layers_1']):
        a = int(params.get(f'a1_{i}', 0))  # Default value if not set
        m = int(params.get(f'm1_{i}', 0))
        hidden_layers_1.append((a, m))

    hidden_layers_2 = []
    for i in range(params['num_layers_2']):
        a = int(params.get(f'a2_{i}', 0))  # Default value if not set
        m = int(params.get(f'm2_{i}', 0))
        hidden_layers_2.append((a, m))

    return hidden_layers_1, hidden_layers_2

# define the KAN model
class KANModel(nn.Module):
    def __init__(
        self, config
    ):
        super(KANModel, self).__init__()
        self.layer1 = KAN(
            width=[4] + config['hidden_layers_1'] + [ config['transition_dim'] ], grid=config['grid1'], k=3, save_act=False
        )
        self.layer2 = KAN(
            width=[ config['transition_dim'] ] + config['hidden_layers_2'] + [3], grid=config['grid2'], k=3, save_act=False
        )

        self.agg_function = config['agg_function']

    def forward(self, point_set):
        layer1_out = self.layer1(point_set)  # Output shape (n, transition_dim)
        if self.agg_function == 'mean':
            agg_out = torch.mean(layer1_out, dim=0).unsqueeze(0)  # Shape: (1, transition_dim)
        elif self.agg_function == 'sum':
            agg_out = torch.sum(layer1_out, dim=0).unsqueeze(0)  # Shape: (1, transition_dim)
        elif self.agg_function == 'std':
            agg_out = torch.std(layer1_out, dim=0).unsqueeze(0)  # Shape: (1, transition_dim)
        output = self.layer2(agg_out).squeeze(0)  # Shape (3) representing (x,y,z)
        return output

def trainKAN(config, logger, verbose):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KANModel(config).to(device)

    if config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == "SGD":
        momentum = config.get("momentum", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=momentum, weight_decay=config['weight_decay'])
    elif config['optimizer'] == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr=config['lr'])
        # defaults used in original MultKAN docs
        # optimizer = LBFGS(self.get_params(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
        
    criterion = nn.MSELoss()
    scaler = config['scaler']

    for epoch in range(config['num_epochs']):
        
        model.train()
        total_train_loss = 0
        total_test_loss = 0

        for i, (point_set, target) in enumerate(zip(config["X_train"], config["y_train"])):
            optimizer.zero_grad()
            # print(f"DEBUG: shape of point_set before being sent to model: {np.shape(point_set)}")
            outputs = model(point_set.to(device))
            # print(f"DEBUG: shape of outputs: {np.shape(outputs)}")
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
        # with torch.no_grad():
        #     for i,(point_set, target) in enumerate(zip(config['X_test'], config['y_test'])):
        #         pred = model(point_set.to(device))
        #         pred = scaler.inverse_transform(pred.cpu().numpy().reshape(1, -1)).squeeze()
        #         preds.append(pred)
        #         total_test_loss += mean_squared_error(pred, target)
                        
        # # log test loss per epoch
        # avg_test_loss = total_test_loss / len(config["X_test"])
        # logger.report_scalar(title='mse_test_loss', series='test', iteration=epoch, value=avg_test_loss)

        test_losses = []
        with torch.no_grad():
            for i, (point_set, target) in enumerate(zip(config['X_test'], config['y_test'])):
                pred = model(point_set.to(device))
                pred = scaler.inverse_transform(pred.cpu().numpy().reshape(1, -1)).squeeze()
                preds.append(pred)
                test_losses.append(mean_squared_error(pred, target))

        # Compute average MSE over the entire test set
        avg_test_loss = np.mean(test_losses)
        logger.report_scalar(title='mse_test_loss', series='test', iteration=epoch, value=avg_test_loss)

        if verbose and (epoch + 1) % (0.1 * config["num_epochs"]) == 0:
            print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {avg_train_loss:.4e}")
    
    return model, preds


def main():

    task, params = init_task(project_name="KAN-DS optimization", task_name="Template Trainer")

    logger = task.get_logger()
    
    # retrieve training and test data
    train_test_data = retrieve_data(params["data_task_id"])
    X_train, y_train, X_test, y_test, scaler = unpack_and_convert(train_test_data)

    # generate the hidden layers
    hidden_layers_1, hidden_layers_2 = generate_layers(params)

    config = {
        "num_epochs": 10,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "hidden_layers_1": hidden_layers_1,
        "hidden_layers_2": hidden_layers_2,
        "transition_dim": params['transition_dim'],
        "grid1": params['grid1'],
        "grid2": params['grid2'],
        "optimizer": params['optimizer'],
        "lr": params['lr'],
        "weight_decay": params['weight_decay'],
        "momentum": params.get('momentum', 0.),
        "agg_function": params['agg_function'],
    }

    # perform the training and log the results
    model, preds = trainKAN(
        config=config,  # Pass dictionary instead of many arguments
        logger=logger,
        verbose=True
    )

    save_artifacts(task, preds, config['y_test'], config['lr'])

    logger.flush()
    
    task.close()

if __name__ == "__main__":
    main()