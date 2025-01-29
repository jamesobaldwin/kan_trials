import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from clearml import Task, Dataset
from kan import KAN

task = Task.init(
    project_name="KAN deep set optimization", task_name="optuna controller"
)

# load dataset
dataset_path = Dataset.get(dataset_name="train_test_data").get_local_copy()
with open(f"{dataset_path}/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
y = np.load(f"{dataset_path}/y_train.npy")
X_test = np.load(f"{dataset_path}/X_test.npy")
y_test = np.load(f"{dataset_path}/y_test.npy")
scaler = joblib.load(f"{dataset_path}/scaler.pkl")


def objective(trial):
    # optimize number of hidden layers (1, 2, or 3)
    num_layers = trial.suggest_int("num_layers", 1, 3)

    # optimize hidden layers for layer 1
    hidden_layers_1 = []
    for i in range(num_layers):
        a = trial.suggest_int(f"a1_{i}", 0, 4)  # addition nodes, from 0 to 4
        m = trial.suggest_int(f"m1_{i}", 1, 4)  # addition nodes, from 1 to 4
        hidden_layers_1.append([a, m])

    # optimize output size of layer 1 (input size of layer 2)
    out_size_layer1 = trial.suggest_int("out_size_layer1", 3, 10)

    # optimize hidden layers for layer 2
    hidden_layers_2 = []
    for i in range(num_layers):
        a = trial.suggest_int(f"a2_{i}", 0, 4)  # addition nodes, from 0 to 4
        m = trial.suggest_int(f"m2_{i}", 1, 4)  # addition nodes, from 1 to 4
        hidden_layers_2.append([a, m])

    # optimize grid sizes
    grid1 = trial.suggest_int("grid1", 3, 10)
    grid2 = trial.suggest_int("grid2", 3, 10)

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
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
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
    preds_dataset = Dataset.create(dataset_name="predictions", dataset_project="KAN deep set optimization")
    np.save("preds.npy", preds)
    preds_dataset.add_files("preds.npy")
    preds_dataset.upload()
    preds_dataset.finalize()
    
    # calculate and store the MSE
    mse = np.mean((preds - y_test)**2)

    task.get_logger().report_scalar("MSE", "test", value=mse, iteration=1)

    return mse

# === Run Optuna Optimization ===
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

task.close()