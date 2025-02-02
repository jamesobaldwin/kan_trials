import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Task
from mlp_main import trainMLP
from mlp_plot import plot_results
from mlp_data import generate_train_test_sets  # Assuming this function creates datasets
from sklearn.preprocessing import MinMaxScaler

def retrieve_best_hyperparams():
    """ Retrieve the best hyperparameters from the Optuna Controller task. """
    optuna_task = Task.get_task(project_name="MLP Optimization", task_name="optuna controller")
    best_params = optuna_task.get_reported_scalars()["best_hyperparams"]
    return best_params

def run_experiment(N, best_params, task):
    """ Train MLP model on dataset of size N using best hyperparameters and log results. """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # generate train and test sets for N
    X_train, X_test = generate_train_test_sets(N)
    scaler = MinMaxScaler(feature_range=(-1,1))
    
    # normalize targets
    y_train = np.array([np.mean(set, axis=0) for set in X_train])
    y_test = np.array([np.mean(set, axis=0) for set in X_test])
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    
    # convert to tensors
    X_train_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in X_train]
    y_train_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in y_train_scaled]
    X_test_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in X_test]

    # train model using best hyperparameters
    model, losses = trainMLP(
        trial=None,  # We are no longer in an Optuna trial
        config={
            "input_size": X_train_tensors[0].shape[1],
            "init_size": best_params["init_size"],
            "phi_depth": best_params["phi_depth"],
            "rho_depth": best_params["rho_depth"],
            "num_epochs": 300,
            "X_train": X_train_tensors,
            "y_train": y_train_tensors,
        },
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        optimizer_type=best_params["optimizer"],
        momentum=best_params.get("momentum", 0.9),  # Only relevant for SGD
        verbose=True
    ).to(device)

    # generate predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for x in X_test_tensors:
            pred = model(x.unsqueeze(0))
            preds.append(pred.cpu().numpy())
    preds = np.array(preds)
    preds = scaler.inverse_transform(preds)

    # run Plotting and Upload to ClearML
    fig_name = plot_results(y_test, preds, learning_rate=best_params["lr"])
    task.upload_artifact(name=f"MLP_Plot_N{N}", artifact_object=fig_name)

def run(optuna_task_id):
    """ Runs final training experiments for different N values using best Optuna hyperparameters. """
    task = Task.init(project_name="MLP Optimization", task_name="final training")

    # retrieve best hyperparameters
    best_params = retrieve_best_hyperparams()

    # train models on different N values
    for N in [500, 1000, 5000]:
        print(f"\n🔹 Running Training for N={N} 🔹\n")
        run_experiment(N, best_params, task)

    task.close()

if __name__ == "__main__":
    run()
