from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from kan import KAN
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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

class KANModel(nn.Module):
    def __init__(self, model1, model2):
        super(KANModel, self).__init__()
        self.layer1 = model1
        self.layer2 = model2

    def forward(self, point_set):
        layer1_out = self.layer1(
            point_set
        )  # output shape (n, 8) where n is number of points in set
        agg_out = torch.mean(layer1_out, dim=0).unsqueeze(0)  # shape (1,8)
        output = self.layer2(agg_out).squeeze(
            0
        )  # shape (3) representing (x,y,z) of vector
        return output


# training loop for DeepSets MLP model
def train_KAN_model(
    X_set,
    y_set,
    num_epochs: int = 10,
    lr: float = 1e-5,
    verbose: bool = False,
    **kwargs
):

    model = get_kan_ds(**kwargs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []

    for epoch in range(num_epochs):
        # print('Training loop entered')
        model.train()
        ds_total_loss = 0  # Accumulate total loss over all sets in this epoch

        for i in range(len(X_set)):
            # print('Point set loop entered')
            point_set = X_set[i]  # One set of points (3D positions and masses)
            true_center = y_set[i]  # True target center for this set

            # zero out the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(point_set)
            # print(f'shape of output: {np.shape(outputs)}   shape of targets: {np.shape(true_center)}')
            loss = criterion(outputs, true_center)

            # accumulate the loss for this set
            ds_total_loss += loss.item()

            # backward pass and optimization
            loss.backward()
            optimizer.step()

        # Calculate and store the average loss per set for this epoch
        avg_epoch_loss = ds_total_loss / len(X_set)
        epoch_losses.append(avg_epoch_loss)

        # Print progress if verbose
        if verbose and (epoch + 1) % int(0.1 * num_epochs) == 0:
            print(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.4e}")

    # Plot the epoch-level training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label="Average Training Loss per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss vs. Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, epoch_losses


# Test model after training
def get_predictions(model, test_data):
    model.eval()
    with torch.no_grad():
        predicted_coords = model(test_data).numpy()
    return predicted_coords


def create_sets(
    lower: int = 200, upper: int = 1000, N: int = 1000, random_state: int = 42
):
    """
    Args:
        low (int): lower bound for number of points in point set
        upper (int): upper bound for number of points in point set
        N (int): number of point sets
        random_state (int): random seed used for reproducibility

    Return:
        point_sets (ndarray): array of point sets of size (N, n, 4) where
        n is a variable number of points in the point set
    """
    if random_state is not None:
        np.random.seed(random_state)

    point_sets = [
        np.column_stack(
            (
                np.random.rand(n := np.random.randint(lower, upper)),  # x
                np.random.rand(n),  # y
                np.random.rand(n),  # z
                np.random.rand(n),  # masses
            )
        )
        for _ in range(N)
    ]

    return point_sets


def create_test_sets(N: int = 1000):
    """
    Args:
        N (int): number of test sets

    Return:
        test_sets (ndarray): array of test sets of size (N, 1000, 4)
    """
    test_sets = [
        np.column_stack(
            (
                np.random.rand(n := 1000),  # x
                np.random.rand(n),  # y
                np.random.rand(n),  # z
                np.random.rand(n),  # masses
            )
        )
        for _ in range(N)
    ]

    return test_sets


def get_test_sets(N: int = 1000):
    center = np.array([0.5, 0.5, 0.5])
    test_sets = create_test_sets(N=N)
    test_targets = np.array(
        [(CoM(test_sets[i]) - center) for i in range(len(test_sets))]
    )
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)
    test_tensors = [
        torch.tensor(test_sets[i], dtype=torch.float32) for i in range(len(test_sets))
    ]
    return test_tensors, test_targets_tensor


def get_train_sets(N: int = 100):
    N_sets = 100
    center = np.array([0.5, 0.5, 0.5])
    point_sets = create_sets(N=N_sets)
    ps_targets = np.array(
        [(CoM(point_sets[i]) - center) for i in range(len(point_sets))]
    )
    scaler = MinMaxScaler(feature_range=(-1, 1))
    targets_scaled = scaler.fit_transform(ps_targets)
    point_sets_tensor = [
        torch.tensor(point_sets[i], dtype=torch.float32)
        for i in np.arange(len(point_sets))
    ]
    ps_target_tensor = [
        torch.tensor(targets_scaled[i], dtype=torch.float32)
        for i in np.arange(len(targets_scaled))
    ]

    return scaler, point_sets_tensor, ps_target_tensor


def init_model(
    hidden_layers: List[List[int]],
    input_size: int = 4,
    output_size: int = 3,
    grid: int = 5,
    k: int = 3,
    seed: int = 42,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KAN(
        width=[input_size, *hidden_layers, output_size],
        grid=grid,
        k=k,
        seed=seed,
        device=device,
    )
    return model


def run_trial(**kwargs):
    return init_model(**kwargs).__repr__()


def get_kan_ds(
    hidden_layers1: List[List[int]],
    hidden_layers2: List[List[int]],
    grid1: int = 5,
    grid2: int = 10,
    input_size: int = 4,
    shared_size: int = 5,
    output_size: int = 3,
    **kwargs,
):
    model1_args = kwargs.copy()
    model1_args.update(
        {
            "input_size": input_size,
            "output_size": shared_size,
            "hidden_layers": hidden_layers1,
            "grid": grid1,
        }
    )
    model2_args = kwargs.copy()
    model2_args.update(
        {
            "input_size": shared_size,
            "output_size": output_size,
            "hidden_layers": hidden_layers2,
            "grid": grid2,
        }
    )
    model1 = init_model(**model1_args)
    model2 = init_model(**model2_args)
    kanModel = KANModel(model1, model2)
    return kanModel


def run_trial2(**kwargs):
    # model = get_kan_ds(**kwargs)
    scaler, X_train, y_train = get_train_sets()
    X_test, y_test = get_test_sets()
    kanModel, losses = train_KAN_model(X_train, y_train, **kwargs)
    
    return kanModel.__repr__()


if __name__ == "__main__":
    o = run_trial2(
        hidden_layers1=[[1, 3]],
        hidden_layers2=[[3, 3]],
        grid1=5,
        grid2=10,
        input_size=4,
        shared_size=5,
        output_size=3,
    )
    print(o)
