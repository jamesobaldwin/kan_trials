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


class KANModel(nn.Module):
    def __init__(self):
        super(KANModel, self).__init__()
        self.layer1 = KAN(width=[4, [3, 3], 8], grid=5, k=3)
        self.layer2 = KAN(width=[8, 3, 3], grid=10, k=3)

    def forward(self, point_set):
        # print("point_set shape:", point_set.shape)  # Debugging line
        layer1_out = self.layer1(
            point_set
        )  # output shape (n, 8) where n is number of points in set
        # print("layer1_out shape:", layer1_out.shape)
        agg_out = torch.mean(layer1_out, dim=0).unsqueeze(0)  # shape (1,8)
        # print("agg_out shape:", agg_out.shape)
        output = self.layer2(agg_out).squeeze(
            0
        )  # shape (3) representing (x,y,z) of vector
        return output


# training loop for DeepSets MLP model
def train_KAN_model(
    X_set, y_set, num_epochs: int = 10, lr: float = 1e-5, verbose: bool = False
):

    # print('Creating KANModel...')
    model = KANModel()
    # print('KANModel created')
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


def create_test_sets(N: int = 1000):
    center = np.array([0.5, 0.5, 0.5])
    test_sets = create_test_sets(N=1000)
    test_targets = np.array(
        [(CoM(test_sets[i]) - center) for i in range(len(test_sets))]
    )
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)
    test_tensors = [
        torch.tensor(test_sets[i], dtype=torch.float32) for i in range(len(test_sets))
    ]
    return test_tensors, test_targets_tensor


def create_train_test_sets(N: int = 100, lr: float = 1e-5):
    N_sets = 100
    learning_rate = 1e-5
    point_sets = create_sets(N=N_sets)
    ps_targets = np.array(
        [(CoM(point_sets[i]) - center) for i in range(len(point_sets))]
    )
    # print(f'shape of ps_targets: {np.shape(ps_targets)}')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    targets_scaled = ds_scaler.fit_transform(ps_targets)
    point_sets_tensor = [
        torch.tensor(point_sets[i], dtype=torch.float32)
        for i in np.arange(len(point_sets))
    ]
    ps_target_tensor = [
        torch.tensor(targets_scaled[i], dtype=torch.float32)
        for i in np.arange(len(targets_scaled))
    ]

    return scaler, point_sets_tensor, ps_target_tensor


def run_trial(
    hidden_layers: List[List[int]],
    input_size: int = 4,
    output_size: int = 3,
    grid: int = 5,
    k: int = 3,
    seed: int = 42,
) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KAN(
        width=[input_size, *hidden_layers, output_size],
        grid=grid,
        k=k,
        seed=seed,
        device=device,
    )
    return model.__repr__()


if __name__ == "__main__":
    o = run_trial(hidden_layers=[3, 0])
    print(o)
