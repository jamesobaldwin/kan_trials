import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from kan import KAN  # Ensure you have this module available
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter("ignore")

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

def create_train_test_sets(lower: int = 200, upper: int = 1000, n: int = 1000, random_state: int = 42):
    """
    Generate train and test sets (lists of arrays) where each array has a variable 
    number of points (between `lower` and `upper`) and each point has 4 coordinates.
    
    Args:
        lower (int): lower bound for the number of points in a set.
        upper (int): upper bound for the number of points in a set.
        n (int): number of point sets.
        random_state (int): seed for reproducibility.
    
    Returns:
        tuple: (train_sets, test_sets) where each is a list of n arrays with shape (N, 4)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random sizes for each set.
    sizes = np.random.randint(lower, upper, size=n)
    total_points = sizes.sum()
    
    # Generate all train and test points at once.
    train_all_points = np.random.rand(total_points, 4)
    test_all_points = np.random.rand(total_points, 4)
    
    # Get the indices to split.
    split_indices = np.cumsum(sizes)[:-1]
    
    # Split into individual sets.
    train_sets = np.split(train_all_points, split_indices)
    test_sets = np.split(test_all_points, split_indices)
    
    return train_sets, test_sets  # a list of numpy arrays, length n, where each array in the list is variable size [N, 4]

def create_targets(point_sets, scale: bool=True):
    center = np.array([0.5,0.5,0.5])
    ps_targets = np.array([(CoM(point_sets[i]) - center) for i in range(len(point_sets))])
    if scale:
        scaler = MinMaxScaler(feature_range=(-1,1))
        targets = scaler.fit_transform(ps_targets)
    else:
        return ps_targets
    return scaler, targets

def generate_data() -> dict:
    
    X_train, X_test = create_train_test_sets()
    scaler, y_train = create_targets(X_train)
    y_test = create_targets(X_test, scale=False)

    train_test_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "scaler": scaler,
            }

    return train_test_data

def unpack_and_convert(train_test_data: dict):
    '''
    Return the object of train test data, with X_train, y_train, and X_test converted to tensors
    '''
    X_train = [torch.tensor(train_test_data['X_train'][i], dtype=torch.float32) for i in np.arange(len(train_test_data['X_train']))]
    y_train = [torch.tensor(train_test_data['y_train'][i], dtype=torch.float32) for i in np.arange(len(train_test_data['y_train']))]
    X_test = [torch.tensor(train_test_data['X_test'][i], dtype=torch.float32) for i in np.arange(len(train_test_data['X_test']))]

    return (X_train, 
           y_train, 
           X_test, 
           train_test_data['y_test'], 
           train_test_data['scaler']
           )

# Define the model class
class KANModel(nn.Module):
    def __init__(self, config):
        super(KANModel, self).__init__()
        self.layer1 = KAN(
            width=[4] + config['hidden_layers_1'] + [config['transition_dim']], grid=config['grid1'], k=3
        )
        self.layer2 = KAN(
            width=[config['transition_dim']] + config['hidden_layers_2'] + [3], grid=config['grid2'], k=3
        )
        self.agg_function = config['agg_function']

    def forward(self, point_set):
        layer1_out = self.layer1(point_set)
        if self.agg_function == 'mean':
            agg_out = torch.mean(layer1_out, dim=0).unsqueeze(0)
        elif self.agg_function == 'sum':
            agg_out = torch.sum(layer1_out, dim=0).unsqueeze(0)
        elif self.agg_function == 'std':
            agg_out = torch.std(layer1_out, dim=0).unsqueeze(0)
        return self.layer2(agg_out).squeeze(0)

# Define the training function
def trainKAN(config, trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KANModel(config).to(device)
    
    if config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == "SGD":
        momentum = config.get("momentum", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=momentum, weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    criterion = nn.MSELoss()
    epoch_losses = []

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        for point_set, target in zip(config['X_train'], config['y_train']):
            optimizer.zero_grad()
            outputs = model(point_set.to(device))
            loss = criterion(outputs, target.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
        # Calculate and store the average loss per set for this epoch
        avg_epoch_loss = total_loss / len(config['X_train'])
        epoch_losses.append(avg_epoch_loss)

        # Report the intermediate value for pruning
        trial.report(avg_epoch_loss, epoch)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Print progress if verbose
        if config['verbose'] and (epoch + 1) % int(0.1 * config['num_epochs']) == 0:
            print(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.4e}")

    return epoch_losses[-1]

# Define the objective function for Optuna
def objective(trial):
    num_layers_1 = trial.suggest_int("num_layers_1", 1, 3)
    num_layers_2 = trial.suggest_int("num_layers_2", 1, 3)
    
    hidden_layers_1 = [(trial.suggest_int(f"a1_{i}", 0, 4), trial.suggest_int(f"m1_{i}", 1, 4)) for i in range(num_layers_1)]
    hidden_layers_2 = [(trial.suggest_int(f"a2_{i}", 0, 4), trial.suggest_int(f"m2_{i}", 1, 4)) for i in range(num_layers_2)]

    # Load training data (replace with actual data loading logic)
    train_test_data = generate_data()
    X_train, y_train, X_test, y_test, scaler = unpack_and_convert(train_test_data)
    
    config = {
        "num_epochs": 30,
        "verbose": False,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "hidden_layers_1": hidden_layers_1,
        "hidden_layers_2": hidden_layers_2,
        "transition_dim": trial.suggest_int("transition_dim", 1, 10),
        "grid1": trial.suggest_int("grid1", 3, 10),
        "grid2": trial.suggest_int("grid2", 3, 10),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"]),
        "lr": trial.suggest_categorical("lr", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-6, 1e-5, 1e-4, 1e-3]),
        "momentum": trial.suggest_categorical("momentum", [0.99, 0.9, 0.8, 0.7]),
        "agg_function": trial.suggest_categorical("agg_function", ["mean", "sum", "std"]),
    }
    final_loss = trainKAN(config, trial)
    return final_loss

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)  # Adjust number of trials as needed

print("Best trial:", study.best_trial)
