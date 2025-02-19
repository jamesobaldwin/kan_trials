import numpy as np
import matplotlib.pyplot as plt
from clearml import Task
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import linregress


def init_task(project_name: str, task_name: str) -> tuple[Task, dict[str, any]]:
    task = Task.init(project_name=project_name, task_name=task_name)

    params = {
        "optuna_task_id": "8c8f87f43aa54ce7b508b29cba60c77d",
    }

    params = task.connect(params)
    
    return task, params

def load_artifacts(optuna_task_id):
    """ Retrieve test targets and predictions from the ClearML Optuna Controller Task. """
    optuna_task = Task.get_task(task_id=optuna_task_id)

    preds = optuna_task.artifacts["preds"].get()
    y_test = optuna_task.artifacts["y_test"].get()
    lr = optuna_task.artifacts["lr"].get()

    return preds, y_test, lr

def calculate_metrics(y_test, preds):
    """ Compute R² scores, MAE, and regression slopes for each dimension (X, Y, Z). """
    metrics = {}
    for i, axis in enumerate(['x', 'y', 'z']):
        
        slope, intercept, r_value, _, _ = linregress(y_test[:, i], preds[:, i])
        
        metrics[f'r2_{axis}'] = r_value*r_value
        metrics[f'mae_{axis}'] = mean_absolute_error(y_test[:, i], preds[:, i])
        metrics[f'slope_{axis}'] = slope
        metrics[f'intercept_{axis}'] = intercept
    
    return metrics

def plot_with_legend(ax, true_vals, preds, axis_label):
    """
    Helper function to plot predicted vs. true values with legend.
    
    Parameters:
    ax : matplotlib axis
        The subplot axis to plot on.
    true_vals : np.array
        True target values for the given axis.
    preds : np.array
        Model predictions for the given axis.
    axis_label : str
        Label for the axis (e.g., 'X', 'Y', 'Z').
    
    Returns:
    None
    """
    # Scatter plot of predictions
    scatter = ax.scatter(true_vals, preds, label="Predicted Points")

    # Fit a regression line
    slope, intercept, _, _, _ = linregress(true_vals, preds)
    fit_line = slope * true_vals + intercept
    fit_plot, = ax.plot(true_vals, fit_line, 'r--', lw=2, label=f'Fit line (slope={slope:.2f})')

    # Reference slope=1 line (perfect prediction line)
    ref_plot, = ax.plot(true_vals, true_vals, color='yellow', lw=2, label="slope = 1")

    # Calculate metrics
    r2 = r2_score(true_vals, preds)
    mse = mean_squared_error(true_vals, preds)

    # Create legend using actual elements
    metrics_legend = [
        f'R2 Score: {r2:.04}',
        f'MSE: {mse:.04e}'
    ]

    data_legend = ax.legend(
        handles=[scatter, fit_plot, ref_plot],
        labels=["Predicted Points", f'Fit line (slope={slope:.2f})', "slope = 1"],
        loc='upper left'
    )
    ax.add_artist(data_legend)

        # --- Legend 2: Metrics (Lower Right) ---
    metrics_legend = ax.legend(
        handles=[plt.Line2D([0], [0], color='none')] * len(metrics_legend),  # Invisible line for spacing
        labels=[f'R2 Score: {r2:.04}', f'MSE: {mse:.04e}'],
        loc='lower right',
        frameon=True
    )
    
    # Add the second legend to the subplot
    ax.add_artist(metrics_legend)

    # Axis labels and title
    ax.set_xlabel(f'{axis_label} true')
    ax.set_ylabel(f'{axis_label} pred')
    ax.set_title(f'{axis_label}')
    ax.grid(alpha=0.5)

def plot_results(y_test, preds, metrics, learning_rate):
    """ Generate scatter plots with regression lines and metric annotations. """
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    plot_with_legend(axs[0], test_targets[:, 0], x_preds, 'X')
    plot_with_legend(axs[1], test_targets[:, 1], y_preds, 'Y')
    plot_with_legend(axs[2], test_targets[:, 2], z_preds, 'Z')
    
    fig.suptitle(f'{N_sets} sets, Learning Rate: {learning_rate}')
    plt.close()

    return fig_name  # Return filename for artifact upload

def run():
    """ ClearML Task for Plotting Predictions vs Ground Truth """
    task, params = init_task(project_name="MLP Optimization", task_name="Plot Results")

    # retrieve stored predictions and y_test from the Optuna Controller task
    preds, y_test, lr = load_artifacts(params['optuna_task_id'])

    # compute evaluation metrics
    metrics = calculate_metrics(y_test, preds)

    # log metrics to ClearML
    for key, value in metrics.items():
        task.get_logger().report_scalar(title="Metrics", series=key, value=value, iteration=0)

    # generate and log plots
    plot_results(y_test, preds, metrics, learning_rate=lr)  # Learning rate is a placeholder; retrieve dynamically if needed.

    # save and upload the plot to ClearML
    fig_name = "MLP_results.png"
    plt.savefig(fig_name)
    task.upload_artifact(name="MLP_Plot", artifact_object=fig_name)

    task.close()

if __name__ == "__main__":
    run()
