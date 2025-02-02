import numpy as np
import matplotlib.pyplot as plt
from clearml import Task
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import linregress


def load_artifacts():
    """ Retrieve test targets and predictions from the ClearML Optuna Controller Task. """
    optuna_task = Task.get_task(project_name="MLP Optimization", task_name="Optuna Controller")

    preds = optuna_task.artifacts["preds"].get()
    y_test = optuna_task.artifacts["y_test"].get()
    lr = optuna_task.artifacts["lr"].get()

    return preds, y_test, lr

def calculate_metrics(y_test, preds):
    """ Compute R² scores, MAE, and regression slopes for each dimension (X, Y, Z). """
    metrics = {}
    for i, axis in enumerate(['x', 'y', 'z']):
        metrics[f'r2_{axis}'] = r2_score(y_test[:, i], preds[:, i])
        metrics[f'mae_{axis}'] = mean_absolute_error(y_test[:, i], preds[:, i])
        slope, intercept, _, _, _ = linregress(y_test[:, i], preds[:, i])
        metrics[f'slope_{axis}'] = slope
        metrics[f'intercept_{axis}'] = intercept
    return metrics

def plot_results(y_test, preds, learning_rate):
    """ Generate scatter plots with regression lines and metric annotations. """
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True, sharex=True)
    axes_labels = ['x', 'y', 'z']

    for i, axis in enumerate(axes_labels):
        # Compute linear regression
        slope, intercept, r_value, _, _ = linregress(y_test[:, i], preds[:, i])
        r2 = r_value**2
        mae = np.mean(np.abs(y_test[:, i] - preds[:, i]))

        # Scatter plot
        scatter = axs[i].scatter(y_test[:, i], preds[:, i], alpha=0.5, label="Data Points", edgecolors='k')

        # Regression line
        line_fit = slope * y_test[:, i] + intercept
        fit_line, = axs[i].plot(y_test[:, i], line_fit, 'r--', lw=2, label=f"Fit: y={slope:.2f}x+{intercept:.2f}")

        # Formatting
        axs[i].set_xlabel(f"True {axis.upper()}")
        axs[i].set_ylabel(f"Predicted {axis.upper()}")
        axs[i].set_title(f"{axis.upper()} - Predictions vs Truth")
        axs[i].grid(alpha=0.5)

        # each subplot gets its own unique legend
        legend = axs[i].legend(
            handles=[scatter, fit_line],
            labels=[
                f"Data Points",
                f"Fit: y={slope:.2f}x+{intercept:.2f}\nR²: {r2:.4f}, MAE: {mae:.4e}"
            ],
            loc='upper left',
            fontsize=10,
            frameon=True
        )

    fig.suptitle(f"MLP Predictions vs Ground Truth\nLearning Rate: {learning_rate}", fontsize=16)
    
    # Save figure before showing
    fig_name = "MLP_results.png"
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.show()

    return fig_name  # Return filename for artifact upload

def run(optuna_task_id):
    """ ClearML Task for Plotting Predictions vs Ground Truth """
    task = Task.init(project_name="MLP Optimization", task_name="Plot Results", script_path="MLP_optuna/mlp_plot.py")

    # retrieve stored predictions and y_test from the Optuna Controller task
    preds, y_test, lr = load_artifacts()

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
