import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_absolute_error
from clearml import Task, Dataset

task = Task.init(project_name="KAN deep set optimization", task_name="plotting"
                )

# retrieve required datasets
preds_dataset_path = Dataset.get(dataset_name="predictions").get_local_copy()
preds = np.load(f"{preds_dataset_path}/preds.npy")
dataset_path = Dataset.get(dataset_name="train_test_data").get_local_copy()
test_targets = np.load(f"{dataset_path}/y_test.npy")

# store individual axes values
x_preds = preds[:,0]
y_preds = preds[:,1]
z_preds = preds[:,2]
x_true = test_targets[:,0]
y_true = test_targets[:,1]
z_true = test_targets[:,2]

# calculate metrics

# r2 score
r2x = r2_score(x_true, x_preds)
r2y = r2_score(y_true, y_preds)
r2z = r2_score(z_true, z_preds)
# mean absolute error
x_mae = mean_absolute_error(x_true, x_preds)
y_mae = mean_absolute_error(y_true, y_preds)
z_mae = mean_absolute_error(z_true, z_preds)

# Plot with legends for metrics
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# X axis plot
axs[0].scatter(x_true, x_preds)
# Fit a line to the data and plot it
slope_x, intercept_x, _, _, _ = linregress(x_true, x_preds)
line_x = slope_x * x_true + intercept_x
axs[0].plot(x_true, line_x, 'r--', lw=2)

axs[0].set_xlabel('x true')
axs[0].set_ylabel('x pred')
axs[0].set_title('X')
axs[0].grid(alpha=0.5)

x_metrics_legend = [f'Fit line slope: {slope_x:.2f}', f'R2 Score: {r2x:.04}', f'MAE: {x_mae:.04e}']
axs[0].legend(handles=[plt.Line2D([0], [0], color='none')] * len(x_metrics_legend),
              labels=x_metrics_legend,
              loc='upper left')

# Y axis plot
axs[1].scatter(y_true, y_preds)
# Fit a line to the data and plot it
slope_y, intercept_y, _, _, _ = linregress(y_true, y_preds)
line_y = slope_y * y_true + intercept_y
axs[1].plot(y_true, line_y, 'r--', lw=2, label=f'Fit line (slope={slope_y:.2f})')

axs[1].set_xlabel('y true')
axs[1].set_ylabel('y pred')
axs[1].set_title('Y')
axs[1].grid(alpha=0.5)
# axs[1].legend(loc='best')
y_metrics_legend = [f'Fit line slope: {slope_y:.2f}', f'R2 Score: {r2y:.04}', f'MAE: {y_mae:.04e}']
axs[1].legend(handles=[plt.Line2D([0], [0], color='none')] * len(y_metrics_legend),
              labels=y_metrics_legend,
              loc='upper left')


# Z axis plot
axs[2].scatter(z_true, z_preds)
# Fit a line to the data and plot it
slope_z, intercept_z, _, _, _ = linregress(z_true, z_preds)
line_z = slope_z * z_true + intercept_z
axs[2].plot(z_true, line_z, 'r--', lw=2, label=f'Fit line (slope={slope_z:.2f})')

axs[2].set_xlabel('z true')
axs[2].set_ylabel('z pred')
axs[2].set_title('Z')
axs[2].grid(alpha=0.5)
# axs[2].legend(loc='best')
z_metrics_legend = [f'Fit line slope: {slope_z:.2f}', f'R2 Score: {r2z:.04}', f'MAE: {z_mae:.04e}']
axs[2].legend(handles=[plt.Line2D([0], [0], color='none')] * len(z_metrics_legend),
              labels=z_metrics_legend,
              loc='upper left')

fig.suptitle(f'1000 sets, Learning Rate: 0.001')

plot_filename = f'plot_ds_1000_1e3.png'
plt.savefig(plot_filename)
plt.show()

# save to clearml
task.get_logger().report_image("Test Predictions", "Regression Plots", iteration=1, local_path=plot_filename)

task.close()