from nebulaX.experiment import ExperimentTracker
from nebulaX.storage import save_to_json
from nebulaX.visualization import plot_metric_trends

# Initialize an experiment tracker
experiment = ExperimentTracker(name="Model_1", description="First Experiment")

# Log some hyperparameters and metrics
experiment.log_param("learning_rate", 0.01)
experiment.log_param("batch_size", 32)
experiment.log_metric("accuracy", 0.8)
experiment.log_metric("loss", 0.2)

# Save the experiment to a JSON file
experiment.save("experiment_1.json")

# Example metric trends for visualization (two experiments)
metrics = [
    [0.1, 0.3, 0.5, 0.7, 0.9],  # Experiment 1
    [0.2, 0.4, 0.6, 0.8, 1.0],  # Experiment 2
]
labels = ["Experiment 1", "Experiment 2"]

# Plot the metric trends
plot_metric_trends(metrics, labels, title="Accuracy Trends")
