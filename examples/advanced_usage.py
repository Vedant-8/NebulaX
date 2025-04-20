from nebulaPy.experiment import ExperimentTracker
from nebulaPy.storage import save_to_sqlite, load_from_sqlite
from nebulaPy.visualization import compare_metrics

# Initialize multiple experiments
experiment_1 = ExperimentTracker(name="Model_1", description="First Experiment")
experiment_2 = ExperimentTracker(name="Model_2", description="Second Experiment")

# Log parameters and metrics for each experiment
experiment_1.log_param("learning_rate", 0.01)
experiment_1.log_param("batch_size", 32)
experiment_1.log_metric("accuracy", 0.8)
experiment_1.log_metric("loss", 0.2)

experiment_2.log_param("learning_rate", 0.02)
experiment_2.log_param("batch_size", 64)
experiment_2.log_metric("accuracy", 0.85)
experiment_2.log_metric("loss", 0.15)

# Save experiments to SQLite database
save_to_sqlite("experiments.db", "experiment_results", {
    "name": experiment_1.name,
    "description": experiment_1.description,
    "timestamp": experiment_1.timestamp,
    "parameters": experiment_1.params,
    "metrics": experiment_1.metrics
})

save_to_sqlite("experiments.db", "experiment_results", {
    "name": experiment_2.name,
    "description": experiment_2.description,
    "timestamp": experiment_2.timestamp,
    "parameters": experiment_2.params,
    "metrics": experiment_2.metrics
})

# Load experiment data from SQLite database
experiment_1_data = load_from_sqlite("experiments.db", "experiment_results", "Model_1")
experiment_2_data = load_from_sqlite("experiments.db", "experiment_results", "Model_2")

# Compare final accuracy of both experiments
metrics_dict = {
    "Model_1": experiment_1_data["metrics"]["accuracy"],
    "Model_2": experiment_2_data["metrics"]["accuracy"],
}

# Visualize the comparison
compare_metrics(metrics_dict, title="Final Accuracy Comparison")
