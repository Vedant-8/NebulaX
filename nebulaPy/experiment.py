import json
from datetime import datetime

class ExperimentTracker:
    def __init__(self, name: str, description: str = "", timestamp: str = None):
        """
        Initialize a new experiment tracker.

        Args:
            name (str): Name of the experiment.
            description (str): A short description of the experiment.
            timestamp (str): Timestamp of experiment creation. Defaults to current time.
        """
        self.name = name
        self.description = description
        self.timestamp = timestamp or datetime.now().isoformat()
        self.params = {}
        self.metrics = {}

    def log_param(self, param_name: str, value):
        """Log a hyperparameter."""
        self.params[param_name] = value

    def log_metric(self, metric_name: str, value):
        """Log a metric."""
        self.metrics[metric_name] = value

    def save(self, filepath: str):
        """Save experiment data to a JSON file."""
        experiment_data = {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "parameters": self.params,
            "metrics": self.metrics,
        }
        with open(filepath, "w") as f:
            json.dump(experiment_data, f, indent=4)

    @classmethod
    def load(cls, filepath: str):
        """Load experiment data from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        tracker = cls(data["name"], data["description"], data["timestamp"])
        tracker.params = data["parameters"]
        tracker.metrics = data["metrics"]
        return tracker
