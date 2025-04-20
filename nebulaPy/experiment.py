import json
from datetime import datetime

class ExperimentTracker:
    def __init__(self, name: str, description: str = "", timestamp: str = None, version: int = 1):
        self.name = name
        self.description = description
        self.timestamp = timestamp or datetime.now().isoformat()
        self.version = version
        self.params = {}
        self.metrics = {}
        self.tags = []
        self.history = []

    # Add the get_tags method
    def get_tags(self):
        """Retrieve the tags of the experiment."""
        return self.tags

    def log_param(self, param_name: str, value):
        """Log a hyperparameter."""
        self.params[param_name] = value
        self._log_change("Parameter change", param_name, value)

    def log_metric(self, metric_name: str, value):
        """Log a metric."""
        self.metrics[metric_name] = value
        self._log_change("Metric change", metric_name, value)

    def add_tag(self, tag: str):
        """Add a tag to the experiment."""
        if tag not in self.tags:
            self.tags.append(tag)
            self._log_change("Tag added", tag)

    def remove_tag(self, tag: str):
        """Remove a tag from the experiment."""
        if tag in self.tags:
            self.tags.remove(tag)
            self._log_change("Tag removed", tag)

    def _log_change(self, change_type: str, name: str, value=None):
        """Log changes made to experiment and update version."""
        self.version += 1
        self.history.append({
            "version": self.version,
            "change_type": change_type,
            "name": name,
            "value": value,
            "timestamp": datetime.now().isoformat(),
        })

    def save(self, filepath: str):
        """Save experiment data to a JSON file."""
        experiment_data = {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "version": self.version,
            "parameters": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "history": self.history,  # Save the version history
        }
        with open(filepath, "w") as f:
            json.dump(experiment_data, f, indent=4)

    @classmethod
    def load(cls, filepath: str):
        """Load experiment data from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        tracker = cls(data["name"], data["description"], data["timestamp"], data["version"])
        tracker.params = data["parameters"]
        tracker.metrics = data["metrics"]
        tracker.tags = data["tags"]
        tracker.history = data["history"]
        return tracker

    def get_version_history(self):
        """Get the history of all versions."""
        return self.history
