# Usage Guide

## Quick Start

### 1. Setting Up an Experiment

```python
from nebulapy.experiment import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(name="My Experiment", description="Testing NebulaPy features")

# Log parameters
tracker.log_params({
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 10
})

# Log initial metrics
tracker.log_metrics({"accuracy": 0.85, "loss": 0.5})
```

### 2. Visualizing Metrics

```python
from nebulapy.visualization import plot_metrics

# Generate a sample plot
plot_metrics(tracker, metric="accuracy")
```

### 3. Saving and Loading Experiments

```python
from nebulapy.storage import save_to_json, load_from_json

# Save experiment to JSON
save_to_json(tracker, "experiment.json")

# Load experiment from JSON
loaded_tracker = load_from_json("experiment.json")
```

## Advanced Features

### Framework-Specific Logging

#### TensorFlow Integration

```python
from nebulapy.integrations.tensorflow import track_tf_experiment

# Assuming `model` is a TensorFlow model
track_tf_experiment(tracker, model, train_data, val_data, epochs=10)
```

#### PyTorch Integration

```python
from nebulapy.integrations.pytorch import track_torch_experiment

# Assuming `model` and `optimizer` are PyTorch objects
track_torch_experiment(tracker, model, optimizer, train_loader, val_loader, epochs=10)
```

#### Scikit-learn Integration

```python
from nebulapy.integrations.sklearn import track_sklearn_experiment

# Assuming `model` is a Scikit-learn model
track_sklearn_experiment(tracker, model, X_train, y_train, X_test, y_test)
```

### Notifications

```python
# Add a notification for when accuracy exceeds 90%
tracker.add_notification(metric="accuracy", threshold=0.9, message="Congrats! Training accuracy exceeded 90%.")
```

### Version Control

```python
# Roll back to a previous version
tracker.rollback(version=1)
```
