# NebulaX

**NebulaX** is an open-source Python library designed to empower data scientists, machine learning engineers, and researchers to track, organize, and manage their machine learning experiments with ease. NebulaX integrates seamlessly with popular frameworks like TensorFlow, PyTorch, and Scikit-learn, making experiment management efficient and intuitive.

## Features

### 🧪 Experiment Tracking

- Log hyperparameters, metrics, and other experiment details in an organized manner.
- Rollback to previous versions of experiments with ease.

### 📊 Visualization and Comparisons

- Visualize key metrics (e.g., accuracy, loss) over time.
- Compare metrics across experiments for deeper insights.

### 📂 Storage Options

- Save and load experiment data in JSON or SQLite format for easy sharing and reproducibility.

### ⚙️ Framework Integrations

- Track experiments from TensorFlow, PyTorch, and Scikit-learn models seamlessly with built-in integrations.

### 🔔 Notifications and Alerts

- Set notifications for specific metric thresholds to stay informed during training.

### 🚀 Advanced Features

- Tag experiments for better organization.
- Add, update, and retrieve experiment details effortlessly.

---

## Installation

Install NebulaX using pip:

```bash
pip install NebulaX
```

---

## Usage

### Initialize an Experiment

```python
from nebulaX.experiment import ExperimentTracker

# Create an experiment tracker
experiment = ExperimentTracker(name="Experiment 1", description="Testing model performance.")
experiment.log_param("learning_rate", 0.01)
experiment.log_param("batch_size", 32)
```

### Log Metrics

```python
experiment.log_metric("accuracy", 0.85)
experiment.log_metric("loss", 0.35)
```

### Save and Load Experiments

```python
# Save to JSON
experiment.save("experiment_data.json")

# Load from JSON
loaded_experiment = ExperimentTracker.load("experiment_data.json")
```

### Visualize Metrics

```python
from nebulaX.visualization import compare_experiments

# Compare experiments
compare_experiments([experiment], metric_name="accuracy", title="Accuracy Comparison")
```

### Framework Integrations

#### TensorFlow

```python
from nebulaX.experiment_integrations import TensorFlowTracker

# Wrap your TensorFlow model training
tf_tracker = TensorFlowTracker(experiment_tracker=experiment)
tf_tracker.model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[tf_tracker])
```

#### PyTorch

```python
from nebulaX.experiment_integrations import PyTorchTracker

# Wrap PyTorch training with tracker
pytorch_tracker = PyTorchTracker(model, optimizer, loss_fn, train_loader, val_loader, experiment_tracker=experiment, epochs=10)
pytorch_tracker.train()
```

#### Scikit-learn

```python
from nebulaX.experiment_integrations import SklearnTracker

# Train and track a Scikit-learn model
sklearn_tracker = SklearnTracker(model, X_train, y_train, X_val, y_val, experiment_tracker=experiment)
sklearn_tracker.train()
```

### Set Notifications

```python
experiment.set_notification("accuracy", lambda value: value > 0.9, "Accuracy exceeded 90%!")
```

### Rollback to Previous Versions

```python
experiment.rollback(version=1)
```

---

## Documentation

For detailed guides and examples, refer to the [Documentation](https://github.com/Vedant-8/NebulaX/tree/main/docs) folder, which includes:

- [Index](docs/index.md)
- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)

---

## Contributions

Contributions are welcome! If you’d like to report an issue, suggest a feature, or contribute code, feel free to open a pull request or issue on our [GitHub repository](https://github.com/Vedant-8/NebulaX).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Contact

For inquiries or support, feel free to reach out via [GitHub Issues](https://github.com/Vedant-8/NebulaX/issues).