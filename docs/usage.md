# Usage Guide for NebulaPy

This document explains how to use **NebulaPy** for tracking and managing machine learning experiments.

## Basic Usage Example

### Importing and Initializing an Experiment

```python
from nebulaPy.experiment import ExperimentTracker

# Create an ExperimentTracker instance
experiment = ExperimentTracker(name="Model_1", description="First Experiment")

# Log parameters
experiment.log_param("learning_rate", 0.01)
experiment.log_param("batch_size", 32)

# Log metrics
experiment.log_metric("accuracy", 0.8)
experiment.log_metric("loss", 0.2)

# Save the experiment
experiment.save("experiment_1.json")
```
