#!/usr/bin/env python3
"""
Advanced Usage Example for nebulaPy

This script demonstrates how to use nebulaPy for tracking machine learning experiments
across different frameworks (TensorFlow, PyTorch, and scikit-learn) with advanced
features like:
- Parameter and metric tracking
- Version control and rollback
- Experiment comparison
- Data persistence
- Visualizations
- Notifications
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ML Frameworks
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import nebulaPy modules
from nebulaPy.experiment import ExperimentTracker
from nebulaPy.experiment_integrations import TensorFlowTracker, PyTorchTracker, SklearnTracker
from nebulaPy.storage import save_to_json, load_from_json, save_to_sqlite, load_from_sqlite
from nebulaPy.visualization import (
    plot_metric_trends, 
    compare_metrics, 
    compare_experiments,
    interactive_plot_metric_trends,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_residuals,
    visualize_experiment
)

# Create output directory
os.makedirs("experiment_results", exist_ok=True)

def run_tensorflow_experiment():
    print("\n=== Running TensorFlow Experiment ===")
    
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Create a simple model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create experiment tracker
    experiment = ExperimentTracker(
        name="tensorflow_mnist",
        description="MNIST classification with TensorFlow",
    )
    
    # Add tags
    experiment.add_tag("tensorflow")
    experiment.add_tag("mnist")
    experiment.add_tag("classification")
    
    # Set notification for high accuracy
    experiment.set_notification(
        metric_name="accuracy",
        condition=lambda x: x > 0.95,
        message="High accuracy achieved! Model is performing well."
    )
    
    # Log hyperparameters
    experiment.log_param("batch_size", 128)
    experiment.log_param("epochs", 5)
    experiment.log_param("optimizer", "adam")
    experiment.log_param("learning_rate", 0.001)
    
    # Create TensorFlow tracker
    tf_tracker = TensorFlowTracker(experiment)
    
    # Train the model
    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=5,
        validation_data=(x_test, y_test),
        callbacks=[tf_tracker]
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save experiment
    experiment.save("experiment_results/tensorflow_experiment.json")
    save_to_sqlite("experiment_results/experiments.db", "tensorflow_experiments", {
        "name": experiment.name,
        "description": experiment.description,
        "timestamp": experiment.timestamp,
        "parameters": experiment.params,
        "metrics": experiment.metrics
    })
    
    return experiment

def run_pytorch_experiment():
    print("\n=== Running PyTorch Experiment ===")
    
    # Load and preprocess MNIST data for PyTorch
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train).unsqueeze(1).float()  # Add channel dimension
    y_train_tensor = torch.tensor(y_train).long()
    x_test_tensor = torch.tensor(x_test).unsqueeze(1).float()
    y_test_tensor = torch.tensor(y_test).long()
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Define a simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Initialize model, optimizer, and loss function
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create experiment tracker
    experiment = ExperimentTracker(
        name="pytorch_mnist",
        description="MNIST classification with PyTorch",
    )
    
    # Add tags
    experiment.add_tag("pytorch")
    experiment.add_tag("mnist")
    experiment.add_tag("classification")
    
    # Log hyperparameters
    experiment.log_param("batch_size", 128)
    experiment.log_param("epochs", 5)
    experiment.log_param("optimizer", "adam")
    experiment.log_param("learning_rate", 0.001)
    
    # Create PyTorch tracker and train
    pt_tracker = PyTorchTracker(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=test_loader,
        experiment_tracker=experiment,
        epochs=5
    )
    
    # Train the model
    pt_tracker.train()
    
    # Save experiment
    experiment.save("experiment_results/pytorch_experiment.json")
    
    return experiment

def run_sklearn_experiment():
    print("\n=== Running Scikit-learn Experiment ===")
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create experiment tracker
    experiment = ExperimentTracker(
        name="sklearn_breast_cancer",
        description="Breast cancer classification with RandomForest",
    )
    
    # Add tags
    experiment.add_tag("sklearn")
    experiment.add_tag("random_forest")
    experiment.add_tag("classification")
    experiment.add_tag("breast_cancer")
    
    # Set notification for high accuracy
    experiment.set_notification(
        metric_name="val_accuracy",
        condition=lambda x: x > 0.95,
        message="High validation accuracy achieved!"
    )
    
    # Log hyperparameters - try different values
    for n_estimators in [50, 100, 200]:
        # Create and configure the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        
        # Log the current hyperparameter configuration
        experiment.log_param("n_estimators", n_estimators)
        experiment.log_param("max_features", "sqrt")
        experiment.log_param("random_state", 42)
        
        # Create sklearn tracker
        sk_tracker = SklearnTracker(
            model=model,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_test_scaled,
            y_val=y_test,
            experiment_tracker=experiment
        )
        
        # Train the model and log metrics
        sk_tracker.train()
        
        # Save the current version for comparison
        experiment.log_param("version", experiment.version)
    
    # Save experiment
    experiment.save("experiment_results/sklearn_experiment.json")
    save_to_sqlite("experiment_results/experiments.db", "sklearn_experiments", {
        "name": experiment.name,
        "description": experiment.description,
        "timestamp": experiment.timestamp,
        "parameters": experiment.params,
        "metrics": experiment.metrics
    })
    
    return experiment

def compare_and_visualize_experiments(experiments):
    print("\n=== Comparing and Visualizing Experiments ===")
    
    # 1. Compare metrics across experiments
    tf_metrics = experiments[0].metrics.get("val_accuracy", 0)
    pt_metrics = experiments[1].metrics.get("epoch_4_val_accuracy", 0)
    sk_metrics = experiments[2].metrics.get("val_accuracy", 0)
    
    metrics_comparison = {
        "TensorFlow": tf_metrics,
        "PyTorch": pt_metrics,
        "Scikit-learn": sk_metrics
    }
    
    # Compare metrics using a bar chart
    compare_metrics(metrics_comparison, title="Model Accuracy Comparison")
    
    # 2. Compare experiments directly
    print("\nComparing TensorFlow and PyTorch experiments:")
    result = ExperimentTracker.compare_experiments(
        experiments[0], 
        experiments[1], 
        "val_accuracy"  # This will need to be handled differently in a real scenario
    )
    print(f"Comparison result: {result}")
    
    # 3. Filter experiments by tag
    print("\nFiltering experiments with 'classification' tag:")
    classification_exps = ExperimentTracker.filter_experiments(
        experiments,
        tags=lambda tags: "classification" in tags
    )
    print(f"Found {len(classification_exps)} experiments with 'classification' tag")
    
    # 4. Visualize one experiment in detail
    print("\nVisualizing scikit-learn experiment metrics:")
    visualize_experiment(experiments[2])
    
    # 5. Demonstrate version rollback
    print("\nDemonstrating version rollback:")
    sklearn_exp = experiments[2]
    current_version = sklearn_exp.version
    print(f"Current version: {current_version}")
    
    # Roll back to an earlier version
    rollback_version = current_version - 5  # Arbitrary earlier version
    if rollback_version > 0:
        print(f"Rolling back to version {rollback_version}")
        sklearn_exp.rollback(rollback_version)
        print(f"After rollback, version is now: {sklearn_exp.version}")
        
        # Save the rolled-back version
        sklearn_exp.save("experiment_results/sklearn_experiment_rollback.json")

def load_and_analyze_experiments():
    print("\n=== Loading and Analyzing Saved Experiments ===")
    
    # Load experiments from different storage types
    tf_exp = ExperimentTracker.load("experiment_results/tensorflow_experiment.json")
    pt_exp = ExperimentTracker.load("experiment_results/pytorch_experiment.json")
    
    # Load from SQLite
    try:
        sk_exp_data = load_from_sqlite(
            "experiment_results/experiments.db", 
            "sklearn_experiments", 
            "sklearn_breast_cancer"
        )
        sk_exp = ExperimentTracker(
            name=sk_exp_data["name"],
            description=sk_exp_data["description"],
            timestamp=sk_exp_data["timestamp"]
        )
        sk_exp.params = sk_exp_data["parameters"]
        sk_exp.metrics = sk_exp_data["metrics"]
    except Exception as e:
        print(f"Error loading from SQLite: {e}")
        # Fallback to JSON
        sk_exp = ExperimentTracker.load("experiment_results/sklearn_experiment.json")
    
    print(f"Loaded experiments: {tf_exp.name}, {pt_exp.name}, {sk_exp.name}")
    
    # Analyze version history
    print("\nAnalyzing version history of TensorFlow experiment:")
    tf_history = tf_exp.get_version_history()
    print(f"Total versions: {len(tf_history)}")
    print(f"Latest changes: {tf_history[-3:]}")
    
    # Get experiment tags
    print("\nExperiment tags:")
    print(f"TensorFlow: {tf_exp.get_tags()}")
    print(f"PyTorch: {pt_exp.get_tags()}")
    print(f"Scikit-learn: {sk_exp.get_tags()}")
    
    return [tf_exp, pt_exp, sk_exp]

def main():
    print("=== nebulaPy Advanced Usage Example ===")
    
    # Run experiments with different ML frameworks
    tf_experiment = run_tensorflow_experiment()
    pt_experiment = run_pytorch_experiment()
    sk_experiment = run_sklearn_experiment()
    
    # Compare and visualize the experiments
    compare_and_visualize_experiments([tf_experiment, pt_experiment, sk_experiment])
    
    # Load experiments from storage and analyze them
    loaded_experiments = load_and_analyze_experiments()
    
    print("\n=== Example Complete ===")
    print("Experiment results saved in the 'experiment_results' directory.")

if __name__ == "__main__":
    main()