# examples/advanced_usage.py 
import tensorflow as tf
import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nebulaPy.experiment import ExperimentTracker
from nebulaPy.experiment_integrations import TensorFlowTracker, PyTorchTracker, SklearnTracker
from nebulaPy.storage import load_from_sqlite, save_to_sqlite
from nebulaPy.visualization import compare_experiments, plot_metric_trends

# --- Simulating a Real Life Project ---

# Initialize ExperimentTracker
experiment = ExperimentTracker(name="AI Experiment 1", description="Experiment using multiple frameworks (TF, PyTorch, Sklearn).")

# Simulate experiment parameters
experiment.log_param("learning_rate", 0.01)
experiment.log_param("batch_size", 32)
experiment.log_param("epochs", 10)

# --- TensorFlow Model Simulation ---
# Creating a simple TensorFlow model
print("Creating TensorFlow model...")
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])
experiment.add_tag("tensorflow")

# Initialize TensorFlowTracker
tf_tracker = TensorFlowTracker(experiment_tracker=experiment)
tf_tracker.model = tf_model

# Compile and train the model
tf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

# Simulate training process
print("Training TensorFlow model...")
tf_tracker.params = {"batch_size": 32, "epochs": 10}
tf_model.fit(np.random.rand(150, 4), np.random.randint(0, 3, size=(150,)), epochs=10, batch_size=32, callbacks=[tf_tracker])

# --- PyTorch Model Simulation ---
print("Creating PyTorch model...")
# Create a simple PyTorch model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(4, 32)
        self.fc2 = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

pytorch_model = SimpleModel()
experiment.add_tag("pytorch")

# Prepare PyTorch DataLoader (simulated data)
train_data = torch.rand(150, 4)
train_labels = torch.randint(0, 3, (150,))
val_data = torch.rand(50, 4)
val_labels = torch.randint(0, 3, (50,))
train_loader = torch.utils.data.DataLoader(list(zip(train_data, train_labels)), batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(list(zip(val_data, val_labels)), batch_size=32, shuffle=False)

# Initialize PyTorchTracker
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
pytorch_tracker = PyTorchTracker(pytorch_model, optimizer, loss_fn, train_loader, val_loader, experiment_tracker=experiment, epochs=10)

# Train the PyTorch model
print("Training PyTorch model...")
pytorch_tracker.train()

# --- Scikit-learn Model Simulation ---
print("Creating Scikit-learn model...")
# Load Iris dataset
X, y = load_iris(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Train a simple SVM model
svm_model = SVC(kernel='linear', probability=True)
experiment.add_tag("sklearn")

# Initialize SklearnTracker
sklearn_tracker = SklearnTracker(svm_model, X_train, y_train, X_val, y_val, experiment_tracker=experiment)

# Train and log metrics
print("Training Scikit-learn model...")
sklearn_tracker.train()

# --- Comparison of Experiments ---
print("Comparing experiments based on final accuracy...")
# Collect final metrics for comparison
metrics = {
    "TensorFlow": experiment.metrics.get("epoch_9_train_accuracy", 0),
    "PyTorch": experiment.metrics.get("epoch_9_train_accuracy", 0),
    "Scikit-learn": experiment.metrics.get("train_accuracy", 0)
}

compare_experiments([experiment], metric_name="train_accuracy", title="Model Comparison")

# --- Plotting Metric Trends ---
print("Plotting training metrics over time...")
train_metrics_tf = [experiment.metrics.get(f"epoch_{epoch}_train_accuracy", 0) for epoch in range(10)]
train_metrics_pt = [experiment.metrics.get(f"epoch_{epoch}_train_accuracy", 0) for epoch in range(10)]

# Simulate different experiment trends
plot_metric_trends([train_metrics_tf, train_metrics_pt], labels=["TensorFlow", "PyTorch"])

# --- Notifications ---
print("Setting up notifications for metrics...")
experiment.set_notification("train_accuracy", lambda value: value > 0.9, "Training accuracy exceeded 90%! You did it!")
experiment.set_notification("val_accuracy", lambda value: value > 0.8, "Validation accuracy exceeded 80%! Great progress!")

# Simulating metric logging that will trigger notifications
experiment.log_metric("train_accuracy", 0.91)  # This should trigger the notification for training accuracy
experiment.log_metric("val_accuracy", 0.82)   # This should trigger the notification for validation accuracy

# --- Rollback to a Previous Version ---
print("Rolling back experiment to version 1...")
experiment.rollback(1)

# --- Saving and Loading the Experiment Data ---
print("Saving the experiment data to JSON and SQLite...")
experiment.save("experiment_data.json")
save_to_sqlite("experiments.db", "experiments", {
    "name": experiment.name,
    "description": experiment.description,
    "timestamp": experiment.timestamp,
    "parameters": experiment.params,
    "metrics": experiment.metrics
})

# Loading the experiment from JSON and SQLite
loaded_experiment_json = ExperimentTracker.load("experiment_data.json")
loaded_experiment_sqlite = load_from_sqlite("experiments.db", "experiments", experiment.name)

print("Experiment loaded from JSON:", loaded_experiment_json.name)
print("Experiment loaded from SQLite:", loaded_experiment_sqlite["name"])
