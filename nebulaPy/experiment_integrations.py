import tensorflow as tf
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from .experiment import ExperimentTracker


# TensorFlow Integration (Class-based)
class TensorFlowTracker(tf.keras.callbacks.Callback):
    def __init__(self, experiment_tracker: ExperimentTracker):
        super().__init__()
        self._model = None  # Internal model storage
        self.experiment_tracker = experiment_tracker
        self.params = {}  # Initialize params to avoid AttributeError

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if not isinstance(value, tf.keras.Model):
            raise ValueError("The model should be an instance of tf.keras.Model")
        self._model = value

    def on_train_begin(self, logs=None):
        # Log model parameters (e.g., architecture, optimizer config, etc.)
        if self._model:
            self.experiment_tracker.log_param("optimizer", self._model.optimizer.get_config())
            self.experiment_tracker.log_param("batch_size", self.params.get('batch_size', 'Not set'))
            self.experiment_tracker.log_param("epochs", self.params.get('epochs', 'Not set'))
            self.experiment_tracker.log_param("model_summary", str(self._model.summary()))

    def on_epoch_end(self, epoch, logs=None):
        # Log metrics after each epoch
        if logs:
            for metric_name, metric_value in logs.items():
                self.experiment_tracker.log_metric(metric_name, metric_value)


# PyTorch Integration (Class-based)
class PyTorchTracker:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, experiment_tracker: ExperimentTracker, epochs=10):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_tracker = experiment_tracker
        self.epochs = epochs

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        train_loss = running_loss / len(self.train_loader)

        self.experiment_tracker.log_metric(f"epoch_{epoch}_train_loss", train_loss)
        self.experiment_tracker.log_metric(f"epoch_{epoch}_train_accuracy", train_accuracy)

    def _validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / len(self.val_loader.dataset)
        self.experiment_tracker.log_metric(f"epoch_{epoch}_val_loss", val_loss)
        self.experiment_tracker.log_metric(f"epoch_{epoch}_val_accuracy", val_accuracy)


# Scikit-learn Integration (Class-based)
class SklearnTracker:
    def __init__(self, model, X_train, y_train, X_val, y_val, experiment_tracker: ExperimentTracker):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.experiment_tracker = experiment_tracker

    def train(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Log metrics
        train_accuracy = self.model.score(self.X_train, self.y_train)
        self.experiment_tracker.log_metric("train_accuracy", train_accuracy)

        val_accuracy = self.model.score(self.X_val, self.y_val)
        self.experiment_tracker.log_metric("val_accuracy", val_accuracy)

        # Optionally log precision
        y_pred = self.model.predict(self.X_val)
        val_precision = accuracy_score(self.y_val, y_pred)
        self.experiment_tracker.log_metric("val_precision", val_precision)
