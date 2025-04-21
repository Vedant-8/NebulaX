import unittest
import tensorflow as tf
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from nebulaX.experiment import ExperimentTracker
from nebulaX.experiment_integrations import TensorFlowTracker, PyTorchTracker, SklearnTracker
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class TestExperimentIntegrations(unittest.TestCase):
    def setUp(self):
        # Initialize ExperimentTracker
        self.experiment_tracker = ExperimentTracker(name="Test Experiment")

    def test_tensorflow_integration(self):
        """Test TensorFlow integration by simulating a model training session."""
        
        # Create a simple TensorFlow model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Initialize TensorFlowTracker with ExperimentTracker
        tf_tracker = TensorFlowTracker(self.experiment_tracker)
        
        # Manually set the model (since we can't access model directly in the callback)
        tf_tracker.model = model
        
        # Simulate the training process
        tf_tracker.params = {'batch_size': 32, 'epochs': 5}  # Set params manually
        tf_tracker.on_train_begin()
    
        # Simulate the end of an epoch with some fake metrics
        tf_tracker.on_epoch_end(0, {"accuracy": 0.85, "loss": 0.45})
        tf_tracker.on_epoch_end(1, {"accuracy": 0.88, "loss": 0.42})
    
        # Check if metrics were logged
        self.assertIn('accuracy', self.experiment_tracker.metrics)
        self.assertIn('loss', self.experiment_tracker.metrics)
        self.assertEqual(self.experiment_tracker.metrics['accuracy'], 0.88)  # Check latest epoch accuracy
        self.assertEqual(self.experiment_tracker.metrics['loss'], 0.42)  # Check latest epoch loss
        
        # Check if parameters were logged
        self.assertIn('optimizer', self.experiment_tracker.params)
        self.assertIn('batch_size', self.experiment_tracker.params)
        self.assertIn('epochs', self.experiment_tracker.params)
        self.assertIn('model_summary', self.experiment_tracker.params)
        
        # Test if the model summary was logged (check that it's not empty)
        self.assertTrue(self.experiment_tracker.params['model_summary'])

    
    def test_pytorch_integration(self):
        """Test PyTorch integration by simulating a training process."""
        
        # Create a simple PyTorch model
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output a single value for binary classification
            nn.Sigmoid()
        )
        
        # Set up optimizer, loss function, and data loaders
        optimizer = optim.Adam(model.parameters())
        loss_fn = nn.BCELoss()
    
        # Generate dummy data for training and validation
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
        
        # Create TensorDataset and DataLoader
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).view(-1, 1))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
    
        # Initialize PyTorchTracker
        pytorch_tracker = PyTorchTracker(model, optimizer, loss_fn, train_loader, val_loader, self.experiment_tracker, epochs=3)
    
        # Train the model and log metrics
        pytorch_tracker.train()
    
        # Check if metrics were logged
        self.assertIn('epoch_0_train_accuracy', self.experiment_tracker.metrics)
        self.assertIn('epoch_1_train_accuracy', self.experiment_tracker.metrics)
        self.assertIn('epoch_2_train_accuracy', self.experiment_tracker.metrics)



    def test_sklearn_integration(self):
        """Test Scikit-learn integration by simulating a model training session."""
        
        # Create a simple sklearn model (e.g., Logistic Regression)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()

        # Generate dummy data for training and validation
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

        # Initialize SklearnTracker
        sklearn_tracker = SklearnTracker(model, X_train, y_train, X_val, y_val, self.experiment_tracker)

        # Train the model and log metrics
        sklearn_tracker.train()

        # Check if metrics were logged
        self.assertIn('train_accuracy', self.experiment_tracker.metrics)
        self.assertIn('val_accuracy', self.experiment_tracker.metrics)
        self.assertIn('val_precision', self.experiment_tracker.metrics)

if __name__ == "__main__":
    unittest.main()
