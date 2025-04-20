import os
import unittest
from nebulaPy.experiment import ExperimentTracker

class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        """Set up a fresh ExperimentTracker for each test."""
        self.tracker = ExperimentTracker(name="Test Experiment", description="Testing features")

    def test_initialization(self):
        """Test if the ExperimentTracker initializes correctly."""
        self.assertEqual(self.tracker.name, "Test Experiment")
        self.assertEqual(self.tracker.description, "Testing features")
        self.assertIsInstance(self.tracker.timestamp, str)
        self.assertEqual(self.tracker.params, {})
        self.assertEqual(self.tracker.metrics, {})

    def test_log_param(self):
        """Test if parameters are logged correctly."""
        self.tracker.log_param("learning_rate", 0.001)
        self.assertIn("learning_rate", self.tracker.params)
        self.assertEqual(self.tracker.params["learning_rate"], 0.001)

    def test_log_metric(self):
        """Test if metrics are logged correctly."""
        self.tracker.log_metric("accuracy", 0.95)
        self.assertIn("accuracy", self.tracker.metrics)
        self.assertEqual(self.tracker.metrics["accuracy"], 0.95)

    def test_save_and_load(self):
        """Test if saving and loading experiment data works correctly."""
        self.tracker.log_param("batch_size", 32)
        self.tracker.log_metric("loss", 0.05)

        test_file = "test_experiment.json"
        self.tracker.save(test_file)

        loaded_tracker = ExperimentTracker.load(test_file)
        self.assertEqual(loaded_tracker.name, "Test Experiment")
        self.assertEqual(loaded_tracker.params["batch_size"], 32)
        self.assertEqual(loaded_tracker.metrics["loss"], 0.05)

        # Clean up test file
        os.remove(test_file)

if __name__ == "__main__":
    unittest.main()
