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

    def test_add_tag(self):
        """Test adding a tag."""
        self.tracker.add_tag("hyperparameter_tuning")
        self.assertIn("hyperparameter_tuning", self.tracker.get_tags())

    def test_remove_tag(self):
        """Test removing a tag."""
        self.tracker.add_tag("hyperparameter_tuning")
        self.tracker.remove_tag("hyperparameter_tuning")
        self.assertNotIn("hyperparameter_tuning", self.tracker.get_tags())

    def test_get_tags(self):
        """Test retrieving tags."""
        self.tracker.add_tag("baseline")
        self.tracker.add_tag("testing")
        self.assertEqual(self.tracker.get_tags(), ["baseline", "testing"])

    def test_save_load_tags(self):
        """Test saving and loading tags."""
        self.tracker.add_tag("model_1")
        self.tracker.save("experiment_with_tags.json")
        loaded_tracker = ExperimentTracker.load("experiment_with_tags.json")
        self.assertIn("model_1", loaded_tracker.get_tags())
        os.remove("experiment_with_tags.json")
    
    def test_versioning(self):
        """Test version tracking."""
        self.tracker.log_param("param_1", 5)
        self.assertEqual(self.tracker.version, 2)  # Version should increment
        
        self.tracker.log_metric("accuracy", 0.85)
        self.assertEqual(self.tracker.version, 3)  # Version should increment

    def test_experiment_comparison(self):
        """Test experiment comparison with paired t-tests."""
        exp_1 = ExperimentTracker("Exp 1")
        exp_1.log_metric("accuracy", [0.8, 0.82, 0.78])
        exp_1.log_metric("loss", [0.2, 0.22, 0.18])

        exp_2 = ExperimentTracker("Exp 2")
        exp_2.log_metric("accuracy", [0.85, 0.87, 0.83])
        exp_2.log_metric("loss", [0.25, 0.27, 0.23])

        result_accuracy = ExperimentTracker.compare_experiments(exp_1, exp_2, "accuracy")
        self.assertTrue("t_stat" in result_accuracy)
        self.assertTrue("p_value" in result_accuracy)
        self.assertTrue("significant" in result_accuracy)

        result_loss = ExperimentTracker.compare_experiments(exp_1, exp_2, "loss")
        self.assertTrue("t_stat" in result_loss)
        self.assertTrue("p_value" in result_loss)
        self.assertTrue("significant" in result_loss)

    def test_notifications(self):
        """Test notification system for metrics."""
        def condition(value):
            return value > 0.9
    
        self.tracker.set_notification("accuracy", condition, "High accuracy achieved!")
    
        with self.assertLogs(level='INFO') as cm:  # Captures INFO level logs
            self.tracker.log_metric("accuracy", 0.95)
    
        # Verify the notification message appears in the captured logs
        self.assertTrue(
            any("Notification: High accuracy achieved!" in msg for msg in cm.output)
        )



if __name__ == "__main__":
    unittest.main()
