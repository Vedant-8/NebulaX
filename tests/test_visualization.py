import unittest
import matplotlib.pyplot as plt
from nebulaPy.visualization import plot_metric_trends, compare_metrics, interactive_plot_metric_trends, plot_confusion_matrix, plot_roc_curve, plot_residuals, plot_experiment_timeline
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np

class TestVisualization(unittest.TestCase):

    def test_plot_metric_trends(self):
        # Test data: two experiments with their accuracy over epochs
        metrics = [
            [0.1, 0.3, 0.5, 0.7, 0.9],  # Experiment 1
            [0.2, 0.4, 0.6, 0.8, 1.0],  # Experiment 2
        ]
        labels = ["Experiment 1", "Experiment 2"]

        # This will display the plot but we won't actually validate the plot since it's graphical
        try:
            plot_metric_trends(metrics, labels, title="Accuracy Trends")
            plt.close()  # Close the plot after displaying it to prevent it from hanging tests
        except Exception as e:
            self.fail(f"plot_metric_trends raised an exception: {e}")

    def test_compare_metrics(self):
        # Test data: dictionary with final accuracy for each experiment
        metrics_dict = {
            "Experiment 1": 0.9,
            "Experiment 2": 1.0,
            "Experiment 3": 0.85,
        }

        # This will display the plot but we won't validate the plot since it's graphical
        try:
            compare_metrics(metrics_dict, title="Final Accuracy Comparison")
            plt.close()  # Close the plot after displaying it to prevent it from hanging tests
        except Exception as e:
            self.fail(f"compare_metrics raised an exception: {e}")

    def test_interactive_plot_metric_trends(self):
        # Test data: two experiments with their accuracy over epochs
        metrics = [
            [0.1, 0.3, 0.5, 0.7, 0.9],  # Experiment 1
            [0.2, 0.4, 0.6, 0.8, 1.0],  # Experiment 2
        ]
        labels = ["Experiment 1", "Experiment 2"]

        # This will display the interactive plot but we won't validate it since it's graphical
        try:
            interactive_plot_metric_trends(metrics, labels, title="Accuracy Trends (Interactive)")
        except Exception as e:
            self.fail(f"interactive_plot_metric_trends raised an exception: {e}")

    def test_plot_confusion_matrix(self):
        # Test data: true labels and predicted labels for confusion matrix
        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1]
        class_names = ['Class 0', 'Class 1']

        # This will display the confusion matrix plot but we won't validate it since it's graphical
        try:
            plot_confusion_matrix(y_true, y_pred, class_names)
        except Exception as e:
            self.fail(f"plot_confusion_matrix raised an exception: {e}")

    def test_plot_roc_curve(self):
        # Test data: true labels and predicted probabilities for ROC curve
        y_true = [0, 1, 1, 0, 1, 0]
        y_pred_prob = [0.1, 0.9, 0.8, 0.3, 0.7, 0.2]

        # This will display the ROC curve plot but we won't validate it since it's graphical
        try:
            plot_roc_curve(y_true, y_pred_prob)
        except Exception as e:
            self.fail(f"plot_roc_curve raised an exception: {e}")

    def test_plot_residuals(self):
        # Test data: true values and predicted values for residual plot
        y_true = np.array([3.0, 5.0, 7.0, 9.0])
        y_pred = np.array([2.9, 5.1, 6.9, 9.2])

        # This will display the residual plot but we won't validate it since it's graphical
        try:
            plot_residuals(y_true, y_pred)
        except Exception as e:
            self.fail(f"plot_residuals raised an exception: {e}")

    def test_plot_experiment_timeline(self):
        # Test data: List of event dictionaries
        events = [
            {"timestamp": "2025-04-20T10:00:00", "event": "Parameter update", "parameter": "learning_rate"},
            {"timestamp": "2025-04-20T11:00:00", "event": "Metric update", "parameter": "accuracy"},
            {"timestamp": "2025-04-20T12:00:00", "event": "Tag added", "parameter": "test_experiment"},
        ]

        # This will display the experiment timeline plot but we won't validate it since it's graphical
        try:
            plot_experiment_timeline(events)
        except Exception as e:
            self.fail(f"plot_experiment_timeline raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
