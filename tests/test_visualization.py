import unittest
import matplotlib.pyplot as plt
from nebulaPy.visualization import plot_metric_trends, compare_metrics

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

if __name__ == '__main__':
    unittest.main()
