import matplotlib.pyplot as plt

def plot_metric_trends(metrics, labels, title="Metric Trends"):
    """
    Plot line graphs for metric trends across experiments.

    Args:
        metrics (list of list of float): List of metric values for each experiment.
        labels (list of str): Labels for each experiment.
        title (str): Title of the plot.
    """
    for metric, label in zip(metrics, labels):
        plt.plot(metric, label=label)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_metrics(metrics_dict, title="Metric Comparison"):
    """
    Bar chart to compare final metrics across experiments.

    Args:
        metrics_dict (dict): Dictionary where keys are experiment names and values are final metric values.
        title (str): Title of the plot.
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.bar(names, values, color='skyblue')
    plt.title(title)
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
