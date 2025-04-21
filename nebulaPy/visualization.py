import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import pandas as pd
import dash
from dash import dcc, html
import random


# --- 1. Metric Trends ---
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


def compare_experiments(experiments, metric_name, title="Comparison"):
    """
    Compare a specified metric across multiple experiments and plot a bar chart.
    
    Args:
        experiments (list): List of ExperimentTracker instances.
        metric_name (str): Name of the metric to compare (e.g., "accuracy").
        title (str): Title for the bar chart.
    """
    names = []
    values = []

    for exp in experiments:
        metric_values = exp.metrics.get(metric_name)
        if metric_values is None:
            print(f"Warning: Metric '{metric_name}' not found in experiment '{exp.name}'. Skipping.")
            continue
        if isinstance(metric_values, list):
            values.append(metric_values[-1])
        else:
            values.append(metric_values)
        names.append(exp.name)

    if not names or not values:
        raise ValueError(f"No valid data to plot for metric '{metric_name}'.")

    plt.bar(names, values, color='skyblue')
    plt.xlabel("Experiments")
    plt.ylabel(metric_name.capitalize())
    plt.title(title)
    plt.show()


# --- 2. Interactive Plots ---
def interactive_plot_metric_trends(metrics, labels, title="Metric Trends"):
    """
    Plot interactive line graphs for metric trends across experiments.

    Args:
        metrics (list of list of float): List of metric values for each experiment.
        labels (list of str): Labels for each experiment.
        title (str): Title of the plot.
    """
    fig = go.Figure()

    for metric, label in zip(metrics, labels):
        fig.add_trace(go.Scatter(x=list(range(len(metric))), y=metric, mode='lines', name=label))

    fig.update_layout(
        title=title,
        xaxis_title="Epochs",
        yaxis_title="Value",
        showlegend=True
    )
    fig.show()


# --- 3. Classification Metrics ---
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix for classification tasks.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        class_names (list): List of class names for the labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="True"))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")
    fig.show()


def plot_roc_curve(y_true, y_pred_prob):
    """
    Plot ROC curve and AUC score for classification tasks.

    Args:
        y_true (list): True labels.
        y_pred_prob (list): Predicted probabilities for positive class.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True
    )
    fig.show()


# --- 4. Regression Metrics ---
def plot_residuals(y_true, y_pred):
    """
    Plot residuals for regression tasks.

    Args:
        y_true (list): True labels (actual values).
        y_pred (list): Predicted values.
    """
    residuals = y_true - y_pred
    sns.residplot(x=y_pred, y=residuals, lowess=False, color="blue", line_kws={'color': 'red'})
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()


# --- 5. Real-Time Monitoring ---
app = dash.Dash(__name__)

x_vals = [i for i in range(100)]
y_vals = [random.random() for _ in range(100)]

app.layout = html.Div([
    dcc.Graph(id='live-update-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1 * 1000,  # Update every 1 second
        n_intervals=0
    )
])


@app.callback(
    dash.dependencies.Output('live-update-graph', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    y_vals.append(random.random())
    y_vals.pop(0)
    
    return {
        'data': [go.Scatter(x=x_vals, y=y_vals, mode='lines+markers')],
        'layout': go.Layout(
            title="Live Metric Monitoring",
            xaxis=dict(title='Epochs'),
            yaxis=dict(title='Metric Value')
        )
    }


# --- 6. Experiment Timeline ---
def plot_experiment_timeline(events):
    """
    Plot experiment timeline with parameter changes and key events.

    Args:
        events (list): List of event dictionaries with 'timestamp', 'event', and 'parameter' keys.
    """
    df = pd.DataFrame(events)
    fig = px.timeline(df, x_start="timestamp", x_end="timestamp", y="event", color="parameter")
    fig.update_layout(title="Experiment Timeline")
    fig.show()


def visualize_experiment(exp, metrics_to_plot=None, limit_versions=10):
    """
    Visualize an experiment's metrics and timeline.
    
    Args:
        exp (ExperimentTracker): The experiment to visualize.
        metrics_to_plot (list): List of metrics to plot (optional).
        limit_versions (int): Limit the number of versions to visualize.
    """
    history = exp.history[-limit_versions:]
    metrics_to_plot = metrics_to_plot or exp.metrics.keys()
    
    for metric in metrics_to_plot:
        data = exp.get_visualization_data([metric])
        print(f"Visualizing {metric}: {data[metric]}")  # Replace with actual plotting code
