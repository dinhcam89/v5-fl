"""
Visualization utilities for federated learning metrics.
Creates charts and visualizations to track model performance across rounds.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger("FL-Visualization")

def setup_visualization_dir(base_dir="visualizations"):
    """Create and return a directory for the current training run visualizations."""
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a timestamped directory for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logger.info(f"Created visualization directory: {run_dir}")
    return run_dir

def plot_metrics_over_rounds(metrics_history, metric_names, output_dir, title=None, figsize=(12, 8)):
    """
    Plot specified metrics over training rounds.
    
    Args:
        metrics_history: List of dictionaries containing metrics for each round
        metric_names: List of metric names to plot
        output_dir: Directory to save the plot
        title: Optional title for the plot
        figsize: Figure size tuple (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Extract rounds and metrics
    rounds = []
    metrics_data = {metric: [] for metric in metric_names}
    
    for entry in metrics_history:
        round_num = entry.get("round", 0)
        rounds.append(round_num)
        
        # Extract metrics from different possible locations
        metrics = entry.get("metrics", {})
        if "eval_metrics" in entry:
            metrics.update(entry.get("eval_metrics", {}))
        
        # Add metrics data
        for metric in metric_names:
            # Handle different metric name patterns
            value = None
            if metric in metrics:
                value = metrics[metric]
            elif f"avg_{metric}" in metrics:
                value = metrics[f"avg_{metric}"]
            elif f"global_{metric}" in metrics:
                value = metrics[f"global_{metric}"]
            
            metrics_data[metric].append(value if value is not None else 0)
    
    # Plot each metric
    for metric in metric_names:
        plt.plot(rounds, metrics_data[metric], marker='o', linestyle='-', label=metric)
    
    plt.xlabel('Round')
    plt.ylabel('Value')
    plt.title(title or f"Metrics over {len(rounds)} Rounds")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure integer ticks for rounds
    plt.xticks(rounds)
    
    # Save the plot
    filename = f"metrics_{'_'.join(metric_names)}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved metrics plot to {filepath}")
    return filepath

def plot_confusion_matrix(cm, output_dir, class_names=None, title="Confusion Matrix", figsize=(10, 8)):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix as a numpy array
        output_dir: Directory to save the plot
        class_names: List of class names (optional)
        title: Title for the plot
        figsize: Figure size tuple (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Use seaborn for a nice heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    # Save the plot
    filename = f"confusion_matrix_{datetime.now().strftime('%H-%M-%S')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {filepath}")
    return filepath

def generate_confusion_matrix_from_metrics(metrics_history, output_dir, round_num=None):
    """
    Generate a confusion matrix from metrics history.
    
    Args:
        metrics_history: List of dictionaries containing metrics for each round
        output_dir: Directory to save the plot
        round_num: Specific round number to visualize (None = use latest)
    """
    # Get the metrics for the specified round or the latest round
    if round_num is None and metrics_history:
        metrics_entry = metrics_history[-1]  # Latest round
    else:
        metrics_entry = next((entry for entry in metrics_history if entry.get("round") == round_num), None)
    
    if not metrics_entry:
        logger.warning(f"No metrics found for round {round_num}")
        return None
    
    # Get round number
    round_num = metrics_entry.get("round", "unknown")
    
    # Extract confusion matrix components from metrics
    metrics = metrics_entry.get("metrics", {})
    if "eval_metrics" in metrics_entry:
        metrics.update(metrics_entry.get("eval_metrics", {}))
    
    # Check if we have confusion matrix components
    cm_components = ["true_positives", "false_positives", "true_negatives", "false_negatives"]
    if all(component in metrics for component in cm_components):
        # Construct confusion matrix for binary classification
        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        tn = metrics["true_negatives"]
        fn = metrics["false_negatives"]
        
        confusion_matrix = np.array([
            [tn, fp],
            [fn, tp]
        ])
        
        # Generate and save the plot
        title = f"Confusion Matrix - Round {round_num}"
        class_names = ["Negative", "Positive"]
        return plot_confusion_matrix(confusion_matrix, output_dir, class_names, title)
    else:
        logger.warning(f"Confusion matrix components not found in metrics for round {round_num}")
        return None

def create_metrics_dashboard(metrics_history, output_dir, figsize=(15, 10)):
    """
    Create a comprehensive dashboard with multiple metric plots.
    
    Args:
        metrics_history: List of dictionaries containing metrics for each round
        output_dir: Directory to save the dashboard
        figsize: Figure size tuple (width, height)
    """
    if not metrics_history:
        logger.warning("No metrics history provided for dashboard")
        return None
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Federated Learning Metrics Dashboard - {len(metrics_history)} Rounds", fontsize=16)
    
    # Extract rounds and metrics
    rounds = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    loss = []
    
    for entry in metrics_history:
        round_num = entry.get("round", 0)
        rounds.append(round_num)
        
        # Extract metrics from different possible locations
        metrics = entry.get("metrics", {})
        if "eval_metrics" in entry:
            metrics.update(entry.get("eval_metrics", {}))
        if "eval_loss" in entry:
            loss.append(entry.get("eval_loss", 0))
        elif "loss" in metrics:
            loss.append(metrics.get("loss", 0))
        else:
            loss.append(0)
        
        # Extract performance metrics with fallbacks
        accuracy.append(metrics.get("accuracy", metrics.get("avg_accuracy", metrics.get("global_accuracy", 0))))
        precision.append(metrics.get("precision", metrics.get("avg_precision", metrics.get("global_precision", 0))))
        recall.append(metrics.get("recall", metrics.get("avg_recall", metrics.get("global_recall", 0))))
        f1.append(metrics.get("f1_score", metrics.get("f1", metrics.get("avg_f1_score", metrics.get("global_f1", 0)))))
    
    # Plot accuracy
    axs[0, 0].plot(rounds, accuracy, marker='o', linestyle='-', color='blue')
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].set_xlabel('Round')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].set_xticks(rounds)
    
    # Plot precision, recall, F1
    axs[0, 1].plot(rounds, precision, marker='o', linestyle='-', label='Precision', color='green')
    axs[0, 1].plot(rounds, recall, marker='s', linestyle='-', label='Recall', color='red')
    axs[0, 1].plot(rounds, f1, marker='^', linestyle='-', label='F1', color='purple')
    axs[0, 1].set_title('Precision, Recall, F1')
    axs[0, 1].set_xlabel('Round')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].set_xticks(rounds)
    
    # Plot loss
    axs[1, 0].plot(rounds, loss, marker='o', linestyle='-', color='orange')
    axs[1, 0].set_title('Loss')
    axs[1, 0].set_xlabel('Round')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].set_xticks(rounds)
    
    # Plot confusion matrix components
    tp = []
    fp = []
    tn = []
    fn = []
    
    for entry in metrics_history:
        metrics = entry.get("metrics", {})
        if "eval_metrics" in entry:
            metrics.update(entry.get("eval_metrics", {}))
        
        tp.append(metrics.get("true_positives", 0))
        fp.append(metrics.get("false_positives", 0))
        tn.append(metrics.get("true_negatives", 0))
        fn.append(metrics.get("false_negatives", 0))
    
    axs[1, 1].plot(rounds, tp, marker='o', linestyle='-', label='TP', color='green')
    axs[1, 1].plot(rounds, fp, marker='s', linestyle='-', label='FP', color='red')
    axs[1, 1].plot(rounds, tn, marker='^', linestyle='-', label='TN', color='blue')
    axs[1, 1].plot(rounds, fn, marker='x', linestyle='-', label='FN', color='orange')
    axs[1, 1].set_title('Confusion Matrix Components')
    axs[1, 1].set_xlabel('Round')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].set_xticks(rounds)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the dashboard
    filename = "metrics_dashboard.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved metrics dashboard to {filepath}")
    return filepath

def generate_client_comparison(client_metrics, metric_name, output_dir, title=None, figsize=(12, 8)):
    """
    Generate a comparison chart of metrics across different clients.
    
    Args:
        client_metrics: Dictionary mapping client IDs to lists of round metrics
        metric_name: Name of the metric to compare
        output_dir: Directory to save the plot
        title: Optional title for the plot
        figsize: Figure size tuple (width, height)
    """
    plt.figure(figsize=figsize)
    
    max_rounds = 0
    for client_id, metrics_list in client_metrics.items():
        # Find the maximum number of rounds across all clients
        max_rounds = max(max_rounds, len(metrics_list))
        
        # Extract round numbers and metric values
        rounds = [m.get("round", i+1) for i, m in enumerate(metrics_list)]
        values = [m.get(metric_name, 0) for m in metrics_list]
        
        # Plot this client's metrics
        plt.plot(rounds, values, marker='o', linestyle='-', label=f"Client {client_id[-6:]}")
    
    plt.xlabel('Round')
    plt.ylabel(metric_name.capitalize())
    plt.title(title or f"{metric_name.capitalize()} Comparison Across Clients")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set integer ticks for rounds
    plt.xticks(range(1, max_rounds + 1))
    
    # Save the plot
    filename = f"client_comparison_{metric_name}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved client comparison plot for {metric_name} to {filepath}")
    return filepath

def generate_all_visualizations(metrics_history, client_metrics, output_dir=None):
    """
    Generate all visualizations for a training session.
    
    Args:
        metrics_history: List of dictionaries containing metrics for each round
        client_metrics: Dictionary mapping client IDs to lists of round metrics
        output_dir: Directory to save visualizations (created if None)
    
    Returns:
        Dictionary with paths to all generated visualizations
    """
    # Create visualization directory if needed
    if output_dir is None:
        output_dir = setup_visualization_dir()
    
    # Store paths to all generated visualizations
    visualizations = {}
    
    # Generate basic metric plots
    try:
        visualizations["accuracy"] = plot_metrics_over_rounds(
            metrics_history, ["accuracy"], output_dir, "Accuracy over Rounds")
    except Exception as e:
        logger.error(f"Error generating accuracy plot: {str(e)}")
    
    try:
        visualizations["precision_recall_f1"] = plot_metrics_over_rounds(
            metrics_history, ["precision", "recall", "f1_score"], 
            output_dir, "Precision, Recall, and F1 Score over Rounds")
    except Exception as e:
        logger.error(f"Error generating precision/recall/f1 plot: {str(e)}")
    
    try:
        visualizations["loss"] = plot_metrics_over_rounds(
            metrics_history, ["loss"], output_dir, "Loss over Rounds")
    except Exception as e:
        logger.error(f"Error generating loss plot: {str(e)}")
    
    # Generate confusion matrix for the latest round
    try:
        visualizations["confusion_matrix"] = generate_confusion_matrix_from_metrics(
            metrics_history, output_dir)
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {str(e)}")
    
    # Generate comprehensive dashboard
    try:
        visualizations["dashboard"] = create_metrics_dashboard(
            metrics_history, output_dir)
    except Exception as e:
        logger.error(f"Error generating metrics dashboard: {str(e)}")
    
    # Generate client comparisons
    if client_metrics:
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            try:
                visualizations[f"client_{metric}"] = generate_client_comparison(
                    client_metrics, metric, output_dir)
            except Exception as e:
                logger.error(f"Error generating client comparison for {metric}: {str(e)}")
    
    logger.info(f"Generated {len(visualizations)} visualizations in {output_dir}")
    return visualizations