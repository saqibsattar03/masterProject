import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns


def plot_histogram(data, column, title, xlabel, ylabel, filename):
    """
    Plot a histogram of the data column and save the figure.
    """
    try:
        plt.figure(figsize=(10, 6))
        data[column].value_counts().plot(kind='bar', color='skyblue', alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Ensure the directory for results exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
        print(f"Histogram saved as {filename}")
    except Exception as e:
        print(f"Error plotting histogram: {e}")


def plot_balanced_histogram(y_train, title, xlabel, ylabel, filename):
    """
    Plot a histogram of the balanced labels and save the figure with enhanced aesthetics.
    """
    try:
        # Create a new figure with a larger size
        plt.figure(figsize=(12, 7))

        # Plot the histogram with enhanced aesthetics
        ax = pd.Series(y_train).value_counts().sort_index().plot(
            kind='bar',
            color=['#4CAF50', '#2196F3'],  # Use a consistent color palette
            alpha=0.85,
            edgecolor='black'
        )

        # Set title and labels with larger font sizes
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)

        # Add value annotations on top of the bars
        for bar in ax.patches:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{int(bar.get_height())}',  # Convert to integer for better readability
                ha='center',
                fontsize=12,
                color='black',
                weight='bold'
            )

        # Add gridlines for better readability
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        # Set a tight layout and save the figure
        plt.tight_layout()

        # Ensure the directory for results exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)  # Save with high DPI for better quality
        plt.close()
        print(f"Balanced histogram saved as {filename}")
    except Exception as e:
        print(f"Error plotting balanced histogram: {e}")


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the trained model on the test set and compute performance metrics.

    Args:
        model: Trained model.
        X_test: Test data.
        y_test: Test labels.
        model_name: Name of the model for reference.

    Returns:
        metrics: A dictionary containing calculated metrics.
    """
    print(f"Evaluating {model_name} model...")
    y_pred = model.predict(X_test).flatten()  # Get predictions as probabilities
    y_pred_labels = (y_pred >= 0.5).astype(int)  # Convert probabilities to binary labels

    # Compute performance metrics
    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels, zero_division=0)
    recall = recall_score(y_test, y_pred_labels, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_labels).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Print metrics
    print(f"---- {model_name} Metrics ----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc,
        "FPR": fpr,
        "FNR": fnr,
        "y_pred": y_pred,
    }


def plot_comparison_bar_chart(metrics_list):
    """
    Plot a bar chart comparing performance metrics of different models.

    Args:
        metrics_list: List of dictionaries containing model metrics.
    """
    # Prepare data for plotting
    model_names = [metrics["Model"] for metrics in metrics_list]
    accuracies = [metrics["Accuracy"] for metrics in metrics_list]
    precisions = [metrics["Precision"] for metrics in metrics_list]
    recalls = [metrics["Recall"] for metrics in metrics_list]
    roc_aucs = [metrics["ROC-AUC"] for metrics in metrics_list]

    # Bar chart for comparison
    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, accuracies, width, label="Accuracy")
    plt.bar(x, precisions, width, label="Precision")
    plt.bar(x + width, recalls, width, label="Recall")
    plt.bar(x + 2 * width, roc_aucs, width, label="ROC-AUC")
    plt.xticks(x, model_names)
    plt.xlabel("Models")
    plt.ylabel("Performance")
    plt.title("Performance Comparison of Models")
    plt.legend(loc="right")
    plt.show()


# def plot_heatmap(confusion, model_name):
#     """
#     Plot a heatmap for the confusion matrix.
#
#     Args:
#         confusion: Confusion matrix as a 2x2 array.
#         model_name: Name of the model.
#     """
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
#     plt.title(f"Confusion Matrix for {model_name}")
#     plt.xlabel("Predicted Labels")
#     plt.ylabel("True Labels")
#     plt.show()


def plot_heatmaps(confusions, model_names):
    """
    Plot combined heatmaps for multiple models in a single frame.

    Args:
        confusions: List of confusion matrices.
        model_names: List of model names.
    """
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))

    for i, (confusion, model_name) in enumerate(zip(confusions, model_names)):
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=axes[i], xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        axes[i].set_title(f"{model_name} Confusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_tests, y_preds, model_names):
    """
    Plot ROC curves for multiple models in a single frame.

    Args:
        y_tests: List of true label arrays.
        y_preds: List of predicted probabilities.
        model_names: List of model names.
    """
    plt.figure(figsize=(10, 8))
    for y_test, y_pred, model_name in zip(y_tests, y_preds, model_names):
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC: {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curves for Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()

# def plot_computational_cost(metrics_list):
#     """
#     Plot a bar chart comparing computational costs of different models.
#
#     Args:
#         metrics_list: List of dictionaries containing computational costs.
#     """
#     model_names = [metrics["Model"] for metrics in metrics_list]
#     training_times = [metrics["Training Time"] for metrics in metrics_list]
#     memory_usages = [metrics["Memory Usage (MB)"] for metrics in metrics_list]
#
#     x = np.arange(len(model_names))
#     width = 0.35
#
#     plt.figure(figsize=(10, 6))
#     plt.bar(x - width / 2, training_times, width, label="Training Time (s)", color="skyblue")
#     plt.bar(x + width / 2, memory_usages, width, label="Memory Usage (MB)", color="lightgreen")
#     plt.xticks(x, model_names)
#     plt.xlabel("Models")
#     plt.ylabel("Cost")
#     plt.title("Computational Cost Comparison")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('src/results/computational_cost_comparison.png')
#     plt.show()
#     print("Computational cost chart saved as 'src/results/computational_cost_comparison.png'")

def plot_computational_cost(metrics_list):
    """
    Plot a bar chart comparing computational costs of different models with enhanced aesthetics.

    Args:
        metrics_list: List of dictionaries containing computational costs.
    """
    model_names = [metrics["Model"] for metrics in metrics_list]
    training_times = [metrics["Training Time"] for metrics in metrics_list]
    memory_usages = [metrics["Memory Usage (MB)"] for metrics in metrics_list]

    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(12, 7))

    bars1_colors = ["#FF5733", "#33FF57", "#3357FF"]
    bars2_colors = ["#F1C40F", "#8E44AD", "#16A085"]

    # Create bars
    bars1 = plt.bar(x - width / 2, training_times, width, label="Training Time (s)", color=bars1_colors, alpha=0.85)
    bars2 = plt.bar(x + width / 2, memory_usages, width, label="Memory Usage (MB)", color=bars2_colors, alpha=0.85)

    # Add labels, title, and legend
    plt.xticks(x, model_names, fontsize=12, rotation=15)
    plt.xlabel("Models", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    plt.title("Computational Cost Comparison of Models", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)

    # Add gridlines
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value annotations on top of bars
    def add_annotations(bars, values):
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.2f}",
                ha="center",
                fontsize=10,
                color="black",
                weight="bold"
            )

    add_annotations(bars1, training_times)
    add_annotations(bars2, memory_usages)

    # Save the plot
    plt.tight_layout()
    plt.savefig('src/results/computational_cost_comparison.png', dpi=300)
    plt.show()
    print("Computational cost chart saved as 'src/results/computational_cost_comparison.png'")


def plot_subplots(histories, model_names):
    """
    Plot training and validation accuracy and loss for multiple models in subplots.
    Args:
        histories (list): List of model history objects.
        model_names (list): List of model names corresponding to the histories.
    """
    fig, axes = plt.subplots(2, len(histories), figsize=(16, 8))

    for i, (history, model_name) in enumerate(zip(histories, model_names)):
        # Accuracy Plot
        axes[0, i].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, i].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, i].set_title(f'{model_name} Accuracy')
        axes[0, i].set_xlabel('Epochs')
        axes[0, i].set_ylabel('Accuracy')
        axes[0, i].legend()
        axes[0, i].grid()

        # Loss Plot
        axes[1, i].plot(history.history['loss'], label='Training Loss')
        axes[1, i].plot(history.history['val_loss'], label='Validation Loss')
        axes[1, i].set_title(f'{model_name} Loss')
        axes[1, i].set_xlabel('Epochs')
        axes[1, i].set_ylabel('Loss')
        axes[1, i].legend()
        axes[1, i].grid()

    plt.tight_layout()
    plt.savefig('src/results/performance_comparison.png')
    plt.show()
    print("Performance comparison plot saved as 'src/results/performance_comparison.png'")

