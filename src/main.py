from sklearn.utils import compute_class_weight

from preprocessing import load_and_preprocess_data
from train_model import build_cnn_model, build_lstm_model, build_hybrid_model, train_model
from results.visualizations import evaluate_model, plot_comparison_bar_chart, plot_roc_curves, \
    plot_heatmaps, plot_computational_cost, plot_subplots
from sklearn.metrics import confusion_matrix
import numpy as np

DATA_PATH = "data/raw/sql_injection_dataset.csv"

def main():
    X_train, X_test, y_train, y_test, tokenizer, max_len = load_and_preprocess_data(DATA_PATH)
    print(f"Training data size: {len(X_train)} , {len(y_train)}")
    vocab_size = len(tokenizer.word_index) + 1

    metrics = []

    # Train CNN model
    cnn_model = build_cnn_model(vocab_size, max_len)
    cnn_model, cnn_history, cnn_time, cnn_memory = train_model(cnn_model, X_train, y_train, X_test, y_test, "CNN")
    metrics.append({"Model": "CNN", "Training Time": cnn_time, "Memory Usage (MB)": cnn_memory})

   
    # Train LSTM model
    lstm_model = build_lstm_model(vocab_size, max_len)
    lstm_model, lstm_history, lstm_time, lstm_memory = train_model(lstm_model, X_train, y_train, X_test, y_test, "LSTM")
    metrics.append({"Model": "LSTM", "Training Time": lstm_time, "Memory Usage (MB)": lstm_memory})

    # Train Hybrid model
    hybrid_model = build_hybrid_model(vocab_size, max_len)
    hybrid_model, hybrid_history, hybrid_time, hybrid_memory = train_model(hybrid_model, [X_train, X_train], y_train,
                                                                           [X_test, X_test], y_test, "Hybrid")
    metrics.append({"Model": "Hybrid", "Training Time": hybrid_time, "Memory Usage (MB)": hybrid_memory})

    # Evaluate Models
    cnn_metrics = evaluate_model(cnn_model, X_test, y_test, "CNN")
    lstm_metrics = evaluate_model(lstm_model, X_test, y_test, "LSTM")
    hybrid_metrics = evaluate_model(hybrid_model,[ X_test,X_test], y_test, "Hybrid")

    # Plot Combined Subplots
    plot_subplots([cnn_history, lstm_history, hybrid_history], ["CNN", "LSTM", "Hybrid"])

    # Plot Performance Comparison
    plot_comparison_bar_chart([cnn_metrics, lstm_metrics, hybrid_metrics])

    # Confusion Matrices
    cnn_confusion = confusion_matrix(y_test, (cnn_model.predict(X_test) >= 0.5).astype(int))
    lstm_confusion = confusion_matrix(y_test, (lstm_model.predict(X_test) >= 0.5).astype(int))
    hybrid_confusion = confusion_matrix(y_test, (hybrid_model.predict([X_test, X_test]) >= 0.5).astype(int))

    # Plot Combined Heatmaps
    plot_heatmaps([cnn_confusion, lstm_confusion, hybrid_confusion], ["CNN", "LSTM", "Hybrid"])

    # Plot Combined ROC Curves
    plot_roc_curves(
        [y_test, y_test, y_test],
        [
            cnn_metrics["y_pred"],
            lstm_metrics["y_pred"],
            hybrid_metrics["y_pred"]
        ],
        ["CNN", "LSTM", "Hybrid"]
    )

    # Plot computational cost
    plot_computational_cost(metrics)

if __name__ == "__main__":
    main()
