from pathlib import Path
import numpy as np
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_splits
from src.sup.preprocessing import preprocess_and_pca
from src.sup.models import train_baseline_models
from src.sup.training import train_dnn_with_early_stopping
from src.sup.evaluation import evaluate_model_performance
from src.sup.plots import (
    plot_confusion_matrix, plot_accuracy_comparison,
    plot_per_class_accuracy, plot_per_class_heatmap,
    plot_dnn_loss_curves, plot_dnn_accuracy_curves,
)
from src.sup.results import save_results_csv, save_predictions

SEED = 42
OUTPUT_DIR = Path("outputs") / "supervised"
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    data = load_splits()
    X_train = data["x_train"].numpy()
    y_train = data["y_train"].numpy()
    X_val = data["x_val"].numpy()
    y_val = data["y_val"].numpy()
    X_test = data["x_test"].numpy()
    y_test = data["y_test"].numpy()

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")


    n_components = 50
    X_train_pca, X_val_pca, X_test_pca, _, _ = preprocess_and_pca(
        X_train, X_val, X_test, n_components=n_components, whiten=True)


    baseline_predictions = train_baseline_models(X_train_pca, y_train, X_test_pca)

    results = []
    predictions = []
    for model_name, y_pred in baseline_predictions.items():
        result = evaluate_model_performance(model_name, y_test, y_pred)
        results.append(result)
        predictions.append((model_name, y_test, y_pred))
        plot_confusion_matrix(result["confusion_matrix"], model_name, OUTPUT_DIR)

    #DNN
    checkpoint_file = MODEL_DIR / "best_dnn_model.pth"
    dnn_model, dnn_history = train_dnn_with_early_stopping(
        X_train_pca, y_train, X_val_pca, y_val,
        input_dim=n_components, max_epochs=40, batch_size=64,
        lr=0.001, patience=5, save_path=checkpoint_file,
    )

    dnn_model.eval()
    with torch.no_grad():
        logits = dnn_model(torch.as_tensor(X_test_pca, dtype=torch.float32))
        y_pred_dnn = logits.argmax(dim=1).cpu().numpy()

    dnn_result = evaluate_model_performance("Deep Neural Network (MLP)", y_test, y_pred_dnn)
    results.append(dnn_result)
    predictions.append(("Deep Neural Network (MLP)", y_test, y_pred_dnn))
    plot_confusion_matrix(dnn_result["confusion_matrix"], dnn_result["model"], OUTPUT_DIR)

    #Outputs
    results_df, results_path = save_results_csv(results, OUTPUT_DIR)
    _, predictions_path = save_predictions(predictions, OUTPUT_DIR)
    plot_accuracy_comparison(results, OUTPUT_DIR)
    plot_per_class_accuracy(results, OUTPUT_DIR)
    plot_per_class_heatmap(results, OUTPUT_DIR)
    plot_dnn_loss_curves(dnn_history, OUTPUT_DIR)
    plot_dnn_accuracy_curves(dnn_history, OUTPUT_DIR)

    summary = results_df.copy()
    for col in summary.columns:
        if "accuracy" in col:
            summary[col] *= 100
    print(summary.to_string(index=False))
    print(f"\nResults CSV: {results_path}")
    print(f"Predictions CSV: {predictions_path}")
    print(f"Best DNN model: {checkpoint_file}")
    print(f"All outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
