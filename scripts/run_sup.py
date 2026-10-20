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
from src.sup.training import (
    set_seed, train_dnn_with_early_stopping,
    train_cnn_with_early_stopping, predict_in_batches,
)
from src.sup.evaluation import evaluate_model_performance
from src.sup.plots import (
    plot_confusion_matrix, plot_accuracy_comparison,
    plot_per_class_accuracy, plot_per_class_heatmap,
    plot_dnn_loss_curves, plot_dnn_accuracy_curves,
    plot_individual_model_accuracy,
)
from src.sup.results import save_results_csv, save_predictions

SEED = 42
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "supervised"
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

    #DNN (on the 50 whitened PCA features)
    set_seed(SEED)
    checkpoint_file = MODEL_DIR / "best_dnn_model.pth"
    dnn_model, dnn_history = train_dnn_with_early_stopping(
        X_train_pca, y_train, X_val_pca, y_val,
        input_dim=n_components, max_epochs=40, batch_size=64,
        lr=0.001, patience=5, save_path=checkpoint_file, seed=SEED,
    )

    y_pred_dnn = predict_in_batches(dnn_model, X_test_pca)

    dnn_result = evaluate_model_performance("Deep Neural Network (MLP)", y_test, y_pred_dnn)
    results.append(dnn_result)
    predictions.append(("Deep Neural Network (MLP)", y_test, y_pred_dnn))
    plot_confusion_matrix(dnn_result["confusion_matrix"], dnn_result["model"], OUTPUT_DIR)

    #CNN (on the raw 28x28 images: PCA would destroy the spatial structure)
    def to_images(X):
        X = np.asarray(X, dtype=np.float32).reshape(-1, 1, 28, 28)
        return X / 255.0 if X.max() > 1.5 else X   # make sure pixels are in [0, 1]

    X_train_img, X_val_img, X_test_img = to_images(X_train), to_images(X_val), to_images(X_test)

    set_seed(SEED)
    cnn_checkpoint = MODEL_DIR / "best_cnn_model.pth"
    cnn_model, cnn_history = train_cnn_with_early_stopping(
        X_train_img, y_train, X_val_img, y_val,
        max_epochs=40, batch_size=64, lr=0.001, patience=5,
        save_path=cnn_checkpoint, seed=SEED,
    )

    y_pred_cnn = predict_in_batches(cnn_model, X_test_img)

    cnn_result = evaluate_model_performance("CNN", y_test, y_pred_cnn)
    results.append(cnn_result)
    predictions.append(("CNN", y_test, y_pred_cnn))
    plot_confusion_matrix(cnn_result["confusion_matrix"], cnn_result["model"], OUTPUT_DIR)
    n_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"CNN trainable parameters: {n_params}")

    #Outputs
    results_df, results_path = save_results_csv(results, OUTPUT_DIR)
    _, predictions_path = save_predictions(predictions, OUTPUT_DIR)
    plot_accuracy_comparison(results, OUTPUT_DIR)
    plot_per_class_accuracy(results, OUTPUT_DIR)
    plot_per_class_heatmap(results, OUTPUT_DIR)
    plot_individual_model_accuracy(results, OUTPUT_DIR)
    plot_dnn_loss_curves(dnn_history, OUTPUT_DIR, name="DNN")
    plot_dnn_accuracy_curves(dnn_history, OUTPUT_DIR, name="DNN")
    plot_dnn_loss_curves(cnn_history, OUTPUT_DIR, name="CNN")
    plot_dnn_accuracy_curves(cnn_history, OUTPUT_DIR, name="CNN")

    summary = results_df.copy()
    for col in summary.columns:
        if "accuracy" in col:
            summary[col] *= 100
    print(summary.to_string(index=False))
    print(f"\nResults CSV: {results_path}")
    print(f"Predictions CSV: {predictions_path}")
    print(f"Best DNN model: {checkpoint_file}")
    print(f"Best CNN model: {cnn_checkpoint}")
    print(f"All outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
