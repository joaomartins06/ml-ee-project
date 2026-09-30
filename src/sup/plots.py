from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _save_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_confusion_matrix(cm, model_name, output_dir):
    output_dir = _save_dir(output_dir)
    safe = model_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Digit")
    plt.ylabel("Actual Digit")
    plt.tight_layout()
    path = output_dir / f"confusion_matrix_{safe}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_accuracy_comparison(results, output_dir):
    output_dir = _save_dir(output_dir)

    models = [r["model"] for r in results]
    overall = [r["overall_accuracy"] * 100 for r in results]
    balanced = [r["balanced_accuracy"] * 100 for r in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(
        x - width / 2,
        overall,
        width,
        label="Overall Accuracy"
    )

    bars2 = ax.bar(
        x + width / 2,
        balanced,
        width,
        label="Balanced Accuracy"
    )


    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height - 2,
            f"{height:.2f}%",
            ha="center",
            va="top",
            fontweight="bold",
            color="white"
        )

    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height - 2,
            f"{height:.2f}%",
            ha="center",
            va="top",
            fontweight="bold",
            color="white"
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_title("Model Accuracy Comparison")

    ax.set_xticks(x)
    ax.set_xticklabels(
        models,
        rotation=15,
        ha="right"
    )

    ax.set_ylim(0, 100)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )

    plt.tight_layout()

    path = output_dir / "accuracy_by_model.png"
    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    return path

def plot_per_class_accuracy(results, output_dir):
    output_dir = _save_dir(output_dir)

    classes = np.arange(10)
    width = 0.30

    fig, ax = plt.subplots(figsize=(16, 8))

    bars = []


    for i, result in enumerate(results):
        bar = ax.bar(
            classes + (i - 1) * width,
            result["per_class_accuracy"] * 100,
            width,
            label=result["model"]
        )
        bars.append(bar)

    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height - 2.5,
                f"{height:.1f}%",
                ha="center",
                va="top",
                fontweight="bold",
                color="white",
                fontsize=9
            )

    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Test Accuracy")

    ax.set_xticks(classes)
    ax.set_ylim(0, 100)


    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )

    plt.tight_layout()

    path = output_dir / "per_class_accuracy.png"

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return path

def plot_per_class_heatmap(results, output_dir):
    output_dir = _save_dir(output_dir)
    df = pd.DataFrame([r["per_class_accuracy"] * 100 for r in results],
                      index=[r["model"] for r in results],
                      columns=[f"Class {i}" for i in range(10)])
    plt.figure(figsize=(12, 5)); sns.heatmap(df, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100)
    plt.title("Per-Class Test Accuracy Heatmap"); plt.xlabel("Digit Class"); plt.ylabel("Model"); plt.tight_layout()
    path = output_dir / "per_class_heatmap.png"; plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(); return path


def plot_dnn_loss_curves(history, output_dir):
    output_dir = _save_dir(output_dir); epochs = range(1, len(history["train_losses"]) + 1)
    plt.figure(figsize=(10, 6)); plt.plot(epochs, history["train_losses"], label="Training Loss"); plt.plot(epochs, history["val_losses"], label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("DNN Training and Validation Loss"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = output_dir / "dnn_loss_curves.png"; plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(); return path


def plot_dnn_accuracy_curves(history, output_dir):
    output_dir = _save_dir(output_dir); epochs = range(1, len(history["train_accuracies"]) + 1)
    plt.figure(figsize=(10, 6)); plt.plot(epochs, np.array(history["train_accuracies"]) * 100, label="Training Accuracy"); plt.plot(epochs, np.array(history["val_accuracies"]) * 100, label="Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("DNN Training and Validation Accuracy"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = output_dir / "dnn_accuracy_curves.png"; plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(); return path
