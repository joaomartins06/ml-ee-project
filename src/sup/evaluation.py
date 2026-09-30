import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_model_performance(model_name, y_true, y_pred, num_classes=10):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    overall_acc = np.trace(cm) / np.sum(cm)
    class_totals = np.maximum(cm.sum(axis=1), 1)
    per_class_acc = np.diag(cm) / class_totals
    balanced_acc = np.mean(per_class_acc)

    print(f"\n--- Performance Summary: {model_name} ---")
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")

    return {
        "model": model_name,
        "overall_accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
    }
