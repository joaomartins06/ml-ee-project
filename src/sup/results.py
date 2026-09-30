from pathlib import Path
import pandas as pd


def save_results_csv(results, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        row = {"model": result["model"], "overall_accuracy": result["overall_accuracy"], "balanced_accuracy": result["balanced_accuracy"]}
        row.update({f"class_{c}_accuracy": acc for c, acc in enumerate(result["per_class_accuracy"])})
        rows.append(row)
    df = pd.DataFrame(rows)
    path = output_dir / "results_supervised.csv"; df.to_csv(path, index=False)
    return df, path


def save_predictions(predictions, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, y_true, y_pred in predictions:
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            rows.append({"model": model_name, "sample_index": i, "true_label": int(true), "predicted_label": int(pred), "correct": int(true == pred)})
    df = pd.DataFrame(rows)
    path = output_dir / "predictions.csv"; df.to_csv(path, index=False)
    return df, path
