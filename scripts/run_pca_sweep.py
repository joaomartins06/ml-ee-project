import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_splits

SEED = 42
OUTPUT_DIR = Path("outputs") / "supervised" / "pca_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPONENTS = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 784]

#Returns validation accuracy, training time and whether it failed to converge.
def fit_logreg(X_train, y_train, X_val, y_val):
    clf = LogisticRegression(max_iter=200, random_state=SEED)
    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_train, y_train)
    not_converged = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return clf.score(X_val, y_val), time.time() - start, not_converged


def main():
    data = load_splits()
    X_train = data["x_train"].numpy().reshape(len(data["x_train"]), -1)
    y_train = data["y_train"].numpy()
    X_val = data["x_val"].numpy().reshape(len(data["x_val"]), -1)
    y_val = data["y_val"].numpy()

    n_features = X_train.shape[1]
    ks = sorted({k for k in COMPONENTS if k <= n_features})
    max_k = max(ks)
    print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)} | Features: {n_features}")
    print(f"Components to test: {ks}")

    # 1) Standardise (fitted on the training set only, as in run_sup.py)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xva_s = scaler.transform(X_val)

    # 2) One PCA with the maximum number of components (no whitening)
    pca = PCA(n_components=max_k, whiten=False, random_state=SEED)
    Ztr = pca.fit_transform(Xtr_s)
    Zva = pca.transform(Xva_s)
    std = np.sqrt(pca.explained_variance_)          # whitening = dividing by this
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100

    # 3) Sweep
    rows = []
    for k in ks:
        for whiten in (True, False):
            Xtr_k, Xva_k = Ztr[:, :k], Zva[:, :k]
            if whiten:
                Xtr_k, Xva_k = Xtr_k / std[:k], Xva_k / std[:k]
            acc, secs, not_conv = fit_logreg(Xtr_k, y_train, Xva_k, y_val)
            rows.append({
                "n_components": k,
                "whiten": whiten,
                "variance_retained_%": cum_var[k - 1],
                "val_accuracy_%": acc * 100,
                "train_time_s": secs,
                "not_converged": not_conv,
            })
            print(f"k={k:4d} | whiten={str(whiten):5s} | variance={cum_var[k - 1]:6.2f}% | "
                  f"val acc={acc * 100:6.2f}% | {secs:5.1f}s | converged={not not_conv}")

    # 4) Baseline without PCA: raw pixels scaled to [0, 1]
    scale = 255.0 if X_train.max() > 1.5 else 1.0
    acc, secs, not_conv = fit_logreg(X_train / scale, y_train, X_val / scale, y_val)
    print(f"No PCA (raw pixels, {n_features} features) | val acc={acc * 100:.2f}% | "
          f"{secs:.1f}s | converged={not not_conv}")

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "pca_sweep_logreg.csv"
    df.to_csv(csv_path, index=False)

    # 5) Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for whiten, label in ((True, "PCA + whitening (used in run_sup.py)"), (False, "PCA, no whitening")):
        sub = df[df["whiten"] == whiten]
        ax.plot(sub["n_components"], sub["val_accuracy_%"], marker="o", label=label)
    ax.axhline(acc * 100, color="gray", linestyle="--", label=f"No PCA, raw pixels ({acc * 100:.2f}%)")
    ax.axvline(50, color="red", linestyle=":", alpha=0.6, label="50 components (current choice)")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Logistic Regression: validation accuracy vs number of PCA components")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "pca_sweep_logreg.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 6) Summary
    best = df.loc[df["val_accuracy_%"].idxmax()]
    ref = df[(df["n_components"] == 50) & (df["whiten"])]
    print("\n===== Summary =====")
    print(f"Best setting: {int(best['n_components'])} components, whiten={best['whiten']} "
          f"-> {best['val_accuracy_%']:.2f}% validation accuracy")
    if len(ref):
        print(f"Current setting (50 components, whiten=True): {ref['val_accuracy_%'].iloc[0]:.2f}%")
    print(f"No PCA baseline: {acc * 100:.2f}%")
    if df["not_converged"].any():
        print("Note: some runs hit max_iter=200 without converging (see 'not_converged' in the CSV).")
    print(f"\nCSV:  {csv_path}\nPlot: {plot_path}")


if __name__ == "__main__":
    main()
