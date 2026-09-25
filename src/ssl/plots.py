"""Per-run figures (logged as MLflow artifacts). Every function returns a matplotlib Figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import NUM_CLASSES

COLORS = {"frozen": "#0072B2", "finetuned": "#E69F00", "scratch": "#6B6B6B"}
NAMES = {"frozen": "Frozen SSL encoder", "finetuned": "Fine-tuned SSL encoder", "scratch": "From scratch"}
SPLIT_COLORS = {"train": "#56B4E9", "val": "#E69F00", "test": "#009E73"}
GOOD, BAD = "#1a7f37", "#c62828"

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10, "axes.titlesize": 11})


def plot_pretrain_curve(hist, title):
    """Train / validation reconstruction MSE per epoch, best epoch marked."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ep = np.arange(len(hist["train_mse"]))
    ax.plot(ep, hist["train_mse"], color=SPLIT_COLORS["train"], marker="o", ms=3, label="train")
    ax.plot(ep, hist["val_mse"], color=SPLIT_COLORS["val"], marker="o", ms=3, label="validation")
    ax.axvline(hist["best_epoch"], color="k", ls="--", lw=1, label=f"best epoch ({hist['best_epoch']})")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE (per pixel)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_training_curves(res, title):
    """Left: training loss. Right: train / val / test accuracy. Best epoch (by val accuracy) marked."""
    h = res["history"]
    ep = np.arange(len(h["train_loss"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(ep, h["train_loss"], color=SPLIT_COLORS["train"], marker="o", ms=3)
    a1.set_ylabel("cross-entropy (train)")
    a1.set_yscale("log")
    for k in ("train", "val", "test"):
        a2.plot(ep, h[f"{k}_acc"], color=SPLIT_COLORS[k], label=k)
    a2.set_ylabel("accuracy")
    a2.set_ylim(0, 1.02)
    a2.legend(loc="lower right")
    for ax in (a1, a2):
        ax.axvline(res["best_epoch"], color="k", ls="--", lw=1)
        ax.set_xlabel("epoch")
    a2.set_title(f"dashed: best val epoch ({res['best_epoch']})", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_per_class(acc, title):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(range(NUM_CLASSES), acc, color=COLORS["frozen"])
    for i, a in enumerate(acc):
        ax.text(i, a + 0.01, f"{a * 100:.0f}", ha="center", fontsize=8)
    ax.axhline(float(np.nanmean(acc)), color="k", ls="--", lw=1, label=f"mean {np.nanmean(acc) * 100:.1f}%")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xlabel("digit class")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion(pred, y, title):
    cm = torch.bincount(y * NUM_CLASSES + pred, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES).numpy()
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_predictions(x, mask, pred, y, title):
    """Digits as the classifier sees them (masked). Top: first 10 test images. Bottom: first 10 errors."""
    wrong = torch.nonzero(pred != y).flatten()[:10]
    blocks = [("first 10 test images", torch.arange(10)), ("first 10 errors", wrong)]
    fig, axes = plt.subplots(2, 10, figsize=(12, 3.4))
    for r, (name, idx) in enumerate(blocks):
        for c in range(10):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(idx):
                i = int(idx[c])
                ax.imshow((x[i] * mask).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"pred {int(pred[i])} / true {int(y[i])}", fontsize=7, color=GOOD if pred[i] == y[i] else BAD)
        axes[r, 0].text(-0.12, 0.5, name, rotation=90, va="center", ha="right", fontsize=8, transform=axes[r, 0].transAxes)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.05, top=0.85)
    return fig


def plot_reconstructions(x, x_tilde, x_hat, title, n=10):
    """Rows: original x, masked x_tilde (classifier input), reconstruction x_hat."""
    fig, axes = plt.subplots(3, n, figsize=(n * 1.1, 3.8))
    for r, (name, imgs) in enumerate([("x", x), ("x̃ (masked)", x_tilde), ("x̂ (reconstruction)", x_hat)]):
        for i in range(n):
            axes[r, i].imshow(imgs[i].detach().cpu().reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[r, i].axis("off")
        axes[r, 0].text(-0.1, 0.5, name, rotation=90, va="center", ha="right", fontsize=8, transform=axes[r, 0].transAxes)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.06, top=0.9)
    return fig
