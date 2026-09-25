"""Cross-condition aggregation, tables and comparison figures (m-level and summary-level MLflow artifacts)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import NUM_CLASSES
from src.ssl.plots import COLORS, NAMES

ORDER = ["frozen", "finetuned", "scratch"]


def _variants(df):
    return [v for v in ORDER if v in set(df["variant"])]


def _pct(f):
    return f"{f * 100:g}"


def aggregate(df):
    """Mean / std over seeds of test accuracy per (m, variant, label_fraction)."""
    agg = (df.groupby(["m", "variant", "label_fraction"])
             .agg(n_labeled=("n_labeled", "first"), n_seeds=("seed", "nunique"),
                  test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
                  val_acc_mean=("val_acc", "mean"))
             .reset_index())
    agg["test_acc_std"] = agg["test_acc_std"].fillna(0.0)
    return agg


def gap_table(df):
    """Per (m, label_fraction): accuracy gain (percentage points) of each SSL variant over scratch, mean/std over seeds."""
    p = df.pivot_table(index=["m", "label_fraction", "seed"], columns="variant", values="test_acc")
    if "scratch" not in p:
        return None
    out = []
    for v in [v for v in ("frozen", "finetuned") if v in p]:
        g = ((p[v] - p["scratch"]).dropna() * 100).groupby(["m", "label_fraction"]).agg(["mean", "std"]).reset_index()
        g["variant"] = v
        out.append(g.rename(columns={"mean": "gain_pp_mean", "std": "gain_pp_std"}))
    if not out:
        return None
    gap = pd.concat(out, ignore_index=True)
    gap["gain_pp_std"] = gap["gain_pp_std"].fillna(0.0)
    return gap


def results_table(agg):
    """Wide 'mean ± std (%)' table: one row per (m, label fraction), one column per variant."""
    a = agg.assign(cell=(agg.test_acc_mean * 100).map("{:.2f}".format) + " ± " + (agg.test_acc_std * 100).map("{:.2f}".format))
    wide = a.pivot(index=["m", "label_fraction", "n_labeled"], columns="variant", values="cell")
    wide = wide[[v for v in ORDER if v in wide.columns]].rename(columns=NAMES).reset_index()
    wide["label_fraction"] = wide["label_fraction"].map(lambda f: f"{_pct(f)}%")
    return wide.rename(columns={"label_fraction": "label fraction", "n_labeled": "# labeled"})


def table_html(wide, title):
    css = ("body{font:14px system-ui,sans-serif;margin:24px;color:#1b2127}table{border-collapse:collapse}"
           "th,td{padding:6px 12px;border-bottom:1px solid #dde2e6;text-align:right}th{background:#f2f4f6}"
           "td:nth-child(-n+3),th:nth-child(-n+3){text-align:left}")
    return (f"<html><head><meta charset='utf-8'><style>{css}</style></head><body><h3>{title}</h3>"
            f"<p>Test accuracy (%), mean ± std over seeds.</p>{wide.to_html(index=False, border=0)}</body></html>")


def _panels(agg, figsize_per=4.2):
    ms = sorted(agg["m"].unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(figsize_per * len(ms), 3.8), sharey=True, squeeze=False)
    return ms, fig, axes[0]


def plot_acc_vs_fraction(agg, title):
    """Test accuracy vs label fraction, one panel per m, one line per variant (error bars: std over seeds)."""
    ms, fig, axes = _panels(agg)
    for ax, m in zip(axes, ms):
        for v in _variants(agg):
            s = agg[(agg.m == m) & (agg.variant == v)].sort_values("label_fraction")
            ax.errorbar(s.label_fraction * 100, s.test_acc_mean, yerr=s.test_acc_std, color=COLORS[v], marker="o",
                        capsize=3, label=NAMES[v])
        ax.set_xscale("log")
        fracs = sorted(agg.label_fraction.unique())
        ax.set_xticks([f * 100 for f in fracs], [f"{_pct(f)}%" for f in fracs])
        ax.minorticks_off()
        ax.set_xlabel("labeled fraction of the 54k train pool")
        ax.set_title(f"m = {m}")
    axes[0].set_ylabel("test accuracy")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_grouped_bars(agg, title):
    """Grouped bars: label fraction on x, one bar per variant, one panel per m."""
    ms, fig, axes = _panels(agg, 5.2)
    variants, fracs = _variants(agg), sorted(agg.label_fraction.unique())
    width = 0.8 / len(variants)
    lo = max(0.0, agg.test_acc_mean.min() - 0.1)
    for ax, m in zip(axes, ms):
        for k, v in enumerate(variants):
            s = agg[(agg.m == m) & (agg.variant == v)].set_index("label_fraction").reindex(fracs)
            xs = np.arange(len(fracs)) + (k - (len(variants) - 1) / 2) * width
            ax.bar(xs, s.test_acc_mean, width, yerr=s.test_acc_std, color=COLORS[v], capsize=2, label=NAMES[v])
            for x, val, err in zip(xs, s.test_acc_mean, s.test_acc_std):
                if not np.isnan(val):
                    ax.text(x, val + err + 0.006, f"{val * 100:.1f}", ha="center", fontsize=6, rotation=90)
        ax.set_xticks(np.arange(len(fracs)), [f"{_pct(f)}%" for f in fracs])
        ax.set_ylim(lo, 1.0)
        ax.set_xlabel("labeled fraction")
        ax.set_title(f"m = {m}")
    axes[0].set_ylabel(f"test accuracy (axis starts at {lo:.2f})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(variants), fontsize=8, frameon=False)
    return fig


def plot_gap(gap, title):
    """Accuracy gain over the from-scratch baseline (percentage points) vs label fraction."""
    ms = sorted(gap["m"].unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 3.8), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], ms):
        for v in [v for v in ("frozen", "finetuned") if v in set(gap.variant)]:
            s = gap[(gap.m == m) & (gap.variant == v)].sort_values("label_fraction")
            ax.errorbar(s.label_fraction * 100, s.gain_pp_mean, yerr=s.gain_pp_std, color=COLORS[v], marker="o",
                        capsize=3, label=NAMES[v])
        ax.axhline(0, color="k", lw=1)
        ax.set_xscale("log")
        fracs = sorted(gap.label_fraction.unique())
        ax.set_xticks([f * 100 for f in fracs], [f"{_pct(f)}%" for f in fracs])
        ax.minorticks_off()
        ax.set_xlabel("labeled fraction")
        ax.set_title(f"m = {m}")
    axes[0][0].set_ylabel("test accuracy minus scratch (pp)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_per_class_heatmap(df, title):
    """Mean per-class test accuracy (%) for every (m, variant, label fraction) condition."""
    cols = [f"acc_class_{c}" for c in range(NUM_CLASSES)]
    g = df.groupby(["m", "variant", "label_fraction"])[cols].mean().reset_index()
    g["order"] = g["variant"].map(ORDER.index)
    g = g.sort_values(["m", "order", "label_fraction"])
    labels = [f"m={r.m}  {r.variant}  {_pct(r.label_fraction)}%" for r in g.itertuples()]
    vals = g[cols].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(g) + 1.6))
    im = ax.imshow(vals, cmap="YlGnBu", vmin=max(0, np.nanmin(vals) - 5), vmax=100, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if vals[i, j] > (np.nanmin(vals) + 100) / 2 + 8 else "black")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("digit class")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="test accuracy (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pretrain_overlay(hists, title):
    """Train (solid) and validation (dashed) reconstruction MSE for every seed of one m. hists: {seed: history}."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, (seed, h) in enumerate(sorted(hists.items())):
        c = f"C{k}"
        ep = np.arange(len(h["train_mse"]))
        ax.plot(ep, h["train_mse"], color=c, label=f"seed {seed} train")
        ax.plot(ep, h["val_mse"], color=c, ls="--", label=f"seed {seed} val")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE (per pixel)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_recon_comparison(recons, title):
    """recons: {m: (x, x_tilde, x_hat)}. Rows: original, then masked + reconstruction for each m."""
    n = 10
    ms = sorted(recons, reverse=True)
    rows = [("original x", recons[ms[0]][0])]
    for m in ms:
        rows += [(f"masked, m={m}", recons[m][1]), (f"recon, m={m}", recons[m][2])]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.1, len(rows) * 1.15 + 0.6))
    for r, (name, imgs) in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(imgs[i].detach().cpu().reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[r, i].axis("off")
        axes[r, 0].text(-0.1, 0.5, name, va="center", ha="right", fontsize=8, transform=axes[r, 0].transAxes)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.13, top=0.92)
    return fig
