"""Run the SSL / semi-supervised grid with nested MLflow runs.

    .venv/bin/python scripts/run_ssl.py            # full grid from configs/ssl.yaml
    .venv/bin/python scripts/run_ssl.py --smoke    # tiny run: check the pipeline and the MLflow layout

View the results (from the repo root, no flags needed):

    .venv/bin/mlflow ui

Run hierarchy in one experiment, one top-level run per m per invocation:

    exp{N}_{pretrain_epochs}_{downstream_epochs}_m{m}     top level: this m's full config + cross-
    │                                                      variant/fraction comparison (tables, bar
    │                                                      charts, gap-over-scratch, per-class heatmap)
    ├── pretrain                                           group: mean ± std over seeds
    │   ├── seed=0                                         loss curve, reconstructions, encoder model
    │   ├── seed=1
    │   └── seed=2
    ├── frozen                                              group: this variant across fractions
    │   ├── frac=1%                                         group: mean ± std over seeds
    │   │   ├── seed=0                                      training curves, confusion matrix, ...
    │   │   ├── seed=1
    │   │   └── seed=2
    │   ├── frac=5%
    │   ├── frac=10%
    │   └── frac=25%
    ├── finetuned  (same shape as frozen)
    └── scratch    (same shape, no link to pretrain)

`N` auto-increments per invocation (shared by every m produced in that run), so successive
experiments (different LRs, schedules, epoch budgets, ...) don't collide in the run list.
"""
import argparse
import copy
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # MLflow's default store (./mlflow.db, ./mlruns) is then what a plain `mlflow ui` reads

from src.data import apply_mask, load_splits, make_mask, stratified_subset  # noqa: E402
from src.ssl import plots, report  # noqa: E402
from src.ssl.downstream import train_classifier  # noqa: E402
from src.ssl.models import architecture_str  # noqa: E402
from src.ssl.pretrain import pretrain  # noqa: E402
from src.ssl.utils import set_seed  # noqa: E402

OUT = ROOT / "outputs"


def log_fig(fig, artifact_file, local_dir=None):
    """Log a figure as an MLflow artifact (and optionally keep a local copy for the report)."""
    mlflow.log_figure(fig, artifact_file, save_kwargs={"dpi": 130, "bbox_inches": "tight"})
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(local_dir / Path(artifact_file).name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def setup_experiment(name):
    """Create the experiment with a local artifact folder (the store's default needs a running server),
    and drop the empty built-in Default experiment so a plain `mlflow ui` opens on real data."""
    client = mlflow.MlflowClient()
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlartifacts" / name).as_uri())
    mlflow.set_experiment(name)
    try:
        if not client.search_runs(["0"], max_results=1):
            client.delete_experiment("0")
    except Exception:
        pass
    return mlflow.get_experiment_by_name(name).experiment_id


def next_exp_index(experiment_id):
    """Smallest unused N for the `exp{N}_...` top-level run name, so reruns don't collide."""
    client = mlflow.MlflowClient()
    runs = client.search_runs([experiment_id], max_results=5000)
    mx = 0
    for r in runs:
        m = re.match(r"^exp(\d+)_", r.info.run_name or "")
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def log_table_artifacts(agg, title, prefix=""):
    wide = report.results_table(agg)
    mlflow.log_text(report.table_html(wide, title), f"{prefix}results_table.html")
    mlflow.log_text(agg.to_csv(index=False), f"{prefix}results_table.csv")
    mlflow.log_table(agg, f"{prefix}results_table.json")


def log_comparison_metrics(agg, gap, prefix=""):
    metrics = {}
    for r in agg.itertuples():
        base = f"{prefix}{r.variant}_frac{report._pct(r.label_fraction)}"
        metrics[f"{base}_test_acc_mean"], metrics[f"{base}_test_acc_std"] = r.test_acc_mean, r.test_acc_std
    if gap is not None:
        for r in gap.itertuples():
            metrics[f"{prefix}{r.variant}_minus_scratch_frac{report._pct(r.label_fraction)}_pp"] = r.gain_pp_mean
    mlflow.log_metrics(metrics)


def run_pretrain_group(m, seeds, pcfg, d, arch, data, mask_by_seed, device, local_dir):
    """`pretrain` group run: trains every seed's encoder, then logs the mean ± std aggregate on top."""
    encoders, pre_run_id, pre_hist = {}, {}, {}
    with mlflow.start_run(run_name="pretrain", nested=True):
        mlflow.set_tags({"stage": "pretrain-group", "m": m})
        for seed in seeds:
            set_seed(seed)
            mask = mask_by_seed[seed]
            with mlflow.start_run(run_name=f"seed={seed}", nested=True) as run:
                pre_run_id[seed] = run.info.run_id
                mlflow.set_tags({"stage": "pretrain", "m": m, "seed": seed})
                mlflow.log_params({"m": m, "mask_seed": seed, "bottleneck_dim": d, "architecture": arch,
                                   "lr": pcfg["lr"], "epochs": pcfg["epochs"], "batch_size": pcfg["batch_size"],
                                   "n_pretrain_images": len(data["x_train"])})
                t0 = time.time()
                encoder, decoder, hist = pretrain(
                    data["x_train"], data["x_val"], mask, pcfg, device,
                    on_epoch=lambda e, tr, va: mlflow.log_metrics({"train_mse": tr, "val_mse": va}, step=e))
                b = hist["best_epoch"]
                mlflow.log_metrics({"best_val_mse": hist["val_mse"][b], "final_train_mse": hist["train_mse"][b],
                                    "best_epoch": b, "epochs_run": len(hist["val_mse"]), "duration_s": time.time() - t0})
                encoders[seed], pre_hist[seed] = encoder, hist
                with torch.no_grad():
                    x = data["x_val"][:10].to(device)
                    xt = apply_mask(x, mask.to(device))
                    xh = decoder(encoder(xt))
                log_fig(plots.plot_pretrain_curve(hist, f"Pretraining loss, m={m}, seed {seed}"), "figures/loss_curve.png")
                log_fig(plots.plot_reconstructions(x, xt, xh, f"Reconstructions, m={m}, seed {seed}"),
                        "figures/reconstructions.png")
                torch.save(encoder.state_dict(), OUT / "checkpoints" / f"enc_m{m}_seed{seed}.pt")
                mlflow.pytorch.log_model(copy.deepcopy(encoder).cpu(), name="encoder",
                                         input_example=xt[:2].cpu().numpy(), serialization_format="pickle")
                print(f"[pretrain m={m} seed={seed}] best val MSE {hist['val_mse'][b]:.5f} @ epoch {b}")

        log_fig(report.plot_pretrain_meanstd(pre_hist, f"Pretraining loss (mean ± std over seeds), m={m}"),
                "figures/pretrain_loss_meanstd.png", local_dir / "pretrain")
        vals = [pre_hist[s]["val_mse"][pre_hist[s]["best_epoch"]] for s in seeds]
        mlflow.log_metrics({"best_val_mse_mean": float(np.mean(vals)), "best_val_mse_std": float(np.std(vals))})
    return encoders, pre_run_id


def run_fraction_group(variant, m, frac, seeds, data, mask_by_seed, dcfg, device,
                        encoders, pre_run_id, local_dir):
    """`frac=X%` group run: trains every seed's classifier, then logs the mean ± std aggregate on top."""
    results, rows = {}, []
    with mlflow.start_run(run_name=f"frac={report._pct(frac)}%", nested=True):
        mlflow.set_tags({"stage": "fraction-group", "variant": variant, "m": m, "label_fraction": frac})
        for seed in seeds:
            set_seed(seed)
            mask = mask_by_seed[seed]
            lab_idx = stratified_subset(data["y_train"], frac, seed)
            n_lab = len(lab_idx)
            # low label budgets: validation set is subsampled to a comparable size
            val_idx = (torch.arange(len(data["y_val"])) if frac >= 0.10
                       else stratified_subset(data["y_val"], n_lab / len(data["y_val"]), seed))
            ds = {"x_lab": data["x_train"][lab_idx], "y_lab": data["y_train"][lab_idx],
                  "x_val": data["x_val"][val_idx], "y_val": data["y_val"][val_idx],
                  "x_test": data["x_test"], "y_test": data["y_test"]}
            cond = f"{variant}, m={m}, {report._pct(frac)}% labels, seed {seed}"

            with mlflow.start_run(run_name=f"seed={seed}", nested=True):
                mlflow.set_tags({"stage": "downstream", "variant": variant, "m": m,
                                 "label_fraction": frac, "seed": seed})
                mlflow.log_params({
                    "m": m, "label_fraction": frac, "label_subset_seed": seed, "mask_seed": seed,
                    "variant": variant, "n_labeled": n_lab, "n_val": len(val_idx),
                    "lr_head": dcfg["lr_head"], "lr_finetune": dcfg["lr_finetune"] if variant == "finetuned" else "",
                    "epochs": dcfg["epochs"], "batch_size": dcfg["batch_size"]})
                if pre_run_id.get(seed) and variant != "scratch":
                    mlflow.set_tag("pretrain_run_id", pre_run_id[seed])
                t0 = time.time()
                res = train_classifier(variant, dcfg, mask, ds, device, pretrained_encoder=encoders.get(seed),
                                       on_epoch=lambda e, mt: mlflow.log_metrics(mt, step=e))
                h, b = res["history"], res["best_epoch"]
                mlflow.log_metrics({
                    "final_test_acc": res["test_acc"], "final_test_loss": res["test_loss"],
                    "best_val_acc": res["val_acc"], "best_val_loss": h["val_loss"][b],
                    "final_train_acc": h["train_acc"][b], "final_train_loss": h["train_loss"][b],
                    "train_val_gap": h["train_acc"][b] - res["val_acc"],
                    "best_epoch": b, "epochs_run": len(h["val_acc"]), "duration_s": time.time() - t0,
                    **{f"test_acc_class_{c}": a for c, a in enumerate(res["per_class_acc"])}})
                log_fig(plots.plot_training_curves(res, f"Training curves: {cond}"), "figures/training_curves.png")
                log_fig(plots.plot_per_class(res["per_class_acc"], f"Per-class test accuracy: {cond}"),
                        "figures/per_class_accuracy.png")
                log_fig(plots.plot_confusion(res["pred"], data["y_test"], f"Confusion matrix: {cond}"),
                        "figures/confusion_matrix.png")
                log_fig(plots.plot_predictions(data["x_test"], mask, res["pred"], data["y_test"], f"Predictions: {cond}"),
                        "figures/predictions.png")
                results[seed] = res
                rows.append({"m": m, "seed": seed, "label_fraction": frac, "n_labeled": n_lab, "variant": variant,
                             "test_acc": res["test_acc"], "val_acc": res["val_acc"], "best_epoch": b,
                             "pretrain_run_id": pre_run_id.get(seed, ""),
                             **{f"acc_class_{c}": a for c, a in enumerate(res["per_class_acc"])}})
                print(f"[{variant} m={m} frac={frac} seed={seed}] n_lab={n_lab} "
                      f"val {res['val_acc']:.4f} test {res['test_acc']:.4f} (epoch {b})")

        log_fig(report.plot_training_meanstd(results, f"{variant}, m={m}, {report._pct(frac)}% labels (mean ± std)"),
                "figures/loss_acc_meanstd.png", local_dir / variant / f"frac{report._pct(frac)}")
        accs = [results[s]["test_acc"] for s in seeds]
        mlflow.log_metrics({"test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs))})
    return rows


def run_variant_group(variant, m, fractions, seeds, data, mask_by_seed, dcfg, device,
                       encoders, pre_run_id, local_dir):
    """`{variant}` group run: one fraction-group per label fraction, then this variant's own
    accuracy-vs-label-fraction figure across fractions."""
    rows = []
    with mlflow.start_run(run_name=variant, nested=True):
        mlflow.set_tags({"stage": "variant-group", "variant": variant, "m": m})
        for frac in fractions:
            rows += run_fraction_group(variant, m, frac, seeds, data, mask_by_seed, dcfg, device,
                                       encoders, pre_run_id, local_dir)
        agg = report.aggregate(pd.DataFrame(rows))
        log_fig(report.plot_acc_vs_fraction(agg, f"{variant}: test accuracy vs label fraction, m={m}"),
                "figures/accuracy_vs_label_fraction.png", local_dir / variant)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "ssl.yaml"))
    ap.add_argument("--smoke", action="store_true", help="tiny run: 2 seeds, 2 fractions, all 3 variants, few epochs")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    exp_name = cfg["experiment"]
    if args.smoke:
        exp_name += "-smoke"
        cfg["seeds"], cfg["label_fractions"] = cfg["seeds"][:2], [0.01, 0.10]
        cfg["variants"] = ["frozen", "finetuned", "scratch"]
        cfg["pretrain"].update(epochs=3)
        cfg["downstream"].update(epochs=5)

    OUT.mkdir(exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    local_dir = OUT / ("report_smoke" if args.smoke else "report")
    exp_id = setup_experiment(exp_name)
    exp_n = next_exp_index(exp_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splits()
    d, hidden = cfg["model"]["d"], cfg["model"]["hidden"]
    arch = architecture_str(d, hidden)
    needs_pretrained = any(v != "scratch" for v in cfg["variants"])
    pretrain_epochs, downstream_epochs = cfg["pretrain"]["epochs"], cfg["downstream"]["epochs"]
    results_path = OUT / ("results_ssl_smoke.csv" if args.smoke else "results_ssl.csv")
    all_rows = []
    t_start = time.time()

    for m in cfg["m_values"]:
        top_name = f"exp{exp_n}_{pretrain_epochs}_{downstream_epochs}_m{m}"
        m_local_dir = local_dir / top_name
        with mlflow.start_run(run_name=top_name):
            mlflow.set_tags({"stage": "config", "m": m, "exp_index": exp_n})
            mlflow.log_params({
                "m": m, "d": d, "architecture": arch, "label_fractions": str(cfg["label_fractions"]),
                "seeds": str(cfg["seeds"]), "variants": str(cfg["variants"]),
                "n_train_pool": len(data["x_train"]), "n_val": len(data["x_val"]), "n_test": len(data["x_test"]),
                **{f"pretrain_{k}": v for k, v in cfg["pretrain"].items()},
                **{f"downstream_{k}": v for k, v in cfg["downstream"].items()}})

            mask_by_seed = {seed: make_mask(m, seed) for seed in cfg["seeds"]}
            pcfg, dcfg = {**cfg["model"], **cfg["pretrain"]}, {**cfg["model"], **cfg["downstream"]}
            rows_m = []

            encoders, pre_run_id = ({}, {})
            if needs_pretrained:
                encoders, pre_run_id = run_pretrain_group(m, cfg["seeds"], pcfg, d, arch, data, mask_by_seed,
                                                          device, m_local_dir)

            for variant in cfg["variants"]:
                rows_m += run_variant_group(variant, m, cfg["label_fractions"], cfg["seeds"], data, mask_by_seed,
                                            dcfg, device, encoders, pre_run_id, m_local_dir)

            all_rows += rows_m
            pd.DataFrame(all_rows).to_csv(results_path, index=False)

            # ---- this m's cross-variant / cross-fraction comparison (top level) ----
            df_m = pd.DataFrame(rows_m)
            agg_m, gap_m = report.aggregate(df_m), report.gap_table(df_m)
            log_table_artifacts(agg_m, f"m = {m}")
            log_comparison_metrics(agg_m, gap_m)
            mlflow.log_artifact(str(results_path))
            log_fig(report.plot_grouped_bars(agg_m, f"Test accuracy by condition, m={m} (mean ± std over seeds)"),
                    "figures/accuracy_by_condition.png", m_local_dir)
            log_fig(report.plot_acc_vs_fraction(agg_m, f"Test accuracy vs label fraction, m={m}"),
                    "figures/accuracy_vs_label_fraction.png", m_local_dir)
            if gap_m is not None:
                log_fig(report.plot_gap(gap_m, f"SSL gain over the from-scratch baseline, m={m}"),
                        "figures/gain_over_scratch.png", m_local_dir)
                mlflow.log_table(gap_m, "gain_over_scratch.json")
            log_fig(report.plot_per_class_heatmap(df_m, f"Per-class test accuracy by condition, m={m}"),
                    "figures/per_class_heatmap.png", m_local_dir)
            mlflow.log_metric("total_duration_s", time.time() - t_start)
            print(f"[{top_name}] done")

    print(f"\ntotal time {time.time() - t_start:.0f}s\nresults -> {results_path}\nfigures -> {local_dir}\n"
          f"view    -> .venv/bin/mlflow ui   (run from {ROOT})")


if __name__ == "__main__":
    main()
