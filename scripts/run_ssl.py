"""Run the SSL / semi-supervised grid with nested MLflow runs.

    .venv/bin/python scripts/run_ssl.py            # full grid from configs/ssl.yaml
    .venv/bin/python scripts/run_ssl.py --smoke    # tiny run: check the pipeline and the MLflow layout

View the results (from the repo root, no flags needed):

    .venv/bin/mlflow ui

Run hierarchy in one experiment:
    summary                                  cross-condition tables and figures
    └── m=<m>                                per-m tables and figures
        ├── pretrain m=<m> seed=<s>          loss curve, reconstructions, encoder model
        └── <variant> m=<m> frac=<f> seed=<s>   curves, per-class accuracy, confusion matrix, predictions
"""
import argparse
import copy
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
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
    setup_experiment(exp_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splits()
    d, hidden = cfg["model"]["d"], cfg["model"]["hidden"]
    arch = architecture_str(d, hidden)
    needs_pretrained = any(v != "scratch" for v in cfg["variants"])
    results_path = OUT / ("results_ssl_smoke.csv" if args.smoke else "results_ssl.csv")
    rows, pre_hists, recons = [], {}, {}
    t_start = time.time()

    with mlflow.start_run(run_name="summary"):
        mlflow.set_tag("stage", "summary")
        mlflow.log_params({
            "m_values": str(cfg["m_values"]), "label_fractions": str(cfg["label_fractions"]),
            "seeds": str(cfg["seeds"]), "variants": str(cfg["variants"]), "architecture": arch,
            "n_train_pool": len(data["x_train"]), "n_val": len(data["x_val"]), "n_test": len(data["x_test"]),
            **{f"pretrain_{k}": v for k, v in cfg["pretrain"].items()},
            **{f"downstream_{k}": v for k, v in cfg["downstream"].items()}})

        for m in cfg["m_values"]:
            with mlflow.start_run(run_name=f"m={m}", nested=True):
                mlflow.set_tags({"stage": "m-summary", "m": m})
                mlflow.log_params({"m": m, "d": d})
                for seed in cfg["seeds"]:
                    set_seed(seed)
                    mask = make_mask(m, seed)
                    pre_id, encoder = None, None

                    if needs_pretrained:
                        pcfg = {**cfg["model"], **cfg["pretrain"]}
                        with mlflow.start_run(run_name=f"pretrain m={m} seed={seed}", nested=True) as run:
                            pre_id = run.info.run_id
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
                                                "best_epoch": b, "epochs_run": len(hist["val_mse"]),
                                                "duration_s": time.time() - t0})
                            pre_hists[(m, seed)] = hist
                            with torch.no_grad():
                                x = data["x_val"][:10].to(device)
                                xt = apply_mask(x, mask.to(device))
                                xh = decoder(encoder(xt))
                            if seed == cfg["seeds"][0]:
                                recons[m] = (x.cpu(), xt.cpu(), xh.cpu())
                            log_fig(plots.plot_pretrain_curve(hist, f"Pretraining loss, m={m}, seed {seed}"),
                                    "figures/loss_curve.png")
                            log_fig(plots.plot_reconstructions(x, xt, xh, f"Reconstructions, m={m}, seed {seed}"),
                                    "figures/reconstructions.png")
                            torch.save(encoder.state_dict(), OUT / "checkpoints" / f"enc_m{m}_seed{seed}.pt")
                            mlflow.pytorch.log_model(copy.deepcopy(encoder).cpu(), name="encoder", input_example=xt[:2].cpu().numpy(),
                                     serialization_format="pickle")
                            print(f"[pretrain m={m} seed={seed}] best val MSE {hist['val_mse'][b]:.5f} @ epoch {b}")

                    for frac in cfg["label_fractions"]:
                        lab_idx = stratified_subset(data["y_train"], frac, seed)
                        n_lab = len(lab_idx)
                        # low label budgets: validation set is subsampled to a comparable size
                        val_idx = (torch.arange(len(data["y_val"])) if frac >= 0.10
                                   else stratified_subset(data["y_val"], n_lab / len(data["y_val"]), seed))
                        ds = {"x_lab": data["x_train"][lab_idx], "y_lab": data["y_train"][lab_idx],
                              "x_val": data["x_val"][val_idx], "y_val": data["y_val"][val_idx],
                              "x_test": data["x_test"], "y_test": data["y_test"]}
                        dcfg = {**cfg["model"], **cfg["downstream"]}

                        for variant in cfg["variants"]:
                            set_seed(seed)
                            cond = f"{variant}, m={m}, {report._pct(frac)}% labels, seed {seed}"
                            with mlflow.start_run(run_name=f"{variant} m={m} frac={frac} seed={seed}", nested=True):
                                mlflow.set_tags({"stage": "downstream", "variant": variant, "m": m,
                                                 "label_fraction": frac, "seed": seed})
                                mlflow.log_params({
                                    "m": m, "label_fraction": frac, "label_subset_seed": seed, "mask_seed": seed,
                                    "variant": variant, "n_labeled": n_lab, "n_val": len(val_idx), "architecture": arch,
                                    "lr_head": dcfg["lr_head"],
                                    "lr_finetune": dcfg["lr_finetune"] if variant == "finetuned" else "",
                                    "epochs": dcfg["epochs"], "batch_size": dcfg["batch_size"]})
                                if pre_id and variant != "scratch":
                                    mlflow.set_tag("pretrain_run_id", pre_id)
                                t0 = time.time()
                                res = train_classifier(
                                    variant, dcfg, mask, ds, device, pretrained_encoder=encoder,
                                    on_epoch=lambda e, mt: mlflow.log_metrics(mt, step=e))
                                h, b = res["history"], res["best_epoch"]
                                mlflow.log_metrics({
                                    "final_test_acc": res["test_acc"], "best_val_acc": res["val_acc"],
                                    "final_train_acc": h["train_acc"][b], "train_val_gap": h["train_acc"][b] - res["val_acc"],
                                    "best_epoch": b, "epochs_run": len(h["val_acc"]), "duration_s": time.time() - t0,
                                    **{f"test_acc_class_{c}": a for c, a in enumerate(res["per_class_acc"])}})
                                log_fig(plots.plot_training_curves(res, f"Training curves: {cond}"), "figures/training_curves.png")
                                log_fig(plots.plot_per_class(res["per_class_acc"], f"Per-class test accuracy: {cond}"),
                                        "figures/per_class_accuracy.png")
                                log_fig(plots.plot_confusion(res["pred"], data["y_test"], f"Confusion matrix: {cond}"),
                                        "figures/confusion_matrix.png")
                                log_fig(plots.plot_predictions(data["x_test"], mask, res["pred"], data["y_test"],
                                                               f"Predictions: {cond}"), "figures/predictions.png")
                                rows.append({"m": m, "seed": seed, "label_fraction": frac, "n_labeled": n_lab,
                                             "variant": variant, "test_acc": res["test_acc"], "val_acc": res["val_acc"],
                                             "best_epoch": b, "pretrain_run_id": pre_id or "",
                                             **{f"acc_class_{c}": a for c, a in enumerate(res["per_class_acc"])}})
                                print(f"[{variant} m={m} frac={frac} seed={seed}] n_lab={n_lab} "
                                      f"val {res['val_acc']:.4f} test {res['test_acc']:.4f} (epoch {b})")
                    pd.DataFrame(rows).to_csv(results_path, index=False)  # written after every (m, seed)

                # ---- m-level comparison (this m only) ----
                df_m = pd.DataFrame([r for r in rows if r["m"] == m])
                agg_m, gap_m = report.aggregate(df_m), report.gap_table(df_m)
                log_table_artifacts(agg_m, f"m = {m}")
                log_comparison_metrics(agg_m, gap_m)
                log_fig(report.plot_acc_vs_fraction(agg_m, f"Test accuracy vs label fraction, m={m}"),
                        "figures/accuracy_vs_label_fraction.png", local_dir / f"m{m}")
                if needs_pretrained:
                    log_fig(report.plot_pretrain_overlay({s: pre_hists[(m, s)] for s in cfg["seeds"]},
                                                         f"Pretraining loss, all seeds, m={m}"),
                            "figures/pretrain_loss_overlay.png", local_dir / f"m{m}")

        # ---- summary-level comparison (all conditions) ----
        df = pd.DataFrame(rows)
        agg, gap = report.aggregate(df), report.gap_table(df)
        log_table_artifacts(agg, "All conditions")
        log_comparison_metrics(agg, gap, prefix="")
        mlflow.log_artifact(str(results_path))
        log_fig(report.plot_grouped_bars(agg, "Test accuracy by condition (mean ± std over seeds)"),
                "figures/accuracy_by_condition.png", local_dir)
        log_fig(report.plot_acc_vs_fraction(agg, "Test accuracy vs label fraction"),
                "figures/accuracy_vs_label_fraction.png", local_dir)
        if gap is not None:
            log_fig(report.plot_gap(gap, "SSL gain over the from-scratch baseline"), "figures/gain_over_scratch.png", local_dir)
            mlflow.log_table(gap, "gain_over_scratch.json")
        log_fig(report.plot_per_class_heatmap(df, "Per-class test accuracy by condition"),
                "figures/per_class_heatmap.png", local_dir)
        if recons:
            log_fig(report.plot_recon_comparison(recons, f"Reconstructions across m (seed {cfg['seeds'][0]})"),
                    "figures/reconstructions_across_m.png", local_dir)
        mlflow.log_metric("total_duration_s", time.time() - t_start)

    print(f"\ntotal time {time.time() - t_start:.0f}s\nresults -> {results_path}\nfigures -> {local_dir}\n"
          f"view    -> .venv/bin/mlflow ui   (run from {ROOT})")


if __name__ == "__main__":
    main()
