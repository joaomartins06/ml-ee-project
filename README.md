# SparSSL: Exploiting Sparsity for MNIST Classification via Self-Supervised Pretraining

EE4C12 (Machine Learning for Electrical Engineering) final project, TU Delft.

## What this is

MNIST digit classification with two approaches:

- **Supervised:** logistic regression and a linear SVM on PCA features, plus a DNN and a CNN.
- **Self-supervised (SSL):** an MLP encoder-decoder is pretrained to reconstruct images from
  randomly masked pixels, using no labels. A classifier is then trained on top using only a
  small fraction of the labels, and compared against the same network trained from scratch.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

MNIST is downloaded automatically on the first run.

## Running

Run everything from the repo root.

**Supervised models**

```bash
python scripts/run_sup.py
```

**PCA sweep**

```bash
python scripts/run_pca_sweep.py
```

**SSL**

```bash
python scripts/run_ssl.py --smoke   # tiny test run, takes about a minute
python scripts/run_ssl.py           # full run, settings in configs/ssl.yaml
```

The main settings in `configs/ssl.yaml`:

| Setting | Meaning |
|---|---|
| `m_values` | pixels kept by the random mask during pretraining (784 = no masking) |
| `masked_training` | `true`: the classifier also sees masked images, `false`: full images |
| `label_fractions` | fraction of the training labels the classifier gets |
| `seeds` | random seeds to repeat each experiment with |
| `variants` | `frozen` (encoder fixed), `finetuned` (encoder trained too), `scratch` (no pretraining) |

SSL runs are logged with MLflow. To browse the curves, figures and tables, run this in a
second terminal and open http://localhost:5000:

```bash
mlflow ui
```

## Outputs

Everything is written to `outputs/` and is not tracked by git. Supervised results go to
`outputs/supervised/`. SSL results go to `outputs/` as a CSV per run, and the MLflow data is
stored in `mlflow.db` and `mlartifacts/`.

## Project layout

```
scripts/       entry points: run_sup.py, run_pca_sweep.py, run_ssl.py
configs/       ssl.yaml
src/data.py    data split and masking, shared by all branches
src/sup/       supervised models
src/ssl/       self-supervised models and training
```

## Authors

João Martins, Sara Pereira

## License

MIT, see LICENSE.
