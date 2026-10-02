# SSL / semi-supervised branch

This is the self-supervised + semi-supervised branch of the project: pretrain an MLP
encoder-decoder to reconstruct images from a masked subset of pixels (no labels), then
test whether that pretrained encoder gives a classification advantage over training from
scratch when only a **small fraction** of labels is available.

If you just want to run it, skip to [Running it](#running-it).

## The idea, in short

- **Masking.** For a chosen `m`, every image gets its own random mask `Φ` that keeps exactly
  `m` of the 784 pixels and zeros the rest: `x̃ = Φ ⊙ x`. A fresh mask is drawn every time an
  image is seen in training. Validation and test use fixed per-image masks, drawn once per
  seed, so the curves and metrics are reproducible. See `random_masks` and `make_eval_masks`
  in `src/data.py`, shared by every branch.
- **Pretraining (unsupervised).** An encoder `784 → 256 → 32` and decoder `32 → 256 → 784`
  are trained to reconstruct the full image `x` from the masked `x̃`, using **all** unlabeled
  training images. No labels touch this stage.
- **Downstream (semi-supervised).** A linear head `32 → 10` is attached to the encoder and
  trained on a **small labeled subset** (1%, 5%, 10%, or 25% of the training pool). Three
  variants, all compared at the same `(m, label_fraction)`:
  - `frozen` — encoder weights frozen, only the head trains. Isolates representation quality.
  - `finetuned` — encoder unfrozen too, trained end-to-end at a lower LR.
  - `scratch` — same architecture, random init, no pretraining. The fair baseline.
- **`masked_training`** (config flag, downstream stage only). `true`: the classifier also sees
  `m`-pixel masked images, with the same random masking as pretraining. `false`: it trains and
  is evaluated on full images, so `m` only controls the pretraining. It applies to every
  variant, `scratch` included, so the baseline always sees the same kind of input.
- **The question this branch answers:** does masked-reconstruction pretraining on unlabeled
  images help classification when labels are scarce? See `frozen`/`finetuned` vs `scratch`
  at low `label_fraction`.

## Repo layout

```
src/data.py              SHARED code — every branch must use this, not a private copy:
                          load_splits()       10k test / 6k val / 54k train, class-stratified
                          random_masks(n, m)  per-image random masks, exactly m ones each
                          make_eval_masks()   fixed per-image masks for train/val/test curves
                          apply_mask(x, mask) x̃ = mask ⊙ x (one global mask or per-image masks)
                          make_mask(m, seed)  one fixed global mask (fixed-sensor setting)
                          stratified_subset() class-balanced labeled subset, nested across fractions

src/ssl/
  models.py               Encoder / Decoder / Head (plain MLPs, see architecture above)
  pretrain.py             pretraining loop (MSE reconstruction, early stopping on val MSE)
  downstream.py           frozen / finetuned / scratch training loop (cross-entropy, early
                           stopping on val accuracy; also tracks val/test loss per epoch)
  plots.py                per-run figures (loss curve, training curves, confusion matrix, ...)
  report.py               cross-seed / cross-condition aggregation, tables, comparison figures
  utils.py                seeding, per-class accuracy

configs/ssl.yaml          all grid values and hyperparameters (see below)
scripts/run_ssl.py        orchestrates the full grid, writes everything to MLflow

outputs/                  gitignored — local results, not shared via git (see below)
mlartifacts/, mlflow.db   gitignored — local MLflow store, not shared via git
data/mnist.npz            gitignored — MNIST data (or it downloads automatically if absent)
```

## Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Everything below assumes you run with this venv's Python (`.venv/bin/python`, or an
activated shell). MNIST is expected at `data/mnist.npz`; if it's not there, `src/data.py`
falls back to downloading it via `torchvision`.

## Configuration — `configs/ssl.yaml`

```yaml
m_values: [196]                          # pixels kept by the random mask in pretraining (784 = no masking)
masked_training: false                   # true: downstream sees m-pixel masked inputs; false: full images
label_fractions: [0.01, 0.05, 0.1, 0.25] # fraction of the 54k train pool that gets labels
seeds: [0, 1, 2]                         # each seed drives the masks, labeled subset, and init
variants: [frozen, finetuned, scratch]   # which downstream variants to run

model:
  d: 32           # bottleneck / latent dimension
  hidden: 256      # hidden layer width (encoder and decoder)

pretrain:
  lr: 1.0e-3
  epochs: 200      # cap — early stopping (patience) decides the real stopping point
  batch_size: 256
  patience: 20     # epochs without val-MSE improvement before stopping

downstream:
  lr_head: 1.0e-3       # head LR (frozen and scratch), and head's LR within finetuned
  lr_finetune: 1.0e-4    # encoder LR, finetuned variant only
  epochs: 100
  batch_size: 64
  patience: 10
```

Change this file to change what runs — the script takes no other flags for the grid
itself. Runs with different `masked_training` values are different experiments: each
invocation gets its own `exp{N}` and the flag is part of the name, so nothing is overwritten.
To compare both settings, run the script twice.

## Running it

**Smoke test** — tiny grid (2 seeds, 2 label fractions, a handful of epochs), just to check
the pipeline and MLflow layout still work after a code change. Takes under a minute.

```bash
.venv/bin/python scripts/run_ssl.py --smoke
```

Writes to a separate `sparssl-ssl-smoke` MLflow experiment, so it never mixes with real
results.

**Real run** — the full grid from `configs/ssl.yaml`:

```bash
.venv/bin/python scripts/run_ssl.py
```

This can take a while (many small training runs across the full grid); it's meant to be
run directly in your own terminal, not backgrounded silently. **The terminal stays mostly
quiet** — it only prints one line per *completed* run (one per pretraining seed, one per
downstream seed), not per epoch. Per-epoch progress goes to MLflow instead (see below), so
a silent terminal for a while during pretraining is normal, not stuck.

## Watching it live in MLflow

In a second terminal, from the repo root, while the run is going:

```bash
.venv/bin/mlflow ui
```

Open `http://localhost:5000`. Both processes share the same local `mlflow.db`, so you can
watch live without stopping training.

### Run hierarchy

One experiment, `sparssl-ssl` (or `sparssl-ssl-smoke`). One top-level run per `m` per
invocation of the script:

```
exp{N}_{pre_epochs}_{down_epochs}_m{m}_{maskedtrain|fulltrain}
│                                                       top level: this m's full config +
│                                                       cross-variant/fraction comparison
│                                                       (tables, bar charts, gap-over-scratch,
│                                                       per-class heatmap)
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
├── finetuned   (same shape as frozen)
└── scratch     (same shape, no link to pretrain — it's the baseline)
```

`N` auto-increments each time you run the script (shared across every `m` in that
invocation), so rerunning with different settings never overwrites a previous run — you
end up with `exp1_...`, `exp2_...`, etc. side by side in the same experiment. The suffix
says what the downstream classifier saw: `maskedtrain` (`masked_training: true`) or
`fulltrain` (`false`). In `fulltrain` runs `m` is the pretraining mask size only.

Every group node (`pretrain`, every `frac=X%`) logs a **mean ± std across seeds** figure —
thin lines are the individual seeds, the bold line + shaded band is the aggregate,
truncated to the shortest seed (a seed that stops earlier than the others, via early
stopping, is marked where its own curve ends). Individual `seed=N` runs keep their own
full per-epoch curves and figures underneath.

Downstream runs (`frozen`/`finetuned`) carry a `pretrain_run_id` tag pointing at the exact
pretraining run (same `m`, same `seed`) whose encoder they started from — click that tag's
value in the UI to jump straight to it. `scratch` runs have no such tag.

## Where things land on disk

| What | Path |
|---|---|
| MLflow store | `mlflow.db` (repo root) |
| MLflow artifacts (figures, tables, models) | `mlartifacts/<experiment>/` |
| Encoder checkpoints (plain `.pt`, outside MLflow too) | `outputs/checkpoints/enc_exp{N}_..._m{m}_{maskedtrain\|fulltrain}_seed{seed}.pt` |
| Combined results table (all m/variant/fraction/seed of one invocation) | `outputs/results_ssl_exp{N}_{maskedtrain\|fulltrain}.csv` (`results_ssl_smoke_...` for smoke) |
| Copies of every comparison figure, mirroring the MLflow tree | `outputs/report/exp{N}_..._m{m}/...` |

None of the above is committed — `outputs/`, `mlartifacts/`, `mlflow.db`, and `mlruns/` are
all gitignored. **MLflow results are local to whoever ran the script** and are not shared
via git; only the code (`src/`, `scripts/`, `configs/ssl.yaml`) is. If you want to hand
someone your actual numbers/figures, send the `outputs/results_ssl_exp{N}_...csv` file and/or
`outputs/report/` directly — those don't need MLflow installed to open.

## Why this matters for the other branches

`src/data.py` is shared on purpose: `load_splits()`, `apply_mask()`, `random_masks()`,
`make_eval_masks()`, `make_mask()` and `stratified_subset()` are meant to be the **only**
implementation of the split and the masking in the whole repo. If another branch forks its
own masking or splitting logic, the train/val/test split and any masked evaluation stop being
comparable across branches. Import from here rather than reimplementing.

If another branch evaluates on masked inputs, use `make_eval_masks(n, m, seed, "test")` to
get the same per-image test masks this branch uses.
