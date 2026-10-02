"""Shared data utilities: MNIST loading, train/val/test split, measurement mask, labeled subsets.

Every branch (supervised, unsupervised, SSL) must use these functions so that
the split and the measurement operator (m) are identical across branches.
"""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" #MNIST dataset 
NUM_PIXELS = 784
NUM_CLASSES = 10

VAL_SIZE = 6000
SPLIT_SEED = 0


def _load_raw():
    """Return uint8 arrays (x_train, y_train, x_test, y_test) of the standard 60k/10k MNIST."""
    npz = DATA_DIR / "mnist.npz"
    if npz.exists():
        d = np.load(npz)
        #MNIST dataset is already divided between train and test
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    from torchvision.datasets import MNIST  # fallback: download

    tr, te = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False, download=True)
    return tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()


def load_splits(val_size=VAL_SIZE, split_seed=SPLIT_SEED):
    """Fixed split: standard 10k test, `val_size` validation carved from the 60k train, rest = train pool.

    The val/train split is class-stratified (each class contributes val_size * (its share of the
    60k) images to validation), so validation mirrors the training-pool class balance instead of
    whatever a single uniform shuffle happens to give.

    Returns a dict of tensors: x_{train,val,test} float32 in [0, 1] with shape (N, 784),
    y_{train,val,test} int64 with shape (N,).
    """
    x_tr, y_tr, x_te, y_te = _load_raw()
    rng = np.random.default_rng(split_seed)
    val_parts, train_parts = [], []
    #ensures that the validation set is balanced when it comes to its classes
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(y_tr == c)
        idx = idx[rng.permutation(len(idx))]
        n_val_c = round(val_size * len(idx) / len(y_tr))
        val_parts.append(idx[:n_val_c])
        train_parts.append(idx[n_val_c:])
    val_idx, train_idx = np.sort(np.concatenate(val_parts)), np.sort(np.concatenate(train_parts))

    def prep(x):
        return torch.from_numpy(x.reshape(len(x), -1).astype(np.float32) / 255.0)

    def lab(y):
        return torch.from_numpy(y.astype(np.int64))

    return {
        "x_train": prep(x_tr[train_idx]), "y_train": lab(y_tr[train_idx]),
        "x_val": prep(x_tr[val_idx]), "y_val": lab(y_tr[val_idx]),
        "x_test": prep(x_te), "y_test": lab(y_te),
    }


def make_mask(m, seed):
    """Fixed random binary mask with exactly `m` ones out of 784 (float32, shape (784,)).

    Sampled once per (m, seed) and reused for every image and every model.
    """
    if not 0 < m <= NUM_PIXELS:
        raise ValueError(f"m must be in [1, {NUM_PIXELS}], got {m}")
    rng = np.random.default_rng([seed, m])
    mask = np.zeros(NUM_PIXELS, dtype=np.float32)
    mask[rng.choice(NUM_PIXELS, size=m, replace=False)] = 1.0
    return torch.from_numpy(mask)


def apply_mask(x, mask):
    """Zero-filled measurement: x_tilde = mask * x (input size stays 784).

    `mask` is either one global (784,) mask or per-image masks of shape (len(x), 784).
    """
    #just apply the mask element-wise
    return x * mask.to(x.device)


EVAL_SPLITS = {"train": 0, "val": 1, "test": 2}


def random_masks(n, m, device="cpu", generator=None):
    """Per-image random binary masks, shape (n, 784), each row with exactly `m` ones.

    Redrawing these every time an image is seen is the masking used for training. `generator`
    (if given) must live on `device`. For m = 784 every mask is all ones (no masking).
    """
    if not 0 < m <= NUM_PIXELS:
        raise ValueError(f"m must be in [1, {NUM_PIXELS}], got {m}")
    if m == NUM_PIXELS:
        return torch.ones(n, NUM_PIXELS, device=device)
    keep = torch.rand(n, NUM_PIXELS, device=device, generator=generator).topk(m, dim=1).indices
    return torch.zeros(n, NUM_PIXELS, device=device).scatter_(1, keep, 1.0)


def make_eval_masks(n, m, seed, split):
    """Fixed per-image masks for evaluation: `n` images, exactly `m` pixels kept in each.

    Drawn once per (seed, m, split) and identical across epochs, variants and branches, so
    validation/test numbers are reproducible and comparable. split: 'train' | 'val' | 'test'.
    """
    state = np.random.SeedSequence([seed, m, EVAL_SPLITS[split]]).generate_state(1)[0]
    return random_masks(n, m, "cpu", torch.Generator().manual_seed(int(state)))


def stratified_subset(y, fraction, seed):
    """Indices of a class-stratified subset containing `fraction` of each class.

    Subsets are nested for a given seed: the indices for a smaller fraction are a
    prefix of those for a larger fraction (per class).
    """
    y_np = y.cpu().numpy()
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(y_np == c)
        idx = idx[rng.permutation(len(idx))]
        chosen.append(idx[: max(1, round(fraction * len(idx)))])
    return torch.from_numpy(np.sort(np.concatenate(chosen)))
