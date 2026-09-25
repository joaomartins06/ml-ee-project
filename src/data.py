"""Shared data utilities: MNIST loading, train/val/test split, measurement mask, labeled subsets.

Every branch (supervised, unsupervised, SSL) must use these functions so that
the split and the measurement operator (m) are identical across branches.
"""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
NUM_PIXELS = 784
NUM_CLASSES = 10

VAL_SIZE = 6000
SPLIT_SEED = 0


def _load_raw():
    """Return uint8 arrays (x_train, y_train, x_test, y_test) of the standard 60k/10k MNIST."""
    npz = DATA_DIR / "mnist.npz"
    if npz.exists():
        d = np.load(npz)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    from torchvision.datasets import MNIST  # fallback: download

    tr, te = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False, download=True)
    return tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()


def load_splits(val_size=VAL_SIZE, split_seed=SPLIT_SEED):
    """Fixed split: standard 10k test, `val_size` validation carved from the 60k train, rest = train pool.

    Returns a dict of tensors: x_{train,val,test} float32 in [0, 1] with shape (N, 784),
    y_{train,val,test} int64 with shape (N,).
    """
    x_tr, y_tr, x_te, y_te = _load_raw()
    perm = np.random.default_rng(split_seed).permutation(len(x_tr))
    val_idx, train_idx = perm[:val_size], perm[val_size:]

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
    """Zero-filled measurement: x_tilde = mask * x (input size stays 784)."""
    return x * mask.to(x.device)


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
