import random

import numpy as np
import torch

from src.data import NUM_CLASSES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_class_accuracy(pred, y):
    """List of 10 accuracies, one per digit class."""
    return [((pred[y == c] == c).float().mean().item() if (y == c).any() else float("nan")) for c in range(NUM_CLASSES)]
