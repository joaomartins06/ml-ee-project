"""MLP encoder / decoder / classification head for masked-reconstruction pretraining."""
import torch.nn as nn

from src.data import NUM_CLASSES, NUM_PIXELS


class Encoder(nn.Module):
    """784 -> 256 (ReLU) -> d (linear latent)."""

    def __init__(self, d=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NUM_PIXELS, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """d -> 256 (ReLU) -> 784 (sigmoid, pixels in [0, 1])."""

    def __init__(self, d=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, NUM_PIXELS), nn.Sigmoid())

    def forward(self, z):
        return self.net(z)


class Head(nn.Module):
    """Linear classification head d -> 10 (logits; softmax lives in the cross-entropy loss)."""

    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Linear(d, NUM_CLASSES)

    def forward(self, z):
        return self.net(z)


def architecture_str(d=32, hidden=256):
    return (f"enc[784-{hidden}(ReLU)-{d}] dec[{d}-{hidden}(ReLU)-784(Sigmoid)] head[{d}-{NUM_CLASSES}]")
