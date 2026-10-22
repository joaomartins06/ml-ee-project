import torch.nn as nn

from src.data import NUM_CLASSES, NUM_PIXELS


#784 -> 256 -> d, linear latent layer
class Encoder(nn.Module):

    def __init__(self, d=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NUM_PIXELS, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


#d -> 256 -> 784, sigmoid output
class Decoder(nn.Module):

    def __init__(self, d=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, NUM_PIXELS), nn.Sigmoid())

    def forward(self, z):
        return self.net(z)


#linear head d -> 10, logits
class Head(nn.Module):

    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Linear(d, NUM_CLASSES)

    def forward(self, z):
        return self.net(z)


def architecture_str(d=32, hidden=256):
    return (f"enc[784-{hidden}(ReLU)-{d}] dec[{d}-{hidden}(ReLU)-784(Sigmoid)] head[{d}-{NUM_CLASSES}]")
