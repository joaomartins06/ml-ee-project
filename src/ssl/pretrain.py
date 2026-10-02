"""Self-supervised pretraining: reconstruct the full image x from a randomly masked x_tilde.

Every time an image is seen in training it gets a fresh random mask keeping exactly `m` pixels
(per image). Train/validation MSE for monitoring and early stopping use fixed per-image masks
(`make_eval_masks`), so the curves are reproducible and not noisy from mask resampling.
"""
import copy

import torch
import torch.nn.functional as F

from src.data import apply_mask, make_eval_masks, random_masks
from src.ssl.models import Decoder, Encoder


@torch.no_grad()
def _recon_mse(encoder, decoder, x, masks, batch_size):
    total = 0.0
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size]
        total += F.mse_loss(decoder(encoder(apply_mask(xb, masks[i : i + batch_size]))), xb, reduction="sum").item()
    return total / x.numel()  # mean over all pixels


def pretrain(x_train, x_val, m, seed, cfg, device, on_epoch=None):
    """Train encoder+decoder on unlabeled images. Loss: MSE(x, g(f(mask * x))) over all 784 pixels.

    m: pixels kept per image by the random mask. seed: drives the training masks and the fixed eval masks.
    cfg keys: d, hidden, lr, epochs, batch_size, patience.
    Returns (best_encoder, best_decoder, history) where the weights are from the best val-MSE epoch.
    """
    encoder, decoder = Encoder(cfg["d"], cfg["hidden"]).to(device), Decoder(cfg["d"], cfg["hidden"]).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    x_train, x_val = x_train.to(device), x_val.to(device)
    train_masks = make_eval_masks(len(x_train), m, seed, "train").to(device)
    val_masks = make_eval_masks(len(x_val), m, seed, "val").to(device)
    gen = torch.Generator(device=device).manual_seed(seed)

    best = (float("inf"), None, None, -1)
    history = {"train_mse": [], "val_mse": []}
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(len(x_train), device=device)
        for i in range(0, len(x_train), cfg["batch_size"]):
            xb = x_train[perm[i : i + cfg["batch_size"]]]
            xt = apply_mask(xb, random_masks(len(xb), m, device, gen))
            loss = F.mse_loss(decoder(encoder(xt)), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            tr = _recon_mse(encoder, decoder, x_train, train_masks, 2048)
            va = _recon_mse(encoder, decoder, x_val, val_masks, 2048)
        history["train_mse"].append(tr)
        history["val_mse"].append(va)
        if on_epoch:
            on_epoch(epoch, tr, va)
        if va < best[0]:
            best = (va, copy.deepcopy(encoder.state_dict()), copy.deepcopy(decoder.state_dict()), epoch)
        elif epoch - best[3] >= cfg["patience"]:
            break

    encoder.load_state_dict(best[1])
    decoder.load_state_dict(best[2])
    history["best_epoch"] = best[3]
    return encoder, decoder, history
