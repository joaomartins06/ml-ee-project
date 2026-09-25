"""Downstream classification on a small labeled subset: frozen / fine-tuned / from-scratch."""
import copy

import torch
import torch.nn.functional as F

from src.data import apply_mask
from src.ssl.models import Encoder, Head
from src.ssl.utils import per_class_accuracy

VARIANTS = ("frozen", "finetuned", "scratch")


@torch.no_grad()
def _predict(encoder, head, x, mask):
    encoder.eval()
    return torch.cat([head(encoder(apply_mask(x[i : i + 4096], mask))).argmax(1) for i in range(0, len(x), 4096)])


def train_classifier(variant, cfg, mask, data, device, pretrained_encoder=None, on_epoch=None):
    """Train one classifier variant.

    variant: 'frozen'    -> pretrained encoder frozen, only the head is trained (lr = cfg['lr_head'])
             'finetuned' -> pretrained encoder + head trained end-to-end (encoder lr = cfg['lr_finetune'])
             'scratch'   -> random-init encoder + head trained end-to-end (lr = cfg['lr_head'])
    data: dict with x_lab, y_lab (labeled subset) and x_val, y_val, x_test, y_test.
    cfg keys: d, hidden, lr_head, lr_finetune, epochs, batch_size, patience.
    Model selection: weights from the epoch with best validation accuracy; test accuracy is reported at that epoch.
    """
    if variant not in VARIANTS:
        raise ValueError(variant)
    if variant != "scratch" and pretrained_encoder is None:
        raise ValueError(f"{variant} needs a pretrained encoder")

    encoder = Encoder(cfg["d"], cfg["hidden"]).to(device)
    if pretrained_encoder is not None and variant != "scratch":
        encoder.load_state_dict(pretrained_encoder.state_dict())
    head = Head(cfg["d"]).to(device)

    if variant == "frozen":
        for p in encoder.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(head.parameters(), lr=cfg["lr_head"])
    elif variant == "finetuned":
        opt = torch.optim.Adam(
            [{"params": encoder.parameters(), "lr": cfg["lr_finetune"]},
             {"params": head.parameters(), "lr": cfg["lr_head"]}])
    else:
        opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=cfg["lr_head"])

    x_lab, y_lab = data["x_lab"].to(device), data["y_lab"].to(device)
    mask = mask.to(device)
    ev = {k: (data[f"x_{k}"].to(device), data[f"y_{k}"].to(device)) for k in ("val", "test")}

    def acc(split):
        x, y = ev[split]
        return (_predict(encoder, head, x, mask) == y).float().mean().item()

    history = {"train_acc": [], "val_acc": [], "test_acc": [], "train_loss": []}
    best = (-1.0, None, None, -1)
    for epoch in range(cfg["epochs"]):
        encoder.train(variant != "frozen")
        perm = torch.randperm(len(x_lab), device=device)
        epoch_loss = 0.0
        for i in range(0, len(x_lab), cfg["batch_size"]):
            idx = perm[i : i + cfg["batch_size"]]
            xb = apply_mask(x_lab[idx], mask)
            if variant == "frozen":
                with torch.no_grad():
                    z = encoder(xb)
            else:
                z = encoder(xb)
            loss = F.cross_entropy(head(z), y_lab[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)

        tr_acc = (_predict(encoder, head, x_lab, mask) == y_lab).float().mean().item()
        va_acc, te_acc = acc("val"), acc("test")
        for k, v in (("train_acc", tr_acc), ("val_acc", va_acc), ("test_acc", te_acc), ("train_loss", epoch_loss / len(x_lab))):
            history[k].append(v)
        if on_epoch:
            on_epoch(epoch, {"train_loss": epoch_loss / len(x_lab), "train_acc": tr_acc, "val_acc": va_acc, "test_acc": te_acc})
        if va_acc > best[0]:
            best = (va_acc, copy.deepcopy(encoder.state_dict()), copy.deepcopy(head.state_dict()), epoch)
        elif epoch - best[3] >= cfg["patience"]:
            break

    encoder.load_state_dict(best[1])
    head.load_state_dict(best[2])
    x_te, y_te = ev["test"]
    pred = _predict(encoder, head, x_te, mask)
    return {
        "best_epoch": best[3],
        "val_acc": best[0],
        "test_acc": (pred == y_te).float().mean().item(),
        "per_class_acc": per_class_accuracy(pred, y_te),
        "pred": pred.cpu(),
        "history": history,
    }
