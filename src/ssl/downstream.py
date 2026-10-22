import copy

import torch
import torch.nn.functional as F

from src.data import apply_mask, random_masks
from src.ssl.models import Encoder, Head
from src.ssl.utils import per_class_accuracy

VARIANTS = ("frozen", "finetuned", "scratch")


#loss, accuracy and predictions over a full set (masks=None -> full images)
@torch.no_grad()
def _eval(encoder, head, x, y, masks, batch_size=4096):
    encoder.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size], y[i : i + batch_size]
        if masks is not None:
            xb = apply_mask(xb, masks[i : i + batch_size])
        logits = head(encoder(xb))
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        preds.append(logits.argmax(1))
    pred = torch.cat(preds)
    return total_loss / len(x), (pred == y).float().mean().item(), pred


def _to(t, device):
    return None if t is None else t.to(device)


#frozen: pretrained encoder frozen, head only | finetuned: pretrained encoder + head | scratch: random init, encoder + head
#train_m = pixels kept by a fresh random mask every batch, None -> full images
#best epoch picked on val accuracy
def train_classifier(variant, cfg, data, device, train_m=None, seed=0, pretrained_encoder=None, on_epoch=None):
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

    masked = train_m is not None
    x_lab, y_lab = data["x_lab"].to(device), data["y_lab"].to(device)
    masks_lab = _to(data.get("masks_lab"), device) if masked else None
    ev = {k: (data[f"x_{k}"].to(device), data[f"y_{k}"].to(device),
              _to(data.get(f"masks_{k}"), device) if masked else None) for k in ("val", "test")}
    gen = torch.Generator(device=device).manual_seed(seed) if masked else None

    history = {"train_loss": [], "val_loss": [], "test_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
    best = (-1.0, None, None, -1)
    for epoch in range(cfg["epochs"]):
        encoder.train(variant != "frozen")
        perm = torch.randperm(len(x_lab), device=device)
        for i in range(0, len(x_lab), cfg["batch_size"]):
            idx = perm[i : i + cfg["batch_size"]]
            xb = x_lab[idx]
            if masked:
                xb = apply_mask(xb, random_masks(len(idx), train_m, device, gen))
            if variant == "frozen":
                with torch.no_grad():
                    z = encoder(xb)
            else:
                z = encoder(xb)
            loss = F.cross_entropy(head(z), y_lab[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        #eval-mode loss/acc on full sets (fixed masks if masked)
        tr_loss, tr_acc, _ = _eval(encoder, head, x_lab, y_lab, masks_lab)
        va_loss, va_acc, _ = _eval(encoder, head, *ev["val"])
        te_loss, te_acc, _ = _eval(encoder, head, *ev["test"])
        epoch_metrics = {"train_loss": tr_loss, "val_loss": va_loss, "test_loss": te_loss,
                          "train_acc": tr_acc, "val_acc": va_acc, "test_acc": te_acc}
        for k, v in epoch_metrics.items():
            history[k].append(v)
        if on_epoch:
            on_epoch(epoch, epoch_metrics)
        if va_acc > best[0]:
            best = (va_acc, copy.deepcopy(encoder.state_dict()), copy.deepcopy(head.state_dict()), epoch)
        elif epoch - best[3] >= cfg["patience"]:
            break

    encoder.load_state_dict(best[1])
    head.load_state_dict(best[2])
    test_loss, test_acc, pred = _eval(encoder, head, *ev["test"])
    return {
        "best_epoch": best[3],
        "val_acc": best[0],
        "test_acc": test_acc,
        "test_loss": test_loss,
        "per_class_acc": per_class_accuracy(pred, ev["test"][1]),
        "pred": pred.cpu(),
        "history": history,
    }
