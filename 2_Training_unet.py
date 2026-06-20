"""
==============================================================================
 Training_unet.py  --  Train a standard UNet on the prepared dataset
==============================================================================

Consumes `prepared/unet/{images,masks}/{train,val}` produced by data_reader.py.
Loss = Dice + BCE (binary foreground).  Optimizer = Adam + ReduceLROnPlateau.
Best checkpoint (by val Dice) and all outputs are saved under `Unet_run/`.

Run:
    python data_reader.py          # once, to build prepared/
    python Training_unet.py
==============================================================================
"""

import os
# Work around duplicate OpenMP runtimes on Windows/Anaconda (libiomp5md.dll vs
# libomp.dll). Must be set BEFORE numpy / torch / matplotlib are imported, or
# the process aborts with "OMP: Error #15". See http://openmp.llvm.org/
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# =============================================================================
# CONFIG
# =============================================================================
class CONFIG:
    PROJECT_ROOT = Path(__file__).resolve().parent
    PREPARED_DIR = PROJECT_ROOT / "prepared" / "unet"
    RUN_DIR      = PROJECT_ROOT / "Unet_run"

    IMG_SIZE      = 256        # must be divisible by 32 for the ResNet encoder
    SEED          = 42
    BATCH_SIZE    = 8
    EPOCHS        = 80
    LR            = 1e-4        # lower LR: encoder is ImageNet-pretrained
    WEIGHT_DECAY  = 1e-4
    BASE_CHANNELS = 32         # kept for checkpoint compatibility (decoder width)
    NUM_WORKERS   = 0          # Windows-safe; raise on Linux
    AUGMENT       = True

    # ---- model: pretrained ResNet-backbone Attention-UNet -------------------
    ENCODER     = "resnet34"   # resnet18 / resnet34 / resnet50
    PRETRAINED  = True         # load ImageNet weights for the encoder
    ATTENTION   = True         # attention gates on the skip connections

    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# =============================================================================
# Dataset (reads the resized images/masks; augmentations applied to both)
# =============================================================================
class UNetDataset(Dataset):
    def __init__(self, split, augment=False):
        img_dir = CONFIG.PREPARED_DIR / "images" / split
        self.items = sorted(img_dir.glob("*.png"))
        self.mask_dir = CONFIG.PREPARED_DIR / "masks" / split
        self.augment = augment
        if not self.items:
            raise RuntimeError(f"No images in {img_dir}. Run data_reader.py first.")

    def __len__(self):
        return len(self.items)

    def _aug(self, img, mask):
        if random.random() < 0.5:
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            a = random.uniform(-20, 20)
            img = img.rotate(a, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            mask = mask.rotate(a, resample=Image.NEAREST, fillcolor=0)
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        return img, mask

    def __getitem__(self, i):
        ip = self.items[i]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(self.mask_dir / ip.name).convert("L")
        if self.augment:
            img, mask = self._aug(img, mask)
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1).contiguous()
        y = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32)).unsqueeze(0)
        return x, y, ip.stem


# =============================================================================
# Model: Attention-UNet with a pretrained ResNet encoder
# -----------------------------------------------------------------------------
# A from-scratch UNet underfits a ~90-image dataset.  Using an ImageNet-
# pretrained ResNet encoder + attention gates on the skip connections gives a
# large quality boost.  Input is ImageNet-normalized INSIDE forward(), so every
# caller (training, hybrid, heatmaps) can keep feeding plain [0,1] RGB tensors.
# The class is still named `UNet(in_ch, out_ch, base, ...)` so existing imports
# and checkpoints keep working.
# =============================================================================
_RESNET = {
    "resnet18": (models.resnet18, "ResNet18_Weights", [64, 64, 128, 256, 512]),
    "resnet34": (models.resnet34, "ResNet34_Weights", [64, 64, 128, 256, 512]),
    "resnet50": (models.resnet50, "ResNet50_Weights", [64, 256, 512, 1024, 2048]),
}


class AttentionGate(nn.Module):
    """Additive attention gate (Attention-UNet) applied to a skip connection."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.wx = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(True)

    def forward(self, gate, skip):
        a = self.psi(self.relu(self.wg(gate) + self.wx(skip)))
        return skip * a


class DecoderBlock(nn.Module):
    """Bilinear-upsample -> (attention) -> concat skip -> double conv."""
    def __init__(self, in_ch, skip_ch, out_ch, attention=True):
        super().__init__()
        self.att = AttentionGate(in_ch, skip_ch, max(skip_ch // 2, 1)) \
            if (attention and skip_ch > 0) else None
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if self.att is not None:
                skip = self.att(x, skip)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32,
                 encoder="resnet34", pretrained=True, attention=True):
        super().__init__()
        ctor, wname, ch = _RESNET[encoder]
        weights = None
        if pretrained:
            try:                                   # offline-safe pretrained load
                weights = getattr(models, wname).IMAGENET1K_V1
            except Exception:
                weights = None
        enc = ctor(weights=weights)

        # --- encoder stages (channels c0..c4 at strides /2,/4,/8,/16,/32) ----
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # -> c0, /2
        self.maxpool = enc.maxpool
        self.layer1, self.layer2 = enc.layer1, enc.layer2         # c1 /4, c2 /8
        self.layer3, self.layer4 = enc.layer3, enc.layer4         # c3 /16, c4 /32
        c0, c1, c2, c3, c4 = ch

        # --- decoder (mirror of the encoder, with attention skips) -----------
        self.d4 = DecoderBlock(c4, c3, 256, attention)
        self.d3 = DecoderBlock(256, c2, 128, attention)
        self.d2 = DecoderBlock(128, c1, 64, attention)
        self.d1 = DecoderBlock(64, c0, base, attention)
        self.d0 = DecoderBlock(base, 0, max(base // 2, 16), False)  # final /1 upsample
        self.head = nn.Conv2d(max(base // 2, 16), out_ch, 1)

        # ImageNet normalization constants (applied in forward)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x0 = self.stem(x)                 # /2
        x1 = self.layer1(self.maxpool(x0))  # /4
        x2 = self.layer2(x1)              # /8
        x3 = self.layer3(x2)              # /16
        x4 = self.layer4(x3)              # /32
        d = self.d4(x4, x3)               # /16
        d = self.d3(d, x2)                # /8
        d = self.d2(d, x1)                # /4
        d = self.d1(d, x0)                # /2
        d = self.d0(d, None)              # /1
        return self.head(d)


# =============================================================================
# loss + metric
# =============================================================================
def dice_loss(logits, targets, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * targets).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + targets.sum((1, 2, 3)) + eps
    return (1 - num / den).mean()


class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__(); self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, t):
        return self.bce(logits, t) + dice_loss(logits, t)


def metrics_from_counts(tp, fp, fn, eps=1e-7):
    """Dice, IoU, Precision, Recall, F1 from pixel confusion counts."""
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return {"dice": (2*tp) / (2*tp + fp + fn + eps),
            "iou": tp / (tp + fp + fn + eps),
            "precision": prec, "recall": rec,
            "f1": (2*prec*rec) / (prec + rec + eps)}


@torch.no_grad()
def dataset_metrics(model, loader):
    """Aggregate (micro) Dice/IoU/Precision/Recall/F1 over a loader."""
    model.eval(); tp = fp = fn = 0.0
    for x, y, _ in loader:
        p = (torch.sigmoid(model(x.to(CONFIG.DEVICE))) > 0.5).cpu()
        g = y > 0.5
        tp += float((p & g).sum()); fp += float((p & ~g).sum()); fn += float((~p & g).sum())
    return metrics_from_counts(tp, fp, fn)


def write_excel(path, sheets):
    """Write {sheet_name: list_of_dicts} to an .xlsx (CSV fallback if no pandas)."""
    try:
        import pandas as pd
        with pd.ExcelWriter(path) as xls:
            for name, rows in sheets.items():
                pd.DataFrame(rows).to_excel(xls, sheet_name=name, index=False)
        print(f"[unet] metrics workbook -> {path}")
    except Exception as e:
        print(f"[unet] pandas/openpyxl unavailable ({e}); writing CSVs instead")
        for name, rows in sheets.items():
            if not rows:
                continue
            csv = Path(path).with_name(f"{Path(path).stem}_{name}.csv")
            cols = list(rows[0].keys())
            with open(csv, "w") as f:
                f.write(",".join(cols) + "\n")
                for r in rows:
                    f.write(",".join(str(r[c]) for c in cols) + "\n")


# =============================================================================
# training
# =============================================================================
def main():
    set_seed(CONFIG.SEED)
    CONFIG.RUN_DIR.mkdir(parents=True, exist_ok=True)

    tr = DataLoader(UNetDataset("train", CONFIG.AUGMENT), batch_size=CONFIG.BATCH_SIZE,
                    shuffle=True, num_workers=CONFIG.NUM_WORKERS)
    va = DataLoader(UNetDataset("val", False), batch_size=CONFIG.BATCH_SIZE,
                    shuffle=False, num_workers=CONFIG.NUM_WORKERS)

    model = UNet(3, 1, CONFIG.BASE_CHANNELS, encoder=CONFIG.ENCODER,
                 pretrained=CONFIG.PRETRAINED, attention=CONFIG.ATTENTION).to(CONFIG.DEVICE)
    crit = ComboLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR, weight_decay=CONFIG.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=5)

    def ckpt_meta(**extra):
        """Common checkpoint fields so loaders can rebuild the exact architecture."""
        return {"model": model.state_dict(), "base": CONFIG.BASE_CHANNELS,
                "encoder": CONFIG.ENCODER, "attention": CONFIG.ATTENTION,
                "img_size": CONFIG.IMG_SIZE, **extra}

    hist = {"train_loss": [], "val_loss": [], "val_dice": [], "lr": []}
    batch_log = []      # per-batch rows  -> Excel sheet "per_batch"
    epoch_log = []      # per-epoch rows  -> Excel sheet "per_epoch"
    best = -1.0
    best_path = CONFIG.RUN_DIR / "unet_best.pt"
    print(f"[unet] device={CONFIG.DEVICE} epochs={CONFIG.EPOCHS} batch={CONFIG.BATCH_SIZE}")

    for ep in range(1, CONFIG.EPOCHS + 1):
        model.train(); run = 0.0
        for bi, (x, y, _) in enumerate(tr):
            x, y = x.to(CONFIG.DEVICE), y.to(CONFIG.DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            run += loss.item() * x.size(0)
            batch_log.append({"epoch": ep, "batch": bi, "train_loss": round(loss.item(), 6)})
        train_loss = run / len(tr.dataset)

        model.eval(); vrun = 0.0
        with torch.no_grad():
            for x, y, _ in va:
                x, y = x.to(CONFIG.DEVICE), y.to(CONFIG.DEVICE)
                vrun += crit(model(x), y).item() * x.size(0)
        val_loss = vrun / len(va.dataset)
        vm = dataset_metrics(model, va)           # full val metrics this epoch
        val_dice = vm["dice"]
        sched.step(val_dice)

        lr = opt.param_groups[0]["lr"]
        for k, v in zip(hist, (train_loss, val_loss, val_dice, lr)):
            hist[k].append(v)
        epoch_log.append({"epoch": ep, "train_loss": round(train_loss, 6),
                          "val_loss": round(val_loss, 6), "lr": lr,
                          **{f"val_{k}": round(v, 6) for k, v in vm.items()}})
        flag = ""
        if val_dice > best:
            best = val_dice
            torch.save(ckpt_meta(epoch=ep, val_dice=val_dice), best_path)
            flag = "  <- best"
        print(f"epoch {ep:03d}/{CONFIG.EPOCHS}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  dice={val_dice:.4f}  iou={vm['iou']:.4f}  lr={lr:.2e}{flag}")

    torch.save(ckpt_meta(), CONFIG.RUN_DIR / "unet_last.pt")
    with open(CONFIG.RUN_DIR / "history.json", "w") as f:
        json.dump(hist, f, indent=2)
    # ---- Excel report: per-batch losses + per-epoch metrics -----------------
    write_excel(CONFIG.RUN_DIR / "unet_metrics.xlsx",
                {"per_epoch": epoch_log, "per_batch": batch_log})

    # ---- curves -------------------------------------------------------------
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, hist["train_loss"], label="train"); ax[0].plot(ep, hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, hist["val_dice"], color="green"); ax[1].set_title("Val Dice"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(CONFIG.RUN_DIR / "training_curves.png", dpi=150); plt.close(fig)

    print(f"[unet] done. best val Dice={best:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
