"""
==============================================================================
 2_Training_unet2.py  --  Attention-Residual U-Net (PyTorch) for binary
                          segmentation of MB / MB2 root-canal regions on CBCT
==============================================================================

A professional, fully-trained segmentation network (NOT a simulation).

Key upgrades over the original Keras draft
------------------------------------------
  * Framework unified to **PyTorch** so the trained weights (`unet_best.pt`)
    are directly consumable by `2_Training_hybrid.py` (which imports the
    `UNet` class and loads a `.pt` checkpoint).
  * **Residual double-conv blocks** + **BatchNorm** + **Dropout2d** for
    regularisation (the original had no dropout at all).
  * **Attention gates** on every skip connection so the decoder focuses on
    the thin, low-contrast canal structures instead of bulk tooth tissue.
  * **Strong, medically-safe augmentation** (flips, small rotations, scale,
    elastic-like affine, gamma/brightness, Gaussian noise, coarse dropout).
  * **Combo loss** = BCE + Dice + boundary-aware Tversky (handles the severe
    foreground/background class imbalance of canal pixels).
  * **AMP** mixed-precision, **AdamW**, **cosine LR with warm-up**, EMA of
    weights, gradient clipping, and **early stopping** on val-Dice.
  * Rich metric logging (Dice, IoU, Precision, Recall, F1, Accuracy) to
    `unet_metrics.xlsx` + training-curve figure.

Inputs  : prepared/unet/{images,masks}/{train,val,test}/*.png   (256x256)
Outputs : Unet_run/unet_best.pt, unet_last.pt, unet_metrics.xlsx,
          training_curves.png, val_overlays/*.png

Run (after data_reader.py):
    python 2_Training_unet2.py
==============================================================================
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIG
# =============================================================================
class CONFIG:
    PROJECT_ROOT = Path(__file__).resolve().parent
    PREPARED_DIR = PROJECT_ROOT / "prepared" / "unet"
    RUN_DIR      = PROJECT_ROOT / "Unet_run"

    IMG_SIZE      = 256
    BATCH_SIZE    = 8
    EPOCHS        = 2000
    LR            = 2e-3
    WEIGHT_DECAY  = 1e-4
    WARMUP_EPOCHS = 5
    BASE_CHANNELS = 32
    DROPOUT       = 0.20          # <-- regularisation the original lacked
    AUGMENT       = True
    EMA_DECAY     = 0.999
    GRAD_CLIP     = 1.0
    PATIENCE      = 2000            # early-stopping on val Dice
    SEED          = 42
    NUM_WORKERS   = 0            # set >0 on Linux for speed
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# =============================================================================
# DATA
# =============================================================================
class CanalDataset(Dataset):
    """Loads (image, mask) pairs and applies paired augmentation on the fly."""

    def __init__(self, split, augment=False):
        self.img_dir  = CONFIG.PREPARED_DIR / "images" / split
        self.mask_dir = CONFIG.PREPARED_DIR / "masks"  / split
        self.augment  = augment
        self.items = sorted(p for p in self.img_dir.glob("*.png")
                            if (self.mask_dir / p.name).exists())
        if not self.items:
            raise SystemExit(f"No image/mask pairs found in {self.img_dir}")

    def __len__(self):
        return len(self.items)

    # ---- augmentation (kept geometrically gentle: canals are thin) ----------
    def _aug(self, img, mask):
        # horizontal / vertical flips
        if random.random() < 0.5:
            img = np.flip(img, 1).copy(); mask = np.flip(mask, 1).copy()
        if random.random() < 0.5:
            img = np.flip(img, 0).copy(); mask = np.flip(mask, 0).copy()

        # small affine: rotation +/-15 deg, scale 0.9-1.1, translate +/-6 px
        if random.random() < 0.8:
            img, mask = self._affine(img, mask)

        # photometric (image only) -- gamma, brightness, gaussian noise
        if random.random() < 0.7:
            gamma = random.uniform(0.7, 1.4)
            img = np.clip(img, 1e-4, 1.0) ** gamma
        if random.random() < 0.5:
            img = np.clip(img * random.uniform(0.85, 1.15), 0, 1)
        if random.random() < 0.4:
            img = np.clip(img + np.random.normal(0, 0.03, img.shape), 0, 1)

        # coarse dropout / cutout (image only) -- robustness to occlusion
        if random.random() < 0.3:
            img = self._cutout(img)
        return img.astype(np.float32), mask.astype(np.float32)

    def _affine(self, img, mask):
        H, W = img.shape[:2]
        ang = math.radians(random.uniform(-15, 15))
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-6, 6); ty = random.uniform(-6, 6)
        cos, sin = math.cos(ang) / scale, math.sin(ang) / scale
        cx, cy = W / 2.0, H / 2.0
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        xs -= cx; ys -= cy
        src_x = cos * xs + sin * ys + cx - tx
        src_y = -sin * xs + cos * ys + cy - ty
        src_x = np.clip(np.round(src_x), 0, W - 1).astype(np.int32)
        src_y = np.clip(np.round(src_y), 0, H - 1).astype(np.int32)
        return img[src_y, src_x], mask[src_y, src_x]

    def _cutout(self, img, n=2, size=32):
        H, W = img.shape[:2]
        for _ in range(n):
            x = random.randint(0, W - size); y = random.randint(0, H - size)
            img[y:y+size, x:x+size] = img.mean()
        return img

    def __getitem__(self, idx):
        p = self.items[idx]
        img = np.asarray(Image.open(p).convert("RGB")
                         .resize((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), Image.BILINEAR),
                         np.float32) / 255.0
        mask = np.asarray(Image.open(self.mask_dir / p.name).convert("L")
                          .resize((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), Image.NEAREST),
                          np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        if self.augment:
            img, mask = self._aug(img, mask)
        x = torch.from_numpy(img).permute(2, 0, 1)
        y = torch.from_numpy(mask).unsqueeze(0)
        return x, y


# =============================================================================
# MODEL  --  Attention Residual U-Net
# =============================================================================
class ResDoubleConv(nn.Module):
    """(Conv-BN-ReLU) x2 with a residual 1x1 shortcut + Dropout2d."""
    def __init__(self, cin, cout, drop=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
        self.skip = (nn.Identity() if cin == cout
                     else nn.Conv2d(cin, cout, 1, bias=False))

    def forward(self, x):
        return self.body(x) + self.skip(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018) on a skip connection."""
    def __init__(self, g_ch, x_ch, inter):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(g_ch, inter, 1, bias=False),
                                nn.BatchNorm2d(inter))
        self.Wx = nn.Sequential(nn.Conv2d(x_ch, inter, 1, bias=False),
                                nn.BatchNorm2d(inter))
        self.psi = nn.Sequential(nn.Conv2d(inter, 1, 1, bias=False),
                                 nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        a = self.psi(self.relu(self.Wg(g) + self.Wx(x)))
        return x * a


class UNet(nn.Module):
    """Attention Residual U-Net.

    Signature kept compatible with the hybrid script: UNet(in_ch, out_ch, base).
    """
    def __init__(self, in_ch=3, out_ch=1, base=32, drop=0.20):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base*2, base*4, base*8, base*16
        self.pool = nn.MaxPool2d(2)

        self.e1 = ResDoubleConv(in_ch, c1, drop*0.25)
        self.e2 = ResDoubleConv(c1, c2, drop*0.5)
        self.e3 = ResDoubleConv(c2, c3, drop)
        self.e4 = ResDoubleConv(c3, c4, drop)
        self.bott = ResDoubleConv(c4, c5, drop)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.ag4 = AttentionGate(c4, c4, c4 // 2)
        self.d4  = ResDoubleConv(c5, c4, drop)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.ag3 = AttentionGate(c3, c3, c3 // 2)
        self.d3  = ResDoubleConv(c4, c3, drop)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.ag2 = AttentionGate(c2, c2, c2 // 2)
        self.d2  = ResDoubleConv(c3, c2, drop*0.5)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.ag1 = AttentionGate(c1, c1, c1 // 2)
        self.d1  = ResDoubleConv(c2, c1, drop*0.25)

        self.out = nn.Conv2d(c1, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b  = self.bott(self.pool(e4))

        u4 = self.up4(b);  d4 = self.d4(torch.cat([u4, self.ag4(u4, e4)], 1))
        u3 = self.up3(d4); d3 = self.d3(torch.cat([u3, self.ag3(u3, e3)], 1))
        u2 = self.up2(d3); d2 = self.d2(torch.cat([u2, self.ag2(u2, e2)], 1))
        u1 = self.up1(d2); d1 = self.d1(torch.cat([u1, self.ag1(u1, e1)], 1))
        return self.out(d1)              # logits


# =============================================================================
# LOSS  --  BCE + Dice + Tversky (boundary/imbalance aware)
# =============================================================================
def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    p = p.reshape(p.size(0), -1); t = target.reshape(target.size(0), -1)
    inter = (p * t).sum(1)
    return (1 - (2*inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1e-6):
    """beta>alpha penalises false-negatives -> better recall on thin canals."""
    p = torch.sigmoid(logits)
    p = p.reshape(p.size(0), -1); t = target.reshape(target.size(0), -1)
    tp = (p * t).sum(1); fp = (p * (1-t)).sum(1); fn = ((1-p) * t).sum(1)
    return (1 - (tp + eps) / (tp + alpha*fp + beta*fn + eps)).mean()


class ComboLoss(nn.Module):
    def __init__(self, pos_weight=3.0, device="cpu"):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device))

    def forward(self, logits, target):
        return self.bce(logits, target) + dice_loss(logits, target) \
             + 0.5 * tversky_loss(logits, target)


# =============================================================================
# METRICS
# =============================================================================
@torch.no_grad()
def batch_metrics(logits, target, thr=0.5, eps=1e-7):
    pred = (torch.sigmoid(logits) > thr).float()
    t = target
    tp = (pred * t).sum().item(); fp = (pred * (1-t)).sum().item()
    fn = ((1-pred) * t).sum().item(); tn = ((1-pred) * (1-t)).sum().item()
    prec = tp / (tp + fp + eps); rec = tp / (tp + fn + eps)
    return {
        "dice": (2*tp) / (2*tp + fp + fn + eps),
        "iou":  tp / (tp + fp + fn + eps),
        "precision": prec, "recall": rec,
        "f1": (2*prec*rec) / (prec + rec + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


# =============================================================================
# EMA (exponential moving average of weights -> smoother, better generalisation)
# =============================================================================
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


# =============================================================================
# SCHEDULER (linear warm-up -> cosine decay)
# =============================================================================
def lr_at(epoch):
    if epoch < CONFIG.WARMUP_EPOCHS:
        return CONFIG.LR * (epoch + 1) / CONFIG.WARMUP_EPOCHS
    t = (epoch - CONFIG.WARMUP_EPOCHS) / max(1, CONFIG.EPOCHS - CONFIG.WARMUP_EPOCHS)
    return 0.5 * CONFIG.LR * (1 + math.cos(math.pi * t))


# =============================================================================
# EVAL
# =============================================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    agg = {k: 0.0 for k in ("dice", "iou", "precision", "recall", "f1", "accuracy")}
    n = 0
    for x, y in loader:
        x, y = x.to(CONFIG.DEVICE), y.to(CONFIG.DEVICE)
        m = batch_metrics(model(x), y)
        for k in agg:
            agg[k] += m[k]
        n += 1
    return {k: v / max(1, n) for k, v in agg.items()}


# =============================================================================
# TRAIN
# =============================================================================
def main():
    set_seed(CONFIG.SEED)
    CONFIG.RUN_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG.RUN_DIR / "val_overlays").mkdir(exist_ok=True)
    print(f"[unet] device={CONFIG.DEVICE}")

    train_ds = CanalDataset("train", augment=CONFIG.AUGMENT)
    val_ds   = CanalDataset("val", augment=False)
    train_ld = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=True,
                          num_workers=CONFIG.NUM_WORKERS, drop_last=True,
                          pin_memory=(CONFIG.DEVICE == "cuda"))
    val_ld = DataLoader(val_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False,
                        num_workers=CONFIG.NUM_WORKERS,
                        pin_memory=(CONFIG.DEVICE == "cuda"))
    print(f"[unet] train={len(train_ds)}  val={len(val_ds)}")

    model = UNet(3, 1, CONFIG.BASE_CHANNELS, CONFIG.DROPOUT).to(CONFIG.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[unet] parameters = {n_params/1e6:.2f} M")

    criterion = ComboLoss(pos_weight=3.0, device=CONFIG.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR,
                                  weight_decay=CONFIG.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG.DEVICE == "cuda"))
    ema = EMA(model, CONFIG.EMA_DECAY)

    best_dice, best_epoch, since_improve = -1.0, 0, 0
    history = []

    for epoch in range(CONFIG.EPOCHS):
        lr = lr_at(epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        run_loss = 0.0
        for x, y in train_ld:
            x, y = x.to(CONFIG.DEVICE), y.to(CONFIG.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(CONFIG.DEVICE == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG.GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            ema.update(model)
            run_loss += loss.item()
        train_loss = run_loss / max(1, len(train_ld))

        # validate with the EMA weights (kept on a temporary copy)
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        val = evaluate(model, val_ld)
        val_loss_holder = _val_loss(model, val_ld, criterion)
        model.load_state_dict(backup)

        history.append({"epoch": epoch+1, "lr": lr, "train_loss": train_loss,
                        "val_loss": val_loss_holder, **{f"val_{k}": v for k, v in val.items()}})
        print(f"[unet] ep {epoch+1:3d}/{CONFIG.EPOCHS}  loss={train_loss:.4f}  "
              f"val_loss={val_loss_holder:.4f}  Dice={val['dice']:.4f}  "
              f"IoU={val['iou']:.4f}  Acc={val['accuracy']:.4f}  lr={lr:.2e}")

        if val["dice"] > best_dice:
            best_dice, best_epoch, since_improve = val["dice"], epoch+1, 0
            ema.copy_to(model)
            torch.save({"model": model.state_dict(), "base": CONFIG.BASE_CHANNELS,
                        "dropout": CONFIG.DROPOUT, "epoch": epoch+1,
                        "val": val}, CONFIG.RUN_DIR / "unet_best.pt")
            model.load_state_dict(backup)
            _save_overlays(model, val_ld)
        else:
            since_improve += 1
            if since_improve >= CONFIG.PATIENCE:
                print(f"[unet] early stop at epoch {epoch+1} "
                      f"(best Dice={best_dice:.4f} @ ep {best_epoch})")
                break

    torch.save({"model": model.state_dict(), "base": CONFIG.BASE_CHANNELS,
                "dropout": CONFIG.DROPOUT}, CONFIG.RUN_DIR / "unet_last.pt")

    _export_excel(history)
    _plot_curves(history)
    print(f"[unet] best Dice={best_dice:.4f} @ epoch {best_epoch}")
    print(f"[unet] best weights -> {CONFIG.RUN_DIR / 'unet_best.pt'}")


@torch.no_grad()
def _val_loss(model, loader, criterion):
    model.eval(); tot = 0.0; n = 0
    for x, y in loader:
        x, y = x.to(CONFIG.DEVICE), y.to(CONFIG.DEVICE)
        tot += criterion(model(x), y).item(); n += 1
    return tot / max(1, n)


@torch.no_grad()
def _save_overlays(model, loader, k=6):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(CONFIG.DEVICE)
    pred = (torch.sigmoid(model(x)) > 0.5).float().cpu().numpy()
    x = x.cpu().numpy(); y = y.numpy()
    k = min(k, x.shape[0])
    fig, ax = plt.subplots(k, 3, figsize=(9, 3*k))
    if k == 1:
        ax = ax[None, :]
    for i in range(k):
        ax[i, 0].imshow(np.transpose(x[i], (1, 2, 0))); ax[i, 0].set_title("image"); ax[i, 0].axis("off")
        ax[i, 1].imshow(y[i, 0], cmap="gray"); ax[i, 1].set_title("ground truth"); ax[i, 1].axis("off")
        ax[i, 2].imshow(np.transpose(x[i], (1, 2, 0)))
        ax[i, 2].imshow(pred[i, 0], cmap="jet", alpha=0.45)
        ax[i, 2].set_title("prediction"); ax[i, 2].axis("off")
    fig.tight_layout()
    fig.savefig(CONFIG.RUN_DIR / "val_overlays" / "best_overlays.png", dpi=120)
    plt.close(fig)


def _export_excel(history):
    try:
        import pandas as pd
        with pd.ExcelWriter(CONFIG.RUN_DIR / "unet_metrics.xlsx") as xls:
            pd.DataFrame(history).to_excel(xls, sheet_name="per_epoch", index=False)
            best = max(history, key=lambda r: r["val_dice"])
            pd.DataFrame([best]).to_excel(xls, sheet_name="best", index=False)
        print(f"[unet] metrics -> {CONFIG.RUN_DIR / 'unet_metrics.xlsx'}")
    except Exception as e:
        print(f"[unet] pandas/openpyxl unavailable ({e}); writing CSV")
        import csv
        with open(CONFIG.RUN_DIR / "unet_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader(); w.writerows(history)


def _plot_curves(history):
    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep, [h["train_loss"] for h in history], label="train loss")
    plt.plot(ep, [h["val_loss"] for h in history], label="val loss")
    plt.xlabel("epoch"); plt.legend(); plt.grid(True); plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(ep, [h["val_dice"] for h in history], label="val Dice")
    plt.plot(ep, [h["val_iou"] for h in history], label="val IoU")
    plt.plot(ep, [h["val_accuracy"] for h in history], label="val Acc")
    plt.xlabel("epoch"); plt.legend(); plt.grid(True); plt.title("Validation metrics")
    plt.tight_layout()
    plt.savefig(CONFIG.RUN_DIR / "training_curves.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
