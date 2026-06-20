"""
==============================================================================
 2_Training_unet_crop.py  --  Attention-Residual U-Net trained on the CROPPED
                              (bbox) dataset  image2/unet
==============================================================================

Same professional network and training recipe as `2_Training_unet2.py`
(Attention-Residual U-Net, Combo loss = BCE+Dice+Tversky, AMP, AdamW, cosine
LR + warm-up, EMA, gradient clipping, early stopping, rich metric logging),
but with three project-specific changes you asked for:

  1. TRAINS ON THE CROPS.  Inputs come from `image2/unet/{images,masks}/...`
     (the images cropped to the annotation box by `1_data_reader2.py`), so the
     network learns segmentation inside the localized tooth/canal box.

  2. EVERYTHING IN ONE FOLDER.  All artefacts land under `Unet_crop_run/`:
        Unet_crop_run/
        ├─ unet_best.pt, unet_last.pt
        ├─ unet_metrics.xlsx (+ csv fallback)   <- per-epoch metrics & losses
        ├─ training_curves.png                  <- loss & metric evolution
        ├─ test_set_metrics.json                <- final test/val/train scores
        ├─ test_samples/{train,val,test}/*.png  <- 10 samples / split, 6-panel
        └─ gradcam/{train,val,test}/*.png       <- Grad-CAM, its own sub-folder

  3. FULL-CONTEXT VISUALS.  For every sample shown, a 6-panel figure is built
     whose TOP-LEFT panel is the ORIGINAL image *before* cropping, then the
     full mask, the full overlay, and finally the crop the model actually saw
     with ground-truth vs. prediction -- mirroring the requested layout.

Run (after 1_data_reader2.py has produced image2/):
    python 2_Training_unet_crop.py
==============================================================================
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import json
import math
import base64
import random
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
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
    DATA_DIR     = PROJECT_ROOT / "image2" / "unet"     # <-- CROPPED dataset
    IMAGE2_DIR   = PROJECT_ROOT / "image2"
    RUN_DIR      = PROJECT_ROOT / "Unet_crop_run"       # <-- single output dir

    IMG_SIZE      = 256
    BATCH_SIZE    = 8
    EPOCHS        = 120
    LR            = 2e-3
    WEIGHT_DECAY  = 1e-4
    WARMUP_EPOCHS = 5
    BASE_CHANNELS = 32
    DROPOUT       = 0.20
    AUGMENT       = True
    EMA_DECAY     = 0.999
    GRAD_CLIP     = 1.0
    PATIENCE      = 20
    SEED          = 42
    NUM_WORKERS   = 0
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

    N_SHOW        = 10        # samples per split for test/gradcam visuals


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# =============================================================================
# tiny labelme helpers (self-contained, for reconstructing the pre-crop image)
# =============================================================================
def _safe_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _resolve_full_image(source_image, source_label):
    """Original full image: from disk if present, else from labelme base64."""
    if source_image and Path(source_image).exists():
        return Image.open(source_image).convert("RGB")
    data = _safe_json(source_label)
    if data and data.get("imageData"):
        raw = base64.b64decode(data["imageData"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise FileNotFoundError(f"no pixels for {source_label}")


def _polygons_to_mask(data, h, w):
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for shape in (data or {}).get("shapes", []):
        pts = shape.get("points", [])
        if len(pts) >= 3:
            d.polygon([(float(x), float(y)) for x, y in pts], outline=1, fill=1)
    return np.array(m, dtype=np.uint8)


# =============================================================================
# DATA
# =============================================================================
class CanalDataset(Dataset):
    """Loads (image, mask) crop pairs and applies paired augmentation."""

    def __init__(self, split, augment=False):
        self.img_dir  = CONFIG.DATA_DIR / "images" / split
        self.mask_dir = CONFIG.DATA_DIR / "masks"  / split
        self.augment  = augment
        self.items = sorted(p for p in self.img_dir.glob("*.png")
                            if (self.mask_dir / p.name).exists())
        if not self.items:
            raise SystemExit(f"No image/mask pairs found in {self.img_dir}")

    def __len__(self):
        return len(self.items)

    def _aug(self, img, mask):
        if random.random() < 0.5:
            img = np.flip(img, 1).copy(); mask = np.flip(mask, 1).copy()
        if random.random() < 0.5:
            img = np.flip(img, 0).copy(); mask = np.flip(mask, 0).copy()
        if random.random() < 0.8:
            img, mask = self._affine(img, mask)
        if random.random() < 0.7:
            gamma = random.uniform(0.7, 1.4)
            img = np.clip(img, 1e-4, 1.0) ** gamma
        if random.random() < 0.5:
            img = np.clip(img * random.uniform(0.85, 1.15), 0, 1)
        if random.random() < 0.4:
            img = np.clip(img + np.random.normal(0, 0.03, img.shape), 0, 1)
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
# MODEL  --  Attention Residual U-Net  (identical to 2_Training_unet2.py)
# =============================================================================
class ResDoubleConv(nn.Module):
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
        return self.out(d1)


# =============================================================================
# LOSS / METRICS / EMA / SCHED  (same as 2_Training_unet2.py)
# =============================================================================
def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    p = p.reshape(p.size(0), -1); t = target.reshape(target.size(0), -1)
    inter = (p * t).sum(1)
    return (1 - (2*inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1e-6):
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


def lr_at(epoch):
    if epoch < CONFIG.WARMUP_EPOCHS:
        return CONFIG.LR * (epoch + 1) / CONFIG.WARMUP_EPOCHS
    t = (epoch - CONFIG.WARMUP_EPOCHS) / max(1, CONFIG.EPOCHS - CONFIG.WARMUP_EPOCHS)
    return 0.5 * CONFIG.LR * (1 + math.cos(math.pi * t))


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


@torch.no_grad()
def _val_loss(model, loader, criterion):
    model.eval(); tot = 0.0; n = 0
    for x, y in loader:
        x, y = x.to(CONFIG.DEVICE), y.to(CONFIG.DEVICE)
        tot += criterion(model(x), y).item(); n += 1
    return tot / max(1, n)


# =============================================================================
# TRAIN
# =============================================================================
def train():
    set_seed(CONFIG.SEED)
    CONFIG.RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[unet-crop] device={CONFIG.DEVICE}  data={CONFIG.DATA_DIR}")

    train_ds = CanalDataset("train", augment=CONFIG.AUGMENT)
    val_ds   = CanalDataset("val", augment=False)
    train_ld = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=True,
                          num_workers=CONFIG.NUM_WORKERS, drop_last=True,
                          pin_memory=(CONFIG.DEVICE == "cuda"))
    val_ld = DataLoader(val_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False,
                        num_workers=CONFIG.NUM_WORKERS,
                        pin_memory=(CONFIG.DEVICE == "cuda"))
    print(f"[unet-crop] train={len(train_ds)}  val={len(val_ds)}")

    model = UNet(3, 1, CONFIG.BASE_CHANNELS, CONFIG.DROPOUT).to(CONFIG.DEVICE)
    print(f"[unet-crop] parameters = {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

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

        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        val = evaluate(model, val_ld)
        val_loss = _val_loss(model, val_ld, criterion)
        model.load_state_dict(backup)

        history.append({"epoch": epoch+1, "lr": lr, "train_loss": train_loss,
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"[unet-crop] ep {epoch+1:3d}/{CONFIG.EPOCHS}  loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  Dice={val['dice']:.4f}  "
              f"IoU={val['iou']:.4f}  Acc={val['accuracy']:.4f}  lr={lr:.2e}")

        if val["dice"] > best_dice:
            best_dice, best_epoch, since_improve = val["dice"], epoch+1, 0
            ema.copy_to(model)
            torch.save({"model": model.state_dict(), "base": CONFIG.BASE_CHANNELS,
                        "dropout": CONFIG.DROPOUT, "epoch": epoch+1, "val": val},
                       CONFIG.RUN_DIR / "unet_best.pt")
            model.load_state_dict(backup)
        else:
            since_improve += 1
            if since_improve >= CONFIG.PATIENCE:
                print(f"[unet-crop] early stop at epoch {epoch+1} "
                      f"(best Dice={best_dice:.4f} @ ep {best_epoch})")
                break

    torch.save({"model": model.state_dict(), "base": CONFIG.BASE_CHANNELS,
                "dropout": CONFIG.DROPOUT}, CONFIG.RUN_DIR / "unet_last.pt")

    _export_excel(history)
    _plot_curves(history)
    print(f"[unet-crop] best Dice={best_dice:.4f} @ epoch {best_epoch}")
    print(f"[unet-crop] best weights -> {CONFIG.RUN_DIR / 'unet_best.pt'}")
    return history


def _export_excel(history):
    try:
        import pandas as pd
        with pd.ExcelWriter(CONFIG.RUN_DIR / "unet_metrics.xlsx") as xls:
            pd.DataFrame(history).to_excel(xls, sheet_name="per_epoch", index=False)
            best = max(history, key=lambda r: r["val_dice"])
            pd.DataFrame([best]).to_excel(xls, sheet_name="best", index=False)
        print(f"[unet-crop] metrics -> {CONFIG.RUN_DIR / 'unet_metrics.xlsx'}")
    except Exception as e:
        print(f"[unet-crop] pandas/openpyxl unavailable ({e}); writing CSV")
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


# =============================================================================
# load final model
# =============================================================================
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=CONFIG.DEVICE)
    model = UNet(3, 1, ckpt.get("base", CONFIG.BASE_CHANNELS),
                 ckpt.get("dropout", CONFIG.DROPOUT)).to(CONFIG.DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# =============================================================================
# provenance index: stem -> source paths (to rebuild the pre-crop image)
# =============================================================================
def _manifest_index():
    idx = {}
    mpath = CONFIG.IMAGE2_DIR / "manifest.json"
    if mpath.exists():
        man = json.load(open(mpath, encoding="utf-8"))
        for split, recs in man.get("splits", {}).items():
            for r in recs:
                idx[r["stem"]] = {"split": split,
                                  "source_image": r.get("source_image"),
                                  "source_label": r.get("source_label")}
    return idx


def _crop_bbox(stem):
    """sourceBBox / sourceSize stored by 1_data_reader2.py in image2/json."""
    j = CONFIG.IMAGE2_DIR / "json" / f"{stem}.json"
    d = _safe_json(j)
    if not d:
        return None, None
    return d.get("sourceBBox"), d.get("sourceSize")


# =============================================================================
# visualization helpers
# =============================================================================
def _contour(mask_bool, thickness=2):
    m = mask_bool
    er = m.copy()
    er[1:, :] &= m[:-1, :]; er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]; er[:, :-1] &= m[:, 1:]
    edge = m & ~er
    out = edge.copy()
    for _ in range(max(0, thickness - 1)):
        e = out; out = e.copy()
        out[1:, :] |= e[:-1, :]; out[:-1, :] |= e[1:, :]
        out[:, 1:] |= e[:, :-1]; out[:, :-1] |= e[:, 1:]
    return out


def _paint(base_rgb, mask_bool, color, alpha=0.5):
    out = base_rgb.copy().astype(np.float32)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)


@torch.no_grad()
def _predict_crop(model, crop_img_256):
    """crop_img_256: float32 RGB [256,256,3] in [0,1] -> pred mask 256x256 bool."""
    x = torch.from_numpy(crop_img_256).permute(2, 0, 1).unsqueeze(0).to(CONFIG.DEVICE)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob


def _load_sample(stem, info):
    """Gather everything needed for one panel figure.

    Returns dict with full image/mask, crop image (256), crop gt mask (256),
    bbox in full coords, or None if the original could not be rebuilt.
    """
    bbox, size = _crop_bbox(stem)
    try:
        full = _resolve_full_image(info.get("source_image"), info.get("source_label"))
    except Exception:
        return None
    fdata = _safe_json(info.get("source_label"))
    H, W = (size if size else [full.height, full.width])
    full_mask = _polygons_to_mask(fdata, H, W).astype(bool)
    full_rgb = np.asarray(full.resize((W, H)), np.uint8)

    split = info["split"]
    cimg_p = CONFIG.DATA_DIR / "images" / split / f"{stem}.png"
    cmsk_p = CONFIG.DATA_DIR / "masks"  / split / f"{stem}.png"
    crop256 = np.asarray(Image.open(cimg_p).convert("RGB")
                         .resize((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), Image.BILINEAR),
                         np.float32) / 255.0
    cmask256 = np.asarray(Image.open(cmsk_p).convert("L")
                          .resize((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), Image.NEAREST)) > 127
    return {"full_rgb": full_rgb, "full_mask": full_mask, "bbox": bbox,
            "crop256": crop256, "cmask256": cmask256}


def _figure_six(stem, s, prob, out_path, title_suffix=""):
    """6-panel: original (pre-crop) ... crop GT vs prediction."""
    pred = prob > 0.5
    crop_u8 = (s["crop256"] * 255).astype(np.uint8)
    gt = s["cmask256"]

    # full image with bbox drawn
    full_box = s["full_rgb"].copy()
    if s["bbox"]:
        x0, y0, x1, y1 = s["bbox"]
        full_box = full_box.copy()
        Image_box = Image.fromarray(full_box)
        ImageDraw.Draw(Image_box).rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        full_box = np.asarray(Image_box)

    inter = gt & pred
    union = gt | pred
    dice = 2 * inter.sum() / (gt.sum() + pred.sum() + 1e-7)
    iou = inter.sum() / (union.sum() + 1e-7)

    panels = [
        ("1. Raw (pre-crop)", s["full_rgb"], None),
        ("2. Full mask", np.stack([s["full_mask"].astype(np.uint8)*255]*3, -1), None),
        ("3. Raw + GT(red) + box(green)",
         _paint(full_box, _contour(s["full_mask"], 2), (255, 0, 0), 1.0), None),
        ("4. Crop (model input)", crop_u8, None),
        ("5. Crop  GT(green) / Pred(blue)",
         _paint(_paint(crop_u8, gt, (0, 200, 0), 0.5), pred, (0, 80, 255), 0.5), None),
        (f"6. Intersection  Dice={dice:.2f} IoU={iou:.2f}",
         _paint(_paint(np.zeros_like(crop_u8), gt, (0, 200, 0), 1.0),
                pred, (0, 80, 255), 0.5), None),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(f"{stem}{title_suffix}", fontsize=13)
    for a, (title, im, _) in zip(ax.ravel(), panels):
        a.imshow(im); a.set_title(title, fontsize=10); a.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {"dice": float(dice), "iou": float(iou)}


def export_test_samples(model, idx):
    """For each split: render up to N_SHOW 6-panel figures with the final model."""
    out_root = CONFIG.RUN_DIR / "test_samples"
    summary = {}
    for split in ("train", "val", "test"):
        d = out_root / split
        d.mkdir(parents=True, exist_ok=True)
        stems = sorted(p.stem for p in (CONFIG.DATA_DIR / "images" / split).glob("*.png"))
        stems = stems[:CONFIG.N_SHOW]
        scores = []
        for stem in stems:
            info = idx.get(stem)
            if info is None:
                info = {"split": split, "source_image": None, "source_label": None}
            s = _load_sample(stem, info)
            if s is None:
                continue
            prob = _predict_crop(model, s["crop256"])
            sc = _figure_six(stem, s, prob, d / f"{stem}.png",
                             title_suffix=f"   [{split}]")
            scores.append(sc)
        summary[split] = {
            "n": len(scores),
            "mean_dice": float(np.mean([x["dice"] for x in scores])) if scores else 0.0,
            "mean_iou": float(np.mean([x["iou"] for x in scores])) if scores else 0.0,
        }
        print(f"[unet-crop] {split}: wrote {len(scores)} sample figures -> {d}")
    return summary


# =============================================================================
# GRAD-CAM
# =============================================================================
class GradCAM:
    """Grad-CAM on the U-Net bottleneck for the foreground segmentation score."""
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x):
        self.model.zero_grad()
        logits = self.model(x)
        score = (torch.sigmoid(logits) * (logits > 0).float()).sum()
        score.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)          # GAP on grads
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                            align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        return cam.cpu().numpy(), torch.sigmoid(logits)[0, 0].detach().cpu().numpy()


def export_gradcam(model, idx):
    """Grad-CAM visuals in their OWN folder, same pre-crop->crop panel story."""
    cam_engine = GradCAM(model, model.bott)
    out_root = CONFIG.RUN_DIR / "gradcam"
    for split in ("train", "val", "test"):
        d = out_root / split
        d.mkdir(parents=True, exist_ok=True)
        stems = sorted(p.stem for p in (CONFIG.DATA_DIR / "images" / split).glob("*.png"))
        stems = stems[:CONFIG.N_SHOW]
        for stem in stems:
            info = idx.get(stem, {"split": split, "source_image": None,
                                  "source_label": None})
            s = _load_sample(stem, info)
            if s is None:
                continue
            x = torch.from_numpy(s["crop256"]).permute(2, 0, 1).unsqueeze(0).to(CONFIG.DEVICE)
            cam, prob = cam_engine(x)
            pred = prob > 0.5
            crop_u8 = (s["crop256"] * 255).astype(np.uint8)

            panels = [
                ("1. Raw (pre-crop)", s["full_rgb"], None, None),
                ("2. Raw + GT(red)",
                 _paint(s["full_rgb"], _contour(s["full_mask"], 2), (255, 0, 0), 1.0),
                 None, None),
                ("3. Crop (model input)", crop_u8, None, None),
                ("4. Crop GT(green)/Pred(blue)",
                 _paint(_paint(crop_u8, s["cmask256"], (0, 200, 0), 0.5),
                        pred, (0, 80, 255), 0.5), None, None),
                ("5. Grad-CAM", cam, "jet", None),
                ("6. Grad-CAM overlay", crop_u8, None, cam),
            ]
            fig, ax = plt.subplots(2, 3, figsize=(13, 8))
            fig.suptitle(f"{stem}   [{split}]  Grad-CAM", fontsize=13)
            for a, (title, im, cmap, overlay) in zip(ax.ravel(), panels):
                if cmap:
                    a.imshow(im, cmap=cmap)
                else:
                    a.imshow(im)
                if overlay is not None:
                    a.imshow(overlay, cmap="jet", alpha=0.45)
                a.set_title(title, fontsize=10); a.axis("off")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(d / f"{stem}.png", dpi=110)
            plt.close(fig)
        print(f"[unet-crop] grad-cam {split}: wrote -> {d}")


# =============================================================================
# entry point
# =============================================================================
def main():
    train()
    model = load_model(CONFIG.RUN_DIR / "unet_best.pt")
    idx = _manifest_index()

    summary = export_test_samples(model, idx)
    export_gradcam(model, idx)

    with open(CONFIG.RUN_DIR / "test_set_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[unet-crop] sample summary: {summary}")
    print(f"[unet-crop] all results in {CONFIG.RUN_DIR}")


if __name__ == "__main__":
    main()
