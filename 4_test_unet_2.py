# =============================================================================
# 6_Test_all_visual_confusion.py
# -----------------------------------------------------------------------------
# Test trained Attention-Residual U-Net on image1 and image2 datasets.
#
# Outputs:
#   6_Test_all/
#       image1/
#           panels/
#           image1_confusion_matrix.png
#           image1_confusion_matrix.csv
#           image1_metrics.json
#       image2/
#           panels/
#           image2_confusion_matrix.png
#           image2_confusion_matrix.csv
#           image2_metrics.json
#       all_results_summary.json
#
# Each panel image contains:
#   1) Raw image
#   2) Ground-truth mask
#   3) Raw + GT mask
#   4) GT mask + Network mask
#   5) Raw + GT + Network mask
#   6) Raw + Network heatmap / Grad-CAM
#
# Put this file beside:
#   2_Training_unet_crop.py
#   image1/
#   image2/
#   Unet_crop_run/unet_best.pt
#
# Run:
#   python 6_Test_all_visual_confusion.py
# =============================================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIG
# =============================================================================

class CONFIG:
    PROJECT_ROOT = Path(__file__).resolve().parent

    IMAGE1_DIR = PROJECT_ROOT / "image1"
    IMAGE2_DIR = PROJECT_ROOT / "image2"

    CKPT_PATH = PROJECT_ROOT / "unet_best.pt"
    OUT_DIR = PROJECT_ROOT / "6_Test_all"

    IMG_SIZE = 256
    BASE_CHANNELS = 32
    DROPOUT = 0.20

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # تعداد تصاویر ترکیبی که برای هر دیتاست ذخیره می‌شود
    # اگر می‌خواهی برای همه تصاویر پنل ساخته شود، مقدار را None بگذار.
    N_VIS_PER_DATASET = 10

    # آستانه تبدیل احتمال شبکه به ماسک باینری
    THRESHOLD = 0.5

    # کیفیت خروجی تصاویر
    PANEL_DPI = 220
    CM_DPI = 220

    IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


# =============================================================================
# MODEL: same Attention Residual U-Net
# =============================================================================

class ResDoubleConv(nn.Module):
    def __init__(self, cin, cout, drop=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.body(x) + self.skip(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_ch, inter, 1, bias=False),
            nn.BatchNorm2d(inter)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(x_ch, inter, 1, bias=False),
            nn.BatchNorm2d(inter)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        a = self.psi(self.relu(self.Wg(g) + self.Wx(x)))
        return x * a


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32, drop=0.20):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

        self.pool = nn.MaxPool2d(2)

        self.e1 = ResDoubleConv(in_ch, c1, drop * 0.25)
        self.e2 = ResDoubleConv(c1, c2, drop * 0.5)
        self.e3 = ResDoubleConv(c2, c3, drop)
        self.e4 = ResDoubleConv(c3, c4, drop)

        self.bott = ResDoubleConv(c4, c5, drop)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.ag4 = AttentionGate(c4, c4, c4 // 2)
        self.d4 = ResDoubleConv(c5, c4, drop)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.ag3 = AttentionGate(c3, c3, c3 // 2)
        self.d3 = ResDoubleConv(c4, c3, drop)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.ag2 = AttentionGate(c2, c2, c2 // 2)
        self.d2 = ResDoubleConv(c3, c2, drop * 0.5)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.ag1 = AttentionGate(c1, c1, c1 // 2)
        self.d1 = ResDoubleConv(c2, c1, drop * 0.25)

        self.out = nn.Conv2d(c1, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        b = self.bott(self.pool(e4))

        u4 = self.up4(b)
        d4 = self.d4(torch.cat([u4, self.ag4(u4, e4)], dim=1))

        u3 = self.up3(d4)
        d3 = self.d3(torch.cat([u3, self.ag3(u3, e3)], dim=1))

        u2 = self.up2(d3)
        d2 = self.d2(torch.cat([u2, self.ag2(u2, e2)], dim=1))

        u1 = self.up1(d2)
        d1 = self.d1(torch.cat([u1, self.ag1(u1, e1)], dim=1))

        return self.out(d1)


# =============================================================================
# GRAD-CAM
# =============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.grads = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        prob = torch.sigmoid(logits)

        # foreground score
        score = prob.sum()
        score.backward()

        if self.acts is None or self.grads is None:
            cam = torch.zeros_like(prob[0, 0])
        else:
            weights = self.grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False
            )[0, 0]

        cam = cam.detach()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)

        return cam.cpu().numpy(), prob[0, 0].detach().cpu().numpy()


# =============================================================================
# UTILITIES
# =============================================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_model():
    if not CONFIG.CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CONFIG.CKPT_PATH}\n"
            "ابتدا کد آموزش را اجرا کن تا unet_best.pt ساخته شود."
        )

    ckpt = torch.load(CONFIG.CKPT_PATH, map_location=CONFIG.DEVICE)

    base = ckpt.get("base", CONFIG.BASE_CHANNELS)
    dropout = ckpt.get("dropout", CONFIG.DROPOUT)

    model = UNet(3, 1, base=base, drop=dropout).to(CONFIG.DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[OK] model loaded: {CONFIG.CKPT_PATH}")
    print(f"[INFO] device: {CONFIG.DEVICE}")

    return model


def is_image_file(path):
    return Path(path).suffix.lower() in CONFIG.IMAGE_EXTS


def read_rgb(path):
    return Image.open(path).convert("RGB")


def read_mask(path):
    mask = Image.open(path).convert("L")
    arr = np.asarray(mask, dtype=np.float32)
    if arr.max() > 1:
        arr = arr / 255.0
    return arr > 0.5


def resize_np_rgb(img_pil, size):
    return np.asarray(
        img_pil.resize((size, size), Image.BILINEAR),
        dtype=np.float32
    ) / 255.0


def resize_mask_bool(mask_bool, size_hw):
    h, w = size_hw
    img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
    arr = np.asarray(img.resize((w, h), Image.NEAREST))
    return arr > 127


def normalize_heatmap(hm):
    hm = hm.astype(np.float32)
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-7)
    return hm


def colorize_mask(mask_bool, color):
    """
    color: tuple RGB, e.g. (255, 0, 0)
    """
    h, w = mask_bool.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask_bool] = np.array(color, dtype=np.uint8)
    return out


def overlay_mask(rgb, mask_bool, color=(255, 0, 0), alpha=0.45):
    out = rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_two_masks(rgb, gt_bool, pred_bool):
    """
    GT: green
    Pred: blue
    Intersection: yellow
    """
    out = rgb.copy().astype(np.float32)

    gt_only = gt_bool & (~pred_bool)
    pred_only = pred_bool & (~gt_bool)
    both = gt_bool & pred_bool

    out[gt_only] = 0.50 * out[gt_only] + 0.50 * np.array([0, 220, 0])
    out[pred_only] = 0.50 * out[pred_only] + 0.50 * np.array([0, 80, 255])
    out[both] = 0.35 * out[both] + 0.65 * np.array([255, 230, 0])

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_colormap_on_image(rgb, heatmap, alpha=0.45):
    """
    heatmap must be 2D float [0,1].
    """
    heatmap = normalize_heatmap(heatmap)

    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[..., :3]
    heat_rgb = (heat_rgb * 255).astype(np.uint8)

    out = (1 - alpha) * rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_confusion_csv(cm, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",Pred_Background,Pred_Foreground\n")
        f.write(f"GT_Background,{int(cm[0,0])},{int(cm[0,1])}\n")
        f.write(f"GT_Foreground,{int(cm[1,0])},{int(cm[1,1])}\n")


def plot_confusion_matrix(cm, metrics, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred BG", "Pred FG"])
    ax.set_yticklabels(["GT BG", "GT FG"])

    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")

    ax.set_title(
        f"{title}\n"
        f"Dice={metrics['dice']:.4f} | IoU={metrics['iou']:.4f} | "
        f"Acc={metrics['accuracy']:.4f}"
    )

    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            color = "white" if cm[i, j] > max_val * 0.5 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=CONFIG.CM_DPI)
    plt.close(fig)


# =============================================================================
# DATA DISCOVERY
# =============================================================================

def try_mask_path_for_image(img_path):
    """
    Find corresponding mask by replacing images -> masks
    and trying common extensions.
    """
    img_path = Path(img_path)
    parts = list(img_path.parts)

    candidates = []

    # Replace folder named images with masks
    if "images" in parts:
        idx = parts.index("images")
        mask_parts = parts.copy()
        mask_parts[idx] = "masks"
        base = Path(*mask_parts)
        candidates.append(base)

        for ext in CONFIG.IMAGE_EXTS:
            candidates.append(base.with_suffix(ext))

    # Same folder with suffix patterns
    stem = img_path.stem
    parent = img_path.parent

    possible_names = [
        f"{stem}.png",
        f"{stem}_mask.png",
        f"{stem}-mask.png",
        f"{stem}_label.png",
        f"{stem}-label.png",
    ]

    for name in possible_names:
        candidates.append(parent / name)
        candidates.append(parent.parent / "masks" / name)
        candidates.append(parent.parent / "mask" / name)
        candidates.append(parent.parent / "labels" / name)

    # Remove duplicates
    unique = []
    seen = set()
    for c in candidates:
        c = Path(c)
        if c not in seen:
            seen.add(c)
            unique.append(c)

    for c in unique:
        if c.exists() and is_image_file(c):
            return c

    return None


def collect_pairs_from_base(base_dir):
    """
    Flexible loader.

    Supported examples:
        image1/images/train/*.png + image1/masks/train/*.png
        image1/images/*.png       + image1/masks/*.png
        image2/unet/images/train/*.png + image2/unet/masks/train/*.png
    """
    base_dir = Path(base_dir)
    pairs = []

    if not base_dir.exists():
        return pairs

    # Case 1: base/images exists
    img_root = base_dir / "images"
    if img_root.exists():
        image_files = sorted(
            p for p in img_root.rglob("*")
            if p.is_file() and is_image_file(p)
        )
        for img_path in image_files:
            mask_path = try_mask_path_for_image(img_path)
            if mask_path is not None:
                split = "all"
                rel_parts = img_path.relative_to(img_root).parts
                if len(rel_parts) > 1:
                    split = rel_parts[0]
                pairs.append({
                    "image": img_path,
                    "mask": mask_path,
                    "split": split,
                    "stem": img_path.stem
                })

    # Case 2: no images folder, search recursively and skip masks folders
    if not pairs:
        image_files = []
        for p in base_dir.rglob("*"):
            if not p.is_file() or not is_image_file(p):
                continue
            lower_parts = [x.lower() for x in p.parts]
            if any(x in lower_parts for x in ["masks", "mask", "labels", "label"]):
                continue
            image_files.append(p)

        image_files = sorted(image_files)

        for img_path in image_files:
            mask_path = try_mask_path_for_image(img_path)
            if mask_path is not None and mask_path != img_path:
                pairs.append({
                    "image": img_path,
                    "mask": mask_path,
                    "split": "all",
                    "stem": img_path.stem
                })

    return pairs


def collect_dataset_pairs(dataset_root, dataset_name):
    """
    For image2, prefer image2/unet because training code used cropped dataset.
    For image1, try image1/unet first if exists, otherwise image1 itself.
    """
    dataset_root = Path(dataset_root)

    search_bases = []

    if (dataset_root / "unet").exists():
        search_bases.append(dataset_root / "unet")

    search_bases.append(dataset_root)

    all_pairs = []
    used_base = None

    for base in search_bases:
        pairs = collect_pairs_from_base(base)
        if pairs:
            all_pairs = pairs
            used_base = base
            break

    if not all_pairs:
        print(f"[WARN] No image/mask pairs found for {dataset_name}: {dataset_root}")
    else:
        print(f"[OK] {dataset_name}: found {len(all_pairs)} pairs from {used_base}")

    return all_pairs, used_base


# =============================================================================
# PREDICTION
# =============================================================================

@torch.no_grad()
def predict_prob(model, img_pil):
    """
    Returns:
        prob_256: probability map at 256x256
    """
    img_np_256 = resize_np_rgb(img_pil, CONFIG.IMG_SIZE)
    x = torch.from_numpy(img_np_256).permute(2, 0, 1).unsqueeze(0)
    x = x.to(CONFIG.DEVICE)

    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    return prob


def predict_prob_and_cam(model, cam_engine, img_pil):
    img_np_256 = resize_np_rgb(img_pil, CONFIG.IMG_SIZE)

    x = torch.from_numpy(img_np_256).permute(2, 0, 1).unsqueeze(0)
    x = x.to(CONFIG.DEVICE)

    try:
        cam_256, prob_256 = cam_engine(x)
    except Exception as e:
        print(f"[WARN] Grad-CAM failed, using probability heatmap instead: {e}")
        with torch.no_grad():
            logits = model(x)
            prob_256 = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        cam_256 = prob_256.copy()

    return prob_256, cam_256


def resize_float_map_to_hw(arr, size_hw):
    h, w = size_hw
    img = Image.fromarray((normalize_heatmap(arr) * 255).astype(np.uint8))
    out = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    return out


# =============================================================================
# METRICS / CONFUSION
# =============================================================================

def update_confusion(cm, gt_bool, pred_bool):
    gt = gt_bool.astype(bool)
    pr = pred_bool.astype(bool)

    tn = np.logical_and(~gt, ~pr).sum()
    fp = np.logical_and(~gt, pr).sum()
    fn = np.logical_and(gt, ~pr).sum()
    tp = np.logical_and(gt, pr).sum()

    cm[0, 0] += int(tn)
    cm[0, 1] += int(fp)
    cm[1, 0] += int(fn)
    cm[1, 1] += int(tp)

    return cm


def metrics_from_confusion(cm):
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    eps = 1e-7

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "dice": float(dice),
        "iou": float(iou),
        "f1": float(f1),
    }


def sample_metrics(gt_bool, pred_bool):
    cm = np.zeros((2, 2), dtype=np.int64)
    cm = update_confusion(cm, gt_bool, pred_bool)
    return metrics_from_confusion(cm)


# =============================================================================
# PANEL GENERATION
# =============================================================================

def make_panel_image(
    raw_rgb,
    gt_mask,
    pred_mask,
    heatmap,
    title,
    out_path,
    metrics=None
):
    """
    Save one high quality 6-panel figure.
    """
    raw_rgb = raw_rgb.astype(np.uint8)

    gt_color = colorize_mask(gt_mask, (0, 220, 0))
    pred_color = colorize_mask(pred_mask, (0, 80, 255))

    raw_gt = overlay_mask(raw_rgb, gt_mask, color=(0, 220, 0), alpha=0.45)

    # GT + prediction masks together on black background
    mask_compare = np.zeros_like(raw_rgb)
    gt_only = gt_mask & (~pred_mask)
    pred_only = pred_mask & (~gt_mask)
    both = gt_mask & pred_mask

    mask_compare[gt_only] = np.array([0, 220, 0], dtype=np.uint8)       # GT only
    mask_compare[pred_only] = np.array([0, 80, 255], dtype=np.uint8)    # Pred only
    mask_compare[both] = np.array([255, 230, 0], dtype=np.uint8)        # overlap

    raw_two = overlay_two_masks(raw_rgb, gt_mask, pred_mask)
    raw_heat = apply_colormap_on_image(raw_rgb, heatmap, alpha=0.45)

    panels = [
        ("1) Raw image", raw_rgb),
        ("2) Ground-truth mask", gt_color),
        ("3) Raw + GT mask", raw_gt),
        ("4) GT + Network mask", mask_compare),
        ("5) Raw + two masks", raw_two),
        ("6) Raw + network heatmap", raw_heat),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))
    fig.suptitle(title, fontsize=14)

    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    if metrics is not None:
        txt = (
            f"Dice={metrics['dice']:.4f} | IoU={metrics['iou']:.4f} | "
            f"Precision={metrics['precision']:.4f} | Recall={metrics['recall_sensitivity']:.4f}"
        )
        fig.text(0.5, 0.02, txt, ha="center", fontsize=11)

    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(out_path, dpi=CONFIG.PANEL_DPI)
    plt.close(fig)


# =============================================================================
# PROCESS DATASET
# =============================================================================

def process_dataset(dataset_name, dataset_dir, model, cam_engine):
    out_root = CONFIG.OUT_DIR / dataset_name
    panel_dir = out_root / "panels"

    ensure_dir(out_root)
    ensure_dir(panel_dir)

    pairs, used_base = collect_dataset_pairs(dataset_dir, dataset_name)

    cm_total = np.zeros((2, 2), dtype=np.int64)

    per_sample = []
    n_panels = 0

    if CONFIG.N_VIS_PER_DATASET is None:
        max_vis = len(pairs)
    else:
        max_vis = int(CONFIG.N_VIS_PER_DATASET)

    for idx, item in enumerate(pairs):
        img_path = item["image"]
        mask_path = item["mask"]

        try:
            img_pil = read_rgb(img_path)
            raw_rgb = np.asarray(img_pil, dtype=np.uint8)

            gt_mask = read_mask(mask_path)

            h, w = gt_mask.shape
            if raw_rgb.shape[:2] != gt_mask.shape:
                # اگر اندازه ماسک و تصویر فرق داشت، ماسک را به اندازه تصویر تبدیل می‌کنیم
                gt_mask = resize_mask_bool(gt_mask, raw_rgb.shape[:2])
                h, w = gt_mask.shape

            prob_256, cam_256 = predict_prob_and_cam(model, cam_engine, img_pil)

            prob_full = resize_float_map_to_hw(prob_256, (h, w))
            cam_full = resize_float_map_to_hw(cam_256, (h, w))

            pred_mask = prob_full > CONFIG.THRESHOLD

            cm_total = update_confusion(cm_total, gt_mask, pred_mask)

            sm = sample_metrics(gt_mask, pred_mask)

            per_sample.append({
                "index": idx,
                "stem": item["stem"],
                "split": item["split"],
                "image": str(img_path),
                "mask": str(mask_path),
                **sm
            })

            if n_panels < max_vis:
                panel_name = f"{idx:04d}_{item['split']}_{item['stem']}.png"
                out_panel_path = panel_dir / panel_name

                make_panel_image(
                    raw_rgb=raw_rgb,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    heatmap=cam_full,
                    title=f"{dataset_name} | {item['split']} | {item['stem']}",
                    out_path=out_panel_path,
                    metrics=sm
                )

                n_panels += 1

            print(
                f"[{dataset_name}] {idx+1}/{len(pairs)} | "
                f"{item['stem']} | Dice={sm['dice']:.4f} | IoU={sm['iou']:.4f}"
            )

        except Exception as e:
            print(f"[ERROR] failed on {img_path}: {e}")

    metrics_total = metrics_from_confusion(cm_total)

    # Save confusion matrix
    cm_png = out_root / f"{dataset_name}_confusion_matrix.png"
    cm_csv = out_root / f"{dataset_name}_confusion_matrix.csv"

    plot_confusion_matrix(
        cm=cm_total,
        metrics=metrics_total,
        title=f"{dataset_name} Pixel-level Confusion Matrix",
        out_path=cm_png
    )

    save_confusion_csv(cm_total, cm_csv)

    # Save metrics
    save_json(metrics_total, out_root / f"{dataset_name}_metrics.json")
    save_json(per_sample, out_root / f"{dataset_name}_per_sample_metrics.json")

    summary = {
        "dataset": dataset_name,
        "dataset_dir": str(dataset_dir),
        "used_base": str(used_base) if used_base else None,
        "num_pairs": len(pairs),
        "num_panel_images": n_panels,
        "output_dir": str(out_root),
        "confusion_matrix": cm_total.tolist(),
        "metrics": metrics_total,
    }

    save_json(summary, out_root / f"{dataset_name}_summary.json")

    print("\n" + "=" * 80)
    print(f"[DONE] {dataset_name}")
    print(f"Pairs: {len(pairs)}")
    print(f"Panels: {n_panels}")
    print(f"Dice: {metrics_total['dice']:.4f}")
    print(f"IoU: {metrics_total['iou']:.4f}")
    print(f"Accuracy: {metrics_total['accuracy']:.4f}")
    print(f"Confusion PNG: {cm_png}")
    print("=" * 80 + "\n")

    return summary

# =============================================================================
# EXTRA: image1 crop-style visual panels like the sample image
# =============================================================================

def bbox_from_mask(mask_bool, pad_ratio=0.80, min_side=96):
    """
    Build a square crop box around GT mask.
    Returns x0, y0, x1, y1 in image coordinates.
    """
    h, w = mask_bool.shape
    ys, xs = np.where(mask_bool)

    if len(xs) == 0 or len(ys) == 0:
        # fallback: center crop
        side = min(h, w)
        x0 = max(0, (w - side) // 2)
        y0 = max(0, (h - side) // 2)
        return x0, y0, x0 + side, y0 + side

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    bw = x1 - x0
    bh = y1 - y0

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    side = max(bw, bh)
    side = int(side * (1.0 + pad_ratio))
    side = max(side, min_side)
    side = min(side, max(h, w))

    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = x0 + side
    y1 = y0 + side

    # keep inside image
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        shift = x1 - w
        x0 -= shift
        x1 = w
    if y1 > h:
        shift = y1 - h
        y0 -= shift
        y1 = h

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)

    return x0, y0, x1, y1


def draw_cross_and_bbox(rgb, bbox, mask_bool=None):
    """
    Draw vertical green line, horizontal blue line and red crop box.
    Optionally overlay mask in red.
    """
    out = rgb.copy().astype(np.uint8)

    if mask_bool is not None:
        out = overlay_mask(out, mask_bool, color=(255, 0, 0), alpha=0.45)

    x0, y0, x1, y1 = bbox
    cx = int((x0 + x1) / 2)
    cy = int((y0 + y1) / 2)

    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)

    h, w = out.shape[:2]

    # horizontal blue line
    draw.line([(0, cy), (w, cy)], fill=(0, 0, 255), width=2)

    # vertical green line
    draw.line([(cx, 0), (cx, h)], fill=(0, 255, 0), width=2)

    # red crop rectangle
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)

    return np.asarray(img)


def make_crop_style_panel_image1(
    raw_rgb,
    gt_mask_full,
    crop_rgb_256,
    gt_mask_crop_256,
    pred_mask_crop_256,
    heatmap_crop_256,
    bbox,
    title,
    out_path
):
    """
    Similar to the user's sample:
        1. Raw
        2. Mask
        3. Raw + Mask(red)
        4. Crop loc (G,R)
        5. Crop mask (G,B)
        6. Intersection
    """

    raw_rgb = raw_rgb.astype(np.uint8)

    # 1) Raw + lines and bbox
    raw_with_guides = draw_cross_and_bbox(raw_rgb, bbox, mask_bool=None)

    # 2) Full GT mask
    full_mask_rgb = np.zeros_like(raw_rgb)
    full_mask_rgb[gt_mask_full] = np.array([255, 255, 255], dtype=np.uint8)

    # 3) Raw + GT mask in red + guides
    raw_mask_guides = draw_cross_and_bbox(raw_rgb, bbox, mask_bool=gt_mask_full)

    # 4) Crop location: GT green + Pred red
    crop_loc = crop_rgb_256.copy().astype(np.uint8)
    crop_loc = overlay_mask(crop_loc, pred_mask_crop_256, color=(255, 0, 0), alpha=0.60)
    crop_loc = overlay_mask(crop_loc, gt_mask_crop_256, color=(0, 255, 0), alpha=0.60)

    # 5) Crop mask: GT green + Pred blue
    crop_mask = crop_rgb_256.copy().astype(np.uint8)
    crop_mask = overlay_mask(crop_mask, gt_mask_crop_256, color=(0, 255, 0), alpha=0.60)
    crop_mask = overlay_mask(crop_mask, pred_mask_crop_256, color=(0, 80, 255), alpha=0.60)

    # 6) Intersection / comparison
    intersection = np.zeros_like(crop_rgb_256).astype(np.uint8)

    gt_only = gt_mask_crop_256 & (~pred_mask_crop_256)
    pred_only = pred_mask_crop_256 & (~gt_mask_crop_256)
    both = gt_mask_crop_256 & pred_mask_crop_256

    intersection[gt_only] = np.array([0, 255, 0], dtype=np.uint8)       # GT only
    intersection[pred_only] = np.array([0, 80, 255], dtype=np.uint8)    # Pred only
    intersection[both] = np.array([255, 255, 0], dtype=np.uint8)        # overlap

    # Optional: if you want heatmap instead of crop_loc, uncomment this:
    # heatmap_overlay = apply_colormap_on_image(crop_rgb_256, heatmap_crop_256, alpha=0.45)

    panels = [
        ("1. Raw", raw_with_guides),
        ("2. Mask", full_mask_rgb),
        ("3. Raw+Mask(red)", raw_mask_guides),
        ("4. Crop loc (G,R)", crop_loc),
        ("5. Crop mask (G,B)", crop_mask),
        ("6. Intersection", intersection),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(title, fontsize=14)

    for ax, (name, img) in zip(axes.ravel(), panels):
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def export_image1_crop_style_panels(model, cam_engine, n_samples=None):
    """
    Create sample-like crop visualization panels from image1.
    Output:
        6_Test_all/image1/crop_style_panels/
    """
    out_dir = CONFIG.OUT_DIR / "image1" / "crop_style_panels"
    ensure_dir(out_dir)

    pairs, used_base = collect_dataset_pairs(CONFIG.IMAGE1_DIR, "image1")

    if not pairs:
        print("[WARN] No pairs found for image1 crop-style panels.")
        return {
            "num_pairs": 0,
            "num_saved": 0,
            "out_dir": str(out_dir)
        }

    if n_samples is None:
        selected_pairs = pairs
    else:
        selected_pairs = pairs[:int(n_samples)]

    saved = 0
    summary = []

    for idx, item in enumerate(selected_pairs):
        try:
            img_path = item["image"]
            mask_path = item["mask"]

            img_pil = read_rgb(img_path)
            raw_rgb = np.asarray(img_pil, dtype=np.uint8)

            gt_mask_full = read_mask(mask_path)

            if raw_rgb.shape[:2] != gt_mask_full.shape:
                gt_mask_full = resize_mask_bool(gt_mask_full, raw_rgb.shape[:2])

            # -------------------------------------------------------------
            # 1) crop image using GT mask bounding box
            # -------------------------------------------------------------
            bbox = bbox_from_mask(
                gt_mask_full,
                pad_ratio=0.80,
                min_side=96
            )

            x0, y0, x1, y1 = bbox

            crop_rgb = raw_rgb[y0:y1, x0:x1]
            crop_mask = gt_mask_full[y0:y1, x0:x1]

            crop_pil = Image.fromarray(crop_rgb).convert("RGB")
            crop_rgb_256 = np.asarray(
                crop_pil.resize((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), Image.BILINEAR),
                dtype=np.uint8
            )

            crop_mask_256 = resize_mask_bool(
                crop_mask,
                (CONFIG.IMG_SIZE, CONFIG.IMG_SIZE)
            )

            # -------------------------------------------------------------
            # 2) run network on crop
            # -------------------------------------------------------------
            prob_256, cam_256 = predict_prob_and_cam(
                model=model,
                cam_engine=cam_engine,
                img_pil=crop_pil
            )

            pred_mask_256 = prob_256 > CONFIG.THRESHOLD

            # -------------------------------------------------------------
            # 3) save panel
            # -------------------------------------------------------------
            out_name = f"{idx:04d}_{item['split']}_{item['stem']}_crop_style.png"
            out_path = out_dir / out_name

            make_crop_style_panel_image1(
                raw_rgb=raw_rgb,
                gt_mask_full=gt_mask_full,
                crop_rgb_256=crop_rgb_256,
                gt_mask_crop_256=crop_mask_256,
                pred_mask_crop_256=pred_mask_256,
                heatmap_crop_256=cam_256,
                bbox=bbox,
                title=f"{item['stem']}",
                out_path=out_path
            )

            saved += 1

            summary.append({
                "index": idx,
                "stem": item["stem"],
                "split": item["split"],
                "image": str(img_path),
                "mask": str(mask_path),
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "output": str(out_path)
            })

            print(f"[image1-crop-style] saved {saved}/{len(selected_pairs)} -> {out_path}")

        except Exception as e:
            print(f"[ERROR] image1 crop-style failed on {item.get('image')}: {e}")

    save_json(
        {
            "used_base": str(used_base) if used_base else None,
            "num_pairs": len(pairs),
            "num_saved": saved,
            "out_dir": str(out_dir),
            "items": summary
        },
        CONFIG.OUT_DIR / "image1" / "image1_crop_style_summary.json"
    )

    print("\n" + "=" * 80)
    print("[DONE] image1 crop-style panels")
    print(f"Saved: {saved}")
    print(f"Output: {out_dir}")
    print("=" * 80 + "\n")

    return {
        "num_pairs": len(pairs),
        "num_saved": saved,
        "out_dir": str(out_dir)
    }
# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("6_Test_all_visual_confusion.py")
    print("=" * 80)
    print(f"Project root: {CONFIG.PROJECT_ROOT}")
    print(f"Image1 dir:   {CONFIG.IMAGE1_DIR}")
    print(f"Image2 dir:   {CONFIG.IMAGE2_DIR}")
    print(f"Checkpoint:   {CONFIG.CKPT_PATH}")
    print(f"Output dir:   {CONFIG.OUT_DIR}")
    print(f"Device:       {CONFIG.DEVICE}")
    print("=" * 80)

    ensure_dir(CONFIG.OUT_DIR)

    model = load_model()
    cam_engine = GradCAM(model, model.bott)

    all_summary = {}
    
    all_summary["image1"] = process_dataset(
        dataset_name="image1",
        dataset_dir=CONFIG.IMAGE1_DIR,
        model=model,
        cam_engine=cam_engine
    )
    
    all_summary["image1_crop_style"] = export_image1_crop_style_panels(
    model=model,
    cam_engine=cam_engine,
    n_samples=CONFIG.N_VIS_PER_DATASET
)
    
    
    all_summary["image2"] = process_dataset(
        dataset_name="image2",
        dataset_dir=CONFIG.IMAGE2_DIR,
        model=model,
        cam_engine=cam_engine
    )

    save_json(all_summary, CONFIG.OUT_DIR / "all_results_summary.json")

    print("\nAll results saved in:")
    print(CONFIG.OUT_DIR)
    print("\nFinished successfully.")


if __name__ == "__main__":
    main()