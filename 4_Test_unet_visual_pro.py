# =============================================================================
# 4_Test_unet_visual_pro.py
#
# Professional visual test for U-Net segmentation + training-curve plotting.
#
# What this script does:
#   1) Loads the best U-Net checkpoint:  Unet_run/unet_best.pt
#   2) Samples several crops from image2/<unet>/images/{train,val,test}
#      (or directly from image2/images), predicts masks, computes Dice/IoU.
#   3) Recovers the ORIGINAL (un-cropped) image from images/.
#      Crop naming convention (from 1_data_reader2.py `_unique_stem`):
#          <parentdir>__<originalstem>
#      e.g.  crop   "1__amini2_2 - Copy.png"
#            origin "amini2_2 - Copy.jpg"  inside images/1/
#      so we strip the "<dir>__" prefix and search images/ recursively.
#   4) Saves a HIGH-QUALITY side-by-side panel per sample
#      (original | crop | GT | pred | GT overlay | pred overlay |
#       both overlays | probability map). Each panel is shown full-screen
#       (maximized window) before being saved.
#   5) Parses "log training u net.txt" and writes TWO separate figures:
#          - metrics_curves.png  (Dice / IoU / Accuracy)
#          - loss_curves.png     (train loss / val loss)
#      Marks the BEST weight (max Dice) on both, and writes a summary.
#
# Run:
#   python 4_Test_unet_visual_pro.py
# =============================================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda/Windows OpenMP

import re
import csv
import random
import importlib.util
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
from PIL import Image, ImageDraw

import torch

import matplotlib
matplotlib.use("TkAgg")            # interactive backend so we can show full-screen
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(r"D:\project\0000_OPG\lengani")

TRAIN_SCRIPT = PROJECT_ROOT / "2_Training_unet2.py"
CKPT_PATH    = PROJECT_ROOT / "Unet_run" / "unet_best.pt"

# --- cropped dataset (image2) -------------------------------------------------
IMAGE2_DIR   = PROJECT_ROOT / "image2"
# prefer the split folders produced by the data reader; fall back to image2/images
CROP_SPLIT_IMAGES = IMAGE2_DIR / "unet" / "images"     # {train,val,test}
CROP_SPLIT_MASKS  = IMAGE2_DIR / "unet" / "masks"      # {train,val,test}
CROP_FLAT_IMAGES  = IMAGE2_DIR / "images"              # flat fallback
CROP_FLAT_JSON    = IMAGE2_DIR / "json"

# --- original (un-cropped) images --------------------------------------------
ORIGINAL_ROOT = PROJECT_ROOT / "images"

# --- training log ------------------------------------------------------------
LOG_TXT_CANDIDATES = [
    PROJECT_ROOT / "log training u net.txt",
    PROJECT_ROOT / "Unet_run" / "log training u net.txt",
    PROJECT_ROOT / "training_log.txt",
    PROJECT_ROOT / "Unet_run" / "training_log.txt",
]

OUTPUT_DIR = PROJECT_ROOT / "Unet_run" / "test_visual_pro"

SPLIT       = "test"     # test | val | train   (used only for the split layout)
N_SAMPLES   = 20
IMG_SIZE    = 256
THRESHOLD   = 0.50
SEED        = 42

# image-quality knobs
SAVE_DPI       = 220     # high quality
PANEL_FIGSIZE  = (22, 11)
SHOW_FULLSCREEN = True   # maximize each figure window before saving

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# GENERAL UTILS
# =============================================================================
def safe_float(x):
    x = str(x).strip().lower()
    if x in {"nan", "+nan", "-nan"}:
        return np.nan
    if x in {"inf", "+inf", "infinity", "+infinity"}:
        return np.inf
    if x in {"-inf", "-infinity"}:
        return -np.inf
    return float(x)


def strip_dir_prefix(crop_stem: str) -> str:
    """
    Recover the original stem from a crop stem made by `<parentdir>__<stem>`.
    "1__amini2_2 - Copy"  ->  "amini2_2 - Copy"
    Handles the collision suffix variant "<dir>__<stem>__<n>" by also
    returning the stripped core; the search routine is tolerant either way.
    """
    if "__" in crop_stem:
        # drop only the FIRST "<dir>__" segment (the parent-dir prefix)
        return crop_stem.split("__", 1)[1]
    return crop_stem


def normalize_stem(stem: str) -> str:
    """Lowercase + strip common crop/roi tokens + remove non-alphanumerics."""
    s = stem.lower()
    for token in ("_crop", "-crop", " crop", "_cropped", "-cropped", " cropped",
                  "_roi", "-roi", " roi", "_patch", "-patch", " patch",
                  "_mask", "-mask", " mask", "_256", "-256"):
        if token in s:
            s = s.split(token)[0]
    s = re.sub(r"[^a-z0-9\u0600-\u06ff]+", "", s)
    return s


def make_placeholder(text, size=(256, 256)):
    img = Image.new("RGB", size, color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.multiline_text((15, size[1] // 2 - 35), text, fill=(20, 20, 20), spacing=6)
    return img


def read_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return img


def read_mask(path, size=None):
    mask = Image.open(path).convert("L")
    if size is not None:
        mask = mask.resize((size, size), Image.NEAREST)
    arr = np.asarray(mask).astype(np.float32) / 255.0
    return (arr > 0.5).astype(np.uint8)


# =============================================================================
# LOAD UNET FROM TRAINING SCRIPT
# =============================================================================
def load_unet_class():
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")
    spec = importlib.util.spec_from_file_location("training_unet_module", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "UNet"):
        raise AttributeError("UNet class not found inside 2_Training_unet2.py")
    return module.UNet


def load_model():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    UNet = load_unet_class()
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    base    = ckpt.get("base", 32)
    dropout = ckpt.get("dropout", 0.20)
    model = UNet(in_ch=3, out_ch=1, base=base, drop=dropout).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("=" * 90)
    print(f"Loaded best checkpoint: {CKPT_PATH}")
    print(f"Device: {DEVICE}")
    if "epoch" in ckpt:
        print(f"Best epoch (ckpt): {ckpt['epoch']}")
    if "val" in ckpt:
        print(f"Validation metrics (ckpt): {ckpt['val']}")
    print("=" * 90)
    return model, ckpt


# =============================================================================
# ORIGINAL IMAGE SEARCH
# =============================================================================
def build_original_index(original_root):
    files = []
    if not original_root.exists():
        print(f"[WARNING] Original root not found: {original_root}")
        return files
    print(f"[INFO] Indexing original images under: {original_root}")
    for p in original_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    print(f"[INFO] Original images found: {len(files)}")
    return files


def find_original_image(crop_path, original_files):
    """
    Find the original image before crop. The crop stem encodes the parent dir
    as a prefix ("<dir>__<origstem>"), so we first strip that prefix.
    Priority: exact filename -> exact stem -> normalized stem -> contains -> fuzzy.
    """
    crop_core = strip_dir_prefix(crop_path.stem)          # e.g. "amini2_2 - Copy"
    core_lower = crop_core.lower()
    core_norm  = normalize_stem(crop_core)

    # 1) exact filename (any image ext) e.g. amini2_2 - Copy.jpg/.png
    for p in original_files:
        if p.stem.lower() == core_lower:
            return p, "exact_stem"

    # 2) normalized stem
    norm_matches = [p for p in original_files
                    if core_norm and normalize_stem(p.stem) == core_norm]
    if norm_matches:
        return sorted(norm_matches, key=lambda x: len(str(x)))[0], "normalized_stem"

    # 3) contains
    contains = []
    for p in original_files:
        ps, pn = p.stem.lower(), normalize_stem(p.stem)
        if core_lower in ps or ps in core_lower:
            contains.append(p)
        elif core_norm and (core_norm in pn or pn in core_norm):
            contains.append(p)
    if contains:
        return sorted(contains, key=lambda x: len(str(x)))[0], "contains"

    # 4) fuzzy
    best_path, best_score = None, 0.0
    for p in original_files:
        pn = normalize_stem(p.stem)
        if not pn or not core_norm:
            continue
        score = SequenceMatcher(None, core_norm, pn).ratio()
        if score > best_score:
            best_score, best_path = score, p
    if best_path is not None and best_score >= 0.82:
        return best_path, f"fuzzy_{best_score:.3f}"

    return None, "not_found"


# =============================================================================
# PREDICTION AND METRICS
# =============================================================================
def pil_to_tensor(img_pil):
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x.to(DEVICE)


@torch.no_grad()
def predict_mask(model, crop_img_pil):
    x = pil_to_tensor(crop_img_pil)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    pred = (prob > THRESHOLD).astype(np.uint8)
    return prob, pred


def dice_iou(gt, pred, eps=1e-7):
    gt, pred = gt.astype(bool), pred.astype(bool)
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return dice, iou


def overlay_single_mask(image_pil, mask, color=(255, 0, 0), alpha=0.45):
    out = np.asarray(image_pil).astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def overlay_both_masks(image_pil, gt_mask, pred_mask, alpha=0.50):
    """Green = GT only, Red = Pred only, Yellow = overlap."""
    out = np.asarray(image_pil).astype(np.float32).copy()
    gt, pr = gt_mask.astype(bool), pred_mask.astype(bool)
    both, gt_only, pr_only = gt & pr, gt & ~pr, pr & ~gt
    out[gt_only] = (1 - alpha) * out[gt_only] + alpha * np.array([0, 255, 0], np.float32)
    out[pr_only] = (1 - alpha) * out[pr_only] + alpha * np.array([255, 0, 0], np.float32)
    out[both]    = (1 - alpha) * out[both]    + alpha * np.array([255, 255, 0], np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =============================================================================
# FULL-SCREEN HELPER
# =============================================================================
def show_fullscreen(fig, pause=0.6):
    """Maximize the figure window (best-effort across backends), then draw."""
    if not SHOW_FULLSCREEN:
        return
    try:
        mng = fig.canvas.manager
        backend = matplotlib.get_backend().lower()
        if "tk" in backend:
            try:
                mng.window.state("zoomed")          # Windows
            except Exception:
                mng.window.attributes("-fullscreen", True)
        elif "qt" in backend:
            mng.window.showMaximized()
        elif "wx" in backend:
            mng.frame.Maximize(True)
    except Exception:
        pass
    try:
        plt.show(block=False)
        plt.pause(pause)
    except Exception:
        pass


# =============================================================================
# SAVE VISUAL REPORT (high quality, shown full-screen first)
# =============================================================================
def save_visual_report(out_path, original_pil, crop_pil, gt_mask, pred_mask,
                       prob_mask, sample_name, dice, iou, original_path, match_status):
    gt_mask_img   = Image.fromarray((gt_mask * 255).astype(np.uint8))
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    gt_overlay    = overlay_single_mask(crop_pil, gt_mask, (0, 255, 0), 0.45)
    pred_overlay  = overlay_single_mask(crop_pil, pred_mask, (255, 0, 0), 0.45)
    both_overlay  = overlay_both_masks(crop_pil, gt_mask, pred_mask, 0.50)

    fig = plt.figure(figsize=PANEL_FIGSIZE)
    if original_path is None:
        original_txt = "Original: NOT FOUND"
    else:
        original_txt = f"Original: {Path(original_path).name} | match={match_status}"
    fig.suptitle(
        f"{sample_name} | Dice={dice:.4f} | IoU={iou:.4f} | Threshold={THRESHOLD}\n{original_txt}",
        fontsize=14
    )

    panels = [
        ("1) Original before crop", original_pil, None),
        ("2) Cropped image / U-Net input", crop_pil, None),
        ("3) Ground-truth mask", gt_mask_img, "gray"),
        ("4) Predicted mask", pred_mask_img, "gray"),
        ("5) GT on crop (Green)", gt_overlay, None),
        ("6) Pred on crop (Red)", pred_overlay, None),
        ("7) Both (G=GT R=Pred Y=Overlap)", both_overlay, None),
        ("8) Probability map", prob_mask, "jet"),
    ]
    for i, (title, obj, cmap) in enumerate(panels, start=1):
        ax = plt.subplot(2, 4, i)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        if isinstance(obj, np.ndarray):
            ax.imshow(obj, cmap=cmap, vmin=0, vmax=1)
        elif cmap == "gray":
            ax.imshow(obj, cmap="gray")
        else:
            ax.imshow(obj)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    show_fullscreen(fig)                          # full-screen preview
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# TRAINING LOG PARSER + TWO SEPARATE FIGURES
# =============================================================================
LOG_RE = re.compile(
    r"\[unet\]\s+ep\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\s+"
    r"loss=(?P<loss>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"val_loss=(?P<val_loss>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"Dice=(?P<dice>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"IoU=(?P<iou>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"Acc=(?P<acc>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"lr=(?P<lr>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)


def find_log_file():
    for p in LOG_TXT_CANDIDATES:
        if p.exists():
            return p
    return None


def parse_training_log(log_path):
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            rows.append({
                "epoch": int(m.group("epoch")),
                "total": int(m.group("total")),
                "loss": safe_float(m.group("loss")),
                "val_loss": safe_float(m.group("val_loss")),
                "dice": safe_float(m.group("dice")),
                "iou": safe_float(m.group("iou")),
                "acc": safe_float(m.group("acc")),
                "lr": safe_float(m.group("lr")),
            })
    return rows


def save_log_csv(rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "total", "loss", "val_loss",
                                          "dice", "iou", "acc", "lr"])
        w.writeheader()
        w.writerows(rows)


def finite_rows_until_first_nan(rows):
    clean = []
    for r in rows:
        if any(not np.isfinite(v) for v in
               (r["loss"], r["val_loss"], r["dice"], r["iou"], r["acc"], r["lr"])):
            break
        clean.append(r)
    return clean


def best_row_by_dice(rows):
    valid = [r for r in rows if np.isfinite(r["dice"])]
    return max(valid, key=lambda r: r["dice"]) if valid else None


def plot_loss_curves(rows, out_path, title_suffix=""):
    """ONE figure: train loss + val loss, with best-weight marker."""
    if not rows:
        return
    ep       = np.array([r["epoch"] for r in rows], float)
    loss     = np.array([r["loss"] for r in rows], float)
    val_loss = np.array([r["val_loss"] for r in rows], float)
    best     = best_row_by_dice(rows)

    fig = plt.figure(figsize=(14, 8))
    ax = plt.subplot(1, 1, 1)
    ax.plot(ep, loss, label="train loss", linewidth=1.6)
    ax.plot(ep, val_loss, label="val loss", linewidth=1.6)
    if best is not None:
        ax.axvline(best["epoch"], linestyle="--", alpha=0.6, color="k")
        ax.scatter([best["epoch"]], [best["val_loss"]], zorder=5, color="red")
        ax.annotate(f" best weight\n ep={best['epoch']}\n val_loss={best['val_loss']:.4f}",
                    (best["epoch"], best["val_loss"]), fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss curves" + (f" ({title_suffix})" if title_suffix else ""))
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    show_fullscreen(fig)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(rows, out_path, title_suffix=""):
    """ONE figure: Dice / IoU / Accuracy, with best-weight marker on Dice."""
    if not rows:
        return
    ep   = np.array([r["epoch"] for r in rows], float)
    dice = np.array([r["dice"] for r in rows], float)
    iou  = np.array([r["iou"] for r in rows], float)
    acc  = np.array([r["acc"] for r in rows], float)
    best = best_row_by_dice(rows)

    fig = plt.figure(figsize=(16, 9))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(ep, dice, label="val Dice", linewidth=1.8)
    ax1.plot(ep, iou,  label="val IoU",  linewidth=1.8)
    if best is not None:
        ax1.axvline(best["epoch"], linestyle="--", alpha=0.6, color="k")
        ax1.scatter([best["epoch"]], [best["dice"]], zorder=5, color="red")
        ax1.annotate(f" BEST weight\n ep={best['epoch']}\n Dice={best['dice']:.4f}"
                     f"\n IoU={best['iou']:.4f}",
                     (best["epoch"], best["dice"]), fontsize=10)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Metric")
    ax1.set_title("Validation Dice / IoU")
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(ep, acc, label="val Accuracy", color="green", linewidth=1.8)
    if best is not None:
        ax2.axvline(best["epoch"], linestyle="--", alpha=0.6, color="k")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.grid(True, alpha=0.3); ax2.legend()

    if best is not None:
        fig.suptitle(f"Metric curves {title_suffix} | "
                     f"Best Dice={best['dice']:.4f} @ epoch {best['epoch']}", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    show_fullscreen(fig)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def make_training_curves_from_log():
    log_path = find_log_file()
    if log_path is None:
        print("[WARNING] No training log found. Expected one of:")
        for p in LOG_TXT_CANDIDATES:
            print(f"          {p}")
        return

    print("=" * 90)
    print(f"[INFO] Reading training log: {log_path}")
    rows = parse_training_log(log_path)
    if not rows:
        print("[WARNING] No epoch lines parsed from log.")
        return

    curves_dir = OUTPUT_DIR / "training_curves_from_log"
    curves_dir.mkdir(parents=True, exist_ok=True)

    save_log_csv(rows, curves_dir / "parsed_training_log.csv")

    clean = finite_rows_until_first_nan(rows)
    use   = clean if clean else rows
    tag   = "clean until first NaN" if clean else "all log"

    # TWO separate figures, exactly as requested
    plot_metric_curves(use, curves_dir / "metrics_curves.png", tag)
    plot_loss_curves(use,   curves_dir / "loss_curves.png",    tag)

    best = best_row_by_dice(use)
    with open(curves_dir / "best_epoch_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"log_path: {log_path}\n")
        f.write(f"total_parsed_rows: {len(rows)}\n")
        f.write(f"clean_rows_until_first_nan: {len(clean)}\n")
        if best is not None:
            f.write(f"best_epoch: {best['epoch']}\n")
            f.write(f"best_dice: {best['dice']:.6f}\n")
            f.write(f"best_iou: {best['iou']:.6f}\n")
            f.write(f"best_val_loss: {best['val_loss']:.6f}\n")
            f.write(f"best_acc: {best['acc']:.6f}\n")

    print(f"[INFO] metrics_curves.png + loss_curves.png saved to: {curves_dir}")
    if best is not None:
        print(f"[INFO] BEST weight: epoch={best['epoch']} | Dice={best['dice']:.4f} | "
              f"IoU={best['iou']:.4f} | val_loss={best['val_loss']:.4f}")
    print("=" * 90)


# =============================================================================
# COLLECT CROP / MASK PAIRS  (split layout, with flat fallback)
# =============================================================================
def collect_crop_mask_pairs(split):
    img_dir  = CROP_SPLIT_IMAGES / split
    mask_dir = CROP_SPLIT_MASKS / split
    pairs = []

    if img_dir.exists() and mask_dir.exists():
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                cand = [mask_dir / f"{img_path.stem}{e}" for e in IMAGE_EXTS]
                cand = [c for c in cand if c.exists()]
                if not cand:
                    continue
                mask_path = cand[0]
            pairs.append((img_path, mask_path))
        if pairs:
            print(f"[INFO] Using split layout: {img_dir}")
            return pairs

    # fallback: flat image2/images with no separate masks
    print(f"[INFO] Split layout empty; falling back to flat: {CROP_FLAT_IMAGES}")
    for img_path in sorted(CROP_FLAT_IMAGES.glob("*")):
        if img_path.suffix.lower() in IMAGE_EXTS:
            pairs.append((img_path, None))
    return pairs


# =============================================================================
# VISUAL TEST MAIN
# =============================================================================
def run_visual_tests():
    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model()

    pairs = collect_crop_mask_pairs(SPLIT)
    if not pairs:
        raise RuntimeError(f"No crops found for split '{SPLIT}' in {IMAGE2_DIR}")

    random.shuffle(pairs)
    selected = pairs[:min(N_SAMPLES, len(pairs))]

    original_files = build_original_index(ORIGINAL_ROOT)

    visual_dir = OUTPUT_DIR / f"visual_{SPLIT}"
    visual_dir.mkdir(parents=True, exist_ok=True)
    report_csv = visual_dir / "visual_test_report.csv"

    print("=" * 90)
    print(f"[INFO] Split: {SPLIT} | total pairs: {len(pairs)} | selected: {len(selected)}")
    print(f"[INFO] Output: {visual_dir}")
    print("=" * 90)

    results = []
    for idx, (crop_path, mask_path) in enumerate(selected, start=1):
        crop_pil = read_rgb(crop_path, size=IMG_SIZE)

        if mask_path is not None:
            gt_mask = read_mask(mask_path, size=IMG_SIZE)
        else:
            gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)

        prob_mask, pred_mask = predict_mask(model, crop_pil)
        dsc, iou = dice_iou(gt_mask, pred_mask)

        original_path, match_status = find_original_image(crop_path, original_files)
        if original_path is not None:
            original_pil = read_rgb(original_path, size=IMG_SIZE)
            original_str = str(original_path)
        else:
            original_pil = make_placeholder("Original\nNOT FOUND", (IMG_SIZE, IMG_SIZE))
            original_str = "NOT FOUND"

        out_path = visual_dir / f"{idx:03d}_{crop_path.stem}_visual.png"
        save_visual_report(out_path, original_pil, crop_pil, gt_mask, pred_mask,
                           prob_mask, crop_path.name, dsc, iou, original_path, match_status)

        results.append({
            "index": idx, "sample": crop_path.name,
            "dice": dsc, "iou": iou,
            "crop_path": str(crop_path),
            "mask_path": str(mask_path) if mask_path else "NONE",
            "original_path": original_str,
            "original_match_status": match_status,
            "visual_path": str(out_path),
        })
        print(f"[{idx:03d}] {crop_path.name} | Dice={dsc:.4f} | IoU={iou:.4f} | "
              f"orig={match_status}")

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "index", "sample", "dice", "iou", "crop_path", "mask_path",
            "original_path", "original_match_status", "visual_path"])
        w.writeheader()
        w.writerows(results)

    print("=" * 90)
    print(f"[INFO] Visual testing done. Report: {report_csv}")
    print("=" * 90)


def main():
    make_training_curves_from_log()   # metrics_curves.png + loss_curves.png
    run_visual_tests()                # side-by-side panels


if __name__ == "__main__":
    main()