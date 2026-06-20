# =============================================================================
# 3_Test_unet_visual_samples_v2.py
#
# Visual test for U-Net segmentation + training curves from text log
#
# Features:
#   1) Load best U-Net checkpoint: Unet_run/unet_best.pt
#   2) Test several samples from prepared/unet/images/{train,val,test}
#   3) Search original pre-crop image recursively inside:
#        D:\project\0000_OPG\lengani\images
#   4) Save visual comparison:
#        original image before crop
#        cropped image
#        ground-truth mask
#        predicted mask
#        GT overlay
#        Pred overlay
#        both overlays
#        probability map
#   5) Parse training log txt and draw training_curves from it
#
# Run:
#   python 3_Test_unet_visual_samples_v2.py
# =============================================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import csv
import math
import random
import importlib.util
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
from PIL import Image, ImageDraw

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(r"D:\project\0000_OPG\lengani")

TRAIN_SCRIPT = PROJECT_ROOT / "2_Training_unet2.py"
CKPT_PATH    = PROJECT_ROOT / "Unet_run" / "unet_best.pt"

PREPARED_DIR = PROJECT_ROOT / "prepared" / "unet"
IMAGE_DIR    = PREPARED_DIR / "images"
MASK_DIR     = PREPARED_DIR / "masks"

# تصویرهای اصلی قبل از crop اینجا هستند
ORIGINAL_ROOT = Path(r"D:\project\0000_OPG\lengani\images")

OUTPUT_DIR = PROJECT_ROOT / "Unet_run" / "test_visual_samples_v2"

# split می‌تواند test یا val یا train باشد
SPLIT = "test"

# تعداد نمونه برای تست
N_SAMPLES = 20

# سایز ورودی شبکه
IMG_SIZE = 256

# آستانه تبدیل probability به mask
THRESHOLD = 0.50

# برای تکرارپذیری انتخاب نمونه‌ها
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# مسیرهای احتمالی فایل لاگ آموزش.
# فایل لاگ ارسالی را بهتر است با نام training_log.txt کنار همین فایل یا داخل Unet_run ذخیره کنی.
LOG_TXT_CANDIDATES = [
    PROJECT_ROOT / "training_log.txt",
    PROJECT_ROOT / "Pasted text(31).txt",
    PROJECT_ROOT / "Unet_run" / "training_log.txt",
    PROJECT_ROOT / "Unet_run" / "Pasted text(31).txt",
]


# =============================================================================
# GENERAL UTILS
# =============================================================================
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def safe_float(x):
    """
    Converts string to float.
    Handles nan, inf, and normal scientific notation.
    """
    x = str(x).strip().lower()
    if x in {"nan", "+nan", "-nan"}:
        return np.nan
    if x in {"inf", "+inf", "infinity", "+infinity"}:
        return np.inf
    if x in {"-inf", "-infinity"}:
        return -np.inf
    return float(x)


def normalize_stem(stem):
    """
    Normalizes file stems to improve original-image matching.
    Removes common crop/roi/patch suffixes and separators.
    """
    s = stem.lower()

    remove_tokens = [
        "_crop", "-crop", " crop",
        "_cropped", "-cropped", " cropped",
        "_roi", "-roi", " roi",
        "_patch", "-patch", " patch",
        "_mask", "-mask", " mask",
        "_img", "-img",
        "_image", "-image",
        "_256", "-256",
    ]

    for token in remove_tokens:
        if token in s:
            s = s.split(token)[0]

    s = re.sub(r"[^a-z0-9\u0600-\u06FF]+", "", s)
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
    arr = (arr > 0.5).astype(np.uint8)
    return arr


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

    base = ckpt.get("base", 32)
    dropout = ckpt.get("dropout", 0.20)

    model = UNet(in_ch=3, out_ch=1, base=base, drop=dropout).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("=" * 90)
    print(f"Loaded best checkpoint: {CKPT_PATH}")
    print(f"Device: {DEVICE}")
    if "epoch" in ckpt:
        print(f"Best epoch: {ckpt['epoch']}")
    if "val" in ckpt:
        print(f"Validation metrics: {ckpt['val']}")
    print("=" * 90)

    return model, ckpt


# =============================================================================
# ORIGINAL IMAGE SEARCH
# =============================================================================
def build_original_index(original_root):
    """
    Recursively indexes all images under ORIGINAL_ROOT.
    """
    files = []

    if not original_root.exists():
        print(f"[WARNING] Original root not found: {original_root}")
        return files

    print(f"[INFO] Indexing original images recursively from:")
    print(f"       {original_root}")

    for p in original_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)

    print(f"[INFO] Original images found: {len(files)}")
    return files


def find_original_image(crop_path, original_files):
    """
    Finds the original image before crop.
    Search priority:
      1) exact filename match
      2) exact stem match
      3) normalized stem match
      4) contains match
      5) fuzzy best match
    """
    crop_name = crop_path.name.lower()
    crop_stem = crop_path.stem.lower()
    crop_norm = normalize_stem(crop_path.stem)

    # 1) exact filename
    for p in original_files:
        if p.name.lower() == crop_name:
            return p, "exact_filename"

    # 2) exact stem
    exact_stem = []
    for p in original_files:
        if p.stem.lower() == crop_stem:
            exact_stem.append(p)
    if exact_stem:
        return sorted(exact_stem, key=lambda x: len(str(x)))[0], "exact_stem"

    # 3) normalized stem
    norm_matches = []
    for p in original_files:
        if normalize_stem(p.stem) == crop_norm and crop_norm:
            norm_matches.append(p)
    if norm_matches:
        return sorted(norm_matches, key=lambda x: len(str(x)))[0], "normalized_stem"

    # 4) contains
    contains_matches = []
    for p in original_files:
        p_stem = p.stem.lower()
        p_norm = normalize_stem(p.stem)

        if crop_stem in p_stem or p_stem in crop_stem:
            contains_matches.append(p)
        elif crop_norm and (crop_norm in p_norm or p_norm in crop_norm):
            contains_matches.append(p)

    if contains_matches:
        return sorted(contains_matches, key=lambda x: len(str(x)))[0], "contains"

    # 5) fuzzy best match
    best_path = None
    best_score = 0.0

    for p in original_files:
        p_norm = normalize_stem(p.stem)
        if not p_norm or not crop_norm:
            continue

        score = SequenceMatcher(None, crop_norm, p_norm).ratio()
        if score > best_score:
            best_score = score
            best_path = p

    # آستانه محافظه‌کارانه؛ اگر کمتر باشد ممکن است تصویر اشتباه انتخاب کند
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
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return dice, iou


def overlay_single_mask(image_pil, mask, color=(255, 0, 0), alpha=0.45):
    img = np.asarray(image_pil).astype(np.float32)
    out = img.copy()

    m = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)

    out[m] = (1 - alpha) * out[m] + alpha * color_arr
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


def overlay_both_masks(image_pil, gt_mask, pred_mask, alpha=0.50):
    """
    Green  = GT only
    Red    = Pred only
    Yellow = Overlap
    """
    img = np.asarray(image_pil).astype(np.float32)
    out = img.copy()

    gt = gt_mask.astype(bool)
    pr = pred_mask.astype(bool)

    both = gt & pr
    gt_only = gt & (~pr)
    pr_only = pr & (~gt)

    green = np.array([0, 255, 0], dtype=np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)
    yellow = np.array([255, 255, 0], dtype=np.float32)

    out[gt_only] = (1 - alpha) * out[gt_only] + alpha * green
    out[pr_only] = (1 - alpha) * out[pr_only] + alpha * red
    out[both] = (1 - alpha) * out[both] + alpha * yellow

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# =============================================================================
# SAVE VISUAL REPORT
# =============================================================================
def save_visual_report(
    out_path,
    original_pil,
    crop_pil,
    gt_mask,
    pred_mask,
    prob_mask,
    sample_name,
    dice,
    iou,
    original_path,
    match_status
):
    gt_mask_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))

    gt_overlay = overlay_single_mask(crop_pil, gt_mask, color=(0, 255, 0), alpha=0.45)
    pred_overlay = overlay_single_mask(crop_pil, pred_mask, color=(255, 0, 0), alpha=0.45)
    both_overlay = overlay_both_masks(crop_pil, gt_mask, pred_mask, alpha=0.50)

    fig = plt.figure(figsize=(20, 10))

    if original_path is None:
        original_txt = "Original: NOT FOUND"
    else:
        original_txt = f"Original: {Path(original_path).name} | match={match_status}"

    fig.suptitle(
        f"{sample_name} | Dice={dice:.4f} | IoU={iou:.4f} | Threshold={THRESHOLD}\n"
        f"{original_txt}",
        fontsize=13
    )

    panels = [
        ("1) Original before crop", original_pil, None),
        ("2) Cropped image / U-Net input", crop_pil, None),
        ("3) Ground-truth mask", gt_mask_img, "gray"),
        ("4) Network predicted mask", pred_mask_img, "gray"),
        ("5) GT mask on crop - Green", gt_overlay, None),
        ("6) Pred mask on crop - Red", pred_overlay, None),
        ("7) Both masks\nGreen=GT | Red=Pred | Yellow=Overlap", both_overlay, None),
        ("8) Probability map", prob_mask, "jet"),
    ]

    for i, (title, obj, cmap) in enumerate(panels, start=1):
        ax = plt.subplot(2, 4, i)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

        if isinstance(obj, np.ndarray):
            ax.imshow(obj, cmap=cmap, vmin=0, vmax=1)
        else:
            if cmap == "gray":
                ax.imshow(obj, cmap="gray")
            else:
                ax.imshow(obj)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


# =============================================================================
# TRAINING LOG PARSER AND CURVES
# =============================================================================
LOG_RE = re.compile(
    r"\[unet\]\s+ep\s+"
    r"(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\s+"
    r"loss=(?P<loss>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"val_loss=(?P<val_loss>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"Dice=(?P<dice>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"IoU=(?P<iou>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"Acc=(?P<acc>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"lr=(?P<lr>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)",
    re.IGNORECASE
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


def save_training_log_csv(rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "total", "loss", "val_loss", "dice", "iou", "acc", "lr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_rows_until_first_nan(rows):
    """
    Returns rows until before first NaN/Inf in loss or val_loss.
    This is useful because after NaN, curves are not meaningful.
    """
    clean = []
    for r in rows:
        values = [r["loss"], r["val_loss"], r["dice"], r["iou"], r["acc"], r["lr"]]
        if any((not np.isfinite(v)) for v in values):
            break
        clean.append(r)
    return clean


def best_row_by_dice(rows):
    valid = [r for r in rows if np.isfinite(r["dice"])]
    if not valid:
        return None
    return max(valid, key=lambda r: r["dice"])


def plot_training_curves(rows, out_path, title_suffix=""):
    if not rows:
        return

    ep = np.array([r["epoch"] for r in rows], dtype=float)

    loss = np.array([r["loss"] for r in rows], dtype=float)
    val_loss = np.array([r["val_loss"] for r in rows], dtype=float)
    dice = np.array([r["dice"] for r in rows], dtype=float)
    iou = np.array([r["iou"] for r in rows], dtype=float)
    acc = np.array([r["acc"] for r in rows], dtype=float)
    lr = np.array([r["lr"] for r in rows], dtype=float)

    best = best_row_by_dice(rows)

    fig = plt.figure(figsize=(16, 10))

    # Loss
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(ep, loss, label="train loss")
    ax1.plot(ep, val_loss, label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss curves")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Dice / IoU
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(ep, dice, label="val Dice")
    ax2.plot(ep, iou, label="val IoU")
    if best is not None:
        ax2.axvline(best["epoch"], linestyle="--", alpha=0.6)
        ax2.scatter([best["epoch"]], [best["dice"]])
        ax2.text(
            best["epoch"],
            best["dice"],
            f" best Dice={best['dice']:.4f}\n ep={best['epoch']}",
            fontsize=9
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Validation Dice / IoU")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Accuracy
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(ep, acc, label="val Acc")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Validation Accuracy")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # LR
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(ep, lr, label="learning rate")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("LR")
    ax4.set_title("Learning rate")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    if best is not None:
        fig.suptitle(
            f"Training curves {title_suffix} | Best Dice={best['dice']:.4f} @ epoch {best['epoch']}",
            fontsize=14
        )
    else:
        fig.suptitle(f"Training curves {title_suffix}", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def make_training_curves_from_log():
    log_path = find_log_file()

    if log_path is None:
        print("[WARNING] No training log file found.")
        print("          Put your log file here, for example:")
        print(f"          {PROJECT_ROOT / 'training_log.txt'}")
        return

    print("=" * 90)
    print(f"[INFO] Reading training log: {log_path}")

    rows = parse_training_log(log_path)

    if not rows:
        print("[WARNING] No epoch lines were parsed from log.")
        return

    curves_dir = OUTPUT_DIR / "training_curves_from_log"
    curves_dir.mkdir(parents=True, exist_ok=True)

    csv_path = curves_dir / "parsed_training_log.csv"
    save_training_log_csv(rows, csv_path)

    all_curve_path = curves_dir / "training_curves_all_log.png"
    plot_training_curves(rows, all_curve_path, title_suffix="all log")

    clean = finite_rows_until_first_nan(rows)
    clean_curve_path = curves_dir / "training_curves_clean_until_nan.png"
    plot_training_curves(clean, clean_curve_path, title_suffix="clean until first NaN")

    best = best_row_by_dice(clean if clean else rows)

    summary_path = curves_dir / "best_epoch_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"log_path: {log_path}\n")
        f.write(f"total_parsed_rows: {len(rows)}\n")
        f.write(f"clean_rows_until_first_nan: {len(clean)}\n")
        if best is not None:
            f.write(f"best_epoch: {best['epoch']}\n")
            f.write(f"best_dice: {best['dice']:.6f}\n")
            f.write(f"best_iou: {best['iou']:.6f}\n")
            f.write(f"best_val_loss: {best['val_loss']:.6f}\n")
            f.write(f"best_acc: {best['acc']:.6f}\n")

    print(f"[INFO] Parsed log CSV saved: {csv_path}")
    print(f"[INFO] Training curves all log saved: {all_curve_path}")
    print(f"[INFO] Training curves clean saved: {clean_curve_path}")
    print(f"[INFO] Best summary saved: {summary_path}")

    if best is not None:
        print(
            f"[INFO] Best from parsed clean log: "
            f"epoch={best['epoch']} | Dice={best['dice']:.4f} | "
            f"IoU={best['iou']:.4f} | val_loss={best['val_loss']:.4f}"
        )

    print("=" * 90)


# =============================================================================
# VISUAL TEST MAIN
# =============================================================================
def collect_image_mask_pairs(split):
    img_dir = IMAGE_DIR / split
    mask_dir = MASK_DIR / split

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    pairs = []

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        # ماسک معمولاً هم‌نام با تصویر است
        mask_path = mask_dir / img_path.name

        if not mask_path.exists():
            # اگر پسوند متفاوت بود، با stem بگرد
            possible = []
            for ext in IMAGE_EXTS:
                p = mask_dir / f"{img_path.stem}{ext}"
                if p.exists():
                    possible.append(p)
            if possible:
                mask_path = possible[0]
            else:
                continue

        pairs.append((img_path, mask_path))

    return pairs


def run_visual_tests():
    random.seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model()

    pairs = collect_image_mask_pairs(SPLIT)

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found for split: {SPLIT}")

    random.shuffle(pairs)
    selected_pairs = pairs[:min(N_SAMPLES, len(pairs))]

    original_files = build_original_index(ORIGINAL_ROOT)

    visual_dir = OUTPUT_DIR / f"visual_{SPLIT}"
    visual_dir.mkdir(parents=True, exist_ok=True)

    report_csv = visual_dir / "visual_test_report.csv"

    results = []

    print("=" * 90)
    print(f"[INFO] Split: {SPLIT}")
    print(f"[INFO] Total pairs: {len(pairs)}")
    print(f"[INFO] Selected samples: {len(selected_pairs)}")
    print(f"[INFO] Output visual dir: {visual_dir}")
    print("=" * 90)

    for idx, (crop_path, mask_path) in enumerate(selected_pairs, start=1):
        crop_pil = read_rgb(crop_path, size=IMG_SIZE)
        gt_mask = read_mask(mask_path, size=IMG_SIZE)

        prob_mask, pred_mask = predict_mask(model, crop_pil)
        dsc, iou = dice_iou(gt_mask, pred_mask)

        original_path, match_status = find_original_image(crop_path, original_files)

        if original_path is not None:
            # برای چیدن کنار crop، تصویر اصلی را هم در سایز 256 نشان می‌دهیم
            original_pil = read_rgb(original_path, size=IMG_SIZE)
            original_str = str(original_path)
        else:
            original_pil = make_placeholder(
                "Original image\nbefore crop\nNOT FOUND",
                size=(IMG_SIZE, IMG_SIZE)
            )
            original_str = "NOT FOUND"

        out_path = visual_dir / f"{idx:03d}_{crop_path.stem}_visual.png"

        save_visual_report(
            out_path=out_path,
            original_pil=original_pil,
            crop_pil=crop_pil,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            prob_mask=prob_mask,
            sample_name=crop_path.name,
            dice=dsc,
            iou=iou,
            original_path=original_path,
            match_status=match_status
        )

        results.append({
            "index": idx,
            "sample": crop_path.name,
            "dice": dsc,
            "iou": iou,
            "crop_path": str(crop_path),
            "mask_path": str(mask_path),
            "original_path": original_str,
            "original_match_status": match_status,
            "visual_path": str(out_path),
        })

        print(
            f"[{idx:03d}] {crop_path.name} | "
            f"Dice={dsc:.4f} | IoU={iou:.4f} | "
            f"Original={match_status}"
        )
        print(f"      visual -> {out_path}")

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "sample",
            "dice",
            "iou",
            "crop_path",
            "mask_path",
            "original_path",
            "original_match_status",
            "visual_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 90)
    print("[INFO] Visual testing done.")
    print(f"[INFO] Report saved: {report_csv}")
    print("=" * 90)


def main():
    # 1) ساخت نمودارهای آموزش از فایل لاگ ارسالی
    make_training_curves_from_log()

    # 2) تست تصویری چند نمونه با وزن بهترین مدل
    run_visual_tests()


if __name__ == "__main__":
    main()