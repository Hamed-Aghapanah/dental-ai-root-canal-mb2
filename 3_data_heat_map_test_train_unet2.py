# =============================================================================
# 3_Test_unet_visual_samples.py
# Test and visualize several U-Net segmentation samples
#
# Panels:
# 1) Original image before crop       اگر پیدا شود
# 2) Cropped image used by U-Net
# 3) Ground-truth mask
# 4) Predicted mask by network
# 5) Ground-truth mask overlay on crop
# 6) Predicted mask overlay on crop
# 7) Both masks overlay on crop
#
# Run:
#   python 3_Test_unet_visual_samples.py
# =============================================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_SCRIPT = PROJECT_ROOT / "2_Training_unet2.py"
CKPT_PATH    = PROJECT_ROOT / "Unet_run" / "unet_best.pt"

PREPARED_DIR = PROJECT_ROOT / "prepared" / "unet"
IMAGE_DIR    = PREPARED_DIR / "images"
MASK_DIR     = PREPARED_DIR / "masks"

OUTPUT_DIR   = PROJECT_ROOT / "Unet_run" / "test_visual_samples"

SPLIT        = "test"      # "test" یا "val" یا "train"
N_SAMPLES    = 10          # تعداد نمونه‌هایی که می‌خواهی تست شود
IMG_SIZE     = 256
THRESHOLD    = 0.50
SEED         = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# اگر تصویر اصلی قبل از برش در مسیر خاصی داری، اینجا بگذار.
# اگر نداری، برنامه خودش تلاش می‌کند با نام مشابه پیدا کند.
ORIGINAL_IMAGE_DIRS = [
    PROJECT_ROOT / "original_images",
    PROJECT_ROOT / "prepared" / "original_images",
    PROJECT_ROOT / "prepared" / "full_images",
    PROJECT_ROOT / "images",
    PROJECT_ROOT / "data",
]


# =============================================================================
# UTILS
# =============================================================================
def load_unet_class():
    """
    فایل 2_Training_unet2.py با عدد شروع می‌شود و مستقیم import نمی‌شود.
    برای همین با importlib کلاس UNet را از داخل آن لود می‌کنیم.
    """
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

    print("=" * 80)
    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"Device: {DEVICE}")
    if "epoch" in ckpt:
        print(f"Best epoch: {ckpt['epoch']}")
    if "val" in ckpt:
        print(f"Validation metrics: {ckpt['val']}")
    print("=" * 80)

    return model


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


def find_original_image(crop_path):
    """
    تلاش برای پیدا کردن تصویر اصلی قبل از crop.
    اگر نام فایل crop با نام تصویر اصلی یکی باشد یا base مشترک داشته باشد، پیدا می‌شود.
    """
    stem = crop_path.stem
    suffixes = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    candidates = []

    for root in ORIGINAL_IMAGE_DIRS:
        if not root.exists():
            continue

        # exact same file name
        for ext in suffixes:
            candidates.append(root / f"{stem}{ext}")

        # recursive approximate search
        for ext in suffixes:
            candidates.extend(root.rglob(f"*{stem}*{ext}"))

        # اگر اسم crop شامل _crop یا چیز مشابه باشد، base قبل از آن را هم امتحان کن
        for token in ["_crop", "-crop", "_roi", "-roi", "_patch", "-patch"]:
            if token in stem:
                base_stem = stem.split(token)[0]
                for ext in suffixes:
                    candidates.append(root / f"{base_stem}{ext}")
                    candidates.extend(root.rglob(f"*{base_stem}*{ext}"))

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    return None


def make_placeholder(text, size=(256, 256)):
    img = Image.new("RGB", size, color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    msg = text
    draw.multiline_text((15, size[1] // 2 - 25), msg, fill=(20, 20, 20), spacing=6)
    return img


def overlay_single_mask(image_pil, mask, color=(255, 0, 0), alpha=0.45):
    """
    mask روی تصویر.
    color: RGB
    """
    img = np.asarray(image_pil).astype(np.float32)
    out = img.copy()

    color_arr = np.array(color, dtype=np.float32)
    m = mask.astype(bool)

    out[m] = (1 - alpha) * out[m] + alpha * color_arr
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


def overlay_both_masks(image_pil, gt_mask, pred_mask, alpha=0.50):
    """
    هر دو ماسک روی تصویر:
    سبز = ماسک واقعی فقط
    قرمز = ماسک شبکه فقط
    زرد = همپوشانی ماسک واقعی و شبکه
    """
    img = np.asarray(image_pil).astype(np.float32)
    out = img.copy()

    gt = gt_mask.astype(bool)
    pr = pred_mask.astype(bool)

    both = gt & pr
    gt_only = gt & (~pr)
    pr_only = pr & (~gt)

    green = np.array([0, 255, 0], dtype=np.float32)
    red   = np.array([255, 0, 0], dtype=np.float32)
    yellow = np.array([255, 255, 0], dtype=np.float32)

    out[gt_only] = (1 - alpha) * out[gt_only] + alpha * green
    out[pr_only] = (1 - alpha) * out[pr_only] + alpha * red
    out[both]    = (1 - alpha) * out[both]    + alpha * yellow

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def dice_iou(gt, pred, eps=1e-7):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return dice, iou


def save_visual_report(
    out_path,
    original_pil,
    crop_pil,
    gt_mask,
    pred_mask,
    prob_mask,
    sample_name,
    dice,
    iou
):
    gt_mask_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))

    gt_overlay = overlay_single_mask(crop_pil, gt_mask, color=(0, 255, 0), alpha=0.45)
    pred_overlay = overlay_single_mask(crop_pil, pred_mask, color=(255, 0, 0), alpha=0.45)
    both_overlay = overlay_both_masks(crop_pil, gt_mask, pred_mask, alpha=0.50)

    fig = plt.figure(figsize=(18, 9))

    fig.suptitle(
        f"{sample_name} | Dice={dice:.4f} | IoU={iou:.4f} | Threshold={THRESHOLD}",
        fontsize=14
    )

    items = [
        ("1) Original before crop", original_pil),
        ("2) Cropped image / U-Net input", crop_pil),
        ("3) Ground-truth mask", gt_mask_img),
        ("4) Network predicted mask", pred_mask_img),
        ("5) GT mask on crop - Green", gt_overlay),
        ("6) Pred mask on crop - Red", pred_overlay),
        ("7) Both masks on crop\nGreen=GT | Red=Pred | Yellow=Overlap", both_overlay),
        ("8) Probability map", prob_mask),
    ]

    for i, (title, obj) in enumerate(items, start=1):
        ax = plt.subplot(2, 4, i)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

        if isinstance(obj, np.ndarray):
            ax.imshow(obj, cmap="jet", vmin=0, vmax=1)
        else:
            if obj.mode == "L":
                ax.imshow(obj, cmap="gray")
            else:
                ax.imshow(obj)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_dir = IMAGE_DIR / SPLIT
    mask_dir = MASK_DIR / SPLIT

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_paths = sorted([
        p for p in img_dir.glob("*.png")
        if (mask_dir / p.name).exists()
    ])

    if not image_paths:
        raise RuntimeError(f"No image/mask pairs found in: {img_dir}")

    random.shuffle(image_paths)
    image_paths = image_paths[:min(N_SAMPLES, len(image_paths))]

    model = load_model()

    print(f"Testing split: {SPLIT}")
    print(f"Found pairs: {len(image_paths)}")
    print(f"Output dir: {OUTPUT_DIR}")

    results = []

    for idx, crop_path in enumerate(image_paths, start=1):
        mask_path = mask_dir / crop_path.name

        crop_pil = read_rgb(crop_path, size=IMG_SIZE)
        gt_mask = read_mask(mask_path, size=IMG_SIZE)

        prob_mask, pred_mask = predict_mask(model, crop_pil)

        dsc, iou = dice_iou(gt_mask, pred_mask)

        original_path = find_original_image(crop_path)
        if original_path is not None:
            original_pil = read_rgb(original_path, size=IMG_SIZE)
            original_status = str(original_path)
        else:
            original_pil = make_placeholder(
                "Original image\nbefore crop\nNOT FOUND",
                size=(IMG_SIZE, IMG_SIZE)
            )
            original_status = "NOT FOUND"

        out_name = f"{idx:03d}_{crop_path.stem}_visual.png"
        out_path = OUTPUT_DIR / out_name

        save_visual_report(
            out_path=out_path,
            original_pil=original_pil,
            crop_pil=crop_pil,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            prob_mask=prob_mask,
            sample_name=crop_path.name,
            dice=dsc,
            iou=iou
        )

        results.append({
            "sample": crop_path.name,
            "dice": dsc,
            "iou": iou,
            "crop": str(crop_path),
            "mask": str(mask_path),
            "original": original_status,
            "visual": str(out_path),
        })

        print(f"[{idx:03d}] {crop_path.name} | Dice={dsc:.4f} | IoU={iou:.4f}")
        print(f"      saved -> {out_path}")

    # ذخیره گزارش CSV
    csv_path = OUTPUT_DIR / "visual_test_report.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample,dice,iou,crop,mask,original,visual\n")
        for r in results:
            line = (
                f"{r['sample']},"
                f"{r['dice']:.6f},"
                f"{r['iou']:.6f},"
                f"\"{r['crop']}\","
                f"\"{r['mask']}\","
                f"\"{r['original']}\","
                f"\"{r['visual']}\"\n"
            )
            f.write(line)

    print("=" * 80)
    print("Done.")
    print(f"Visual samples saved in: {OUTPUT_DIR}")
    print(f"CSV report saved: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()