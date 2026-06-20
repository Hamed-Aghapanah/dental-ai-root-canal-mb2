# =============================================================================
# 4_test_unet.py
#
# Professional visual testing / reporting tool for the U-Net OPG segmentation
# model produced by 2_Training_unet2.py.
#
# Environment target:
#   * Windows 11, Python 3.10, Anaconda env "yoloxai"
#   * IPython 8 / Spyder / Qt event loop already running
#
# This script REPLACES 4_Test_unet_visual_pro.py and fixes the crash:
#       ImportError: Cannot load backend 'TkAgg' which requires the 'tk'
#       interactive framework, as 'qt' is currently running
#
# Why the crash happened
#   The old script forced  matplotlib.use("TkAgg").  Spyder/IPython already
#   run a Qt event loop, and matplotlib refuses to switch to a *different*
#   interactive GUI backend (Tk) while Qt is live -> ImportError.
#
# What we do instead
#   * matplotlib is forced to the NON-interactive "Agg" backend, which only
#     renders to files (savefig) and never touches any GUI toolkit.  It is
#     safe under Qt, Tk, headless servers, IPython, plain python -- everywhere.
#   * Optional FULL-SCREEN preview of each saved figure is done with OpenCV
#     (cv2) windows, NOT matplotlib.  If OpenCV is missing we fall back to
#     PIL.Image.show(), and if that is unwanted use --no-preview.
#
# Run (default, with full-screen preview if OpenCV is installed):
#       python 4_test_unet.py
#
# Run without any pop-up windows (recommended on a server / for batch):
#       python 4_test_unet.py --no-preview
#
# Spyder / IPython:
#       %runfile D:/project/0000_OPG/lengani/4_test_unet.py --wdir --args --no-preview
# =============================================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# -----------------------------------------------------------------------------
# Matplotlib backend: ALWAYS the safe non-interactive Agg renderer.
# This must happen BEFORE `import matplotlib.pyplot`.
# -----------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")               # safe under Qt / Tk / headless / IPython
import matplotlib.pyplot as plt

import re
import csv
import json
import math
import base64
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
from PIL import Image, ImageDraw

# torch is only needed if a checkpoint is available; import lazily-safe here
try:
    import torch
    _TORCH_OK = True
except Exception as _e:                                    # pragma: no cover
    torch = None
    _TORCH_OK = False
    print(f"[WARN] PyTorch not available ({_e}); segmentation will be skipped.")

# OpenCV is optional (only for full-screen preview)
try:
    import cv2
    _CV2_OK = True
except Exception:
    cv2 = None
    _CV2_OK = False


# =============================================================================
# DEFAULT CONFIG  (everything overridable on the command line)
# =============================================================================
DEFAULT_PROJECT_ROOT = Path(r"D:\project\0000_OPG\lengani")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Log line format, e.g.
# [unet] ep   1/2000  loss=1.8470  val_loss=2.1439  Dice=0.0000  IoU=0.0000  Acc=0.9997  lr=4.00e-04
LOG_RE = re.compile(
    r"\[unet\]\s+ep\s+"
    r"(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\s+"
    r"loss=(?P<loss>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"val_loss=(?P<val_loss>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"Dice=(?P<dice>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"IoU=(?P<iou>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"Acc=(?P<acc>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+"
    r"lr=(?P<lr>nan|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)


# =============================================================================
# SMALL UTILITIES
# =============================================================================
def log(msg):
    print(msg, flush=True)


def safe_float(x):
    """String -> float, tolerant of nan/inf and scientific notation."""
    s = str(x).strip().lower()
    if s in {"nan", "+nan", "-nan"}:
        return np.nan
    if s in {"inf", "+inf", "infinity", "+infinity"}:
        return np.inf
    if s in {"-inf", "-infinity"}:
        return -np.inf
    try:
        return float(s)
    except Exception:
        return np.nan


def normalize_stem(stem):
    """Aggressively normalize a stem to improve fuzzy matching."""
    s = str(stem).lower()
    remove_tokens = [
        "_crop", "-crop", " crop", "_cropped", "-cropped", " cropped",
        "_roi", "-roi", " roi", "_patch", "-patch", " patch",
        "_mask", "-mask", " mask", "_img", "-img", "_image", "-image",
        "_256", "-256", "_unet", "-unet",
    ]
    for token in remove_tokens:
        if token in s:
            s = s.split(token)[0]
    # keep latin letters, digits and persian range only
    s = re.sub(r"[^a-z0-9؀-ۿ]+", "", s)
    return s


def split_crop_stem(crop_stem):
    """
    Crop stems are built by 1_data_reader2.py as  "<parentdir>__<filename>".
    e.g.  "1__amini2_2 - Copy"  ->  folder_hint="1", base="amini2_2 - Copy".
    If there is no "__" we just return (None, crop_stem).
    """
    if "__" in crop_stem:
        folder_hint, base = crop_stem.split("__", 1)
        return folder_hint, base
    return None, crop_stem


def make_placeholder(text, size=(256, 256)):
    img = Image.new("RGB", size, color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.multiline_text((12, size[1] // 2 - 40), text, fill=(20, 20, 20), spacing=6)
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
# FULL-SCREEN PREVIEW (OpenCV first, PIL fallback, or disabled)
# =============================================================================
def preview_fullscreen(image_path, args):
    """
    Show a saved figure full-screen *before* moving on.
    Order of preference:
      1) OpenCV named full-screen window  (cv2)
      2) PIL.Image.show()                 (opens the OS default viewer)
      3) nothing (disabled / unavailable)
    Never raises -- preview problems must not crash the report.
    """
    if args.no_preview:
        return

    if _CV2_OK:
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("cv2.imread returned None")
            win = "U-Net report (press any key / Esc to continue)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            cv2.imshow(win, img)
            cv2.waitKey(int(args.preview_ms))
            cv2.destroyWindow(win)
            try:
                cv2.waitKey(1)        # flush the destroy event on Windows
            except Exception:
                pass
            return
        except Exception as e:
            log(f"[WARN] OpenCV preview failed ({e}); trying PIL.")

    # PIL fallback
    try:
        Image.open(image_path).show()
    except Exception as e:
        log(f"[WARN] PIL preview failed ({e}); continuing without preview.")


# =============================================================================
# LOAD UNET FROM TRAINING SCRIPT  +  CHECKPOINT
# =============================================================================
def load_unet_class(train_script: Path):
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    spec = importlib.util.spec_from_file_location("training_unet_module",
                                                  str(train_script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "UNet"):
        raise AttributeError(f"UNet class not found inside {train_script}")
    return module.UNet


def load_model(args, device):
    """Returns (model, ckpt) or (None, None) if anything is missing."""
    if not _TORCH_OK:
        log("[WARN] torch unavailable -> skipping model load.")
        return None, None
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log(f"[WARN] Checkpoint not found: {ckpt_path}")
        log("       -> original/crop pairs and training curves will still be built.")
        return None, None
    try:
        UNet = load_unet_class(Path(args.train_script))
        ckpt = torch.load(ckpt_path, map_location=device)
        base = ckpt.get("base", 32)
        dropout = ckpt.get("dropout", 0.20)
        model = UNet(in_ch=3, out_ch=1, base=base, drop=dropout).to(device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        log("=" * 90)
        log(f"Loaded checkpoint : {ckpt_path}")
        log(f"Device            : {device}")
        if isinstance(ckpt, dict):
            if "epoch" in ckpt:
                log(f"Checkpoint epoch  : {ckpt['epoch']}")
            if "val" in ckpt:
                log(f"Checkpoint val    : {ckpt['val']}")
        log("=" * 90)
        return model, ckpt
    except Exception as e:
        log(f"[ERROR] Failed to load model ({e}); continuing without segmentation.")
        return None, None


# =============================================================================
# ORIGINAL IMAGE SEARCH (recursive, folder-hint aware)
# =============================================================================
def build_original_index(original_root: Path):
    files = []
    if not original_root.exists():
        log(f"[WARN] Original root not found: {original_root}")
        return files
    log(f"[INFO] Indexing original images recursively under: {original_root}")
    for p in original_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    log(f"[INFO] Original images indexed: {len(files)}")
    return files


def find_original_image(crop_path: Path, original_files):
    """
    Find the pre-crop original for a cropped image.
    Crop stem is "<folder_hint>__<base>" (see 1_data_reader2.py _unique_stem).

    Returns (Path|None, match_status) where match_status is one of:
        exact_stem
        exact_stem_with_folder_hint
        normalized_stem
        fuzzy
        not_found
    """
    crop_stem = crop_path.stem
    folder_hint, base = split_crop_stem(crop_stem)
    base_low = base.lower()
    base_norm = normalize_stem(base)

    # 1) exact base stem AND parent folder == folder_hint  (strongest)
    if folder_hint:
        hits = [p for p in original_files
                if p.stem.lower() == base_low and p.parent.name == folder_hint]
        if hits:
            return sorted(hits, key=lambda x: len(str(x)))[0], "exact_stem_with_folder_hint"

    # 2) exact base stem anywhere
    hits = [p for p in original_files if p.stem.lower() == base_low]
    if hits:
        return sorted(hits, key=lambda x: len(str(x)))[0], "exact_stem"

    # 3) normalized stem match
    if base_norm:
        hits = [p for p in original_files if normalize_stem(p.stem) == base_norm]
        if hits:
            return sorted(hits, key=lambda x: len(str(x)))[0], "normalized_stem"

    # 4) fuzzy best match (conservative threshold)
    best_path, best_score = None, 0.0
    for p in original_files:
        p_norm = normalize_stem(p.stem)
        if not p_norm or not base_norm:
            continue
        score = SequenceMatcher(None, base_norm, p_norm).ratio()
        if score > best_score:
            best_score, best_path = score, p
    if best_path is not None and best_score >= 0.82:
        return best_path, "fuzzy"

    return None, "not_found"


# =============================================================================
# GROUND-TRUTH MASK SEARCH (png in several dirs, else labelme json polygons)
# =============================================================================
def mask_search_dirs(project_root: Path):
    img2 = project_root / "image2"
    return [
        img2 / "unet" / "masks" / "train",
        img2 / "unet" / "masks" / "val",
        img2 / "unet" / "masks" / "test",
        img2 / "masks",
    ]


def json_search_dirs(project_root: Path):
    img2 = project_root / "image2"
    return [img2 / "json"]


def polygons_to_mask_from_json(json_path: Path, out_size):
    """Rasterize every labelme polygon in `json_path` -> binary uint8 mask."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict) or "shapes" not in data:
        return None

    h = int(data.get("imageHeight") or 0)
    w = int(data.get("imageWidth") or 0)
    if h <= 0 or w <= 0:
        # fall back to embedded image size if available
        try:
            raw = base64.b64decode(data["imageData"])
            import io
            im = Image.open(io.BytesIO(raw))
            w, h = im.width, im.height
        except Exception:
            h = w = out_size

    mask = Image.new("L", (w, h), 0)
    drawer = ImageDraw.Draw(mask)
    found = False
    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        drawer.polygon([(float(x), float(y)) for x, y in pts], outline=1, fill=1)
        found = True
    if not found:
        return None
    mask = mask.resize((out_size, out_size), Image.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def find_gt_mask(crop_path: Path, project_root: Path, out_size):
    """
    Find ground-truth mask for a crop. Returns (mask|None, source_str).
    Crop file name matches the mask file name (same stem) in image2/unet/masks.
    """
    stem = crop_path.stem

    # (a) png masks in known dirs
    for d in mask_search_dirs(project_root):
        if not d.exists():
            continue
        cand = d / f"{stem}.png"
        if cand.exists():
            return read_mask(cand, size=out_size), str(cand)
        # try any extension
        for ext in IMAGE_EXTS:
            c = d / f"{stem}{ext}"
            if c.exists():
                return read_mask(c, size=out_size), str(c)

    # (b) labelme json polygons
    for d in json_search_dirs(project_root):
        if not d.exists():
            continue
        cand = d / f"{stem}.json"
        if cand.exists():
            m = polygons_to_mask_from_json(cand, out_size)
            if m is not None:
                return m, f"{cand} (polygons)"

    return None, "NOT FOUND"


# =============================================================================
# PREDICTION + METRICS + OVERLAYS
# =============================================================================
def pil_to_tensor(img_pil, device):
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def predict_mask(model, crop_img_pil, device, threshold):
    with torch.no_grad():
        x = pil_to_tensor(crop_img_pil, device)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    pred = (prob > threshold).astype(np.uint8)
    return prob, pred


def dice_iou(gt, pred, eps=1e-7):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return float(dice), float(iou)


def overlay_single_mask(image_pil, mask, color=(255, 0, 0), alpha=0.45):
    img = np.asarray(image_pil).astype(np.float32)
    out = img.copy()
    m = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * color_arr
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def overlay_both_masks(image_pil, gt_mask, pred_mask, alpha=0.50):
    """Green = GT only, Red = Pred only, Yellow = overlap."""
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
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =============================================================================
# FIGURE BUILDERS
# =============================================================================
def save_pair_figure(out_path, original_pil, crop_pil, crop_name,
                     original_name, match_status):
    """Original-before-crop alongside the cropped image."""
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(
        f"crop file : {crop_name}\n"
        f"original  : {original_name}   |   match = {match_status}",
        fontsize=12,
    )
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("1) Original before crop", fontsize=11)
    ax1.axis("off")
    ax1.imshow(original_pil)

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("2) Cropped image (image2/images)", fontsize=11)
    ax2.axis("off")
    ax2.imshow(crop_pil)

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_full_report(out_path, original_pil, crop_pil, gt_mask, pred_mask,
                     prob_mask, sample_name, dice, iou, threshold,
                     original_name, match_status, mask_source):
    gt_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
    pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    gt_overlay = overlay_single_mask(crop_pil, gt_mask, color=(0, 255, 0), alpha=0.45)
    pred_overlay = overlay_single_mask(crop_pil, pred_mask, color=(255, 0, 0), alpha=0.45)
    both_overlay = overlay_both_masks(crop_pil, gt_mask, pred_mask, alpha=0.50)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        f"{sample_name}  |  Dice={dice:.4f}  |  IoU={iou:.4f}  |  thr={threshold}\n"
        f"original: {original_name} (match={match_status})  |  GT mask: {mask_source}",
        fontsize=13,
    )

    panels = [
        ("1) Original before crop", original_pil, None),
        ("2) Cropped image / U-Net input", crop_pil, None),
        ("3) Ground-truth mask", gt_img, "gray"),
        ("4) Predicted mask", pred_img, "gray"),
        ("5) GT overlay (Green)", gt_overlay, None),
        ("6) Prediction overlay (Red)", pred_overlay, None),
        ("7) Both: Green=GT  Red=Pred  Yellow=Overlap", both_overlay, None),
        ("8) Probability map", prob_mask, "jet"),
    ]
    for i, (title, obj, cmap) in enumerate(panels, start=1):
        ax = plt.subplot(2, 4, i)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if isinstance(obj, np.ndarray):
            ax.imshow(obj, cmap=cmap, vmin=0, vmax=1)
        elif cmap == "gray":
            ax.imshow(obj, cmap="gray")
        else:
            ax.imshow(obj)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =============================================================================
# TRAINING LOG PARSING + CURVES + BEST EPOCH
# =============================================================================
def find_log_file(args):
    if args.log:
        p = Path(args.log)
        if p.exists():
            return p
        log(f"[WARN] --log given but not found: {p}")
    root = Path(args.project_root)
    candidates = [
        root / "log training u net.txt",
        root / "training_log.txt",
        root / "Unet_run" / "training_log.txt",
        root / "Unet_run" / "log training u net.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_training_log(log_path: Path):
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            rows.append({
                "epoch": int(m.group("epoch")),
                "total_epoch": int(m.group("total")),
                "loss": safe_float(m.group("loss")),
                "val_loss": safe_float(m.group("val_loss")),
                "dice": safe_float(m.group("dice")),
                "iou": safe_float(m.group("iou")),
                "acc": safe_float(m.group("acc")),
                "lr": safe_float(m.group("lr")),
            })
    return rows


def save_log_csv(rows, out_csv):
    fields = ["epoch", "total_epoch", "loss", "val_loss", "dice", "iou", "acc", "lr"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def best_epoch_by_dice(rows):
    """Best = max Dice, tie-break max IoU, tie-break min val_loss."""
    valid = [r for r in rows if np.isfinite(r["dice"])]
    if not valid:
        return None
    return max(valid, key=lambda r: (
        r["dice"],
        r["iou"] if np.isfinite(r["iou"]) else -np.inf,
        -(r["val_loss"] if np.isfinite(r["val_loss"]) else np.inf),
    ))


def best_epoch_by_val_loss(rows):
    valid = [r for r in rows if np.isfinite(r["val_loss"])]
    if not valid:
        return None
    return min(valid, key=lambda r: r["val_loss"])


def _plot_xy(out_path, ep, series, title, ylabel, mark_best=None):
    fig = plt.figure(figsize=(12, 7))
    ax = plt.subplot(1, 1, 1)
    for label, y in series:
        ax.plot(ep, y, label=label, linewidth=1.4)
    if mark_best is not None:
        bx, by, txt = mark_best
        ax.axvline(bx, linestyle="--", alpha=0.6, color="k")
        ax.scatter([bx], [by], zorder=5)
        ax.text(bx, by, f"  {txt}", fontsize=9, va="center")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_training_report(args, out_dir, ckpt):
    """Parse the log, write csv + 3 curve figures + best-epoch summaries."""
    log_path = find_log_file(args)
    if log_path is None:
        log("[WARN] No training log file found; skipping training curves.")
        log(f"       Expected e.g.: {Path(args.project_root) / 'log training u net.txt'}")
        return []

    log("=" * 90)
    log(f"[INFO] Parsing training log: {log_path}")
    rows = parse_training_log(log_path)
    if not rows:
        log("[WARN] No epoch lines parsed from the log.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    ep = np.array([r["epoch"] for r in rows], dtype=float)

    # --- CSV ---
    save_log_csv(rows, out_dir / "parsed_training_log.csv")

    # --- best epochs ---
    best_d = best_epoch_by_dice(rows)
    best_v = best_epoch_by_val_loss(rows)

    # --- metrics curves (Dice, IoU, Accuracy) ---
    _plot_xy(
        out_dir / "metrics_curves.png", ep,
        [("val Dice", [r["dice"] for r in rows]),
         ("val IoU",  [r["iou"] for r in rows]),
         ("val Accuracy", [r["acc"] for r in rows])],
        "Validation metrics (Dice / IoU / Accuracy)", "Metric",
        mark_best=(best_d["epoch"], best_d["dice"],
                   f"best Dice={best_d['dice']:.4f} @ ep {best_d['epoch']}")
        if best_d else None,
    )

    # --- losses curves ---
    _plot_xy(
        out_dir / "losses_curves.png", ep,
        [("train loss", [r["loss"] for r in rows]),
         ("val loss",   [r["val_loss"] for r in rows])],
        "Loss curves (train / val)", "Loss",
        mark_best=(best_v["epoch"], best_v["val_loss"],
                   f"min val_loss={best_v['val_loss']:.4f} @ ep {best_v['epoch']}")
        if best_v else None,
    )

    # --- learning-rate curve ---
    _plot_xy(
        out_dir / "learning_rate_curve.png", ep,
        [("learning rate", [r["lr"] for r in rows])],
        "Learning-rate schedule", "LR",
    )

    # --- best-epoch summaries ---
    summary = {
        "log_path": str(log_path),
        "total_parsed_rows": len(rows),
        "total_epoch": rows[0]["total_epoch"] if rows else None,
        "best_by_dice": best_d,
        "best_by_val_loss": best_v,
    }
    with open(out_dir / "best_epoch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    with open(out_dir / "best_epoch_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"log_path                 : {log_path}\n")
        f.write(f"total_parsed_rows        : {len(rows)}\n")
        if best_d:
            f.write("\n[BEST by Dice  (Dice -> IoU -> val_loss)]\n")
            f.write(f"  epoch     : {best_d['epoch']} / {best_d['total_epoch']}\n")
            f.write(f"  Dice      : {best_d['dice']:.6f}\n")
            f.write(f"  IoU       : {best_d['iou']:.6f}\n")
            f.write(f"  val_loss  : {best_d['val_loss']:.6f}\n")
            f.write(f"  Acc       : {best_d['acc']:.6f}\n")
            f.write(f"  loss      : {best_d['loss']:.6f}\n")
            f.write(f"  lr        : {best_d['lr']:.3e}\n")
        if best_v:
            f.write("\n[BEST by val_loss (minimum)]\n")
            f.write(f"  epoch     : {best_v['epoch']} / {best_v['total_epoch']}\n")
            f.write(f"  val_loss  : {best_v['val_loss']:.6f}\n")
            f.write(f"  Dice      : {best_v['dice']:.6f}\n")
            f.write(f"  IoU       : {best_v['iou']:.6f}\n")

    # --- compare checkpoint epoch with best-by-dice epoch ---
    write_best_weight_report(out_dir, ckpt, best_d, best_v)

    log(f"[INFO] metrics_curves.png / losses_curves.png / learning_rate_curve.png saved")
    log(f"[INFO] parsed_training_log.csv + best_epoch_summary.{{txt,json}} saved")
    if best_d:
        log(f"[INFO] BEST by Dice: ep {best_d['epoch']} Dice={best_d['dice']:.4f} "
            f"IoU={best_d['iou']:.4f} val_loss={best_d['val_loss']:.4f}")
    if best_v:
        log(f"[INFO] BEST by val_loss: ep {best_v['epoch']} val_loss={best_v['val_loss']:.4f}")
    log("=" * 90)
    return rows


def write_best_weight_report(out_dir, ckpt, best_d, best_v):
    lines = []
    lines.append("BEST WEIGHT REPORT")
    lines.append("=" * 60)
    ckpt_epoch = None
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        ckpt_epoch = ckpt["epoch"]
        lines.append(f"checkpoint epoch (from .pt) : {ckpt_epoch}")
        if isinstance(ckpt.get("val"), dict):
            lines.append(f"checkpoint val metrics      : {ckpt['val']}")
    else:
        lines.append("checkpoint epoch (from .pt) : NOT AVAILABLE")

    if best_d:
        lines.append(f"best epoch from log (Dice)  : {best_d['epoch']} "
                     f"(Dice={best_d['dice']:.4f})")
    if best_v:
        lines.append(f"best epoch from log (vloss) : {best_v['epoch']} "
                     f"(val_loss={best_v['val_loss']:.4f})")

    lines.append("-" * 60)
    if ckpt_epoch is not None and best_d is not None:
        if int(ckpt_epoch) == int(best_d["epoch"]):
            lines.append("MATCH: checkpoint epoch == best-by-Dice epoch in log.")
        else:
            lines.append("MISMATCH: checkpoint epoch != best-by-Dice epoch in log.")
            lines.append(f"  checkpoint epoch = {ckpt_epoch}, "
                         f"log best-by-Dice epoch = {best_d['epoch']}")
    else:
        lines.append("Cannot compare: missing checkpoint epoch and/or log best epoch.")

    with open(out_dir / "best_weight_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# CROP IMAGE COLLECTION
# =============================================================================
def collect_crops(project_root: Path):
    crop_dir = project_root / "image2" / "images"
    if not crop_dir.exists():
        raise FileNotFoundError(f"Cropped images dir not found: {crop_dir}")
    crops = sorted(p for p in crop_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return crops


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Professional visual test/report for U-Net OPG segmentation.")
    ap.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    ap.add_argument("--checkpoint", default=None,
                    help="Path to unet_best.pt (default: <root>/Unet_run/unet_best.pt)")
    ap.add_argument("--train-script", default=None,
                    help="Path to 2_Training_unet2.py (holds the UNet class)")
    ap.add_argument("--log", default=None,
                    help="Path to training log txt (default: <root>/log training u net.txt)")
    ap.add_argument("--samples", type=int, default=20,
                    help="Number of crop samples to render full reports for")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--no-preview", action="store_true",
                    help="Disable full-screen preview pop-ups (recommended for batch).")
    ap.add_argument("--preview-ms", type=int, default=800,
                    help="Milliseconds to show each full-screen preview (0 = wait key).")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA exists.")
    ap.add_argument("--only-curves", action="store_true",
                    help="Only build training curves from the log.")
    ap.add_argument("--only-pairs", action="store_true",
                    help="Only build original/crop pair figures.")
    return ap


def resolve_args(args):
    root = Path(args.project_root)
    if args.checkpoint is None:
        args.checkpoint = str(root / "Unet_run" / "unet_best.pt")
    if args.train_script is None:
        args.train_script = str(root / "2_Training_unet2.py")
    return args


def main():
    args = resolve_args(build_argparser().parse_args())
    root = Path(args.project_root)

    # device
    if _TORCH_OK and not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # output layout
    out_root = root / "Unet_run" / "professional_report_image2"
    pairs_dir = out_root / "original_crop_pairs"
    samples_dir = out_root / "samples_full_report"
    log_report_dir = out_root / "training_log_report"
    out_root.mkdir(parents=True, exist_ok=True)

    log("#" * 90)
    log("# 4_test_unet.py  --  professional U-Net report")
    log(f"# project root : {root}")
    log(f"# output root  : {out_root}")
    log(f"# device       : {device}")
    log(f"# preview      : {'OFF' if args.no_preview else ('OpenCV' if _CV2_OK else 'PIL fallback')}")
    log(f"# started      : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("#" * 90)

    # ---- (A) training curves from log (independent of model/data) -----------
    if not args.only_pairs:
        # ckpt loaded once below; but training report only needs ckpt epoch, so
        # peek the checkpoint cheaply here if it exists
        ckpt_for_report = None
        if _TORCH_OK and Path(args.checkpoint).exists():
            try:
                ckpt_for_report = torch.load(args.checkpoint, map_location="cpu")
            except Exception as e:
                log(f"[WARN] Could not peek checkpoint for report ({e}).")
        make_training_report(args, log_report_dir, ckpt_for_report)

    if args.only_curves:
        log("[DONE] --only-curves: training curves built; nothing else requested.")
        return

    # ---- collect crops + original index -------------------------------------
    try:
        crops = collect_crops(root)
    except Exception as e:
        log(f"[ERROR] {e}")
        return
    if not crops:
        log("[ERROR] No cropped images found in image2/images.")
        return

    # limit to N samples (deterministic = first N sorted, reproducible)
    selected = crops[:max(0, args.samples)] if args.samples > 0 else crops
    original_files = build_original_index(root / "images")

    pairs_dir.mkdir(parents=True, exist_ok=True)

    # ---- (B) original/crop pairs --------------------------------------------
    log("=" * 90)
    log(f"[INFO] Building original/crop pairs for {len(selected)} samples.")
    for idx, crop_path in enumerate(selected, start=1):
        try:
            crop_pil = read_rgb(crop_path, size=args.img_size)
        except Exception as e:
            log(f"[WARN] cannot read crop {crop_path.name} ({e}); skipped.")
            continue

        original_path, match_status = find_original_image(crop_path, original_files)
        if original_path is not None:
            try:
                original_pil = read_rgb(original_path, size=args.img_size)
            except Exception as e:
                original_pil = make_placeholder(f"Original unreadable\n{e}",
                                                (args.img_size, args.img_size))
            original_name = original_path.name
        else:
            original_pil = make_placeholder("Original before crop\nNOT FOUND",
                                            (args.img_size, args.img_size))
            original_name = "NOT FOUND"

        out_path = pairs_dir / f"{idx:03d}_{crop_path.stem}_pair.png"
        save_pair_figure(out_path, original_pil, crop_pil, crop_path.name,
                         original_name, match_status)
        log(f"[{idx:03d}] pair  {crop_path.name}  <-  {original_name}  ({match_status})")
        preview_fullscreen(out_path, args)

    if args.only_pairs:
        log("[DONE] --only-pairs: original/crop pairs built.")
        return

    # ---- (C) full segmentation reports --------------------------------------
    model, ckpt = load_model(args, device)
    if model is None:
        log("[INFO] No model available -> skipping segmentation full reports.")
        log("[DONE] original/crop pairs + training curves are complete.")
        return

    samples_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_root / "sample_visual_metrics.csv"
    results = []

    log("=" * 90)
    log(f"[INFO] Building full segmentation reports for {len(selected)} samples.")
    for idx, crop_path in enumerate(selected, start=1):
        try:
            crop_pil = read_rgb(crop_path, size=args.img_size)
        except Exception as e:
            log(f"[WARN] cannot read crop {crop_path.name} ({e}); skipped.")
            continue

        # original
        original_path, match_status = find_original_image(crop_path, original_files)
        if original_path is not None:
            try:
                original_pil = read_rgb(original_path, size=args.img_size)
            except Exception:
                original_pil = make_placeholder("Original unreadable",
                                                (args.img_size, args.img_size))
            original_name = original_path.name
        else:
            original_pil = make_placeholder("Original before crop\nNOT FOUND",
                                            (args.img_size, args.img_size))
            original_name = "NOT FOUND"

        # prediction
        try:
            prob_mask, pred_mask = predict_mask(model, crop_pil, device, args.threshold)
        except Exception as e:
            log(f"[WARN] prediction failed for {crop_path.name} ({e}); skipped.")
            continue

        # ground truth
        gt_mask, mask_source = find_gt_mask(crop_path, root, args.img_size)
        if gt_mask is None:
            gt_mask = np.zeros((args.img_size, args.img_size), np.uint8)
            dsc, iou = float("nan"), float("nan")
            mask_source = "NOT FOUND (metrics N/A)"
        else:
            dsc, iou = dice_iou(gt_mask, pred_mask)

        out_path = samples_dir / f"{idx:03d}_{crop_path.stem}_report.png"
        save_full_report(out_path, original_pil, crop_pil, gt_mask, pred_mask,
                         prob_mask, crop_path.name, dsc, iou, args.threshold,
                         original_name, match_status, mask_source)

        results.append({
            "index": idx,
            "sample": crop_path.name,
            "dice": dsc,
            "iou": iou,
            "threshold": args.threshold,
            "crop_path": str(crop_path),
            "original_path": str(original_path) if original_path else "NOT FOUND",
            "original_match_status": match_status,
            "gt_mask_source": mask_source,
            "report_path": str(out_path),
        })
        log(f"[{idx:03d}] report {crop_path.name}  Dice={dsc:.4f}  IoU={iou:.4f}  "
            f"orig={match_status}  gt={'ok' if 'NOT FOUND' not in mask_source else 'missing'}")
        preview_fullscreen(out_path, args)

    # ---- metrics csv ---------------------------------------------------------
    if results:
        fields = ["index", "sample", "dice", "iou", "threshold", "crop_path",
                  "original_path", "original_match_status", "gt_mask_source",
                  "report_path"]
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        valid = [r["dice"] for r in results if np.isfinite(r["dice"])]
        if valid:
            log(f"[INFO] mean Dice over {len(valid)} scored samples = {np.mean(valid):.4f}")
        log(f"[INFO] sample_visual_metrics.csv -> {metrics_csv}")

    log("=" * 90)
    log("[DONE] All reports generated.")
    log(f"       pairs   : {pairs_dir}")
    log(f"       samples : {samples_dir}")
    log(f"       curves  : {log_report_dir}")
    log("=" * 90)


if __name__ == "__main__":
    main()
