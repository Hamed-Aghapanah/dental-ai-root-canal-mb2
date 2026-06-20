"""
==============================================================================
 Training_hybrid.py  --  YOLO (localize) + UNet (segment) hybrid inference
==============================================================================

Pipeline per test image:
  1. Run the trained YOLO detector to find regions of interest (boxes).
  2. Crop each detected region from the ORIGINAL-resolution image.
  3. Resize the crop to the UNet input size and run UNet segmentation.
  4. Paste the predicted crop-mask back into a full-resolution mask canvas
     (union of all ROIs).  Pixels outside every box stay background.
  5. Compare against the ground-truth mask (rebuilt from the labelme polygons
     at original resolution via data_reader helpers).

Outputs (in `Hybrid_run/`):
  * predicted masks            -> Hybrid_run/masks/<stem>.png
  * heatmaps (image|gt|pred)   -> Hybrid_run/heatmaps/<stem>.png
  * per-image + aggregate Dice/IoU/Precision/Recall/F1 -> hybrid_metrics.xlsx

This is an INFERENCE/ASSEMBLY stage: it reuses the two already-trained models
(no new weights are trained).  If a fallback is needed (no YOLO boxes for an
image), the whole image is passed through UNet so coverage never drops to zero.

Requirements: torch, ultralytics, pillow, numpy  (pandas/openpyxl for Excel).

Run (after Training_unet.py and Training_yolo.py):
    python Training_hybrid.py
==============================================================================
"""

import os
# Work around duplicate OpenMP runtimes on Windows/Anaconda (libiomp5md.dll vs
# libomp.dll). Must be set BEFORE numpy / torch / matplotlib are imported, or
# the process aborts with "OMP: Error #15".
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import importlib
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch


def _import_project_module(name):
    """Import `name`, tolerating numbered filenames (e.g. '2_Training_unet.py').

    Tries a normal `import name` first; if that fails, looks for any
    '*<name>.py' file next to this script and loads it via importlib so the
    pipeline works whether files are named 'Training_unet.py' or
    '2_Training_unet.py'."""
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    for cand in sorted(here.glob(f"*{name}.py")):
        spec = importlib.util.spec_from_file_location(name, cand)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError(f"Cannot find '{name}.py' (or '*_{name}.py') in {here}")


# reuse trained UNet architecture + labelme mask rebuild (modular imports)
_tu = _import_project_module("Training_unet")
_dr = _import_project_module("data_reader")
UNet = _tu.UNet
safe_load_labelme = _dr.safe_load_labelme
polygons_to_mask = _dr.polygons_to_mask
image_size = _dr.image_size
resolve_image = _dr.resolve_image


# =============================================================================
# CONFIG
# =============================================================================
class CONFIG:
    PROJECT_ROOT = Path(__file__).resolve().parent
    PREPARED_DIR = PROJECT_ROOT / "prepared"
    UNET_CKPT    = PROJECT_ROOT / "Unet_run" / "unet_best.pt"
    YOLO_WEIGHTS = PROJECT_ROOT / "YOLO_run" / "train" / "weights" / "best.pt"
    RUN_DIR      = PROJECT_ROOT / "Hybrid_run"

    UNET_SIZE = 256          # must match what UNet was trained on
    YOLO_SIZE = 640
    CONF      = 0.25         # YOLO confidence threshold
    BOX_PAD   = 0.10         # expand each box by this fraction before cropping
    DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# model loading
# =============================================================================
def load_unet():
    if not CONFIG.UNET_CKPT.exists():
        raise SystemExit(f"No UNet checkpoint at {CONFIG.UNET_CKPT}. Run Training_unet.py.")
    ckpt = torch.load(CONFIG.UNET_CKPT, map_location=CONFIG.DEVICE)
    # rebuild the exact architecture recorded at train time; pretrained=False
    # since the trained weights are loaded over it anyway (no download needed)
    model = UNet(3, 1, ckpt.get("base", 32), encoder=ckpt.get("encoder", "resnet34"),
                 pretrained=False, attention=ckpt.get("attention", True)).to(CONFIG.DEVICE).eval()
    model.load_state_dict(ckpt["model"])
    return model


def load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics not installed. Run:  pip install ultralytics")
    if not CONFIG.YOLO_WEIGHTS.exists():
        raise SystemExit(f"No YOLO weights at {CONFIG.YOLO_WEIGHTS}. Run Training_yolo.py.")
    return YOLO(str(CONFIG.YOLO_WEIGHTS))


# =============================================================================
# segmentation of a single crop
# =============================================================================
@torch.no_grad()
def segment_crop(unet, img_rgb):
    """Run UNet on a PIL crop -> probability map at the crop's pixel size."""
    w, h = img_rgb.size
    inp = img_rgb.resize((CONFIG.UNET_SIZE, CONFIG.UNET_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(inp, np.float32) / 255.0).permute(2, 0, 1)
    prob = torch.sigmoid(unet(x.unsqueeze(0).to(CONFIG.DEVICE)))[0, 0].cpu().numpy()
    # resize probability back to the crop's native size
    return np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((w, h),
                                                                            Image.BILINEAR)) / 255.0


def pad_box(x1, y1, x2, y2, W, H, frac):
    """Expand a box by `frac` on each side, clamped to image bounds."""
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * frac; x2 += bw * frac; y1 -= bh * frac; y2 += bh * frac
    return (int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2)))


# =============================================================================
# metrics
# =============================================================================
def metrics_from_counts(tp, fp, fn, eps=1e-7):
    prec = tp / (tp + fp + eps); rec = tp / (tp + fn + eps)
    return {"dice": (2*tp)/(2*tp+fp+fn+eps), "iou": tp/(tp+fp+fn+eps),
            "precision": prec, "recall": rec, "f1": (2*prec*rec)/(prec+rec+eps)}


def write_excel(path, sheets):
    """Write {sheet: rows} to xlsx, CSV fallback if pandas/openpyxl missing."""
    try:
        import pandas as pd
        with pd.ExcelWriter(path) as xls:
            for name, rows in sheets.items():
                pd.DataFrame(rows).to_excel(xls, sheet_name=name, index=False)
        print(f"[hybrid] metrics workbook -> {path}")
    except Exception as e:
        print(f"[hybrid] pandas/openpyxl unavailable ({e}); writing CSVs")
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
# main
# =============================================================================
def test_records():
    """Test-split records from the manifest (stem + source label for GT)."""
    manifest = json.loads((CONFIG.PREPARED_DIR / "manifest.json").read_text())
    return manifest["splits"]["test"]


def main():
    (CONFIG.RUN_DIR / "masks").mkdir(parents=True, exist_ok=True)
    (CONFIG.RUN_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)

    unet, yolo = load_unet(), load_yolo()
    records = test_records()
    print(f"[hybrid] device={CONFIG.DEVICE}  test images={len(records)}")

    per_image = []
    agg_tp = agg_fp = agg_fn = 0.0

    for rec in records:
        stem = rec["stem"]
        # original-resolution image (from prepared/yolo or rebuilt from labelme)
        yolo_img = CONFIG.PREPARED_DIR / "yolo" / "images" / "test" / f"{stem}.png"
        if yolo_img.exists():
            image = Image.open(yolo_img).convert("RGB")
        else:
            image = resolve_image({"stem": stem, "image": rec["source_image"],
                                   "label": rec["source_label"]})
        W, H = image.size

        # ---- ground-truth mask at original resolution -----------------------
        data = safe_load_labelme(Path(rec["source_label"]))
        gh, gw = image_size({"stem": stem, "image": rec["source_image"],
                             "label": rec["source_label"]}, data)
        gt = polygons_to_mask(data, gh, gw)
        if (gh, gw) != (H, W):     # robustness: keep GT aligned with the image
            gt = np.asarray(Image.fromarray(gt * 255).resize((W, H), Image.NEAREST)) > 127
        else:
            gt = gt > 0

        # ---- YOLO detection -> ROI crops -> UNet segmentation ---------------
        res = yolo.predict(source=np.asarray(image), conf=CONFIG.CONF,
                           imgsz=CONFIG.YOLO_SIZE, device=CONFIG.DEVICE, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []

        pred_prob = np.zeros((H, W), np.float32)
        if boxes:
            for (x1, y1, x2, y2) in boxes:
                bx1, by1, bx2, by2 = pad_box(x1, y1, x2, y2, W, H, CONFIG.BOX_PAD)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                crop = image.crop((bx1, by1, bx2, by2))
                cp = segment_crop(unet, crop)
                # union: keep the max probability where ROIs overlap
                pred_prob[by1:by2, bx1:bx2] = np.maximum(pred_prob[by1:by2, bx1:bx2], cp)
        else:
            # fallback: no detections -> segment the whole frame so recall != 0
            pred_prob = segment_crop(unet, image)

        pred = pred_prob > 0.5

        # ---- metrics --------------------------------------------------------
        tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum()); fn = float((~pred & gt).sum())
        agg_tp += tp; agg_fp += fp; agg_fn += fn
        m = metrics_from_counts(tp, fp, fn); m = {"stem": stem, "n_boxes": len(boxes), **m}
        per_image.append(m)

        # ---- save predicted mask + heatmap ----------------------------------
        Image.fromarray((pred * 255).astype(np.uint8)).save(CONFIG.RUN_DIR / "masks" / f"{stem}.png")
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].imshow(image); ax[0].set_title("image + YOLO boxes"); ax[0].axis("off")
        for (x1, y1, x2, y2) in boxes:
            ax[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                          edgecolor="lime", linewidth=1.5))
        ax[1].imshow(gt, cmap="gray"); ax[1].set_title("ground truth"); ax[1].axis("off")
        ax[2].imshow(image); ax[2].imshow(pred_prob, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax[2].set_title(f"hybrid pred (Dice={m['dice']:.2f})"); ax[2].axis("off")
        fig.tight_layout(); fig.savefig(CONFIG.RUN_DIR / "heatmaps" / f"{stem}_heatmap.png", dpi=120)
        plt.close(fig)

    # ---- aggregate + Excel --------------------------------------------------
    micro = metrics_from_counts(agg_tp, agg_fp, agg_fn)
    macro = {k: float(np.mean([r[k] for r in per_image])) for k in
             ("dice", "iou", "precision", "recall", "f1")} if per_image else {}
    summary = [{"aggregate": "macro", **macro}, {"aggregate": "micro", **micro}]
    write_excel(CONFIG.RUN_DIR / "hybrid_metrics.xlsx",
                {"summary": summary, "per_image": per_image})
    with open(CONFIG.RUN_DIR / "hybrid_metrics.json", "w") as f:
        json.dump({"micro": micro, "macro": macro, "per_image": per_image}, f, indent=2)

    if per_image:
        print(f"[hybrid] macro Dice={macro['dice']:.4f} IoU={macro['iou']:.4f} "
              f"P={macro['precision']:.4f} R={macro['recall']:.4f} F1={macro['f1']:.4f}")
    print(f"[hybrid] masks    -> {CONFIG.RUN_DIR / 'masks'}")
    print(f"[hybrid] heatmaps -> {CONFIG.RUN_DIR / 'heatmaps'}")


if __name__ == "__main__":
    main()
