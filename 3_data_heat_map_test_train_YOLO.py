"""
==============================================================================
 data_heat_map_test_train_YOLO.py  --  YOLO inference, localization heatmaps
==============================================================================

* Loads the trained YOLO detector from `YOLO_run/train/weights/best.pt`.
* Optionally resumes/fine-tunes on the dataset (--finetune).
* Runs predictions on the test split (`prepared/yolo/images/test`).
* Saves localization heatmaps (confidence-weighted box accumulation) in
  `YOLO_run/heatmaps/`.
* Writes:
    - dataset-level Precision / Recall / mAP via `model.val()` (test split),
    - per-image Precision / Recall (IoU>=0.5 matching vs GT boxes),
  as CSV + JSON in `YOLO_run/`.

Requirements:
    pip install ultralytics

Run:
    python data_heat_map_test_train_YOLO.py
    python data_heat_map_test_train_YOLO.py --finetune --epochs 20
==============================================================================
"""

import os
# Work around duplicate OpenMP runtimes on Windows/Anaconda (libiomp5md.dll vs
# libomp.dll). Must be set BEFORE numpy / torch / matplotlib are imported, or
# the process aborts with "OMP: Error #15".
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch


# =============================================================================
# CONFIG
# =============================================================================
class CONFIG:
    PROJECT_ROOT = Path(__file__).resolve().parent
    YOLO_DIR     = PROJECT_ROOT / "prepared" / "yolo"
    DATA_YAML    = YOLO_DIR / "data.yaml"
    RUN_DIR      = PROJECT_ROOT / "YOLO_run"
    WEIGHTS      = RUN_DIR / "train" / "weights" / "best.pt"
    HEATMAP_DIR  = RUN_DIR / "heatmaps"

    IMG_SIZE  = 640
    CONF      = 0.25          # detection confidence threshold
    IOU_MATCH = 0.5           # IoU threshold for per-image TP matching
    DEVICE    = 0 if torch.cuda.is_available() else "cpu"


# =============================================================================
# helpers
# =============================================================================
def load_gt_boxes(label_path, w, h):
    """Read YOLO-format GT (normalized cx,cy,bw,bh) -> pixel [x1,y1,x2,y2]."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
        boxes.append([x1, y1, x2, y2])
    return boxes


def iou(a, b):
    """IoU of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def per_image_pr(pred_boxes, gt_boxes, thr):
    """Greedy IoU matching -> (precision, recall) for one image."""
    matched = set()
    tp = 0
    for pb in pred_boxes:
        best_j, best_iou = -1, thr
        for j, gb in enumerate(gt_boxes):
            if j in matched:
                continue
            v = iou(pb, gb)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            matched.add(best_j); tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    prec = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    return prec, rec, tp, fp, fn


def make_heatmap(image, pred_boxes, confidences, gt_boxes, out_path, title=""):
    """3-panel localization view:
       [ predictions (+conf) | ground truth | confidence heatmap overlay ].
    The heatmap accumulates a confidence-weighted Gaussian blob per detection."""
    w, h = image.size
    heat = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for (x1, y1, x2, y2), c in zip(pred_boxes, confidences):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sx, sy = max((x2 - x1) / 2, 1), max((y2 - y1) / 2, 1)
        heat += c * np.exp(-(((xx - cx) ** 2) / (2 * sx ** 2) +
                             ((yy - cy) ** 2) / (2 * sy ** 2)))
    if heat.max() > 0:
        heat /= heat.max()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # --- predictions with confidence labels ---------------------------------
    ax[0].imshow(image); ax[0].set_title(f"predictions ({len(pred_boxes)})"); ax[0].axis("off")
    for (x1, y1, x2, y2), c in zip(pred_boxes, confidences):
        ax[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                      edgecolor="lime", linewidth=1.6))
        ax[0].text(x1, max(0, y1 - 3), f"{c:.2f}", color="black", fontsize=7,
                   bbox=dict(facecolor="lime", edgecolor="none", pad=0.5, alpha=0.8))

    # --- ground-truth boxes --------------------------------------------------
    ax[1].imshow(image); ax[1].set_title(f"ground truth ({len(gt_boxes)})"); ax[1].axis("off")
    for (x1, y1, x2, y2) in gt_boxes:
        ax[1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                      edgecolor="red", linewidth=1.6))

    # --- confidence heatmap overlay + colorbar -------------------------------
    ax[2].imshow(image)
    hm = ax[2].imshow(np.ma.masked_less(heat, 0.05), cmap="turbo", alpha=0.6, vmin=0, vmax=1)
    ax[2].set_title("confidence heatmap"); ax[2].axis("off")
    fig.colorbar(hm, ax=ax[2], fraction=0.046, pad=0.02)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130); plt.close(fig)


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", action="store_true", help="fine-tune before testing")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics not installed. Run:  pip install ultralytics")

    if not CONFIG.WEIGHTS.exists():
        raise SystemExit(f"No weights at {CONFIG.WEIGHTS}. Train with Training_yolo.py first.")

    CONFIG.HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(CONFIG.WEIGHTS))
    print(f"[yolo-test] loaded {CONFIG.WEIGHTS}")

    # ---- optional fine-tuning ----------------------------------------------
    if args.finetune:
        model.train(data=str(CONFIG.DATA_YAML), epochs=args.epochs, imgsz=CONFIG.IMG_SIZE,
                    device=CONFIG.DEVICE, project=str(CONFIG.RUN_DIR), name="finetune",
                    exist_ok=True)

    # ---- dataset-level metrics (P/R/mAP) on the test split ------------------
    val = model.val(data=str(CONFIG.DATA_YAML), split="test", imgsz=CONFIG.IMG_SIZE,
                    device=CONFIG.DEVICE, project=str(CONFIG.RUN_DIR), name="test_eval",
                    exist_ok=True)
    dataset_metrics = {
        "precision": float(val.box.mp), "recall": float(val.box.mr),
        "mAP50": float(val.box.map50), "mAP50-95": float(val.box.map),
    }

    # ---- per-image predictions, P/R + heatmaps -----------------------------
    test_imgs = sorted((CONFIG.YOLO_DIR / "images" / "test").glob("*.png"))
    label_dir = CONFIG.YOLO_DIR / "labels" / "test"
    per_image = []
    for ip in test_imgs:
        image = Image.open(ip).convert("RGB")
        w, h = image.size
        res = model.predict(source=str(ip), conf=CONFIG.CONF, imgsz=CONFIG.IMG_SIZE,
                            device=CONFIG.DEVICE, verbose=False)[0]
        pred_boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
        confs = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else []
        gt_boxes = load_gt_boxes(label_dir / f"{ip.stem}.txt", w, h)

        prec, rec, tp, fp, fn = per_image_pr(pred_boxes, gt_boxes, CONFIG.IOU_MATCH)
        per_image.append({"stem": ip.stem, "precision": prec, "recall": rec,
                          "tp": tp, "fp": fp, "fn": fn,
                          "n_pred": len(pred_boxes), "n_gt": len(gt_boxes)})
        make_heatmap(image, pred_boxes, confs, gt_boxes,
                     CONFIG.HEATMAP_DIR / f"{ip.stem}_heatmap.png",
                     title=f"{ip.stem}   P={prec:.3f}  R={rec:.3f}  "
                           f"TP={tp} FP={fp} FN={fn}")

    macro = {k: float(np.mean([r[k] for r in per_image])) for k in ("precision", "recall")} \
        if per_image else {"precision": 0.0, "recall": 0.0}

    report = {"dataset_metrics": dataset_metrics,
              "per_image_macro": macro,
              "num_test_images": len(per_image),
              "per_image": per_image}
    with open(CONFIG.RUN_DIR / "test_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(CONFIG.RUN_DIR / "test_metrics.csv", "w") as f:
        f.write("stem,precision,recall,tp,fp,fn,n_pred,n_gt\n")
        for r in per_image:
            f.write(f"{r['stem']},{r['precision']:.4f},{r['recall']:.4f},"
                    f"{r['tp']},{r['fp']},{r['fn']},{r['n_pred']},{r['n_gt']}\n")

    print(f"[yolo-test] dataset: P={dataset_metrics['precision']:.4f} "
          f"R={dataset_metrics['recall']:.4f} mAP50={dataset_metrics['mAP50']:.4f} "
          f"mAP50-95={dataset_metrics['mAP50-95']:.4f}")
    print(f"[yolo-test] heatmaps -> {CONFIG.HEATMAP_DIR}")
    print(f"[yolo-test] reports  -> {CONFIG.RUN_DIR / 'test_metrics.csv'}")


if __name__ == "__main__":
    main()
