"""
==============================================================================
 Training_yolo.py  --  Train a YOLO detector (localizer) on the prepared data
==============================================================================

Consumes `prepared/yolo/` (images + bbox labels + data.yaml) from data_reader.py.
Bounding boxes were already derived from the polygon masks by data_reader.
Uses Ultralytics YOLOv8.  All outputs go under `YOLO_run/`.

Requirements:
    pip install ultralytics

Run:
    python data_reader.py          # once, to build prepared/
    python Training_yolo.py
==============================================================================
"""

import os
# Work around duplicate OpenMP runtimes on Windows/Anaconda (libiomp5md.dll vs
# libomp.dll). Must be set BEFORE torch / ultralytics are imported, or the
# process aborts with "OMP: Error #15". See http://openmp.llvm.org/
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import torch


# =============================================================================
# CONFIG
# =============================================================================
class CONFIG:
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_YAML    = PROJECT_ROOT / "prepared" / "yolo" / "data.yaml"
    RUN_DIR      = PROJECT_ROOT / "YOLO_run"

    MODEL    = "yolov8n.pt"     # base weights (n/s/m/l/x). 'n' = fastest.
    EPOCHS   = 300
    IMG_SIZE = 640             # YOLO works best at 640; images are upscaled as needed
    BATCH    = 16
    SEED     = 42
    PATIENCE = 20              # early-stopping patience
    DEVICE   = 0 if torch.cuda.is_available() else "cpu"


def main():
    # ultralytics is heavy; import inside main so the file is importable without it
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics not installed. Run:  pip install ultralytics")

    if not CONFIG.DATA_YAML.exists():
        raise SystemExit(f"{CONFIG.DATA_YAML} missing. Run data_reader.py first.")

    CONFIG.RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[yolo] device={CONFIG.DEVICE} model={CONFIG.MODEL} epochs={CONFIG.EPOCHS}")

    model = YOLO(CONFIG.MODEL)

    # ---- train --------------------------------------------------------------
    # Ultralytics writes runs to <project>/<name>/ with weights, curves (results.png),
    # confusion matrix, PR curves, and metrics automatically.
    model.train(
        data=str(CONFIG.DATA_YAML),
        epochs=CONFIG.EPOCHS,
        imgsz=CONFIG.IMG_SIZE,
        batch=CONFIG.BATCH,
        seed=CONFIG.SEED,
        patience=CONFIG.PATIENCE,
        device=CONFIG.DEVICE,
        project=str(CONFIG.RUN_DIR),
        name="train",
        exist_ok=True,
    )

    # ---- validate (writes metrics + PR/confusion plots) ---------------------
    metrics = model.val(
        data=str(CONFIG.DATA_YAML),
        imgsz=CONFIG.IMG_SIZE,
        device=CONFIG.DEVICE,
        project=str(CONFIG.RUN_DIR),
        name="val",
        exist_ok=True,
    )
    print(f"[yolo] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    print(f"[yolo] best weights -> {CONFIG.RUN_DIR / 'train' / 'weights' / 'best.pt'}")

    export_excel(model, metrics)


def export_excel(model, metrics):
    """Log per-epoch curves (from results.csv) + per-class summary to .xlsx."""
    # --- summary: mAP / Precision / Recall / F1, overall + per class ----------
    p, r = float(metrics.box.mp), float(metrics.box.mr)
    f1 = (2 * p * r) / (p + r + 1e-7)
    summary = [{"class": "all", "precision": round(p, 6), "recall": round(r, 6),
                "f1": round(f1, 6), "mAP50": round(float(metrics.box.map50), 6),
                "mAP50-95": round(float(metrics.box.map), 6)}]
    try:
        names = model.names
        for i, ci in enumerate(metrics.box.ap_class_index):
            cp, cr = float(metrics.box.p[i]), float(metrics.box.r[i])
            cf1 = (2 * cp * cr) / (cp + cr + 1e-7)
            summary.append({"class": names[int(ci)], "precision": round(cp, 6),
                            "recall": round(cr, 6), "f1": round(cf1, 6),
                            "mAP50": round(float(metrics.box.ap50[i]), 6),
                            "mAP50-95": round(float(metrics.box.ap[i]), 6)})
    except Exception:
        pass  # per-class breakdown is best-effort

    # --- per-epoch curves: parse ultralytics results.csv ---------------------
    per_epoch = []
    results_csv = CONFIG.RUN_DIR / "train" / "results.csv"
    try:
        import pandas as pd
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            per_epoch = df.to_dict("records")
        with pd.ExcelWriter(CONFIG.RUN_DIR / "yolo_metrics.xlsx") as xls:
            pd.DataFrame(summary).to_excel(xls, sheet_name="summary", index=False)
            if per_epoch:
                pd.DataFrame(per_epoch).to_excel(xls, sheet_name="per_epoch", index=False)
        print(f"[yolo] metrics workbook -> {CONFIG.RUN_DIR / 'yolo_metrics.xlsx'}")
    except Exception as e:
        print(f"[yolo] pandas/openpyxl unavailable ({e}); summary kept in JSON")
        with open(CONFIG.RUN_DIR / "yolo_metrics.json", "w") as f:
            json.dump({"summary": summary}, f, indent=2)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
