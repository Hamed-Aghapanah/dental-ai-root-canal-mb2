"""
==============================================================================
 data_heat_map_test_train_unet.py  --  UNet inference, heatmaps & metrics
==============================================================================

* Loads the trained UNet from `Unet_run/unet_best.pt`.
* Optionally fine-tunes a few epochs on train+val (--finetune).
* Runs predictions on the test set.
* Saves probability heatmaps in `Unet_run/heatmaps/`.
* Writes per-image metrics (Dice, IoU, Precision, Recall, F1) and an aggregate
  report as CSV + JSON in `Unet_run/`.

Run:
    python data_heat_map_test_train_unet.py
    python data_heat_map_test_train_unet.py --finetune --epochs 5
==============================================================================
"""

import os
# Work around duplicate OpenMP runtimes on Windows/Anaconda (libiomp5md.dll vs
# libomp.dll). Must be set BEFORE numpy / torch / matplotlib are imported, or
# the process aborts with "OMP: Error #15".
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import argparse
import importlib
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader


def _import_project_module(name):
    """Import `name`, tolerating numbered filenames (e.g. '2_Training_unet.py')."""
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


# reuse the model + dataset + losses defined for training (modular import)
_tu = _import_project_module("Training_unet")
UNet, UNetDataset, ComboLoss, TRAIN_CFG = _tu.UNet, _tu.UNetDataset, _tu.ComboLoss, _tu.CONFIG


# =============================================================================
# CONFIG
# =============================================================================
class CONFIG:
    PROJECT_ROOT = Path(__file__).resolve().parent
    RUN_DIR      = PROJECT_ROOT / "Unet_run"
    HEATMAP_DIR  = RUN_DIR / "heatmaps"
    CKPT         = RUN_DIR / "unet_best.pt"

    FINETUNE_EPOCHS = 5
    FINETUNE_LR     = 1e-4
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    if not CONFIG.CKPT.exists():
        raise SystemExit(f"No checkpoint at {CONFIG.CKPT}. Train with Training_unet.py first.")
    ckpt = torch.load(CONFIG.CKPT, map_location=CONFIG.DEVICE)
    model = UNet(3, 1, ckpt.get("base", 32), encoder=ckpt.get("encoder", "resnet34"),
                 pretrained=False, attention=ckpt.get("attention", True)).to(CONFIG.DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"[unet-test] loaded {CONFIG.CKPT} (val_dice={ckpt.get('val_dice')})")
    return model


def metrics_from_counts(tp, fp, fn, eps=1e-7):
    dice = (2*tp) / (2*tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = (2*prec*rec) / (prec + rec + eps)
    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec, "f1": f1}


def mask_contour(mask_bool, thickness=2):
    """Boundary of a binary mask (mask AND NOT erosion), thickened -> bool array."""
    m = mask_bool
    er = m.copy()
    er[1:, :] &= m[:-1, :]; er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]; er[:, :-1] &= m[:, 1:]
    out = m & ~er
    for _ in range(max(0, thickness - 1)):
        e = out; out = e.copy()
        out[1:, :] |= e[:-1, :]; out[:-1, :] |= e[1:, :]
        out[:, 1:] |= e[:, :-1]; out[:, :-1] |= e[:, 1:]
    return out


def error_rgb(pred, gt):
    """RGB map: TP=green, FP=red (over-segment), FN=blue (missed)."""
    h, w = pred.shape
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[pred & gt] = (0.0, 1.0, 0.0)
    rgb[pred & ~gt] = (1.0, 0.0, 0.0)
    rgb[~pred & gt] = (0.0, 0.3, 1.0)
    return rgb


def finetune(model, epochs):
    """Optional light fine-tuning on train+val before testing."""
    crit = ComboLoss()
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG.FINETUNE_LR)
    loaders = [DataLoader(UNetDataset(s, augment=(s == "train")), batch_size=TRAIN_CFG.BATCH_SIZE,
                          shuffle=True, num_workers=TRAIN_CFG.NUM_WORKERS) for s in ("train", "val")]
    model.train()
    for ep in range(1, epochs + 1):
        run = n = 0
        for dl in loaders:
            for x, y, _ in dl:
                x, y = x.to(CONFIG.DEVICE), y.to(CONFIG.DEVICE)
                opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
                run += loss.item() * x.size(0); n += x.size(0)
        print(f"[finetune] epoch {ep}/{epochs}  loss={run/max(1,n):.4f}")
    torch.save({"model": model.state_dict(), "base": TRAIN_CFG.BASE_CHANNELS,
                "encoder": TRAIN_CFG.ENCODER, "attention": TRAIN_CFG.ATTENTION,
                "img_size": TRAIN_CFG.IMG_SIZE}, CONFIG.RUN_DIR / "unet_finetuned.pt")


@torch.no_grad()
def run_test(model):
    CONFIG.HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    ds = UNetDataset("test", augment=False)
    per_image = []
    agg_tp = agg_fp = agg_fn = 0.0
    model.eval()

    for i in range(len(ds)):
        x, y, stem = ds[i]
        logits = model(x.unsqueeze(0).to(CONFIG.DEVICE))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = prob > 0.5
        gt = y[0].numpy() > 0.5

        tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum()); fn = float((~pred & gt).sum())
        agg_tp += tp; agg_fp += fp; agg_fn += fn
        m = metrics_from_counts(tp, fp, fn); m["stem"] = stem
        per_image.append(m)

        # ---- rich heatmap figure --------------------------------------------
        #  [ image | GT(green)+pred(red) contours | prob heatmap+cbar |
        #    heatmap overlay | TP/FP/FN error map ]
        img_np = x.permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))

        ax[0].imshow(img_np); ax[0].set_title("image"); ax[0].axis("off")

        contour_vis = img_np.copy()
        contour_vis[mask_contour(gt)] = (0, 1, 0)        # ground truth = green
        contour_vis[mask_contour(pred)] = (1, 0, 0)      # prediction   = red
        ax[1].imshow(contour_vis); ax[1].axis("off")
        ax[1].set_title("contours  GT=green  pred=red")

        hm = ax[2].imshow(prob, cmap="turbo", vmin=0, vmax=1)
        ax[2].set_title("probability"); ax[2].axis("off")
        fig.colorbar(hm, ax=ax[2], fraction=0.046, pad=0.02)

        ax[3].imshow(img_np)
        ax[3].imshow(np.ma.masked_less(prob, 0.05), cmap="turbo", alpha=0.55, vmin=0, vmax=1)
        ax[3].set_title("overlay"); ax[3].axis("off")

        ax[4].imshow(error_rgb(pred, gt))
        ax[4].set_title("TP=g  FP=r  FN=b"); ax[4].axis("off")

        fig.suptitle(f"{stem}   Dice={m['dice']:.3f}  IoU={m['iou']:.3f}  "
                     f"P={m['precision']:.3f}  R={m['recall']:.3f}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(CONFIG.HEATMAP_DIR / f"{stem}_heatmap.png", dpi=130)
        plt.close(fig)

    micro = metrics_from_counts(agg_tp, agg_fp, agg_fn)
    macro = {k: float(np.mean([r[k] for r in per_image])) for k in
             ("dice", "iou", "precision", "recall", "f1")}
    report = {"num_test_images": len(per_image), "micro_average": micro,
              "macro_average": macro, "per_image": per_image}

    with open(CONFIG.RUN_DIR / "test_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(CONFIG.RUN_DIR / "test_metrics.csv", "w") as f:
        f.write("stem,dice,iou,precision,recall,f1\n")
        for r in per_image:
            f.write(f"{r['stem']},{r['dice']:.4f},{r['iou']:.4f},"
                    f"{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")
        f.write(f"MACRO,{macro['dice']:.4f},{macro['iou']:.4f},{macro['precision']:.4f},"
                f"{macro['recall']:.4f},{macro['f1']:.4f}\n")
        f.write(f"MICRO,{micro['dice']:.4f},{micro['iou']:.4f},{micro['precision']:.4f},"
                f"{micro['recall']:.4f},{micro['f1']:.4f}\n")

    print(f"[unet-test] images={len(per_image)}  "
          f"macro Dice={macro['dice']:.4f} IoU={macro['iou']:.4f} "
          f"P={macro['precision']:.4f} R={macro['recall']:.4f}")
    print(f"[unet-test] heatmaps -> {CONFIG.HEATMAP_DIR}")
    print(f"[unet-test] reports  -> {CONFIG.RUN_DIR / 'test_metrics.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", action="store_true", help="fine-tune on train+val first")
    ap.add_argument("--epochs", type=int, default=CONFIG.FINETUNE_EPOCHS)
    args = ap.parse_args()

    model = load_model()
    if args.finetune:
        finetune(model, args.epochs)
    run_test(model)


if __name__ == "__main__":
    main()
