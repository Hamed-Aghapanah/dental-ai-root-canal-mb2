"""
==============================================================================
 data_reader.py  --  Dataset preparation for the OPG segmentation/localization
==============================================================================

Responsibilities
----------------
* Recursively scan `images/` for PNG/JPG images.
* Find each image's labelme label file (.json OR .txt -- both hold labelme JSON
  in this dataset).  ~63% of labels have NO image on disk but embed the original
  image as base64 `imageData`; those are reconstructed so nothing is lost.
* Convert polygon annotations to binary masks.
* Derive YOLO bounding boxes from the polygons.
* Split into train/val/test (reproducible) and materialize TWO ready-to-train
  datasets on disk:

    prepared/
    ├─ unet/
    │  ├─ images/{train,val,test}/<stem>.png   (RGB, resized to IMG_SIZE)
    │  └─ masks/{train,val,test}/<stem>.png    (binary 0/255, resized to IMG_SIZE)
    ├─ yolo/
    │  ├─ images/{train,val,test}/<stem>.png   (RGB, original resolution)
    │  ├─ labels/{train,val,test}/<stem>.txt   (YOLO: "0 cx cy w h", normalized)
    │  └─ data.yaml                            (ultralytics dataset descriptor)
    └─ manifest.json                           (split -> records, for traceability)

Other modules import `CONFIG`, `prepare()` and `load_manifest()` from here.

Run directly to (re)build everything:
    python data_reader.py
==============================================================================
"""

import io
import json
import base64
import random
import logging
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# =============================================================================
# CONFIG
# =============================================================================
class CONFIG:
    PROJECT_ROOT = Path(__file__).resolve().parent
    IMAGES_DIR   = PROJECT_ROOT / "images"          # source images + labelme labels
    PREPARED_DIR = PROJECT_ROOT / "prepared"        # all generated datasets

    IMG_EXTS   = (".png", ".jpg", ".jpeg")
    LABEL_EXTS = (".json", ".txt")                  # both are labelme JSON here

    IMG_SIZE   = 256        # UNet input (square)
    SEED       = 42
    TRAIN_FRAC = 0.8
    VAL_FRAC   = 0.1        # test = remainder

    NUM_SAMPLES = 12        # how many side-by-side visual examples to write to samples/

    YOLO_CLASS_NAME = "tooth"   # single foreground class for detection


# =============================================================================
# logging
# =============================================================================
def get_logger():
    logger = logging.getLogger("data_reader")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    return logger

log = get_logger()


# =============================================================================
# labelme helpers  (shared by the other modules via import)
# =============================================================================
def safe_load_labelme(label_path: Path):
    """Parse a labelme file (.json or .txt). Returns dict or None on failure."""
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("could not parse label %s: %s", label_path, e)
        return None


def label_for_image(img_path: Path):
    """Return the sibling label file for an image (prefer .json), or None."""
    for ext in CONFIG.LABEL_EXTS:
        cand = img_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def image_from_labelme(data):
    """Reconstruct a PIL RGB image from labelme base64 `imageData` (or None)."""
    if not data or not data.get("imageData"):
        return None
    raw = base64.b64decode(data["imageData"])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def resolve_image(record):
    """Load the RGB image for a record: from disk, else from embedded base64."""
    if record.get("image"):
        return Image.open(record["image"]).convert("RGB")
    data = safe_load_labelme(Path(record["label"]))
    img = image_from_labelme(data)
    if img is None:
        raise FileNotFoundError(f"no pixels available for {record['stem']}")
    return img


def image_size(record, data):
    """(H, W) for a record: prefer labelme fields, else open the image."""
    h, w = data.get("imageHeight"), data.get("imageWidth")
    if h and w:
        return int(h), int(w)
    img = resolve_image(record)
    return img.height, img.width


def polygons_to_mask(data, height, width):
    """Rasterize every labelme polygon into a binary (H,W) uint8 mask {0,1}."""
    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)
    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue  # need a real polygon
        drawer.polygon([(float(x), float(y)) for x, y in pts], outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def polygons_to_boxes(data, height, width):
    """Return YOLO-normalized boxes [(cx,cy,w,h), ...] from polygon extents."""
    boxes = []
    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x0, x1 = max(0, min(xs)), min(width, max(xs))
        y0, y1 = max(0, min(ys)), min(height, max(ys))
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1 or bh <= 1:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        boxes.append((cx / width, cy / height, bw / width, bh / height))
    return boxes


# =============================================================================
# scanning
# =============================================================================
def _unique_stem(path: Path, seen: set):
    """Collision-free, filesystem-safe stem: <parentdir>__<filename>."""
    base = f"{path.parent.name}__{path.stem}"
    out, i = base, 1
    while out in seen:
        out, i = f"{base}__{i}", i + 1
    seen.add(out)
    return out


def scan_dataset():
    """Collect labelled records: {stem, image(path|None), label(path)}."""
    seen, records = set(), []

    # (a) image files that have a label
    img_paths = []
    for ext in CONFIG.IMG_EXTS:
        img_paths += glob(str(CONFIG.IMAGES_DIR / "**" / f"*{ext}"), recursive=True)
        img_paths += glob(str(CONFIG.IMAGES_DIR / "**" / f"*{ext.upper()}"), recursive=True)
    have_labels = set()
    for p in sorted(set(img_paths)):
        ip = Path(p)
        lbl = label_for_image(ip)
        if lbl is None:
            continue                      # robustness: image without label -> skip
        have_labels.add(lbl)
        records.append({"stem": _unique_stem(ip, seen), "image": str(ip), "label": str(lbl)})

    # (b) label files whose image is missing but that embed base64 imageData
    label_paths = []
    for ext in CONFIG.LABEL_EXTS:
        label_paths += glob(str(CONFIG.IMAGES_DIR / "**" / f"*{ext}"), recursive=True)
    for p in sorted(set(label_paths)):
        lp = Path(p)
        if lp in have_labels:
            continue
        if any(lp.with_suffix(e).exists() for e in CONFIG.IMG_EXTS):
            continue
        data = safe_load_labelme(lp)
        if data is None or not data.get("imageData"):
            continue                      # no pixels -> skip gracefully
        records.append({"stem": _unique_stem(lp, seen), "image": None, "label": str(lp)})

    log.info("found %d labelled records under %s", len(records), CONFIG.IMAGES_DIR)
    return records


# =============================================================================
# sample visualization  (no cv2: contour computed with numpy morphology)
# =============================================================================
def _mask_contour(mask_bool, thickness=2):
    """Boundary pixels of a binary mask via (mask AND NOT erosion), thickened."""
    m = mask_bool
    er = m.copy()
    er[1:, :] &= m[:-1, :]; er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]; er[:, :-1] &= m[:, 1:]
    edge = m & ~er
    # thicken the edge by OR-ing shifted copies
    out = edge.copy()
    for _ in range(max(0, thickness - 1)):
        e = out
        out = e.copy()
        out[1:, :] |= e[:-1, :]; out[:-1, :] |= e[1:, :]
        out[:, 1:] |= e[:, :-1]; out[:, :-1] |= e[:, 1:]
    return out


def _side_by_side(img_rgb, mask_bool):
    """Compose [ raw | binary mask | raw+red contour ] into one RGB image."""
    h, w = mask_bool.shape
    raw = np.asarray(img_rgb.resize((w, h)), np.uint8)
    mask_vis = np.stack([mask_bool.astype(np.uint8) * 255] * 3, axis=-1)
    overlay = raw.copy()
    overlay[_mask_contour(mask_bool)] = (255, 0, 0)      # red contour
    canvas = np.concatenate([raw, mask_vis, overlay], axis=1)
    return Image.fromarray(canvas)


def generate_samples(records_by_split, n):
    """Write up to `n` side-by-side examples (from train) to prepared/samples/."""
    out_dir = CONFIG.PREPARED_DIR / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = CONFIG.PREPARED_DIR / "unet" / "images" / "train"
    msk_dir = CONFIG.PREPARED_DIR / "unet" / "masks" / "train"
    stems = [r["stem"] for r in records_by_split.get("train", [])][:n]
    for stem in stems:
        ip, mp = img_dir / f"{stem}.png", msk_dir / f"{stem}.png"
        if not ip.exists() or not mp.exists():
            continue
        img = Image.open(ip).convert("RGB")
        mask = np.asarray(Image.open(mp).convert("L")) > 127
        _side_by_side(img, mask).save(out_dir / f"sample_{stem}.png")
    log.info("wrote %d sample visualizations to %s", len(stems), out_dir)


# =============================================================================
# materialization
# =============================================================================
def _mk(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def prepare():
    """Build masks/boxes, split, and write the unet/ and yolo/ datasets."""
    random.seed(CONFIG.SEED)
    np.random.seed(CONFIG.SEED)

    records = scan_dataset()
    if not records:
        raise RuntimeError("No labelled samples found -- check CONFIG.IMAGES_DIR")

    # ---- reproducible 80/10/10 split ----------------------------------------
    random.Random(CONFIG.SEED).shuffle(records)
    n = len(records)
    n_tr = int(n * CONFIG.TRAIN_FRAC)
    n_va = int(n * CONFIG.VAL_FRAC)
    splits = {"train": records[:n_tr],
              "val":   records[n_tr:n_tr + n_va],
              "test":  records[n_tr + n_va:]}

    # ---- output tree --------------------------------------------------------
    root = CONFIG.PREPARED_DIR
    for split in splits:
        _mk(root / "unet" / "images" / split,
            root / "unet" / "masks" / split,
            root / "yolo" / "images" / split,
            root / "yolo" / "labels" / split)

    manifest = {"config": {"img_size": CONFIG.IMG_SIZE, "seed": CONFIG.SEED},
                "splits": {}}

    for split, recs in splits.items():
        kept = []
        for rec in recs:
            data = safe_load_labelme(Path(rec["label"]))
            if data is None:
                continue
            try:
                img = resolve_image(rec)
                h, w = image_size(rec, data)
                mask = polygons_to_mask(data, h, w)
                boxes = polygons_to_boxes(data, h, w)
            except Exception as e:
                log.warning("skip %s (%s)", rec["stem"], e)
                continue
            if mask.sum() == 0:
                log.warning("%s has an empty mask (check label)", rec["stem"])

            stem = rec["stem"]
            # --- UNet copies: resized image + mask ---------------------------
            img_u = img.resize((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), Image.BILINEAR)
            msk_u = Image.fromarray(mask * 255).resize(
                (CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), Image.NEAREST)
            img_u.save(root / "unet" / "images" / split / f"{stem}.png")
            msk_u.save(root / "unet" / "masks" / split / f"{stem}.png")

            # --- YOLO copies: original-resolution image + bbox label ---------
            img.save(root / "yolo" / "images" / split / f"{stem}.png")
            with open(root / "yolo" / "labels" / split / f"{stem}.txt", "w") as f:
                for (cx, cy, bw, bh) in boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            kept.append({"stem": stem, "source_image": rec["image"],
                         "source_label": rec["label"],
                         "height": h, "width": w, "num_boxes": len(boxes)})
        manifest["splits"][split] = kept
        log.info("%-5s : %d samples materialized", split, len(kept))

    # ---- YOLO data.yaml -----------------------------------------------------
    yaml_path = root / "yolo" / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {(root / 'yolo').as_posix()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("nc: 1\n")
        f.write(f"names: ['{CONFIG.YOLO_CLASS_NAME}']\n")

    with open(root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ---- sample visualizations for quick inspection -------------------------
    generate_samples(manifest["splits"], CONFIG.NUM_SAMPLES)

    log.info("prepared dataset written to %s", root)
    log.info("YOLO descriptor: %s", yaml_path)
    return manifest


def load_manifest():
    """Load manifest.json, building the dataset first if it is absent."""
    mpath = CONFIG.PREPARED_DIR / "manifest.json"
    if not mpath.exists():
        return prepare()
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    prepare()
