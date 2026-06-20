"""
==============================================================================
 1_data_reader2.py  --  Dataset preparation for OPG segmentation / localization
==============================================================================

This is the full-featured successor of `1_data_reader.py`.  It keeps **every**
capability of the original script and adds three things the project now needs:

  1. PAIR LISTING / REPORT
     Recursively scans `images/` (including every sub-folder) and lists, as
     image<->json pairs:
        * number of (image, json) PAIRS,
        * number of IMAGES,
        * number of JSON/labelme labels.
     The table is printed and also written to `pairs_report.{txt,csv,json}`.

  2. image1/  --  FULL-IMAGE dataset  (UNet + YOLO training style)
        image1/
        ├─ images/<stem>.png            full RGB image (disk OR base64 rebuilt)
        ├─ json/<stem>.json             normalized labelme json (self-contained)
        ├─ unet/
        │  ├─ images/{train,val,test}/<stem>.png   (RGB, resized IMG_SIZE)
        │  └─ masks/{train,val,test}/<stem>.png    (binary 0/255, IMG_SIZE)
        ├─ yolo/
        │  ├─ images/{train,val,test}/<stem>.png   (RGB, original resolution)
        │  ├─ labels/{train,val,test}/<stem>.txt   ("0 cx cy w h", normalized)
        │  └─ data.yaml
        ├─ samples/                     side-by-side visual checks
        └─ manifest.json

  3. image2/  --  CROPPED-TO-BOX dataset  (so UNet can run on the found box)
     Each image is cropped to the bounding box around its annotation(s) (with a
     small padding margin) and the labelme JSON is UPDATED: polygon points are
     shifted into the crop's coordinate frame, imageHeight/Width are rewritten
     and the crop is re-embedded as base64.  Same unet/ + yolo/ + samples/ +
     manifest.json layout as image1/.
        image2/
        ├─ images/<stem>.png            cropped RGB image
        ├─ json/<stem>.json             updated labelme json (shifted polygons)
        ├─ unet/ ...   yolo/ ...   samples/ ...   manifest.json

Backward-compatible extras (unchanged from the original):
  * `prepared/`  is still produced by `prepare()`.
  * `CONFIG`, `prepare()`, `load_manifest()`, and all labelme helpers remain
    importable by the other modules.

Note on `.txt` labels: in 1-8/5555/70 the `.txt` files ARE labelme JSON, but in
31-33 they are patient-metadata INI files.  `safe_load_labelme()` parses JSON
only, so the INI files are skipped gracefully (no annotation -> not trainable).

Run directly to (re)build everything:
    python 1_data_reader2.py
==============================================================================
"""

import io
import csv
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
    PREPARED_DIR = PROJECT_ROOT / "prepared"        # original generated datasets
    IMAGE1_DIR   = PROJECT_ROOT / "image1"          # full-image dataset
    IMAGE2_DIR   = PROJECT_ROOT / "image2"          # cropped-to-box dataset

    IMG_EXTS   = (".png", ".jpg", ".jpeg")
    LABEL_EXTS = (".json", ".txt")                  # both may hold labelme JSON

    IMG_SIZE   = 256        # UNet input (square)
    SEED       = 42
    TRAIN_FRAC = 0.8
    VAL_FRAC   = 0.1        # test = remainder

    NUM_SAMPLES = 12        # side-by-side visual examples per dataset

    YOLO_CLASS_NAME = "tooth"   # single foreground class for detection

    # --- image2 cropping -----------------------------------------------------
    CROP_PAD_FRAC = 0.08    # padding around the box, as a fraction of box size
    CROP_PAD_MIN  = 8       # ...but at least this many pixels on every side
    EMBED_CROP_B64 = True   # re-embed the crop as base64 in the updated json


# =============================================================================
# logging
# =============================================================================
def get_logger():
    logger = logging.getLogger("data_reader2")
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
    """Parse a labelme file (.json or .txt). Returns dict or None on failure.

    Non-JSON `.txt` files (e.g. the patient-metadata INI files in 31-33) fail to
    parse and yield None, so they are skipped by every caller automatically.
    """
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "shapes" not in data:
            return None
        return data
    except Exception:
        return None


def is_labelme(label_path: Path) -> bool:
    """True iff the file parses as a labelme JSON document."""
    return safe_load_labelme(label_path) is not None


def label_for_image(img_path: Path):
    """Return the sibling labelme file for an image (prefer .json), or None."""
    for ext in CONFIG.LABEL_EXTS:
        cand = img_path.with_suffix(ext)
        if cand.exists() and is_labelme(cand):
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


def union_pixel_bbox(data, height, width, pad_frac, pad_min):
    """Union bounding box (pixels) of every polygon, padded and clamped.

    Returns (x0, y0, x1, y1) as ints, or None if there is no usable polygon.
    """
    xs, ys = [], []
    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        xs += [float(p[0]) for p in pts]
        ys += [float(p[1]) for p in pts]
    if not xs:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x = max(pad_min, (x1 - x0) * pad_frac)
    pad_y = max(pad_min, (y1 - y0) * pad_frac)
    x0 = int(max(0, np.floor(x0 - pad_x)))
    y0 = int(max(0, np.floor(y0 - pad_y)))
    x1 = int(min(width,  np.ceil(x1 + pad_x)))
    y1 = int(min(height, np.ceil(y1 + pad_y)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def encode_image_b64(img_rgb, fmt="PNG"):
    """PIL RGB image -> base64 string suitable for labelme `imageData`."""
    buf = io.BytesIO()
    img_rgb.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_labelme_doc(src_data, points_shapes, height, width,
                     image_path, image_b64=None):
    """Build a fresh, normalized labelme document.

    `points_shapes` is the (possibly shifted) list of shape dicts to store.
    """
    return {
        "version": src_data.get("version", "5.5.0"),
        "flags": src_data.get("flags", {}),
        "shapes": points_shapes,
        "imagePath": image_path,
        "imageData": image_b64,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


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

    # (a) image files that have a labelme sibling label
    img_paths = []
    for ext in CONFIG.IMG_EXTS:
        img_paths += glob(str(CONFIG.IMAGES_DIR / "**" / f"*{ext}"), recursive=True)
        img_paths += glob(str(CONFIG.IMAGES_DIR / "**" / f"*{ext.upper()}"), recursive=True)
    have_labels = set()
    for p in sorted(set(img_paths)):
        ip = Path(p)
        lbl = label_for_image(ip)
        if lbl is None:
            continue                      # image without a labelme label -> skip
        have_labels.add(lbl)
        records.append({"stem": _unique_stem(ip, seen), "image": str(ip), "label": str(lbl)})

    # (b) labelme files whose image is missing but that embed base64 imageData
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
            continue                      # not labelme / no pixels -> skip
        records.append({"stem": _unique_stem(lp, seen), "image": None, "label": str(lp)})

    log.info("found %d labelled records under %s", len(records), CONFIG.IMAGES_DIR)
    return records


# =============================================================================
# PAIR LISTING / REPORT
# =============================================================================
def list_pairs():
    """Recursively pair every image with its labelme json across all subdirs.

    Returns a dict with per-directory rows and global totals. A "pair" is an
    image that has a labelme sibling (.json or labelme-.txt) of the same stem.
    Records reconstructed purely from base64 (no image on disk) are reported
    separately as `embedded_only`.
    """
    base = CONFIG.IMAGES_DIR

    # gather every image / label path
    img_paths, lbl_paths = [], []
    for ext in CONFIG.IMG_EXTS:
        img_paths += glob(str(base / "**" / f"*{ext}"), recursive=True)
        img_paths += glob(str(base / "**" / f"*{ext.upper()}"), recursive=True)
    for ext in CONFIG.LABEL_EXTS:
        lbl_paths += glob(str(base / "**" / f"*{ext}"), recursive=True)
    imgs = sorted({Path(p) for p in img_paths})
    jsons = sorted({Path(p) for p in lbl_paths if Path(p).suffix.lower() == ".json"})
    labelme_labels = sorted({Path(p) for p in lbl_paths if is_labelme(Path(p))})

    # per-directory aggregation
    dirs = sorted({p.parent for p in imgs} | {p.parent for p in labelme_labels})
    rows, pair_list = [], []
    for d in dirs:
        d_imgs   = [p for p in imgs if p.parent == d]
        d_jsons  = [p for p in jsons if p.parent == d]
        d_txt    = [p for p in d.glob("*.txt")]
        d_lblme  = [p for p in labelme_labels if p.parent == d]
        d_pairs  = []
        for ip in d_imgs:
            sib = label_for_image(ip)
            if sib is not None:
                d_pairs.append((ip, sib))
                pair_list.append((str(ip), str(sib)))
        rows.append({
            "dir": str(d.relative_to(base)) if d != base else ".",
            "images": len(d_imgs),
            "json": len(d_jsons),
            "txt": len(d_txt),
            "labelme_labels": len(d_lblme),
            "pairs": len(d_pairs),
        })

    # embedded-only (labelme with base64, no sibling image on disk)
    embedded_only = []
    seen_stems = set()
    for lp in labelme_labels:
        if any(lp.with_suffix(e).exists() for e in CONFIG.IMG_EXTS):
            continue
        data = safe_load_labelme(lp)
        if not data or not data.get("imageData"):
            continue
        key = str(lp.with_suffix(""))
        if key in seen_stems:
            continue                       # json/txt duplicate -> count once
        seen_stems.add(key)
        embedded_only.append(str(lp))

    totals = {
        "images": len(imgs),
        "json_files": len(jsons),
        "labelme_label_files": len(labelme_labels),
        "image_json_pairs": len(pair_list),
        "embedded_only_records": len(embedded_only),
        "total_trainable_records": len(pair_list) + len(embedded_only),
    }
    return {"rows": rows, "totals": totals,
            "pairs": pair_list, "embedded_only": embedded_only}


def report_pairs(write_files=True):
    """Print the pair table + totals and (optionally) persist the report."""
    rep = list_pairs()

    log.info("=" * 70)
    log.info("PAIR LISTING  (recursive over %s)", CONFIG.IMAGES_DIR)
    log.info("=" * 70)
    header = f"{'directory':28} {'imgs':>5} {'json':>5} {'txt':>4} {'labelme':>8} {'pairs':>6}"
    log.info(header)
    log.info("-" * 70)
    for r in rep["rows"]:
        log.info(f"{r['dir']:28} {r['images']:5} {r['json']:5} {r['txt']:4} "
                 f"{r['labelme_labels']:8} {r['pairs']:6}")
    log.info("-" * 70)
    t = rep["totals"]
    log.info("TOTAL images .............. %d", t["images"])
    log.info("TOTAL .json files ......... %d", t["json_files"])
    log.info("TOTAL labelme labels ...... %d  (.json + labelme .txt)", t["labelme_label_files"])
    log.info("image+json PAIRS .......... %d", t["image_json_pairs"])
    log.info("embedded-only (base64) .... %d", t["embedded_only_records"])
    log.info("TOTAL trainable records ... %d", t["total_trainable_records"])
    log.info("=" * 70)

    if write_files:
        out = CONFIG.PROJECT_ROOT
        with open(out / "pairs_report.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        # human-readable txt
        with open(out / "pairs_report.txt", "w", encoding="utf-8") as f:
            f.write(header + "\n" + "-" * 70 + "\n")
            for r in rep["rows"]:
                f.write(f"{r['dir']:28} {r['images']:5} {r['json']:5} {r['txt']:4} "
                        f"{r['labelme_labels']:8} {r['pairs']:6}\n")
            f.write("-" * 70 + "\n")
            for k, v in t.items():
                f.write(f"{k:28}: {v}\n")
        # the actual pair list as CSV
        with open(out / "pairs_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "label_json"])
            for ip, lp in rep["pairs"]:
                w.writerow([ip, lp])
        log.info("wrote pairs_report.{json,txt,csv} to %s", out)
    return rep


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


def _generate_samples_for(root: Path, records_by_split, n):
    """Write up to `n` side-by-side examples (from train) into root/samples/."""
    out_dir = root / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = root / "unet" / "images" / "train"
    msk_dir = root / "unet" / "masks" / "train"
    stems = [r["stem"] for r in records_by_split.get("train", [])][:n]
    written = 0
    for stem in stems:
        ip, mp = img_dir / f"{stem}.png", msk_dir / f"{stem}.png"
        if not ip.exists() or not mp.exists():
            continue
        img = Image.open(ip).convert("RGB")
        mask = np.asarray(Image.open(mp).convert("L")) > 127
        _side_by_side(img, mask).save(out_dir / f"sample_{stem}.png")
        written += 1
    log.info("wrote %d sample visualizations to %s", written, out_dir)


# =============================================================================
# materialization helpers
# =============================================================================
def _mk(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _split_records(records):
    """Reproducible 80/10/10 split (shared by all builders)."""
    recs = list(records)
    random.Random(CONFIG.SEED).shuffle(recs)
    n = len(recs)
    n_tr = int(n * CONFIG.TRAIN_FRAC)
    n_va = int(n * CONFIG.VAL_FRAC)
    return {"train": recs[:n_tr],
            "val":   recs[n_tr:n_tr + n_va],
            "test":  recs[n_tr + n_va:]}


def _write_yolo_yaml(yolo_root: Path):
    with open(yolo_root / "data.yaml", "w") as f:
        f.write(f"path: {yolo_root.as_posix()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("nc: 1\n")
        f.write(f"names: ['{CONFIG.YOLO_CLASS_NAME}']\n")
    return yolo_root / "data.yaml"


def _materialize_unet_yolo(root: Path, splits, get_payload):
    """Build unet/ + yolo/ trees for a dataset.

    `get_payload(rec)` -> (PIL_image_rgb, labelme_data, H, W) or None to skip.
    Returns the manifest dict.
    """
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
            payload = get_payload(rec)
            if payload is None:
                continue
            img, data, h, w = payload
            mask = polygons_to_mask(data, h, w)
            boxes = polygons_to_boxes(data, h, w)
            if mask.sum() == 0:
                log.warning("%s has an empty mask (check label)", rec["stem"])
            stem = rec["stem"]

            # --- UNet copies: resized image + mask ---------------------------
            img_u = img.resize((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), Image.BILINEAR)
            msk_u = Image.fromarray(mask * 255).resize(
                (CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), Image.NEAREST)
            img_u.save(root / "unet" / "images" / split / f"{stem}.png")
            msk_u.save(root / "unet" / "masks" / split / f"{stem}.png")

            # --- YOLO copies: full/native-resolution image + bbox label ------
            img.save(root / "yolo" / "images" / split / f"{stem}.png")
            with open(root / "yolo" / "labels" / split / f"{stem}.txt", "w") as f:
                for (cx, cy, bw, bh) in boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            kept.append({"stem": stem, "height": h, "width": w,
                         "num_boxes": len(boxes)})
        manifest["splits"][split] = kept
        log.info("%-5s : %d samples materialized", split, len(kept))

    _write_yolo_yaml(root / "yolo")
    return manifest


# =============================================================================
# original behaviour:  prepared/   (kept for backward compatibility)
# =============================================================================
def prepare():
    """Build masks/boxes, split, and write the prepared/unet + prepared/yolo."""
    random.seed(CONFIG.SEED)
    np.random.seed(CONFIG.SEED)

    records = scan_dataset()
    if not records:
        raise RuntimeError("No labelled samples found -- check CONFIG.IMAGES_DIR")

    splits = _split_records(records)
    root = CONFIG.PREPARED_DIR

    def payload(rec):
        data = safe_load_labelme(Path(rec["label"]))
        if data is None:
            return None
        try:
            img = resolve_image(rec)
            h, w = image_size(rec, data)
        except Exception as e:
            log.warning("skip %s (%s)", rec["stem"], e)
            return None
        return img, data, h, w

    manifest = _materialize_unet_yolo(root, splits, payload)
    # enrich manifest with source provenance
    src = {r["stem"]: r for r in records}
    for split in manifest["splits"]:
        for item in manifest["splits"][split]:
            r = src.get(item["stem"], {})
            item["source_image"] = r.get("image")
            item["source_label"] = r.get("label")

    with open(root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _generate_samples_for(root, manifest["splits"], CONFIG.NUM_SAMPLES)

    log.info("prepared dataset written to %s", root)
    return manifest


def load_manifest():
    """Load prepared/manifest.json, building the dataset first if it is absent."""
    mpath = CONFIG.PREPARED_DIR / "manifest.json"
    if not mpath.exists():
        return prepare()
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# image1/  --  FULL-IMAGE dataset (images + json mirror + unet/yolo)
# =============================================================================
def build_image1():
    """Materialize image1/: full images + normalized labelme json + unet/yolo."""
    random.seed(CONFIG.SEED)
    np.random.seed(CONFIG.SEED)

    records = scan_dataset()
    if not records:
        raise RuntimeError("No labelled samples found -- check CONFIG.IMAGES_DIR")

    root = CONFIG.IMAGE1_DIR
    _mk(root / "images", root / "json")

    # ---- mirror full images + self-contained labelme json -------------------
    # cache resolved image + data per stem so we don't decode base64 twice
    cache = {}
    for rec in records:
        data = safe_load_labelme(Path(rec["label"]))
        if data is None:
            continue
        try:
            img = resolve_image(rec)
            h, w = image_size(rec, data)
        except Exception as e:
            log.warning("skip %s (%s)", rec["stem"], e)
            continue
        stem = rec["stem"]
        img_name = f"{stem}.png"
        img.save(root / "images" / img_name)

        doc = make_labelme_doc(
            data, data.get("shapes", []), h, w,
            image_path=f"../images/{img_name}",
            image_b64=encode_image_b64(img) if CONFIG.EMBED_CROP_B64 else None,
        )
        with open(root / "json" / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        cache[stem] = (img, data, h, w)

    log.info("image1: mirrored %d images + json", len(cache))

    # ---- unet/ + yolo/ training trees ---------------------------------------
    valid = [r for r in records if r["stem"] in cache]
    splits = _split_records(valid)

    def payload(rec):
        return cache.get(rec["stem"])

    manifest = _materialize_unet_yolo(root, splits, payload)
    src = {r["stem"]: r for r in records}
    for split in manifest["splits"]:
        for item in manifest["splits"][split]:
            r = src.get(item["stem"], {})
            item["source_image"] = r.get("image")
            item["source_label"] = r.get("label")
    with open(root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _generate_samples_for(root, manifest["splits"], CONFIG.NUM_SAMPLES)

    log.info("image1 dataset written to %s", root)
    return manifest


# =============================================================================
# image2/  --  CROPPED-TO-BOX dataset (images + updated json + unet/yolo)
# =============================================================================
def _crop_record(img_rgb, data, h, w):
    """Crop `img` to the padded union bbox of its polygons; shift the json.

    Returns (cropped_img, updated_data, ch, cw, bbox) or None if no usable box.
    """
    bbox = union_pixel_bbox(data, h, w, CONFIG.CROP_PAD_FRAC, CONFIG.CROP_PAD_MIN)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    crop = img_rgb.crop((x0, y0, x1, y1))
    cw, ch = crop.width, crop.height

    new_shapes = []
    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        shifted = [[float(px) - x0, float(py) - y0] for px, py in pts]
        ns = dict(shape)
        ns["points"] = shifted
        new_shapes.append(ns)
    if not new_shapes:
        return None
    return crop, new_shapes, ch, cw, bbox


def build_image2():
    """Materialize image2/: bbox-cropped images + updated json + unet/yolo.

    The crop is the box around each image's annotation(s); the labelme polygon
    points are shifted into the crop frame so UNet can train/run on the box.
    """
    random.seed(CONFIG.SEED)
    np.random.seed(CONFIG.SEED)

    records = scan_dataset()
    if not records:
        raise RuntimeError("No labelled samples found -- check CONFIG.IMAGES_DIR")

    root = CONFIG.IMAGE2_DIR
    _mk(root / "images", root / "json")

    cache = {}      # stem -> (crop_img, crop_data, ch, cw)
    for rec in records:
        data = safe_load_labelme(Path(rec["label"]))
        if data is None:
            continue
        try:
            img = resolve_image(rec)
            h, w = image_size(rec, data)
        except Exception as e:
            log.warning("skip %s (%s)", rec["stem"], e)
            continue

        cropped = _crop_record(img, data, h, w)
        if cropped is None:
            log.warning("%s: no usable bbox, skipped from image2", rec["stem"])
            continue
        crop, new_shapes, ch, cw, bbox = cropped

        stem = rec["stem"]
        img_name = f"{stem}.png"
        crop.save(root / "images" / img_name)

        crop_data = make_labelme_doc(
            data, new_shapes, ch, cw,
            image_path=f"../images/{img_name}",
            image_b64=encode_image_b64(crop) if CONFIG.EMBED_CROP_B64 else None,
        )
        crop_data["sourceBBox"] = list(bbox)          # provenance: crop origin
        crop_data["sourceSize"] = [int(h), int(w)]
        with open(root / "json" / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(crop_data, f, indent=2, ensure_ascii=False)

        cache[stem] = (crop, crop_data, ch, cw)

    log.info("image2: cropped %d images + updated json", len(cache))

    valid = [r for r in records if r["stem"] in cache]
    splits = _split_records(valid)

    def payload(rec):
        return cache.get(rec["stem"])

    manifest = _materialize_unet_yolo(root, splits, payload)
    src = {r["stem"]: r for r in records}
    for split in manifest["splits"]:
        for item in manifest["splits"][split]:
            r = src.get(item["stem"], {})
            item["source_image"] = r.get("image")
            item["source_label"] = r.get("label")
    with open(root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _generate_samples_for(root, manifest["splits"], CONFIG.NUM_SAMPLES)

    log.info("image2 dataset written to %s", root)
    return manifest


# =============================================================================
# entry point
# =============================================================================
def build_all():
    report_pairs(write_files=True)
    prepare()        # backward-compatible prepared/
    build_image1()   # full images
    build_image2()   # cropped-to-box images
    log.info("ALL datasets built: prepared/, image1/, image2/")


if __name__ == "__main__":
    build_all()
