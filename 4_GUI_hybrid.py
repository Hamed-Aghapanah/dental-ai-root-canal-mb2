"""
==============================================================================
 GUI_hybrid.py  --  Desktop GUI: single-image YOLO detection + UNet segmentation
==============================================================================

What it does
------------
* "Open Image..."  : pick any PNG/JPG (an OPG image).
* "1) YOLO Detect" : runs the trained YOLO detector and answers the question
                     "is the object in this image, and WHERE?"  Draws the
                     bounding boxes with confidences on the image.
* "2) UNet Segment": runs the trained UNet on the WHOLE image and overlays
                     the predicted region (red fill + yellow contour).
* "3) Hybrid"      : YOLO localizes -> each padded box is cropped -> UNet
                     segments each crop -> masks are pasted back (same logic
                     as Training_hybrid.py).  Boxes + mask shown together.
* Live "mask threshold" slider re-thresholds the stored probability map
  without re-running the network.  "YOLO conf" / "box pad" apply on the
  next run.
* "Save Result..." : writes annotated image + binary mask + JSON report
                     into  GUI_run/.

Requirements
------------
Trained weights must already exist (run the pipeline first):
    Unet_run/unet_best.pt                  (from Training_unet.py)
    YOLO_run/train/weights/best.pt         (from Training_yolo.py)

Windows / Anaconda notes
------------------------
* Run from a plain terminal / Anaconda Prompt:
      python GUI_hybrid.py
  Do NOT use Spyder's %runfile / IPython magics: Ultralytics +
  multiprocessing need a real __main__ module, otherwise you get the
  `__create_fn__` / multiprocessing SyntaxError.
* KMP_DUPLICATE_LIB_OK is set BELOW, BEFORE numpy/torch are imported, to
  avoid the duplicate-OpenMP abort (libiomp5md.dll vs libomp.dll).
* All inference runs in a background thread so the window never freezes.
==============================================================================
"""

import os
# Must be set BEFORE numpy / torch are imported (OMP: Error #15 workaround).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import queue
import threading
import importlib
import importlib.util
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch


PROJECT_ROOT = Path(__file__).resolve().parent


# =============================================================================
# CONFIG
# =============================================================================
class CONFIG:
    UNET_CKPT    = PROJECT_ROOT / "Unet_run" / "unet_best.pt"
    YOLO_WEIGHTS = PROJECT_ROOT / "YOLO_run" / "train" / "weights" / "best.pt"
    OUT_DIR      = PROJECT_ROOT / "GUI_run"

    UNET_SIZE = 256                    # must match UNet training size
    YOLO_SIZE = 640
    DEFAULT_CONF     = 0.25            # YOLO confidence threshold
    DEFAULT_MASK_THR = 0.50            # probability -> binary mask threshold
    DEFAULT_BOX_PAD  = 0.10            # crop padding fraction (hybrid)

    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"


# =============================================================================
# project-module import that tolerates numbered filenames
# (works whether the file on disk is Training_unet.py OR 2_Training_unet.py)
# =============================================================================
def import_project_module(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    for cand in sorted(PROJECT_ROOT.glob(f"*{name}.py")):
        spec = importlib.util.spec_from_file_location(name, cand)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # register so pickling/imports work
        spec.loader.exec_module(mod)
        return mod
    raise ImportError(
        f"Cannot find '{name}.py' (or '*_{name}.py') next to this script in {PROJECT_ROOT}")


# =============================================================================
# model loading + inference helpers
# =============================================================================
class Models:
    """Lazy holder for the two trained networks."""
    def __init__(self):
        self.unet = None
        self.yolo = None

    def load(self, status_cb=lambda s: None):
        # ---- UNet -----------------------------------------------------------
        status_cb("Loading UNet checkpoint...")
        if not CONFIG.UNET_CKPT.exists():
            raise FileNotFoundError(
                f"No UNet checkpoint at {CONFIG.UNET_CKPT}\nRun Training_unet.py first.")
        tu = import_project_module("Training_unet")
        ckpt = torch.load(CONFIG.UNET_CKPT, map_location=CONFIG.DEVICE)
        unet = tu.UNet(3, 1, ckpt.get("base", 32)).to(CONFIG.DEVICE).eval()
        unet.load_state_dict(ckpt["model"])
        self.unet = unet

        # ---- YOLO -----------------------------------------------------------
        status_cb("Loading YOLO weights...")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("Ultralytics not installed. Run:  pip install ultralytics")
        if not CONFIG.YOLO_WEIGHTS.exists():
            raise FileNotFoundError(
                f"No YOLO weights at {CONFIG.YOLO_WEIGHTS}\nRun Training_yolo.py first.")
        self.yolo = YOLO(str(CONFIG.YOLO_WEIGHTS))
        status_cb(f"Models ready on {CONFIG.DEVICE.upper()}.")


@torch.no_grad()
def unet_prob(unet, pil_rgb):
    """UNet probability map at the input image's native pixel size."""
    w, h = pil_rgb.size
    inp = pil_rgb.resize((CONFIG.UNET_SIZE, CONFIG.UNET_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(inp, np.float32) / 255.0).permute(2, 0, 1)
    prob = torch.sigmoid(unet(x.unsqueeze(0).to(CONFIG.DEVICE)))[0, 0].cpu().numpy()
    back = Image.fromarray((prob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.asarray(back, np.float32) / 255.0


def yolo_detect(yolo, pil_rgb, conf):
    """Run YOLO -> (boxes [[x1,y1,x2,y2],...], confidences [..])."""
    res = yolo.predict(source=np.asarray(pil_rgb), conf=conf,
                       imgsz=CONFIG.YOLO_SIZE, device=CONFIG.YOLO_DEVICE,
                       verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return [], []
    return (res.boxes.xyxy.cpu().numpy().tolist(),
            res.boxes.conf.cpu().numpy().tolist())


def pad_box(x1, y1, x2, y2, W, H, frac):
    """Expand a box by `frac` per side, clamped to the image (as in hybrid)."""
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * frac; x2 += bw * frac; y1 -= bh * frac; y2 += bh * frac
    return (int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2)))


def hybrid_prob(models, pil_rgb, conf, pad_frac):
    """YOLO boxes -> UNet on each padded crop -> full-size probability map.

    Returns (prob HxW float32, boxes, confs, used_fallback)."""
    W, H = pil_rgb.size
    boxes, confs = yolo_detect(models.yolo, pil_rgb, conf)
    prob = np.zeros((H, W), np.float32)
    fallback = False
    if boxes:
        for (x1, y1, x2, y2) in boxes:
            bx1, by1, bx2, by2 = pad_box(x1, y1, x2, y2, W, H, pad_frac)
            if bx2 <= bx1 or by2 <= by1:
                continue
            cp = unet_prob(models.unet, pil_rgb.crop((bx1, by1, bx2, by2)))
            prob[by1:by2, bx1:bx2] = np.maximum(prob[by1:by2, bx1:bx2], cp)
    else:
        # same fallback as Training_hybrid.py: segment the whole frame
        prob = unet_prob(models.unet, pil_rgb)
        fallback = True
    return prob, boxes, confs, fallback


# =============================================================================
# drawing helpers
# =============================================================================
def _mask_contour(m, thickness=2):
    """Boundary of a boolean mask (mask AND NOT erosion), thickened."""
    er = m.copy()
    er[1:, :] &= m[:-1, :]; er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]; er[:, :-1] &= m[:, 1:]
    edge = m & ~er
    out = edge.copy()
    for _ in range(max(0, thickness - 1)):
        e = out.copy()
        out[1:, :] |= e[:-1, :]; out[:-1, :] |= e[1:, :]
        out[:, 1:] |= e[:, :-1]; out[:, :-1] |= e[:, 1:]
    return out


def overlay_mask(pil_rgb, prob, thr, alpha=0.45):
    """Red translucent fill + yellow contour where prob >= thr."""
    arr = np.asarray(pil_rgb, np.uint8).astype(np.float32)
    m = prob >= thr
    if m.any():
        arr[m, 0] = arr[m, 0] * (1 - alpha) + 255 * alpha   # boost red
        arr[m, 1] *= (1 - alpha)
        arr[m, 2] *= (1 - alpha)
        arr = arr.astype(np.uint8)
        arr[_mask_contour(m)] = (255, 255, 0)
        return Image.fromarray(arr)
    return Image.fromarray(arr.astype(np.uint8))


def draw_boxes(pil_rgb, boxes, confs):
    """Lime rectangles + confidence labels (line width scales with image)."""
    img = pil_rgb.copy()
    d = ImageDraw.Draw(img)
    lw = max(2, round(min(img.size) / 250))
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, lw * 6))
    except Exception:
        font = ImageFont.load_default()
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        d.rectangle([x1, y1, x2, y2], outline=(50, 255, 50), width=lw)
        label = f"{c:.2f}"
        ty = y1 - (lw * 8) if y1 - (lw * 8) > 0 else y1 + lw
        d.text((x1 + lw, ty), label, fill=(50, 255, 50), font=font)
    return img


# =============================================================================
# the GUI
# =============================================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("OPG Analyzer  --  YOLO localization + UNet segmentation")
        root.geometry("1180x760")
        root.minsize(900, 560)

        self.models = Models()
        self.models_ready = False
        self.src = None              # original-resolution PIL RGB image
        self.src_path = None
        self.result = None           # dict: mode, boxes, confs, prob, fallback
        self.busy = False
        self.q = queue.Queue()
        self._photo = None           # keep a reference or Tk drops the image
        self._resize_job = None

        self._build_ui()
        self._poll_queue()
        self._run_async(self._task_load_models)   # load nets in the background

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        # left control column ---------------------------------------------------
        left = ttk.Frame(self.root, padding=10)
        left.pack(side="left", fill="y")

        ttk.Button(left, text="Open Image...", command=self.on_open).pack(fill="x")
        ttk.Separator(left).pack(fill="x", pady=8)

        self.btn_yolo = ttk.Button(left, text="1)  YOLO Detect  (where is it?)",
                                   command=lambda: self.on_run("yolo"))
        self.btn_unet = ttk.Button(left, text="2)  UNet Segment  (full image)",
                                   command=lambda: self.on_run("unet"))
        self.btn_hyb = ttk.Button(left, text="3)  Hybrid  YOLO \u2192 UNet",
                                  command=lambda: self.on_run("hybrid"))
        for b in (self.btn_yolo, self.btn_unet, self.btn_hyb):
            b.pack(fill="x", pady=2)

        ttk.Separator(left).pack(fill="x", pady=8)

        self.conf = tk.DoubleVar(value=CONFIG.DEFAULT_CONF)
        self.mask_thr = tk.DoubleVar(value=CONFIG.DEFAULT_MASK_THR)
        self.box_pad = tk.DoubleVar(value=CONFIG.DEFAULT_BOX_PAD)
        self._slider(left, "YOLO confidence (next run)", self.conf, 0.05, 0.90)
        self._slider(left, "Mask threshold (live)", self.mask_thr, 0.10, 0.90,
                     cmd=lambda *_: self._refresh())
        self._slider(left, "Box padding (hybrid, next run)", self.box_pad, 0.0, 0.50)

        ttk.Separator(left).pack(fill="x", pady=8)
        self.show_boxes = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Show YOLO boxes", variable=self.show_boxes,
                        command=self._refresh).pack(anchor="w")
        ttk.Checkbutton(left, text="Show UNet mask overlay", variable=self.show_mask,
                        command=self._refresh).pack(anchor="w")

        ttk.Separator(left).pack(fill="x", pady=8)
        self.btn_save = ttk.Button(left, text="Save Result...", command=self.on_save)
        self.btn_save.pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(left, text="Result").pack(anchor="w")
        self.txt = tk.Text(left, width=38, height=16, wrap="word",
                           state="disabled", font=("Consolas", 9))
        self.txt.pack(fill="both", expand=True)

        # right: image canvas ---------------------------------------------------
        right = ttk.Frame(self.root)
        right.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(right, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # status bar ------------------------------------------------------------
        self.status = tk.StringVar(value="Starting...")
        ttk.Label(self.root, textvariable=self.status, relief="sunken",
                  anchor="w").place(relx=0, rely=1.0, relwidth=1.0, anchor="sw")

        self._set_run_buttons("disabled")

    def _slider(self, parent, label, var, lo, hi, cmd=None):
        row = ttk.Frame(parent); row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text=label, font=("Segoe UI", 8)).pack(anchor="w")
        s = ttk.Scale(row, from_=lo, to=hi, variable=var, command=cmd)
        s.pack(fill="x")
        val = ttk.Label(row, font=("Segoe UI", 8))
        val.pack(anchor="e")
        def upd(*_): val.config(text=f"{var.get():.2f}")
        var.trace_add("write", upd); upd()

    def _set_run_buttons(self, state):
        for b in (self.btn_yolo, self.btn_unet, self.btn_hyb, self.btn_save):
            b.config(state=state)

    # ------------------------------------------------------------- actions --
    def on_open(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self.src = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Open image", f"Could not open:\n{e}")
            return
        self.src_path = Path(path)
        self.result = None
        self.status.set(f"Loaded {self.src_path.name}  ({self.src.width}x{self.src.height})")
        self._set_text("Image loaded. Choose YOLO / UNet / Hybrid.")
        self._refresh()

    def on_run(self, mode):
        if self.src is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        if not self.models_ready or self.busy:
            return
        conf, pad = float(self.conf.get()), float(self.box_pad.get())
        self.status.set(f"Running {mode.upper()} ...")
        self._run_async(self._task_infer, mode, self.src, conf, pad)

    def on_save(self):
        if self.src is None or self.result is None:
            messagebox.showinfo("Nothing to save", "Run an analysis first.")
            return
        CONFIG.OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.src_path.stem if self.src_path else "image"
        mode = self.result["mode"]
        annotated = self._compose()
        ann_p = CONFIG.OUT_DIR / f"{stem}_{mode}_annotated.png"
        annotated.save(ann_p)
        saved = [ann_p]
        if self.result.get("prob") is not None:
            mask = (self.result["prob"] >= float(self.mask_thr.get()))
            mp = CONFIG.OUT_DIR / f"{stem}_{mode}_mask.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(mp)
            saved.append(mp)
        rep = {"image": str(self.src_path), "mode": mode,
               "yolo_conf": float(self.conf.get()),
               "mask_threshold": float(self.mask_thr.get()),
               "box_pad": float(self.box_pad.get()),
               "n_boxes": len(self.result.get("boxes") or []),
               "boxes_xyxy": self.result.get("boxes"),
               "confidences": self.result.get("confs"),
               "found": bool(self.result.get("found")),
               "mask_coverage_pct": self.result.get("coverage")}
        jp = CONFIG.OUT_DIR / f"{stem}_{mode}_report.json"
        jp.write_text(json.dumps(rep, indent=2))
        saved.append(jp)
        self.status.set("Saved: " + ", ".join(p.name for p in saved))
        messagebox.showinfo("Saved", "Saved to GUI_run/:\n" +
                            "\n".join(p.name for p in saved))

    # ----------------------------------------------------- background work --
    def _run_async(self, fn, *args):
        if self.busy:
            return
        self.busy = True
        self._set_run_buttons("disabled")
        def work():
            try:
                self.q.put(("ok", fn(*args)))
            except Exception:
                self.q.put(("err", traceback.format_exc()))
        threading.Thread(target=work, daemon=True).start()

    def _task_load_models(self):
        self.models.load(status_cb=lambda s: self.q.put(("status", s)))
        return {"kind": "models_loaded"}

    def _task_infer(self, mode, image, conf, pad):
        if mode == "yolo":
            boxes, confs = yolo_detect(self.models.yolo, image, conf)
            return {"kind": "result", "mode": mode, "boxes": boxes, "confs": confs,
                    "prob": None, "fallback": False}
        if mode == "unet":
            prob = unet_prob(self.models.unet, image)
            return {"kind": "result", "mode": mode, "boxes": [], "confs": [],
                    "prob": prob, "fallback": False}
        prob, boxes, confs, fb = hybrid_prob(self.models, image, conf, pad)
        return {"kind": "result", "mode": mode, "boxes": boxes, "confs": confs,
                "prob": prob, "fallback": fb}

    def _poll_queue(self):
        try:
            while True:
                tag, payload = self.q.get_nowait()
                if tag == "status":
                    self.status.set(payload)
                elif tag == "err":
                    self.busy = False
                    self._set_run_buttons("normal" if self.models_ready else "disabled")
                    self.status.set("Error -- see message")
                    short = payload.strip().splitlines()[-1]
                    self._set_text("ERROR:\n" + short)
                    messagebox.showerror("Error", payload)
                elif tag == "ok":
                    self.busy = False
                    if payload.get("kind") == "models_loaded":
                        self.models_ready = True
                        self._set_run_buttons("normal")
                        self._set_text("Models loaded.\nOpen an image to begin.")
                    else:
                        self.result = payload
                        self._summarize(payload)
                        self._refresh()
                    self._set_run_buttons("normal")
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    # ----------------------------------------------------------- rendering --
    def _compose(self):
        """Original image + (optional) mask overlay + (optional) boxes."""
        img = self.src
        if img is None:
            return None
        res = self.result
        if res:
            if res.get("prob") is not None and self.show_mask.get():
                img = overlay_mask(img, res["prob"], float(self.mask_thr.get()))
            if res.get("boxes") and self.show_boxes.get():
                img = draw_boxes(img, res["boxes"], res["confs"])
        return img

    def _refresh(self):
        img = self._compose()
        self.canvas.delete("all")
        if img is None:
            return
        cw = max(self.canvas.winfo_width(), 2)
        ch = max(self.canvas.winfo_height(), 2)
        scale = min(cw / img.width, ch / img.height, 1.0)
        disp = img.resize((max(1, int(img.width * scale)),
                           max(1, int(img.height * scale))), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")
        # live coverage update in the summary footer
        if self.result is not None and self.result.get("prob") is not None:
            thr = float(self.mask_thr.get())
            self.result["coverage"] = float((self.result["prob"] >= thr).mean() * 100)

    def _on_canvas_resize(self, _event):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self._refresh)

    # ------------------------------------------------------------- summary --
    def _summarize(self, res):
        mode, boxes, confs = res["mode"], res.get("boxes") or [], res.get("confs") or []
        lines = [f"Mode : {mode.upper()}",
                 f"Image: {self.src_path.name if self.src_path else '-'}", ""]

        if mode in ("yolo", "hybrid"):
            found = len(boxes) > 0
            res["found"] = found
            if found:
                lines.append(f"Object FOUND  --  {len(boxes)} region(s)")
                lines.append("\u0634\u06cc\u0621 \u0645\u0648\u0631\u062f \u0646\u0638\u0631 "
                             "\u067e\u06cc\u062f\u0627 \u0634\u062f \u2713")
                for i, ((x1, y1, x2, y2), c) in enumerate(zip(boxes, confs), 1):
                    lines.append(f"  box {i}: conf={c:.2f}  "
                                 f"[{x1:.0f},{y1:.0f} -> {x2:.0f},{y2:.0f}]")
            else:
                lines.append("Object NOT found by YOLO at this confidence.")
                lines.append("\u0634\u06cc\u0621 \u062f\u0631 \u062a\u0635\u0648\u06cc\u0631 "
                             "\u067e\u06cc\u062f\u0627 \u0646\u0634\u062f \u2717")
            if res.get("fallback"):
                lines.append("")
                lines.append("(no boxes -> UNet ran on the FULL image as fallback)")
        else:
            res["found"] = None

        if res.get("prob") is not None:
            thr = float(self.mask_thr.get())
            cov = float((res["prob"] >= thr).mean() * 100)
            res["coverage"] = cov
            lines += ["", f"UNet mask coverage: {cov:.2f}% of image  (thr={thr:.2f})",
                      "\u0646\u0627\u062d\u06cc\u0647 \u0628\u0627 UNet \u0631\u0648\u06cc "
                      "\u062a\u0635\u0648\u06cc\u0631 \u0645\u0634\u062e\u0635 \u0634\u062f"]

        self._set_text("\n".join(lines))
        self.status.set(f"{mode.upper()} done.")

    def _set_text(self, s):
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", s)
        self.txt.config(state="disabled")


# =============================================================================
def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
