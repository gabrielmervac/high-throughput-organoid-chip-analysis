"""
organoid_annotator.py
=====================
3D organoid segmentation, annotation, and metrics GUI.

Supports:
  • ND2 (Nikon) files  (T × Z × C × Y × X)
  • Multi-frame / Z-stack TIFFs

Segmentation models:
  • Cellpose 3D  — cyto3 / nuclei   (pip install cellpose)
  • StarDist 3D  — 3D_demo pretrained (pip install stardist tensorflow)

Run:
    python organoid_annotator.py
"""

import sys, os, json, warnings
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # OrganoID model is Keras 2
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tifffile
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", message="Sparse invariant", category=UserWarning)

try:
    import torch as _torch  # noqa: F401  import before Qt to avoid c10.dll conflict
except Exception:
    pass

from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QLabel, QComboBox, QSlider, QSpinBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QScrollArea, QFileDialog,
    QStatusBar, QProgressBar, QListWidget, QListWidgetItem,
    QMessageBox, QTabWidget, QFrame, QSizePolicy, QInputDialog,
    QDialog, QDialogButtonBox, QFormLayout, QTextEdit, QProgressDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QLineEdit, QDockWidget, QSplitter as QSpl
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QSize, QTimer
from PyQt6.QtGui import (
    QPixmap, QImage, QIcon, QPainter, QPen, QBrush, QColor, QCursor,
    QFont, QAction, QPalette
)

# Matplotlib (for the metric-over-time plot panel). Guarded so the app still
# runs if matplotlib is unavailable — the plot panel simply degrades to a note.
try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# ── Constants ─────────────────────────────────────────────────────────────────

ORGANOID_TYPES  = ["Tumor", "Normal", "Mixed", "Cystic", "Dead", "Other"]
ORGANOID_STATES = ["Intact", "Growing", "Fragmented", "Dying"]

TYPE_COLORS = {
    "Tumor":      QColor(255,  80,  80),
    "Normal":     QColor( 80, 200,  80),
    "Mixed":      QColor(255, 180,  50),
    "Cystic":     QColor( 80, 160, 255),
    "Dead":       QColor(160, 160, 160),
    "Other":      QColor(200, 200,  80),
    "Unlabelled": QColor( 50, 220, 220),   # bright cyan — clearly visible on gray images
}

VERSION = "2.0.0-TF"

# Pretty labels for metric keys (shared by the metrics table and the plot selector).
# Any key not listed falls back to a generated label (see OrgAnnotator._metric_label).
METRIC_LABELS = {
    "centroid_x":  "Centroid X (px)",
    "centroid_y":  "Centroid Y (px)",
    "centroid_z":  "Centroid Z (plane)",
    "z_extent":    "Z extent (planes)",
    "best_z":      "Best Z (max-area plane)",
    "area_px":     "Area (px, max plane)",
    "area_um2":    "Area (µm², max plane)",
    "diameter_um": "Diameter (µm, max plane)",
    "volume_vox":  "Volume (vox)",
    "volume_um3":  "Volume (µm³)",
}

# Canonical display order for the geometry metrics; intensity keys are ordered
# per-channel afterwards. Any remaining keys are appended alphabetically.
_METRIC_ORDER = [
    "centroid_x", "centroid_y", "centroid_z", "z_extent", "best_z",
    "area_px", "area_um2", "diameter_um", "volume_vox", "volume_um3",
]

# Default fine-tuned OrganoID model (TensorFlow SavedModel directory).
DEFAULT_MODEL_DIR = str((Path(__file__).resolve().parent.parent.parent
                         / "models" / "organoid_finetuned_orgaug_v2_BEST"))

# Phase-contrast channel is the OrganoID segmentation input.
_PHASE_HINTS = ("ph", "phase", "bf", "brightfield", "trans", "dia")


def guess_phase_channel(names):
    """Index of the phase/brightfield channel by name; fallback = last channel."""
    for i, n in enumerate(names):
        if any(h in str(n).lower() for h in _PHASE_HINTS):
            return i
    return (len(names) - 1) if names else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_slice(arr, gains=None, ranges=None, phase_channel=None):
    """
    Convert a 2-D or (C,H,W)/(H,W,C) slice to (H,W,3) uint8 for display.

    gains  : optional list of per-channel contrast multipliers (one per display
             channel R/G/B). A gain of 1.0 == plain min-max stretch. Values >1
             brighten, <1 darken. Result is clipped to [0, 255]. Contrast
             sliders map slider_value/50.0 → gain (50 = 1.0).
    ranges : optional list of per-channel (lo, hi) intensity bounds to stretch
             against. When given, these fixed bounds are used instead of the
             slice's own min/max — this makes the display consistent across all
             timepoints (a channel dimming/brightening over time is visible
             rather than being re-normalised away each frame). Falls back to
             per-slice min/max for any channel without a supplied range.
    """
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] > 4:
        arr = arr[:, :, :3]
    if arr.ndim == 3:
        H, W = arr.shape[:2]
        n_ch = arr.shape[2]
        acc = np.zeros((H, W, 3), dtype=np.float32)
        base_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]   # ch0->R, ch1->G, ch2->B
        for c in range(n_ch):
            # Only the first 3 channels (plus the phase channel) contribute to display.
            if c >= 3 and c != phase_channel:
                continue
            ch = arr[:, :, c].astype(np.float32)
            if ranges is not None and c < len(ranges):
                lo, hi = ranges[c]
            else:
                lo, hi = ch.min(), ch.max()
            norm = (ch - lo) / max(hi - lo, 1) * 255
            if gains is not None and c < len(gains):
                norm = norm * gains[c]
            # Phase-contrast channel is rendered gray (R=G=B); others get an R/G/B tint.
            weight = (1, 1, 1) if c == phase_channel else base_colors[c % 3]
            for k in range(3):
                if weight[k]:
                    acc[:, :, k] += norm * weight[k]
        return np.clip(acc, 0, 255).astype(np.uint8)
    arr = arr.astype(np.float32)
    if ranges is not None and len(ranges) > 0:
        lo, hi = ranges[0]
    else:
        lo, hi = arr.min(), arr.max()
    norm = (arr - lo) / max(hi - lo, 1) * 255
    if gains is not None and len(gains) > 0:
        norm = norm * gains[0]
    arr = np.clip(norm, 0, 255).astype(np.uint8)
    return np.stack([arr] * 3, axis=-1)


def to_raw_float(arr):
    """Return (H,W,3) float32 with original pixel values."""
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] > 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 2:
        return np.concatenate([arr, np.zeros((*arr.shape[:2], 1), np.float32)], axis=-1)
    return arr


def region_sphericity(region, spacing=(1.0, 1.0, 1.0)):
    """
    Shape roundness of a binary region in [0, 1] (1 = perfect sphere/circle).

    For a genuinely 3-D object (Z extent ≥ 2 and ≥2 voxels in Y and X) this is
    the true 3-D sphericity

        Ψ = π^(1/3) · (6V)^(2/3) / A

    where V is the physical volume and A the physical surface area, estimated
    from a marching-cubes mesh (respecting anisotropic voxel spacing). Ψ = 1
    for a sphere and < 1 for any other shape.

    For an essentially planar object (single Z slice, or too thin for a 3-D
    surface) it falls back to 2-D circularity  4π·Area / Perimeter²  on the
    XY footprint, which measures roundness in the imaging plane.

    spacing : (dz, dy, dx) in µm.
    """
    dz, dy, dx = spacing
    vox = int(region.sum())
    if vox == 0:
        return 0.0

    # Crop to the object's bounding box for speed
    zz, yy, xx = np.where(region)
    z0, z1 = zz.min(), zz.max() + 1
    y0, y1 = yy.min(), yy.max() + 1
    x0, x1 = xx.min(), xx.max() + 1
    sub = region[z0:z1, y0:y1, x0:x1]

    if (z1 - z0) >= 2 and sub.shape[1] >= 2 and sub.shape[2] >= 2:
        try:
            from skimage.measure import marching_cubes, mesh_surface_area
            padded = np.pad(sub.astype(np.float32), 1)   # close surface at borders
            verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=(dz, dy, dx))
            area = float(mesh_surface_area(verts, faces))
            V = vox * dz * dy * dx
            if area > 0:
                s = (np.pi ** (1.0 / 3.0)) * ((6.0 * V) ** (2.0 / 3.0)) / area
                return round(min(max(s, 0.0), 1.0), 3)
        except Exception:
            pass
        return 0.0

    # ── 2-D circularity fallback (planar object) ──────────────────────────────
    flat = sub.any(axis=0)
    n_px = int(flat.sum())
    if n_px == 0:
        return 0.0
    area = n_px * dy * dx
    px = 0.5 * (dy + dx)   # nominal in-plane pixel size (XY is usually isotropic)
    perim = None
    try:
        # Crofton perimeter corrects the staircase over-estimate of a digital
        # boundary, so a round disk scores ~1 rather than ~0.63.
        from skimage.measure import perimeter_crofton
        perim = float(perimeter_crofton(flat)) * px
    except Exception:
        # Fallback: staircase perimeter from exposed edges
        f = flat.astype(np.int8)
        t_y = int(np.abs(np.diff(np.pad(f, ((1, 1), (0, 0))), axis=0)).sum())
        t_x = int(np.abs(np.diff(np.pad(f, ((0, 0), (1, 1))), axis=1)).sum())
        perim = t_y * dx + t_x * dy
    if perim and perim > 0:
        c = 4.0 * np.pi * area / (perim ** 2)
        return round(min(max(c, 0.0), 1.0), 3)
    return 0.0


BG_PERCENTILE = 25.0   # background = this percentile of a channel's pixels


def channel_backgrounds(volume, pct=BG_PERCENTILE, mask=None):
    """
    Per-channel background level = `pct`-th percentile of the BACKGROUND pixels in
    that channel of a (Z, C, H, W) volume.  Subtracting this counteracts the global
    fluorescence rise that occurs when fresh media is added.

    When a `mask` (Z, H, W, any nonzero label = a detected object) is given, only
    pixels OUTSIDE every object are used, so real organoid signal does not inflate
    the background estimate.  If the mask leaves no background pixels for a channel,
    that channel falls back to all pixels.
    Returns a list of floats, one per channel.
    """
    if volume is None or volume.ndim != 4:
        return []
    bg_sel = None
    if mask is not None and mask.shape == (volume.shape[0], volume.shape[2], volume.shape[3]):
        bg_sel = (mask == 0)
        if not bg_sel.any():
            bg_sel = None
    out = []
    for c in range(volume.shape[1]):
        ch = volume[:, c]
        pixels = ch[bg_sel] if bg_sel is not None else ch
        out.append(float(np.percentile(pixels, pct)))
    return out


def flatfield_gaussian(img2d, sigma):
    """
    imflatfield-style illumination correction of a single 2-D image (MATLAB's
    imflatfield): estimate the shading as a heavily Gaussian-blurred copy, divide
    it out, and rescale by the shading's mean so overall brightness is preserved.
    Returns float32.

    The shading is a low-frequency field, so it is estimated on a DOWNSAMPLED copy
    (with a correspondingly smaller sigma) and then resized back — this is ~100×
    faster than a full-resolution large-sigma Gaussian and visually identical.
    """
    from scipy.ndimage import gaussian_filter
    import skimage.transform
    img = img2d.astype(np.float32)

    ds = 4 if min(img.shape) >= 256 else 1          # downsample factor
    if ds > 1:
        small = img[::ds, ::ds]
        sh_small = gaussian_filter(small, sigma=max(1.0, float(sigma) / ds))
        shading = skimage.transform.resize(
            sh_small, img.shape, order=1, preserve_range=True).astype(np.float32)
    else:
        shading = gaussian_filter(img, sigma=float(sigma))

    mean_sh = float(shading.mean())
    if mean_sh <= 0:
        return img
    out = img * (mean_sh / np.maximum(shading, 1e-6))
    return out.astype(np.float32)


# Illumination-correction methods offered in the Segment tab (combo index → key).
ILLUM_METHODS = ["none", "tophat", "rolling_ball", "clahe"]
ILLUM_LABELS  = ["None", "Top-hat (opening)", "Rolling ball", "CLAHE"]

# Flat-field / shading-correction methods for the batch run (combo index → key),
# handled by FlatfieldCorrector (applied to the whole volume, all channels).
FLAT_METHODS = ["none", "gaussian", "basic"]
FLAT_LABELS  = ["None", "Gaussian (imflatfield)", "BaSiC"]


def correct_illumination_2d(img2d, method="none", radius=50):
    """
    Flatten uneven background illumination in a single 2-D image so segmentation
    behaves better where the field is unevenly lit (e.g. organoids near a bright
    chamber edge).  Returns a float32 image of the same shape.

    method:
      'none'         — passthrough.
      'tophat'       — white top-hat: estimate a slowly-varying background with a
                       morphological opening (disk of `radius`) and subtract it,
                       removing large-scale shading while keeping organoid-scale
                       detail (this is the "image open" approach).
      'rolling_ball' — estimate the background with a rolling ball of `radius` and
                       subtract it (classic ImageJ-style shading removal).
      'clahe'        — contrast-limited adaptive histogram equalization; normalises
                       LOCAL contrast across the field so a dim edge is not lost.

    For 'tophat'/'rolling_ball' the background is estimated on a downsampled copy
    (with a proportionally smaller radius) and resized back — visually identical to
    a full-resolution estimate but far faster on large frames.
    """
    if method in (None, "none") or img2d is None:
        return img2d
    img = np.asarray(img2d, dtype=np.float32)

    try:
        if method == "clahe":
            from skimage import exposure
            lo, hi = float(img.min()), float(img.max())
            if hi <= lo:
                return img
            norm = (img - lo) / (hi - lo)
            eq   = exposure.equalize_adapthist(norm, clip_limit=0.01)
            return (eq * (hi - lo) + lo).astype(np.float32)

        # tophat / rolling_ball: estimate background on a downsampled copy.
        import skimage.transform
        ds     = 4 if min(img.shape) >= 256 else 1
        radius = max(1, int(radius))
        small  = img[::ds, ::ds] if ds > 1 else img
        r_small = max(1, radius // ds)

        if method == "tophat":
            from skimage.morphology import opening, disk
            bg_small = opening(small, disk(r_small))
        elif method == "rolling_ball":
            from skimage.restoration import rolling_ball
            bg_small = rolling_ball(small, radius=r_small)
        else:
            return img

        if ds > 1:
            bg = skimage.transform.resize(
                bg_small, img.shape, order=1, preserve_range=True).astype(np.float32)
        else:
            bg = bg_small.astype(np.float32)

        out = img - bg
        np.clip(out, 0.0, None, out=out)
        return out
    except Exception:
        # Any missing optional dependency / failure → leave the image unchanged
        # rather than aborting the whole segmentation run.
        return img


class FlatfieldCorrector:
    """
    Per-channel illumination correction applied to (Z, C, H, W) volumes.

    method:
      'none'     — passthrough (no correction)
      'gaussian' — imflatfield-style per-slice Gaussian shading division
      'basic'    — BaSiC (estimates flatfield + darkfield from many images);
                   requires `pip install basicpy`. Fit once on all supplied
                   images per channel, then applied to every volume.
    """

    def __init__(self, method="none", sigma=80.0):
        self.method     = method
        self.sigma      = float(sigma)
        self.flatfields = {}   # channel -> (H,W) flatfield  (basic)
        self.darkfields = {}   # channel -> (H,W) darkfield  (basic)

    def fit(self, images_per_channel):
        """images_per_channel: {c: (N,H,W) float32 stack}. Only used by BaSiC."""
        if self.method != "basic":
            return
        from basicpy import BaSiC
        for c, stack in images_per_channel.items():
            basic = BaSiC(get_darkfield=True, smoothness_flatfield=1.0)
            basic.fit(stack.astype(np.float32))
            self.flatfields[c] = np.asarray(basic.flatfield, dtype=np.float32)
            self.darkfields[c] = np.asarray(basic.darkfield, dtype=np.float32)

    def apply_volume(self, vol):
        """Return a flatfield-corrected copy of a (Z, C, H, W) float32 volume."""
        if self.method == "none":
            return vol
        out = vol.astype(np.float32).copy()
        Z, C = vol.shape[0], vol.shape[1]
        for c in range(C):
            if self.method == "gaussian":
                for z in range(Z):
                    out[z, c] = flatfield_gaussian(vol[z, c], self.sigma)
            elif self.method == "basic" and c in self.flatfields:
                ff = np.maximum(self.flatfields[c], 1e-6)
                df = self.darkfields.get(c, 0.0)
                for z in range(Z):
                    corr = (vol[z, c].astype(np.float32) - df) / ff
                    np.clip(corr, 0.0, None, out=corr)
                    out[z, c] = corr
        return out


# Deterministic RGB palette (mirrors the on-screen track palette) for exports.
_PALETTE_RGB = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


def palette_color(key):
    """Deterministic RGB for a positive integer id (organoid id or track id)."""
    if key is None or key < 1:
        return (50, 220, 220)
    return _PALETTE_RGB[(int(key) - 1) % len(_PALETTE_RGB)]


def _downscale(img, max_dim):
    from PIL import Image as _I
    W, H = img.size
    scale = min(1.0, max_dim / max(H, W))
    if scale < 1.0:
        img = img.resize((max(1, int(W * scale)), max(1, int(H * scale))),
                         _I.Resampling.NEAREST)
    return img


def save_mask_image(path, label2d, scale=0.25):
    """Write a BINARY mask: white (255) where any organoid, black (0) elsewhere.
    Saved LOSSLESS as PNG (JPEG would corrupt the binary values) at `scale`×
    the original resolution (default 0.25×)."""
    from PIL import Image
    import os as _os
    path = _os.path.splitext(path)[0] + ".png"     # force lossless PNG
    H, W = label2d.shape
    binary = np.where(label2d > 0, 255, 0).astype(np.uint8)
    im = Image.fromarray(binary, "L")
    im = im.resize((max(1, int(W * scale)), max(1, int(H * scale))),
                   Image.Resampling.NEAREST)
    im.save(path)


def save_overlay_image(path, base_rgb, label2d, id_color, id_number, scale=0.25,
                       alpha=0.4, font_size=22):
    """
    Write a JPG of the image with the mask overlaid (semi-transparent) and each
    organoid's ID drawn (large) at its centroid. Saved at `scale`× original size.
    The label is drawn AFTER downscaling so it stays large and legible.
    """
    from PIL import Image, ImageDraw, ImageFont
    H, W = label2d.shape
    ids = [int(o) for o in np.unique(label2d) if o != 0]

    base = Image.fromarray(np.ascontiguousarray(base_rgb), "RGB").convert("RGBA")
    ov = np.zeros((H, W, 4), np.uint8)
    for oid in ids:
        col = id_color(oid)
        m = label2d == oid
        ov[m, 0], ov[m, 1], ov[m, 2], ov[m, 3] = col[0], col[1], col[2], int(alpha * 255)
    comp = Image.alpha_composite(base, Image.fromarray(ov, "RGBA")).convert("RGB")
    tw, th = max(1, int(W * scale)), max(1, int(H * scale))
    comp = comp.resize((tw, th), Image.Resampling.BILINEAR)

    draw = ImageDraw.Draw(comp)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    for oid in ids:
        ys, xs = np.where(label2d == oid)
        cx, cy = float(xs.mean()) * scale, float(ys.mean()) * scale
        txt = str(id_number(oid))
        draw.text((cx - 1, cy - 1), txt, fill=(0, 0, 0), font=font)     # outline
        draw.text((cx, cy),         txt, fill=(255, 255, 0), font=font)  # label
    comp.save(path, quality=90)


def compute_3d_metrics(mask_3d, voxel_size_um=(1.0, 1.0, 1.0), volume=None,
                       backgrounds=None, phase_channel=None):
    """
    Return per-organoid 3-D metrics dict keyed by organoid id.

    Geometry (per organoid):
      centroid_x/y/z (px), z_extent (planes), best_z (plane of MAX cross-section),
      area_px / area_um2 / diameter_um (measured on that max-area plane),
      volume_vox / volume_um3.

    Intensity (only for NON-phase channels, i.e. all channels except phase_channel),
    background-subtracted (25th-percentile of that channel over all Z, clipped at 0):
      int_{sum,mean,median}_ch{n}_maxarea  — pixels on the max-area plane only
      int_{sum,mean,median}_ch{n}_vol      — the whole 3-D organoid
      int_bg_ch{n}                         — the background subtracted
    Channel numbering is 1-based over the original channel order (phase kept in the
    numbering but not measured), so ch1=first channel, ch2=second, ...
    """
    ids = np.unique(mask_3d)
    ids = ids[ids != 0]
    metrics = {}
    dz, dy, dx = voxel_size_um
    n_ch = 0
    if volume is not None and volume.ndim == 4 and \
       mask_3d.shape == (volume.shape[0], volume.shape[2], volume.shape[3]):
        n_ch = volume.shape[1]
    if n_ch > 0 and backgrounds is None:
        backgrounds = channel_backgrounds(volume, mask=mask_3d)
    meas_ch = [c for c in range(n_ch) if c != phase_channel]   # fluorescence channels

    for oid in ids:
        region = (mask_3d == oid)
        vol_vox = int(region.sum())
        if vol_vox == 0:
            continue
        Zn = region.shape[0]
        zz, yy, xx = np.where(region)

        # Plane of maximum cross-sectional area.
        areas   = region.reshape(Zn, -1).sum(axis=1)
        # Number of Z-planes the organoid occupies (counts from 1).
        z_extent = int(np.count_nonzero(areas))
        best_z_idx = int(np.argmax(areas))       # 0-based plane index (for indexing arrays)
        max_px  = int(areas[best_z_idx])
        area_um2 = max_px * dy * dx
        diam_um  = 2.0 * np.sqrt(area_um2 / np.pi) if area_um2 > 0 else 0.0
        plane2d  = region[best_z_idx]

        # Physical volume by trapezoidal (midpoint) integration of cross-sectional
        # area along Z between the first and last occupied planes. Integrating over
        # the (k-1) gaps rather than summing k full slabs avoids the extra
        # half-slab of overhang past each end plane that a plain voxel-count gives.
        # Single-plane objects (no gap) yield 0 here, but are discarded upstream by
        # the z_extent >= 2 constraint, so they never reach the output.
        occ = np.nonzero(areas)[0]
        if occ.size >= 2:
            i, j = int(occ[0]), int(occ[-1])
            area_px_trap = float(np.trapz(areas[i:j + 1], dx=1.0))   # in pixel·plane units
        else:
            area_px_trap = 0.0
        volume_um3 = area_px_trap * dz * dy * dx

        m = {
            "centroid_x":  round(float(xx.mean()), 1),
            "centroid_y":  round(float(yy.mean()), 1),
            "centroid_z":  round(float(zz.mean()) + 1.0, 1),   # 1-based plane number
            "z_extent":    z_extent,
            "best_z":      best_z_idx + 1,   # report as a 1-based plane number (like timepoint/XY)
            "area_px":     max_px,
            "area_um2":    round(area_um2, 2),
            "diameter_um": round(diam_um, 2),
            "volume_vox":  vol_vox,
            "volume_um3":  round(volume_um3, 2),
        }

        # Intensities (max-area plane + whole volume), per fluorescence channel.
        if n_ch > 0:
            for c in meas_ch:
                bg = float(backgrounds[c]) if backgrounds and c < len(backgrounds) else 0.0
                pv = volume[best_z_idx, c][plane2d].astype(np.float32) - bg
                np.clip(pv, 0.0, None, out=pv)
                vv = volume[:, c][region].astype(np.float32) - bg
                np.clip(vv, 0.0, None, out=vv)
                ci = c + 1
                m[f"int_sum_ch{ci}_maxarea"]    = round(float(pv.sum()), 2)
                m[f"int_mean_ch{ci}_maxarea"]   = round(float(pv.mean()) if pv.size else 0.0, 2)
                m[f"int_median_ch{ci}_maxarea"] = round(float(np.median(pv)) if pv.size else 0.0, 2)
                m[f"int_sum_ch{ci}_vol"]        = round(float(vv.sum()), 2)
                m[f"int_mean_ch{ci}_vol"]       = round(float(vv.mean()) if vv.size else 0.0, 2)
                m[f"int_median_ch{ci}_vol"]     = round(float(np.median(vv)) if vv.size else 0.0, 2)
            for c in meas_ch:
                bg = float(backgrounds[c]) if backgrounds and c < len(backgrounds) else 0.0
                m[f"int_bg_ch{c + 1}"] = round(bg, 2)

        metrics[int(oid)] = m
    return metrics


# ── 3D Segmentation worker ────────────────────────────────────────────────────

class Seg3DWorker(QThread):
    progress  = pyqtSignal(int, str)
    finished  = pyqtSignal(dict)          # {frame_idx: mask_3d (Z,H,W)}
    error     = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, volumes, params):
        super().__init__()
        self.volumes         = volumes    # {frame_idx: (Z,C,H,W) float32}
        self.params          = params
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def _illum_correct(self, vol):
        """Return a copy of a (Z, C, H, W) volume with illumination correction
        applied ONLY to the phase (segmentation-input) channel; the original volume
        and all fluorescence channels are left untouched. Passthrough when disabled
        or when there is no phase channel."""
        method = self.params.get("illum_method", "none")
        if method in (None, "none") or vol.ndim != 4:
            return vol
        pc = self.params.get("phase_channel", None)
        if pc is None or pc < 0 or pc >= vol.shape[1]:
            return vol
        radius = int(self.params.get("illum_radius", 50))
        out = vol.copy()
        for z in range(out.shape[0]):
            out[z, pc] = correct_illumination_2d(out[z, pc], method, radius)
        return out

    def run(self):
        try:
            model_name = self.params.get("model", "cellpose_cyto3")
            n = len(self.volumes)
            results = {}

            if model_name.startswith("cellpose"):
                self._run_cellpose(n, results)
            elif model_name == "stardist":
                self._run_stardist(n, results)
            elif model_name == "organoid_id":
                self._run_organoid(n, results)
            else:
                self.error.emit(f"Unknown model: {model_name}")
                return

            if not self._stop_requested:
                self.progress.emit(100, "Segmentation complete.")
                self.finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())

    def segment_now(self):
        """Run segmentation synchronously and return {t: mask_3d} (no signals).
        Used by the position batch runner to reuse the exact segmentation code."""
        model_name = self.params.get("model", "cellpose_cyto3")
        results = {}
        n = len(self.volumes)
        if model_name.startswith("cellpose"):
            self._run_cellpose(n, results)
        elif model_name == "stardist":
            self._run_stardist(n, results)
        elif model_name == "organoid_id":
            self._run_organoid(n, results)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        return results

    def _run_cellpose(self, n, results):
        from cellpose import models as cp_models
        self.progress.emit(5, "Loading Cellpose model…")

        cp_model_name = self.params.get("model", "cellpose_cyto3").replace("cellpose_", "")
        use_gpu = self.params.get("gpu", False)
        model   = cp_models.CellposeModel(gpu=use_gpu, model_type=cp_model_name)

        channels  = self.params.get("channels", [0, 0])
        cyto_ch, nuc_ch = channels
        niter     = self.params.get("niter") or None

        if cyto_ch == 0 or nuc_ch == 0:
            cp_channels = [0, 0]
        else:
            cp_channels = [1, 2]

        for i, (fidx, vol) in enumerate(self.volumes.items()):
            if self._stop_requested:
                self.cancelled.emit()
                return

            pct = int(10 + 85 * i / n)
            self.progress.emit(pct, f"Segmenting timepoint {i + 1}/{n}…")

            vol = self._illum_correct(vol)   # flatten uneven illumination (phase ch)

            # Build cp_vol: always (Z, H, W) grayscale — simplest and most compatible.
            # Cellpose channels= param handles cyto/nucleus selection from the flat array.
            if vol.ndim == 4:           # (Z, C, H, W)
                Z, C, H, W = vol.shape
                if cyto_ch == 0 or C == 1:
                    cp_vol = vol.mean(axis=1).astype(np.float32)   # (Z, H, W)
                else:
                    cp_vol = vol[:, cyto_ch - 1].astype(np.float32) # (Z, H, W)
            else:
                cp_vol = vol.astype(np.float32)   # already (Z, H, W)

            # cp_vol is always 3D (Z, H, W) — no channel_axis needed
            stitch_thr = self.params.get("stitch_threshold", 0.5)
            do_3d      = self.params.get("do_3d", False)
            masks_3d, _, _ = model.eval(
                cp_vol,
                do_3D              = do_3d,
                z_axis             = 0,
                diameter           = self.params.get("diameter")          or None,
                flow_threshold     = self.params.get("flow_threshold",     0.4),
                cellprob_threshold = self.params.get("cellprob_threshold", 0.0),
                min_size           = self.params.get("min_size",           500),
                normalize          = self.params.get("normalize",          True),
                augment            = self.params.get("augment",            False),
                niter              = niter,
                channels           = [0, 0],   # grayscale; channel selection done above
                stitch_threshold   = stitch_thr if not do_3d else 0.0,
            )
            results[fidx] = masks_3d.astype(np.int32)

    def _run_stardist(self, n, results):
        try:
            from stardist.models import StarDist3D
            from csbdeep.utils import normalize as csbdeep_normalize
        except ImportError:
            self.error.emit(
                "StarDist is not installed.\n\n"
                "Install with:\n"
                "    pip install stardist tensorflow"
            )
            return

        self.progress.emit(5, "Loading StarDist 3D model…")
        sd_model_name = self.params.get("stardist_model", "3D_demo")
        try:
            model = StarDist3D.from_pretrained(sd_model_name)
        except OSError as e:
            if "1314" in str(e) or "privilege" in str(e).lower():
                self.error.emit(
                    "StarDist model download failed — Windows symlink privilege error.\n\n"
                    "Fix (no admin required):\n"
                    "  Settings → System → For developers\n"
                    "  → Developer Mode  →  On\n\n"
                    "Then restart the app and try again."
                )
            else:
                self.error.emit(f"StarDist model load failed:\n{e}")
            return

        for i, (fidx, vol) in enumerate(self.volumes.items()):
            if self._stop_requested:
                self.cancelled.emit()
                return

            pct = int(10 + 85 * i / n)
            self.progress.emit(pct, f"StarDist: timepoint {i + 1}/{n}…")

            vol = self._illum_correct(vol)   # flatten uneven illumination (phase ch)

            # StarDist expects (Z, Y, X) float, normalized
            if vol.ndim == 4:
                img = vol.mean(axis=1).astype(np.float32)  # (Z,H,W)
            else:
                img = vol.astype(np.float32)

            img = csbdeep_normalize(img, 1, 99.8, axis=(0, 1, 2))

            labels, _ = model.predict_instances(img)
            results[fidx] = labels.astype(np.int32)

    def _run_organoid(self, n, results):
        weights_path = self.params.get("oid_weights_path", "")
        if not weights_path or not os.path.exists(weights_path):
            self.error.emit(
                "OrganoID model not set.\n\n"
                "Select the fine-tuned model folder (a TensorFlow SavedModel, e.g.\n"
                "models/organoid_finetuned_BEST) via the 'Browse…' button on the Segment tab."
            )
            return

        self.progress.emit(5, "Loading OrganoID (TensorFlow) model…")

        try:
            import sys, os as _os
            script_dir = _os.path.dirname(_os.path.abspath(__file__))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            from organoid_id_tf import load_tf_model, segment_volume, segment_volume_per_z
        except ImportError as e:
            self.error.emit(f"Could not import organoid_id_tf:\n{e}")
            return

        try:
            model, meta = load_tf_model(weights_path)
        except Exception as e:
            import traceback
            self.error.emit(f"Failed to load OrganoID model:\n{traceback.format_exc()}")
            return

        phase_channel = self.params.get("phase_channel", None)
        roi = self.params.get("roi", None)
        separate = self.params.get("separate_contours", False)
        multiscale = self.params.get("multiscale", False)

        for i, (fidx, vol) in enumerate(self.volumes.items()):
            if self._stop_requested:
                self.cancelled.emit()
                return

            pct = int(10 + 85 * i / n)
            self.progress.emit(pct, f"OrganoID: timepoint {i + 1}/{n}…")

            vol = self._illum_correct(vol)   # flatten uneven illumination (phase ch)

            try:
                do_per_z = self.params.get("do_3d", False)
                if do_per_z:
                    mask_3d = segment_volume_per_z(
                        model, meta, vol,
                        threshold=0.5,
                        min_area=self.params.get("min_size", 100),
                        fill_holes=True,
                        remove_border=False,
                        separate_contours=separate,
                        stitch_threshold=self.params.get("stitch_threshold", 0.25),
                        phase_channel=phase_channel,
                        roi=roi,
                        multiscale=multiscale,
                    )
                else:
                    mask_3d = segment_volume(
                        model, meta, vol,
                        threshold=0.5,
                        min_area=self.params.get("min_size", 100),
                        fill_holes=True,
                        remove_border=False,
                        separate_contours=separate,
                        phase_channel=phase_channel,
                        roi=roi,
                        multiscale=multiscale,
                    )
                results[fidx] = mask_3d
            except Exception as e:
                import traceback
                self.error.emit(f"OrganoID failed on timepoint {fidx}:\n{traceback.format_exc()}")
                return


# ── Canvas ────────────────────────────────────────────────────────────────────

class OrganoCanvas(QWidget):
    """Displays a single Z-slice of the current volume with 3D mask overlay."""

    organoid_selected = pyqtSignal(int)   # emits organoid_id (0 = deselect)
    roi_changed       = pyqtSignal()      # emitted after an ROI stroke is committed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

        # ROI (region of interest) — a selection mask drawn with the pencil tool.
        # The whole frame is segmented; only organoids lying entirely within the
        # ROI are kept afterwards (anything outside or straddling it is dropped).
        self.roi_mode   = False   # pencil active (left-drag draws instead of selecting)
        self.roi_mask   = None    # (H, W) bool include-mask for the current position
        self._roi_stroke = []     # in-progress freehand path [(ix, iy), ...]
        self._roi_drawing = False

        self.image      = None    # (H, W, 3) uint8 — current Z slice
        self.mask_2d    = None    # (H, W) int32    — current Z mask slice
        self.labels     = {}      # {org_id: OrgAnnotation}
        self.selected   = 0       # currently selected organoid id

        self._zoom      = 1.0
        self._offset    = QPoint(0, 0)
        self._pan_start = None
        self._pan_off   = None

        self.show_masks     = True
        self.show_ids       = True
        self.mask_alpha     = 0.4
        self.color_override = {}   # {org_id: QColor} — set when coloring by track
        self.label_by_track = False  # draw track_id instead of raw org_id

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _img_to_canvas(self, ix, iy):
        return QPoint(int(ix * self._zoom) + self._offset.x(),
                      int(iy * self._zoom) + self._offset.y())

    def _canvas_to_img(self, cx, cy):
        return ((cx - self._offset.x()) / self._zoom,
                (cy - self._offset.y()) / self._zoom)

    def _fit_to_window(self):
        if self.image is None:
            return
        H, W = self.image.shape[:2]
        ww, wh = self.width(), self.height()
        self._zoom   = min(ww / W, wh / H) * 0.95
        self._offset = QPoint(int((ww - W * self._zoom) / 2),
                              int((wh - H * self._zoom) / 2))
        self.update()

    def set_data(self, image, mask_2d, labels, color_override=None, label_by_track=False):
        self.image          = image
        self.mask_2d        = mask_2d
        self.labels         = labels
        self.color_override = color_override or {}   # {org_id: QColor}
        self.label_by_track = label_by_track
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self.image is None:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Load an ND2 or TIFF stack to begin")
            return

        H, W = self.image.shape[:2]
        dw = int(W * self._zoom)
        dh = int(H * self._zoom)
        ox, oy = self._offset.x(), self._offset.y()

        # Draw base image — use bytes() so QImage owns the data
        rgb_bytes = bytes(np.ascontiguousarray(self.image))
        qimg = QImage(rgb_bytes, W, H, 3 * W, QImage.Format.Format_RGB888)
        painter.drawImage(QRect(ox, oy, dw, dh), qimg)

        # ── ROI overlay: dim everything outside the include-mask + green outline ──
        if self.roi_mask is not None and self.roi_mask.shape == (H, W):
            roi_ov = np.zeros((H, W, 4), np.uint8)
            outside = ~self.roi_mask
            roi_ov[outside, 3] = 130                      # darken excluded area
            try:
                edge = self.roi_mask & ~ndimage.binary_erosion(self.roi_mask, iterations=2)
                roi_ov[edge] = (0, 255, 0, 255)           # ROI boundary
            except Exception:
                pass
            rb = bytes(np.ascontiguousarray(roi_ov))
            qroi = QImage(rb, W, H, 4 * W, QImage.Format.Format_RGBA8888)
            painter.drawImage(QRect(ox, oy, dw, dh), qroi)

        # In-progress freehand stroke.
        if self._roi_drawing and len(self._roi_stroke) > 1:
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            pts = [self._img_to_canvas(px, py) for px, py in self._roi_stroke]
            for a, b in zip(pts[:-1], pts[1:]):
                painter.drawLine(a, b)

        if not self.show_masks or self.mask_2d is None:
            return

        ids = np.unique(self.mask_2d)
        ids = ids[ids != 0]
        if len(ids) == 0:
            return

        # Build RGBA overlay array
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        fill_alpha    = int(self.mask_alpha * 255)   # ~102
        border_alpha  = 255

        for oid in ids:
            region = (self.mask_2d == oid)
            ann    = self.labels.get(int(oid))
            otype  = ann.org_type if ann else ""
            col    = self.color_override.get(int(oid)) or \
                     TYPE_COLORS.get(otype, TYPE_COLORS["Unlabelled"])
            alpha  = 230 if int(oid) == self.selected else fill_alpha

            overlay[region, 0] = col.red()
            overlay[region, 1] = col.green()
            overlay[region, 2] = col.blue()
            overlay[region, 3] = alpha

            # Border: erode and subtract
            try:
                eroded = ndimage.binary_erosion(region, iterations=3)
                border = region & ~eroded
                overlay[border, 0] = min(col.red()   + 60, 255)
                overlay[border, 1] = min(col.green() + 60, 255)
                overlay[border, 2] = min(col.blue()  + 60, 255)
                overlay[border, 3] = border_alpha
            except Exception:
                pass

        # Draw overlay — bytes() makes QImage own the buffer (no lifetime issue)
        ov_bytes = bytes(np.ascontiguousarray(overlay))
        qov = QImage(ov_bytes, W, H, 4 * W, QImage.Format.Format_RGBA8888)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawImage(QRect(ox, oy, dw, dh), qov)

        # Draw ID labels. When label_by_track is on and the organoid has a valid
        # track_id, show the (stable) track ID so the number matches the color
        # after tracking; otherwise show the raw per-frame segmentation id.
        if self.show_ids:
            painter.setFont(QFont("Arial", max(8, int(10 * self._zoom))))
            for oid in ids:
                region = (self.mask_2d == oid)
                cy, cx = ndimage.center_of_mass(region)
                px = int(cx * self._zoom) + ox
                py = int(cy * self._zoom) + oy
                ann   = self.labels.get(int(oid))
                if self.label_by_track and ann is not None and ann.track_id >= 1:
                    text = str(ann.track_id)
                else:
                    text = str(int(oid))
                painter.setPen(QColor(0, 0, 0))
                painter.drawText(px - 13, py - 3, 26, 18,
                                 Qt.AlignmentFlag.AlignCenter, text)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(px - 12, py - 2, 24, 16,
                                 Qt.AlignmentFlag.AlignCenter, text)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.pos()
            self._pan_off   = QPoint(self._offset)
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            return

        if event.button() == Qt.MouseButton.LeftButton:
            ix, iy = self._canvas_to_img(event.pos().x(), event.pos().y())
            ix, iy = int(ix), int(iy)
            # Pencil mode: begin a freehand ROI stroke instead of selecting.
            if self.roi_mode and self.image is not None:
                self._roi_drawing = True
                self._roi_stroke  = [(ix, iy)]
                self.update()
                return
            if self.mask_2d is not None:
                H, W = self.mask_2d.shape
                if 0 <= iy < H and 0 <= ix < W:
                    oid = int(self.mask_2d[iy, ix])
                    self.selected = oid
                    self.organoid_selected.emit(oid)
                    self.update()

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._offset = self._pan_off + delta
            self.update()
            return
        if self._roi_drawing:
            ix, iy = self._canvas_to_img(event.pos().x(), event.pos().y())
            self._roi_stroke.append((int(ix), int(iy)))
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            return
        if event.button() == Qt.MouseButton.LeftButton and self._roi_drawing:
            self._roi_drawing = False
            self._commit_roi_stroke()
            self._roi_stroke = []
            self.update()

    def _commit_roi_stroke(self):
        """Rasterise the freehand loop and union it into the ROI include-mask."""
        if self.image is None or len(self._roi_stroke) < 3:
            return
        import skimage.draw
        H, W = self.image.shape[:2]
        if self.roi_mask is None or self.roi_mask.shape != (H, W):
            self.roi_mask = np.zeros((H, W), dtype=bool)
        rows = np.array([p[1] for p in self._roi_stroke])
        cols = np.array([p[0] for p in self._roi_stroke])
        rr, cc = skimage.draw.polygon(rows, cols, shape=(H, W))
        self.roi_mask[rr, cc] = True
        self.roi_changed.emit()

    def clear_roi(self):
        self.roi_mask = None
        self._roi_stroke = []
        self.roi_changed.emit()
        self.update()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        cx, cy = event.position().x(), event.position().y()
        self._offset = QPoint(
            int(cx + (self._offset.x() - cx) * factor),
            int(cy + (self._offset.y() - cy) * factor),
        )
        self._zoom = max(0.05, min(self._zoom * factor, 50.0))
        self.update()

    def resizeEvent(self, event):
        if self.image is not None:
            self._fit_to_window()


# ── Trackers ──────────────────────────────────────────────────────────────────

def _label_sizes(mask):
    """Return {label: voxel_count} for a 3-D int mask (background 0 excluded)."""
    ids, counts = np.unique(mask, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if i != 0}


class IoUTracker:
    """
    Link organoids across timepoints by 3-D mask OVERLAP (IoU) between frames.

    For each object in frame t, its voxel region is intersected with the objects
    of the most recent previous frame(s); it inherits the track of the previous
    object with the highest IoU, provided that IoU >= iou_thresh. Otherwise it
    starts a new track. This mirrors the working cell-annotator tracker and is
    robust to large / crowded / slowly-drifting organoids, where centroid-
    distance linking swaps identities between neighbours.

    Notes
    -----
    * True volumetric IoU: intersection is computed from the previous mask's
      labels lying under the new region, and union = |new| + |old| − |inter|,
      using pre-computed per-frame voxel counts (no expensive full-volume AND/OR).
    * memory : gap closing — how many extra missing frames a track may span and
      still be re-linked. When no match is found in the immediately preceding
      frame, up to `memory` earlier frames are also checked (nearest first).
    * Splits are allowed (two new objects may inherit the same track), matching
      the reference implementation.

    Returns {t: {org_id: track_id (1-based)}}.
    """

    def __init__(self, masks, iou_thresh=0.30, memory=2, progress_cb=None):
        self.masks       = masks
        self.iou_thresh  = float(iou_thresh)
        self.memory      = int(memory)
        self.progress_cb = progress_cb

    def _emit(self, pct, msg):
        if self.progress_cb:
            self.progress_cb(pct, msg)

    def run(self):
        times = sorted(t for t in self.masks if np.any(self.masks[t] != 0))
        if not times:
            return {}

        # Pre-compute per-frame label sizes (voxel counts) for fast IoU.
        sizes = {t: _label_sizes(self.masks[t]) for t in times}

        track_maps = {}
        next_track = 1

        # First frame: every object seeds a new track.
        t0 = times[0]
        track_maps[t0] = {}
        for oid in sizes[t0]:
            track_maps[t0][oid] = next_track
            next_track += 1

        n = len(times)
        for idx in range(1, n):
            t      = times[idx]
            mask_t = self.masks[t]
            track_maps[t] = {}

            # Candidate previous frames, nearest first (for gap closing).
            lookback = times[max(0, idx - 1 - self.memory):idx][::-1]

            self._emit(int(100 * idx / max(1, n - 1)),
                       f"Linking T{times[idx-1]}→T{t}  ({idx}/{n-1})")

            for oid in sizes[t]:
                new_region = (mask_t == oid)
                new_vox    = sizes[t][oid]
                matched_tid = None

                for tp in lookback:
                    prev_mask = self.masks[tp]
                    prev_vals = prev_mask[new_region]     # prev labels under new region
                    cands = np.unique(prev_vals)
                    cands = cands[cands != 0]
                    best_iou, best_old = 0.0, None
                    for old_id in cands:
                        old_id = int(old_id)
                        inter  = int(np.count_nonzero(prev_vals == old_id))
                        union  = new_vox + sizes[tp].get(old_id, 0) - inter
                        iou    = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou, best_old = iou, old_id
                    if best_old is not None and best_iou >= self.iou_thresh:
                        matched_tid = track_maps.get(tp, {}).get(best_old)
                        if matched_tid is not None:
                            break     # good match in nearest frame — stop looking back

                if matched_tid is not None:
                    track_maps[t][oid] = matched_tid
                else:
                    track_maps[t][oid] = next_track
                    next_track += 1

        return track_maps


# ── Track worker (background thread) ─────────────────────────────────────────

class TrackWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, masks, iou_thresh, memory):
        super().__init__()
        self.masks      = masks
        self.iou_thresh = iou_thresh
        self.memory     = memory

    def run(self):
        def _cb(pct, msg):
            self.progress.emit(pct, msg)

        tracker = IoUTracker(
            self.masks,
            iou_thresh  = self.iou_thresh,
            memory      = self.memory,
            progress_cb = _cb,
        )
        try:
            result = tracker.run()
            self.progress.emit(100, "Done.")
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


# ── Position batch worker (segment + track + export each selected position) ────

class PositionBatchWorker(QThread):
    """
    Process selected ND2 positions one at a time (streaming, never all in RAM):
    load → segment all timepoints → track over time → compute metrics → write
    an Excel file (always) plus, for selected positions, low-res mask and overlay
    JPGs (per T, Z) into <out_root>/position_####/.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)      # out_root
    error    = pyqtSignal(str)

    def __init__(self, positions, nd2_path, dims, pos_axis, out_root,
                 seg_params, iou_thresh, memory, voxel,
                 flat_method="none", flat_sigma=80.0,
                 image_positions=None, max_dim=384, experiment="", rois=None,
                 source=None):
        super().__init__()
        self.positions       = positions
        self.nd2_path        = nd2_path
        # Optional format-agnostic VolumeSource (ND2 or TIFF). When set,
        # _read_position delegates to it instead of the built-in ND2 reader,
        # so the exact same pipeline runs on either input format.
        self.source          = source
        self.experiment      = experiment
        self.rois            = rois if rois is not None else {}
        self.phase_channel   = None
        self.time_hours      = []      # per-timepoint acquisition time (hours)
        self.channel_names   = []      # channel names from ND2 metadata
        self._next_global_id = 1       # running experiment-wide organoid id
        self.dims            = dims
        self.pos_axis        = pos_axis
        self.out_root        = out_root
        self.seg_params      = seg_params
        self.iou_thresh      = iou_thresh
        self.memory          = memory
        self.voxel           = voxel
        self.flat_method     = flat_method
        self.flat_sigma      = flat_sigma
        # "all", or a set of position indices to also export JPGs for (else metrics-only)
        self.image_positions = image_positions if image_positions is not None else set()
        self.max_dim         = max_dim
        self._stop           = False

    def request_stop(self):
        self._stop = True

    # ── helpers ───────────────────────────────────────────────────────────────

    def _read_position(self, p):
        if self.source is not None:
            return self.source.read_position(p)     # ND2 or TIFF, canonical (Z,C,H,W)
        import nd2
        with nd2.ND2File(self.nd2_path) as f:
            darr = f.to_dask()
            idx = [slice(None)] * darr.ndim
            idx[self.pos_axis] = p
            sub = np.asarray(darr[tuple(idx)])
        dims_noP  = [d for d in self.dims if d != "P"]
        canonical = [d for d in ["T", "Z", "C", "Y", "X"] if d in dims_noP]
        order     = [dims_noP.index(d) for d in canonical]
        sub       = np.transpose(sub, order).astype(np.float32)
        for ax_i, ax in enumerate(["T", "Z", "C", "Y", "X"]):   # → full (T,Z,C,Y,X)
            if ax not in canonical:
                sub = np.expand_dims(sub, ax_i)
        return {t: sub[t] for t in range(sub.shape[0])}   # {t: (Z,C,H,W)}

    def _display_ranges(self, volumes):
        n_ch = next(iter(volumes.values())).shape[1]
        ranges = []
        for c in range(n_ch):
            lo = min(float(v[:, c].min()) for v in volumes.values())
            hi = max(float(v[:, c].max()) for v in volumes.values())
            ranges.append((lo, hi))
        return ranges

    def _save_images_for(self, p):
        """Whether to write mask/overlay JPGs for this position."""
        return self.image_positions == "all" or p in self.image_positions

    def _merge_split_detections(self, masks, track_maps):
        """Within each timepoint, merge every detection that shares a real track id
        (tid >= 1) into a single labelled object (relabelled to the smallest oid of the
        group). Mutates `masks` in place; leaves `track_maps` valid for the kept oid.

        This removes the double-counting that occurs when the tracker links two objects
        in one frame to the same track (a fragmented organoid) -> one row per organoid
        per timepoint, and the saved masks/overlays show it as one object.
        """
        for t, mask in masks.items():
            tmap = track_maps.get(t, {})
            groups = {}                                  # tid -> [oids in this frame]
            for oid, tid in tmap.items():
                if tid and tid >= 1:
                    groups.setdefault(int(tid), []).append(int(oid))
            for tid, oids in groups.items():
                if len(oids) < 2:
                    continue
                keep = min(oids)                         # merge the rest into this label
                for oid in oids:
                    if oid != keep:
                        mask[mask == oid] = keep
                        track_maps[t].pop(oid, None)     # its pixels now belong to `keep`
        return masks, track_maps

    def _write_position(self, p, volumes, masks, track_maps):
        """Measure + filter + relabel one position; save its JPGs; return the rows
        (the combined workbook is written once, in run())."""
        import os
        pos_dir = os.path.join(self.out_root, f"position_{p + 1:04d}")
        os.makedirs(pos_dir, exist_ok=True)

        times = [t for t in sorted(volumes.keys()) if masks.get(t) is not None]

        # 1) Metrics for every detected instance.
        inst = {}   # (t, oid) -> (track_id, metrics)
        for t in times:
            if self._stop:
                return
            vol, mask = volumes[t], masks[t]
            tmap = track_maps.get(t, {})
            bg = channel_backgrounds(vol, mask=mask)
            mets = compute_3d_metrics(mask, self.voxel, volume=vol, backgrounds=bg,
                                      phase_channel=self.phase_channel)
            for oid, m in mets.items():
                tid = tmap.get(oid, -1)
                inst[(t, oid)] = (tid if tid >= 1 else oid, m)

        # 2) Constraints (discard anything that fails):
        #    - diameter (max-area plane) > 40 µm
        #    - z_extent >= 2 planes
        #    - the track appears (as a valid detection) in > 2 frames
        #    - the track is present in the first OR last frame
        geo_ok = {k for k, (tid, m) in inst.items()
                  if m.get("diameter_um", 0.0) > 40.0 and m.get("z_extent", 0) >= 2}
        first_t = times[0] if times else None
        last_t  = times[-1] if times else None
        track_frames = {}
        for (t, oid) in geo_ok:
            track_frames.setdefault(inst[(t, oid)][0], set()).add(t)
        valid_tracks = {tid for tid, ts in track_frames.items()
                        if len(ts) > 2 and (first_t in ts or last_t in ts)}
        keep = {(t, oid) for (t, oid) in geo_ok
                if inst[(t, oid)][0] in valid_tracks}

        # Relabel surviving tracks to experiment-wide ids 1..N (assigned in order,
        # continuing the running counter across positions).
        gmap = {}
        for tid in sorted(valid_tracks):
            gmap[tid] = self._next_global_id
            self._next_global_id += 1

        # 3) Remove discarded organoids from the masks (so saved images match).
        for t in times:
            keep_oids = {oid for (tt, oid) in keep if tt == t}
            mask = masks[t]
            for oid in (set(int(x) for x in np.unique(mask)) - {0}) - keep_oids:
                mask[mask == oid] = 0

        # 4) Rows — column order & 1-based XY/timepoint exactly as specified;
        #    track_id is the experiment-wide organoid id.
        rows = []
        for (t, oid) in sorted(keep):
            tid, m = inst[(t, oid)]
            th = (self.time_hours[t]
                  if (self.time_hours and t < len(self.time_hours)) else "")
            row = {"experiment": self.experiment,
                   "XY_position": p + 1,          # 1-based
                   "track_id": gmap[tid],         # experiment-wide 1..N id
                   "timepoint": t + 1,            # 1-based
                   "time_hours": th}
            row.update(m)
            rows.append(row)

        # 5) Images (built from the filtered masks; labels = experiment-wide ids).
        if self._save_images_for(p):
            mask_dir    = os.path.join(pos_dir, "masks");    os.makedirs(mask_dir, exist_ok=True)
            overlay_dir = os.path.join(pos_dir, "overlay");  os.makedirs(overlay_dir, exist_ok=True)
            ranges = self._display_ranges(volumes)
            gid_by_t = {}
            for (t, oid) in keep:
                gid_by_t.setdefault(t, {})[oid] = gmap[inst[(t, oid)][0]]
            for t in times:
                if self._stop:
                    return rows
                vol, mask = volumes[t], masks[t]
                gmap_t = gid_by_t.get(t, {})
                _col = lambda o, _m=gmap_t: palette_color(_m.get(o, o))
                _num = lambda o, _m=gmap_t: _m.get(o, o)
                for z in range(vol.shape[0]):
                    base_rgb = normalize_slice(vol[z], ranges=ranges,
                                               phase_channel=self.phase_channel)
                    stem = f"T{t + 1:03d}_Z{z + 1:02d}.jpg"
                    save_mask_image(os.path.join(mask_dir, stem), mask[z], scale=0.5)
                    save_overlay_image(os.path.join(overlay_dir, stem), base_rgb, mask[z],
                                       _col, _num, scale=0.5)

        return rows

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            import os
            os.makedirs(self.out_root, exist_ok=True)
            n = len(self.positions)
            self._next_global_id = 1        # experiment-wide organoid ids start at 1
            all_rows = []
            for i, p in enumerate(self.positions):
                if self._stop:
                    break
                base = int(100 * i / n)
                self.progress.emit(base, f"Position {p + 1} ({i+1}/{n}): loading…")
                volumes = self._read_position(p)

                # Flat-field correction on the PHASE (segmentation) channel only, so
                # it never alters the fluorescence channels used for intensity metrics.
                pc = self.phase_channel
                if self.flat_method != "none" and pc is not None:
                    fc = FlatfieldCorrector(self.flat_method, self.flat_sigma)
                    if fc.method == "basic":
                        imgs = np.stack([v[z, pc] for t, v in sorted(volumes.items())
                                         for z in range(v.shape[0])])
                        if len(imgs) > 64:
                            sel = np.linspace(0, len(imgs) - 1, 64).astype(int)
                            imgs = imgs[sel]
                        try:
                            fc.fit({0: imgs})       # channel key 0 = phase sub-volume
                        except Exception:
                            fc.method = "none"
                    for t in list(volumes.keys()):
                        vol = volumes[t]
                        if pc < vol.shape[1]:
                            vol[:, pc] = fc.apply_volume(vol[:, [pc]])[:, 0]

                self.progress.emit(base, f"Position {p + 1} ({i+1}/{n}): segmenting…")
                # Per-position ROI (full frame segmented; only organoids fully
                # inside the drawn region are kept).
                self.seg_params = dict(self.seg_params, roi=self.rois.get(p))
                masks = Seg3DWorker(volumes, self.seg_params).segment_now()

                self.progress.emit(base, f"Position {p + 1} ({i+1}/{n}): tracking…")
                track_maps = IoUTracker(masks, self.iou_thresh, self.memory).run()
                # The tracker allows "splits" (two detections in one frame can inherit
                # the same track). Collapse them into ONE object per (frame, track) so an
                # organoid that fragmented in a frame is measured & drawn once — no
                # duplicate (track_id, timepoint) rows in the workbook.
                self._merge_split_detections(masks, track_maps)

                self.progress.emit(base, f"Position {p + 1} ({i+1}/{n}): measuring…")
                all_rows.extend(self._write_position(p, volumes, masks, track_maps))

                del volumes, masks, track_maps   # free before next position

            # One workbook for the whole experiment.
            if all_rows and not self._stop:
                self.progress.emit(99, "Writing combined workbook…")
                self._write_workbook(all_rows)

            self.progress.emit(100, "Batch cancelled." if self._stop else "Batch complete.")
            self.finished.emit(self.out_root)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())

    def _write_workbook(self, rows):
        """Write ALL positions to a single Excel file: channel columns renamed to
        the ND2 channel names, sorted by organoid id then timepoint."""
        import os
        import re
        import pandas as pd
        df = pd.DataFrame(rows)
        rename = {}
        for c, name in enumerate(self.channel_names or []):
            if c == self.phase_channel:
                continue
            ci = c + 1
            nm = re.sub(r"\W+", "", str(name)) or f"ch{ci}"
            rename[f"int_sum_ch{ci}_maxarea"]    = f"{nm}_maxarea_sum"
            rename[f"int_mean_ch{ci}_maxarea"]   = f"{nm}_maxarea_mean"
            rename[f"int_median_ch{ci}_maxarea"] = f"{nm}_maxarea_median"
            rename[f"int_sum_ch{ci}_vol"]        = f"{nm}_vol_sum"
            rename[f"int_mean_ch{ci}_vol"]       = f"{nm}_vol_mean"
            rename[f"int_median_ch{ci}_vol"]     = f"{nm}_vol_median"
            rename[f"int_bg_ch{ci}"]             = f"{nm}_bg"
        df = df.rename(columns=rename)
        sort_cols = [c for c in ("track_id", "timepoint") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        stem = self.experiment or "experiment"
        out = os.path.join(self.out_root, f"{stem}_metrics.xlsx")
        try:
            df.to_excel(out, index=False)
        except Exception:
            df.to_csv(os.path.join(self.out_root, f"{stem}_metrics.csv"), index=False)


# ── Organoid annotation object ────────────────────────────────────────────────

class OrgAnnotation:
    __slots__ = ("org_id", "org_type", "state", "notes", "metrics", "track_id")

    def __init__(self):
        self.org_id   = 0
        self.org_type = ""
        self.state    = "Intact"
        self.notes    = ""
        self.metrics  = {}
        self.track_id = -1   # -1 = not yet tracked

    def to_dict(self):
        return {"org_id": self.org_id, "org_type": self.org_type,
                "state": self.state, "notes": self.notes,
                "metrics": self.metrics, "track_id": self.track_id}

    @staticmethod
    def from_dict(d):
        a = OrgAnnotation()
        a.org_id   = d.get("org_id", 0)
        a.org_type = d.get("org_type", "")
        a.state    = d.get("state", "Intact")
        a.notes    = d.get("notes", "")
        a.metrics  = d.get("metrics", {})
        a.track_id = d.get("track_id", -1)
        return a


# ── Main window ───────────────────────────────────────────────────────────────

class OrgAnnotator(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Organoid Annotator  v{VERSION}")
        self.resize(1400, 900)

        # Data
        self.volumes     = {}   # {t: (Z,C,H,W) float32}
        self.masks       = {}   # {t: (Z,H,W) int32}
        self.labels      = {}   # {t: {org_id: OrgAnnotation}}
        self.source_name = ""

        self.current_t   = 0
        self.current_z   = 0
        self.n_timepoints = 0
        self.n_z          = 0
        self.n_channels   = 0
        self._chan_ranges = []   # per-channel (lo, hi) across all timepoints
        self._corr_volumes = {}  # {t: corrected (Z,C,H,W)} shown after segmentation

        # Multi-position (ND2 'P' axis) support — one position held in RAM at a time
        self._nd2_path    = None    # path of a multi-position ND2 (else None)
        self._nd2_stem    = ""      # experiment name (file stem)
        self._nd2_dims    = []      # axis order of the ND2 (e.g. ['T','P','Z','C','Y','X'])
        self._pos_axis    = None    # index of 'P' in _nd2_dims
        self.n_positions  = 1
        self.current_p    = 0
        self._batch_worker = None
        self._loaded_path = None    # last single-file path (for flatfield re-apply)
        self._loaded_kind = None    # 'nd2' | 'tiff'
        self._channel_names = []
        self._phase_channel = None  # index of the phase channel (OrganoID input)
        self._time_hours = []       # per-timepoint acquisition time (hours)
        self._rois = {}             # {position: (H,W) bool include-mask}

        self._seg_worker   = None
        self._seg_start    = None
        self._track_worker = None
        self._seg_timer   = QTimer(self)
        self._seg_timer.setInterval(200)
        self._seg_timer.timeout.connect(self._update_seg_time)

        self._build_ui()
        self._connect_signals()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)
        main_h.setContentsMargins(4, 4, 4, 4)
        main_h.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_h.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())

        # Canvas + Z-strip vertical splitter
        right_v = QSplitter(Qt.Orientation.Vertical)
        right_v.addWidget(self._build_canvas_area())
        right_v.addWidget(self._build_z_strip())
        right_v.setSizes([700, 120])
        splitter.addWidget(right_v)

        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([260, 860, 280])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — load an ND2 or TIFF stack to begin.")

        self._build_menubar()

    def _build_menubar(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction("Load ND2 File",        self._load_nd2)
        fm.addAction("Load TIFF Z-stack",    self._load_tiff_stack)
        fm.addSeparator()
        fm.addAction("Save Project",         self._save_project)
        fm.addAction("Load Project",         self._load_project)
        fm.addSeparator()
        fm.addAction("Export CSV",           self._export_csv)
        fm.addAction("Export HDF5",          self._export_hdf5)

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(300)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(0)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        outer.addWidget(tabs)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — Load & View
        # ══════════════════════════════════════════════════════════════════════
        tab_load = QWidget()
        tl_layout = QVBoxLayout(tab_load)
        tl_layout.setContentsMargins(6, 6, 6, 6)

        # Load
        grp_load = QGroupBox("Load Data")
        ll = QVBoxLayout(grp_load)
        self.btn_load_nd2  = QPushButton("📂  Load ND2 File")
        self.btn_load_tiff = QPushButton("📂  Load TIFF Z-stack")
        self.lbl_source    = QLabel("No file loaded")
        self.lbl_source.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_source.setWordWrap(True)
        ll.addWidget(self.btn_load_nd2)
        ll.addWidget(self.btn_load_tiff)
        ll.addWidget(self.lbl_source)
        tl_layout.addWidget(grp_load)

        # Positions (multi-position ND2 only) — hidden until such a file loads
        self.grp_positions = QGroupBox("Positions")
        pl = QVBoxLayout(self.grp_positions)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("View:"))
        self.spin_pos = QSpinBox()
        self.spin_pos.setRange(0, 0)
        self.spin_pos.setToolTip("Load and view a single position (all its timepoints).")
        self.lbl_pos = QLabel("— / —")
        prow.addWidget(self.spin_pos)
        prow.addWidget(self.lbl_pos)
        prow.addStretch()
        pl.addLayout(prow)

        pl.addWidget(QLabel("Analyze these positions:"))
        self.list_positions = QListWidget()
        self.list_positions.setMaximumHeight(140)
        self.list_positions.setToolTip(
            "Check the positions to include in 'Run selected positions'.")
        pl.addWidget(self.list_positions)
        selrow = QHBoxLayout()
        self.btn_pos_all  = QPushButton("Select all")
        self.btn_pos_none = QPushButton("Select none")
        selrow.addWidget(self.btn_pos_all)
        selrow.addWidget(self.btn_pos_none)
        pl.addLayout(selrow)

        # Metrics (Excel) are always written for analyzed positions; JPG images
        # are only written for the position(s) listed here (usually one, for a
        # human-vs-segmentation benchmark). Blank = no images.
        imgrow = QHBoxLayout()
        imgrow.addWidget(QLabel("Save JPGs for:"))
        self.edit_img_positions = QLineEdit()
        self.edit_img_positions.setPlaceholderText("e.g. 3  or  3,7  or  all  (blank = none)")
        self.edit_img_positions.setToolTip(
            "Which positions also get mask + overlay JPGs written (per timepoint & Z).\n"
            "Metrics Excel is always written for every analyzed position.\n"
            "Leave blank to write no images; use 'all' to image every analyzed position.")
        imgrow.addWidget(self.edit_img_positions)
        pl.addLayout(imgrow)

        self.btn_run_positions = QPushButton("▶  Run selected positions → disk")
        self.btn_run_positions.setStyleSheet(
            "QPushButton{background:#8a5a00;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:hover{background:#a86e10;}"
            "QPushButton:disabled{background:#B0C0D0;color:#EEF1F6;}")
        pl.addWidget(self.btn_run_positions)
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        self.lbl_batch = QLabel("")
        self.lbl_batch.setVisible(False)
        self.lbl_batch.setWordWrap(True)
        self.lbl_batch.setStyleSheet("font-size:10px; color:gray;")
        pl.addWidget(self.batch_progress)
        pl.addWidget(self.lbl_batch)
        self.grp_positions.setVisible(False)
        tl_layout.addWidget(self.grp_positions)

        # Navigation
        grp_nav = QGroupBox("Navigation")
        nl = QFormLayout(grp_nav)
        nl.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.lbl_t    = QLabel("T: —")
        self.slider_t = QSlider(Qt.Orientation.Horizontal)
        self.slider_t.setMinimum(0); self.slider_t.setMaximum(0)
        self.lbl_z    = QLabel("Z: —")
        self.slider_z = QSlider(Qt.Orientation.Horizontal)
        self.slider_z.setMinimum(0); self.slider_z.setMaximum(0)
        self.chk_max_proj = QCheckBox("Max projection")
        self.chk_max_proj.setToolTip("Show maximum intensity projection across Z")
        nl.addRow(self.lbl_t, self.slider_t)
        nl.addRow(self.lbl_z, self.slider_z)
        nl.addRow(self.chk_max_proj)
        tl_layout.addWidget(grp_nav)

        # Per-channel contrast (0–100, 50 = neutral). Channels map to display R/G/B.
        grp_contrast = QGroupBox("Channel Contrast")
        cl = QFormLayout(grp_contrast)
        cl.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.contrast_sliders = []
        self.contrast_labels  = []
        _ch_names = ["Ch 1 (R)", "Ch 2 (G)", "Ch 3 (B)"]
        for name in _ch_names:
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 100)
            s.setValue(50)
            s.setToolTip("Contrast for this channel (50 = neutral, higher = brighter)")
            lbl = QLabel(name)
            cl.addRow(lbl, s)
            self.contrast_sliders.append(s)
            self.contrast_labels.append(lbl)
        self.grp_contrast = grp_contrast
        tl_layout.addWidget(grp_contrast)

        # Voxel size
        grp_vox = QGroupBox("Voxel Size (µm)")
        vl = QFormLayout(grp_vox)
        vl.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.spin_dz = QDoubleSpinBox(); self.spin_dz.setRange(0.01, 100); self.spin_dz.setValue(1.0); self.spin_dz.setSingleStep(0.1)
        self.spin_dy = QDoubleSpinBox(); self.spin_dy.setRange(0.01, 100); self.spin_dy.setValue(0.5); self.spin_dy.setSingleStep(0.1)
        self.spin_dx = QDoubleSpinBox(); self.spin_dx.setRange(0.01, 100); self.spin_dx.setValue(0.5); self.spin_dx.setSingleStep(0.1)
        vl.addRow("Z (µm/slice):", self.spin_dz)
        vl.addRow("Y (µm/px):",    self.spin_dy)
        vl.addRow("X (µm/px):",    self.spin_dx)
        tl_layout.addWidget(grp_vox)

        tl_layout.addStretch()
        tabs.addTab(tab_load, "Load / View")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — Segment
        # ══════════════════════════════════════════════════════════════════════
        tab_seg = QWidget()
        ts_layout = QVBoxLayout(tab_seg)
        ts_layout.setContentsMargins(6, 6, 6, 6)

        # Model & device
        grp_model = QGroupBox("Model & Device")
        ml = QFormLayout(grp_model)
        ml.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "OrganoID (TensorFlow)",
        ])
        self.combo_device = QComboBox()
        self.combo_device.addItem("CPU")
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                self.combo_device.addItem(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        except Exception:
            pass
        ml.addRow("Model:",  self.combo_model)
        ml.addRow("Device:", self.combo_device)

        # OrganoID weights file (.npz produced by convert_organoid_weights.py)
        oid_row = QHBoxLayout()
        # Default to the fine-tuned model if present.
        self._oid_weights_path = DEFAULT_MODEL_DIR if os.path.isdir(DEFAULT_MODEL_DIR) else ""
        self.lbl_oid_weights = QLabel(Path(self._oid_weights_path).name
                                      if self._oid_weights_path else "(no model)")
        self.lbl_oid_weights.setWordWrap(False)
        self.lbl_oid_weights.setToolTip(self._oid_weights_path or
                                        "Path to a fine-tuned OrganoID SavedModel folder")
        btn_oid_browse = QPushButton("Browse…")
        btn_oid_browse.setFixedWidth(80)
        btn_oid_browse.clicked.connect(self._browse_oid_weights)
        oid_row.addWidget(self.lbl_oid_weights, 1)
        oid_row.addWidget(btn_oid_browse)
        self._oid_weights_row_widget = QWidget()
        self._oid_weights_row_widget.setLayout(oid_row)
        ml.addRow("OrganoID model:", self._oid_weights_row_widget)

        # Only OrganoID is offered, so the model row is always shown.
        self._oid_weights_row_widget.setVisible(True)
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)

        ts_layout.addWidget(grp_model)

        # OrganoID parameters
        grp_cp = QGroupBox("OrganoID Parameters")
        cl = QFormLayout(grp_cp)
        cl.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.spin_min_vol = QSpinBox()
        self.spin_min_vol.setRange(1, 100000); self.spin_min_vol.setValue(25)
        self.spin_min_vol.setToolTip(
            "Minimum organoid size in pixels — smaller detections are discarded.")
        cl.addRow("Min organoid size (px):", self.spin_min_vol)
        self.chk_watershed = QCheckBox("Separate touching organoids (watershed)")
        self.chk_watershed.setChecked(False)
        self.chk_watershed.setToolTip(
            "When ON, a watershed step splits touching organoids. When OFF, touching\n"
            "organoids stay merged (avoids over-fragmentation). Default: OFF.")
        cl.addRow(self.chk_watershed)

        # Illumination correction (applied to the phase/segmentation channel only,
        # to help where the field is unevenly lit — e.g. organoids near an edge).
        self.combo_illum = QComboBox()
        self.combo_illum.addItems(ILLUM_LABELS)
        self.combo_illum.setToolTip(
            "Flatten uneven background illumination on the segmentation channel before\n"
            "OrganoID runs — try it when organoids near the chamber edge segment poorly.\n"
            "  • Top-hat (opening): subtract a morphological-opening background.\n"
            "  • Rolling ball: subtract a rolling-ball background estimate.\n"
            "  • CLAHE: normalise local contrast across the field.\n"
            "Only affects segmentation; fluorescence intensity metrics use the raw image.\n"
            "Run 'Segment Current Timepoint' to compare against 'None'.")
        cl.addRow("Illumination correction:", self.combo_illum)

        self.spin_illum_radius = QSpinBox()
        self.spin_illum_radius.setRange(5, 500); self.spin_illum_radius.setValue(50)
        self.spin_illum_radius.setToolTip(
            "Background scale in pixels for Top-hat / Rolling ball. Make it larger than\n"
            "an organoid so real organoids are kept, not removed as background. Ignored\n"
            "by CLAHE.")
        cl.addRow("Illumination radius (px):", self.spin_illum_radius)

        # Flat-field / shading correction — applied to the PHASE (segmentation)
        # channel only, so it never changes fluorescence intensity metrics.
        self.combo_flat = QComboBox()
        self.combo_flat.addItems(FLAT_LABELS)
        self.combo_flat.setToolTip(
            "Correct uneven illumination on the phase (segmentation) channel.\n"
            "Like 'Illumination correction', this touches ONLY the phase channel, so\n"
            "fluorescence intensity metrics are unaffected.\n"
            "  • Gaussian (imflatfield): divide out a Gaussian-blurred shading estimate.\n"
            "  • BaSiC: estimate flatfield + darkfield from the frames (needs basicpy).\n"
            "Applies to interactive segmentation AND the position batch run.")
        cl.addRow("Flat-field:", self.combo_flat)

        self.spin_flat_sigma = QDoubleSpinBox()
        self.spin_flat_sigma.setRange(1.0, 1000.0); self.spin_flat_sigma.setValue(80.0)
        self.spin_flat_sigma.setSingleStep(10.0)
        self.spin_flat_sigma.setToolTip(
            "Gaussian shading sigma in pixels (imflatfield only). Larger = smoother\n"
            "shading estimate. Ignored by None / BaSiC.")
        cl.addRow("Flat-field σ (px):", self.spin_flat_sigma)
        ts_layout.addWidget(grp_cp)

        # 3D mode
        grp_mode = QGroupBox("3D Segmentation Mode")
        mo = QVBoxLayout(grp_mode)
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup
        self.radio_fast    = QRadioButton("Max-Z projection (fast)")
        self.radio_precise = QRadioButton("Per-Z slices + stitch (recommended)")
        self.radio_precise.setChecked(True)
        self.radio_fast.setToolTip(
            "Segment a single max-intensity Z projection, then expand the 2-D mask\n"
            "across Z by intensity. Faster; one shape per organoid.")
        self.radio_precise.setToolTip(
            "Segment EVERY Z slice independently, then link masks across Z by IoU\n"
            "so each organoid keeps one identity through the stack (true per-plane).")
        self._mode_group = QButtonGroup()
        self._mode_group.addButton(self.radio_fast,    0)
        self._mode_group.addButton(self.radio_precise, 1)
        mo.addWidget(self.radio_fast)
        mo.addWidget(self.radio_precise)

        stitch_row = QFormLayout()
        stitch_row.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.spin_stitch = QDoubleSpinBox()
        self.spin_stitch.setRange(0.01, 1.0); self.spin_stitch.setSingleStep(0.05)
        self.spin_stitch.setValue(0.5)
        self.spin_stitch.setToolTip("IoU threshold for Z-slice stitching (Precise / OrganoID per-Z mode)")
        stitch_row.addRow("Stitch threshold:", self.spin_stitch)
        mo.addLayout(stitch_row)

        self.chk_multiscale = QCheckBox("Multi-scale detection (recover very large organoids)")
        self.chk_multiscale.setChecked(False)
        self.chk_multiscale.setToolTip(
            "Also run a zoomed-OUT pass so organoids much larger than the training data\n"
            "(which the normal pass truncates/fragments) are recovered as one solid\n"
            "object. Large organoids come from the zoomed-out pass; normal/small ones\n"
            "from the standard pass. Works with either 3-D mode above. Roughly doubles\n"
            "segmentation time (an extra inference per frame/slice).")
        mo.addWidget(self.chk_multiscale)
        ts_layout.addWidget(grp_mode)

        # Frame range
        grp_range = QGroupBox("Frame Range")
        rl = QFormLayout(grp_range)
        rl.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.spin_t_start = QSpinBox(); self.spin_t_start.setMinimum(0)
        self.spin_t_end   = QSpinBox(); self.spin_t_end.setMinimum(0)
        rl.addRow("From T:", self.spin_t_start)
        rl.addRow("To T:",   self.spin_t_end)
        ts_layout.addWidget(grp_range)

        # Run buttons
        btn_style_blue = (
            "QPushButton{background:#2e6da4;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:hover{background:#3a82c3;}"
            "QPushButton:disabled{background:#B0C0D0;color:#EEF1F6;}"
        )
        btn_style_red = (
            "QPushButton{background:#c0392b;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:disabled{background:#B0C0D0;color:#EEF1F6;}"
        )
        self.btn_seg_frame = QPushButton("▶  Segment Current Timepoint")
        self.btn_seg_frame.setStyleSheet(btn_style_blue)
        self.btn_seg_all   = QPushButton("▶▶  Segment All Timepoints")
        self.btn_seg_all.setStyleSheet(btn_style_blue)
        self.btn_stop      = QPushButton("■  Stop")
        self.btn_stop.setStyleSheet(btn_style_red)
        self.btn_stop.setEnabled(False)

        self.seg_progress  = QProgressBar()
        self.seg_progress.setVisible(False)
        self.lbl_seg_stage = QLabel("")
        self.lbl_seg_stage.setVisible(False)
        self.lbl_seg_stage.setWordWrap(True)
        self.lbl_seg_stage.setStyleSheet("font-size:10px; color:gray;")
        self.lbl_seg_time  = QLabel("")
        self.lbl_seg_time.setVisible(False)
        self.lbl_seg_time.setStyleSheet("font-size:10px; color:gray;")

        ts_layout.addWidget(self.btn_seg_frame)
        ts_layout.addWidget(self.btn_seg_all)
        ts_layout.addWidget(self.btn_stop)
        ts_layout.addWidget(self.seg_progress)
        ts_layout.addWidget(self.lbl_seg_stage)
        ts_layout.addWidget(self.lbl_seg_time)
        ts_layout.addStretch()

        tabs.addTab(tab_seg, "Segment")
        return panel

    def _build_canvas_area(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        vl = QVBoxLayout(frame)
        vl.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        tb = QHBoxLayout()
        self.chk_show_masks = QCheckBox("Show masks")
        self.chk_show_masks.setChecked(True)
        self.chk_show_ids   = QCheckBox("Show IDs")
        self.chk_show_ids.setChecked(True)
        self.btn_roi = QPushButton("✏  Draw ROI")
        self.btn_roi.setCheckable(True)
        self.btn_roi.setToolTip(
            "Pencil: draw a region of interest to select. Left-drag a loop; each loop\n"
            "adds to the ROI. The whole frame is segmented; only organoids lying\n"
            "ENTIRELY within the ROI are kept (anything outside or crossing the\n"
            "boundary is discarded), across ALL timepoints.\n"
            "The ROI is per XY position. Middle-drag pans; wheel zooms.")
        self.btn_roi_clear = QPushButton("Clear ROI")
        self.btn_roi_clear.setToolTip("Remove the ROI for the current position (analyze the full field).")
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setFixedWidth(40)
        tb.addWidget(self.chk_show_masks)
        tb.addWidget(self.chk_show_ids)
        tb.addWidget(self.btn_roi)
        tb.addWidget(self.btn_roi_clear)
        tb.addStretch()
        tb.addWidget(self.btn_fit)
        vl.addLayout(tb)

        self.canvas = OrganoCanvas()
        vl.addWidget(self.canvas)
        return frame

    def _build_metric_plot(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        vl = QVBoxLayout(frame)
        vl.setContentsMargins(4, 2, 4, 2)
        vl.setSpacing(2)

        top = QHBoxLayout()
        top.addWidget(QLabel("Metric over time:"))
        self.combo_plot_metric = QComboBox()
        top.addWidget(self.combo_plot_metric)
        top.addStretch()
        vl.addLayout(top)

        if _HAS_MPL:
            self._plot_fig    = Figure(figsize=(4, 2), tight_layout=True)
            self.plot_canvas  = FigureCanvasQTAgg(self._plot_fig)
            self._plot_ax     = self._plot_fig.add_subplot(111)
            vl.addWidget(self.plot_canvas)
        else:
            self.plot_canvas = None
            vl.addWidget(QLabel("matplotlib not installed — plot unavailable.\n"
                                "Install with:  pip install matplotlib"))

        self._populate_metric_combo()
        return frame

    def _channel_name(self, c):
        """1-based channel index -> display name (falls back to 'Ch{c}')."""
        return (self._channel_names[c - 1]
                if c - 1 < len(self._channel_names) else f"Ch{c}")

    def _metric_label(self, key):
        """Human-readable label for a metric key (shared table + plot)."""
        if key in METRIC_LABELS:
            return METRIC_LABELS[key]
        import re
        m = re.match(r"int_(sum|mean|median)_ch(\d+)_(maxarea|vol)$", key)
        if m:
            kind = {"sum": "Sum", "mean": "Mean", "median": "Median"}[m.group(1)]
            scope = "max-area plane" if m.group(3) == "maxarea" else "volume"
            return f"{kind} {self._channel_name(int(m.group(2)))} ({scope})"
        m = re.match(r"int_bg_ch(\d+)$", key)
        if m:
            return f"Background {self._channel_name(int(m.group(1)))}"
        return key

    def _present_metric_keys(self):
        """Union of every metric key currently on any annotation."""
        present = set()
        for labs in self.labels.values():
            for ann in labs.values():
                present.update(ann.metrics.keys())
        return present

    def _ordered_metric_keys(self, present):
        """Metric keys in a stable display order: geometry block, then per-channel
        intensity (max-area then volume), then background, then leftovers. Guarantees
        the table and plot expose exactly the metrics that land in the Excel file."""
        import re
        ordered = [k for k in _METRIC_ORDER if k in present]
        chans = sorted({int(m.group(1)) for k in present
                        for m in [re.match(r"int_\w+_ch(\d+)_(maxarea|vol)$", k)] if m})
        for c in chans:
            for pat in (f"int_sum_ch{c}_maxarea", f"int_mean_ch{c}_maxarea",
                        f"int_median_ch{c}_maxarea", f"int_sum_ch{c}_vol",
                        f"int_mean_ch{c}_vol", f"int_median_ch{c}_vol"):
                if pat in present:
                    ordered.append(pat)
        for c in chans:
            if f"int_bg_ch{c}" in present:
                ordered.append(f"int_bg_ch{c}")
        for k in sorted(present):
            if k not in ordered:
                ordered.append(k)
        return ordered

    def _populate_metric_combo(self):
        """Rebuild the metric selector from ALL available metrics (== Excel columns)."""
        if not hasattr(self, "combo_plot_metric"):
            return
        prev = self.combo_plot_metric.currentData()
        present = self._present_metric_keys()
        if not present:
            present = {"volume_um3", "area_um2", "diameter_um"}   # before any segmentation
        keys = self._ordered_metric_keys(present)
        self.combo_plot_metric.blockSignals(True)
        self.combo_plot_metric.clear()
        for k in keys:
            self.combo_plot_metric.addItem(self._metric_label(k), k)
        if prev is not None:
            idx = self.combo_plot_metric.findData(prev)
            if idx >= 0:
                self.combo_plot_metric.setCurrentIndex(idx)
        self.combo_plot_metric.blockSignals(False)

    def _series_for_selected(self, metric_key):
        """
        Return (ts, values) for the currently selected organoid across time.

        If the organoid is tracked (track_id >= 1), follow that track across all
        timepoints. Otherwise return just the single current-frame value.
        """
        oid = self.canvas.selected
        if oid == 0:
            return [], []
        cur_ann = self.labels.get(self.current_t, {}).get(oid)
        track_id = cur_ann.track_id if cur_ann else -1

        def _backfill(t, o, ann):
            if metric_key.startswith("int_"):
                self._ensure_intensity(t, o, ann)

        ts, vals = [], []
        if track_id >= 1:
            for t in sorted(self.labels.keys()):
                for o, ann in self.labels[t].items():
                    if ann.track_id != track_id:
                        continue
                    _backfill(t, o, ann)
                    if metric_key in ann.metrics:
                        ts.append(t)
                        vals.append(ann.metrics[metric_key])
                    break
        elif cur_ann:
            _backfill(self.current_t, oid, cur_ann)
            if metric_key in cur_ann.metrics:
                ts.append(self.current_t)
                vals.append(cur_ann.metrics[metric_key])
        return ts, vals

    def _update_metric_plot(self):
        if not _HAS_MPL or self.plot_canvas is None:
            return
        ax = self._plot_ax
        ax.clear()

        key   = self.combo_plot_metric.currentData()
        label = self.combo_plot_metric.currentText()
        if key is None:
            self.plot_canvas.draw_idle()
            return

        ts, vals = self._series_for_selected(key)
        oid = self.canvas.selected

        if ts:
            ax.plot(ts, vals, marker="o", color="#1f77b4", linewidth=1.5, markersize=4)
            # Highlight the current timepoint with a vertical line
            ax.axvline(self.current_t, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
            cur_ann  = self.labels.get(self.current_t, {}).get(oid)
            track_id = cur_ann.track_id if cur_ann else -1
            title = f"Track T{track_id}" if track_id >= 1 else f"Organoid #{oid} (untracked)"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("Timepoint", fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            msg = "Select an organoid" if oid == 0 else "No data for this metric"
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xticks([]); ax.set_yticks([])

        self.plot_canvas.draw_idle()

    def _build_z_strip(self):
        """Horizontal thumbnail strip of Z slices."""
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setMaximumHeight(130)
        vl = QVBoxLayout(frame)
        vl.setContentsMargins(2, 2, 2, 2)
        lbl = QLabel("Timepoints")
        lbl.setStyleSheet("font-size:10px; color:gray;")
        vl.addWidget(lbl)

        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        self._zstrip_layout = QHBoxLayout(inner)
        self._zstrip_layout.setContentsMargins(2, 2, 2, 2)
        self._zstrip_layout.setSpacing(2)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        vl.addWidget(scroll)
        self._zstrip_buttons = []
        return frame

    def _build_right_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(280)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(0)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        outer.addWidget(tabs)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — Annotate  (list + annotation fields + metrics + summary)
        # ══════════════════════════════════════════════════════════════════════
        tab_ann = QWidget()
        ta_layout = QVBoxLayout(tab_ann)
        ta_layout.setContentsMargins(6, 6, 6, 6)

        # Organoid list
        grp_list = QGroupBox("Organoids (current timepoint)")
        gl = QVBoxLayout(grp_list)
        self.org_list = QListWidget()
        self.org_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        gl.addWidget(self.org_list)
        row = QHBoxLayout()
        self.btn_del_org = QPushButton("🗑  Delete")
        self.btn_relabel = QPushButton("🔢  Relabel")
        row.addWidget(self.btn_del_org)
        row.addWidget(self.btn_relabel)
        gl.addLayout(row)
        ta_layout.addWidget(grp_list)

        # Annotation
        grp_ann = QGroupBox("Annotation")
        al = QFormLayout(grp_ann)
        al.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.combo_type  = QComboBox()
        self.combo_type.addItems([""] + ORGANOID_TYPES)
        self.combo_state = QComboBox()
        self.combo_state.addItems(ORGANOID_STATES)
        self.edit_notes  = QLineEdit()
        self.btn_apply   = QPushButton("✔  Apply")
        self.btn_apply.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;padding:4px;border-radius:3px;}"
        )
        al.addRow("Type:",  self.combo_type)
        al.addRow("State:", self.combo_state)
        al.addRow("Notes:", self.edit_notes)
        al.addRow(self.btn_apply)
        ta_layout.addWidget(grp_ann)

        # Summary
        grp_sum = QGroupBox("Summary")
        sl = QFormLayout(grp_sum)
        self.lbl_n_total    = QLabel("0")
        self.lbl_n_labelled = QLabel("0")
        sl.addRow("Total organoids:", self.lbl_n_total)
        sl.addRow("Labelled:",        self.lbl_n_labelled)
        ta_layout.addWidget(grp_sum)

        ta_layout.addStretch()
        tabs.addTab(tab_ann, "Annotate")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — Metrics
        # ══════════════════════════════════════════════════════════════════════
        tab_met = QWidget()
        tm_layout = QVBoxLayout(tab_met)
        tm_layout.setContentsMargins(6, 6, 6, 6)

        grp_metrics = QGroupBox("3D Metrics (selected organoid)")
        ml = QVBoxLayout(grp_metrics)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        ml.addWidget(self.metrics_table)
        tm_layout.addWidget(grp_metrics)

        # Metric-over-time plot lives in this same Metrics tab (below the table)
        tm_layout.addWidget(self._build_metric_plot(), stretch=1)
        tabs.addTab(tab_met, "Metrics")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 — Track & Export
        # ══════════════════════════════════════════════════════════════════════
        tab_trk = QWidget()
        tt_layout = QVBoxLayout(tab_trk)
        tt_layout.setContentsMargins(6, 6, 6, 6)

        # Tracking
        grp_track = QGroupBox("Tracking (IoU overlap)")
        tl = QFormLayout(grp_track)
        tl.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.05, 0.95); self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.30)
        self.spin_iou.setToolTip(
            "Minimum mask overlap (IoU) between frames to link the same organoid.\n"
            "Objects are matched by how much their segmentation masks overlap\n"
            "frame-to-frame, not by centroid distance.\n"
            "Higher = stricter (fewer wrong links); lower = more tolerant of\n"
            "motion / shape change. 0.3 works well for most timelapses.")

        self.spin_memory = QSpinBox()
        self.spin_memory.setRange(0, 10); self.spin_memory.setValue(2)
        self.spin_memory.setToolTip(
            "Gap closing: how many consecutive frames an organoid can be\n"
            "missing (e.g. missed by segmentation) and still be re-linked.\n"
            "0 = no gap closing.")

        self.chk_color_by_track = QCheckBox("Color by track ID")
        self.chk_color_by_track.setChecked(False)
        self.chk_color_by_track.setToolTip(
            "Show a unique stable color per track ID after running tracking.\n"
            "When unchecked, colors come from the organoid type annotation.")

        self.btn_track = QPushButton("▶  Track All Timepoints")
        self.btn_track.setStyleSheet(
            "QPushButton{background:#6a3d9a;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:hover{background:#8a5dba;}"
            "QPushButton:disabled{background:#B0C0D0;color:#EEF1F6;}"
        )
        self.btn_clear_tracks = QPushButton("✖  Clear Tracks")

        self.track_progress  = QProgressBar()
        self.track_progress.setVisible(False)
        self.track_progress.setRange(0, 100)
        self.lbl_track_stage = QLabel("")
        self.lbl_track_stage.setVisible(False)
        self.lbl_track_stage.setWordWrap(True)
        self.lbl_track_stage.setStyleSheet("font-size: 10px; color: gray;")

        tl.addRow("Min overlap (IoU):", self.spin_iou)
        tl.addRow("Gap closing:",       self.spin_memory)
        tl.addRow(self.chk_color_by_track)
        tl.addRow(self.btn_track)
        tl.addRow(self.btn_clear_tracks)
        tl.addRow(self.track_progress)
        tl.addRow(self.lbl_track_stage)
        tt_layout.addWidget(grp_track)

        # Export
        grp_exp = QGroupBox("Export & Project")
        el = QVBoxLayout(grp_exp)
        self.btn_export_csv  = QPushButton("📊  Export CSV")
        self.btn_export_hdf5 = QPushButton("🗄  Export HDF5")
        self.btn_export_hdf5.setStyleSheet(
            "QPushButton{background:#2a7a4b;color:white;font-weight:bold;padding:6px;border-radius:4px;}"
            "QPushButton:hover{background:#339960;}"
        )
        self.btn_save = QPushButton("💾  Save Project")
        self.btn_load = QPushButton("📂  Load Project")
        el.addWidget(self.btn_export_csv)
        el.addWidget(self.btn_export_hdf5)
        el.addWidget(self.btn_save)
        el.addWidget(self.btn_load)
        tt_layout.addWidget(grp_exp)

        tt_layout.addStretch()
        tabs.addTab(tab_trk, "Track & Export")

        return panel

    def _connect_signals(self):
        self.btn_load_nd2.clicked.connect(self._load_nd2)
        self.btn_load_tiff.clicked.connect(self._load_tiff_stack)
        self.slider_t.valueChanged.connect(self._on_t_changed)
        self.slider_z.valueChanged.connect(self._on_z_changed)
        self.chk_max_proj.toggled.connect(self._refresh_canvas)
        self.chk_show_masks.toggled.connect(self._toggle_masks)
        self.chk_show_ids.toggled.connect(self._toggle_ids)
        self.btn_fit.clicked.connect(self.canvas._fit_to_window)
        self.btn_seg_frame.clicked.connect(self._run_seg_frame)
        self.btn_seg_all.clicked.connect(self._run_seg_all)
        self.btn_stop.clicked.connect(self._stop_seg)
        self.btn_apply.clicked.connect(self._apply_label)
        self.btn_del_org.clicked.connect(self._delete_organoid)
        self.btn_relabel.clicked.connect(self._relabel_organoids)
        self.org_list.currentRowChanged.connect(self._on_list_select)
        self.canvas.organoid_selected.connect(self._on_canvas_select)
        self.canvas.roi_changed.connect(self._on_roi_changed)
        self.btn_roi.toggled.connect(self._toggle_roi_mode)
        self.btn_roi_clear.clicked.connect(self._clear_roi)
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_export_hdf5.clicked.connect(self._export_hdf5)
        self.btn_save.clicked.connect(self._save_project)
        self.btn_load.clicked.connect(self._load_project)
        self.btn_track.clicked.connect(self._run_tracking)
        self.btn_clear_tracks.clicked.connect(self._clear_tracks)
        self.chk_color_by_track.toggled.connect(self._refresh_canvas)
        for s in self.contrast_sliders:
            s.valueChanged.connect(self._refresh_canvas)
        self.combo_plot_metric.currentIndexChanged.connect(self._update_metric_plot)
        self.spin_pos.valueChanged.connect(self._on_pos_view_changed)
        self.btn_pos_all.clicked.connect(self._select_all_positions)
        self.btn_pos_none.clicked.connect(self._select_no_positions)
        self.btn_run_positions.clicked.connect(self._run_positions)

    # ── File loading ──────────────────────────────────────────────────────────

    def _load_nd2(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ND2 File", "", "ND2 files (*.nd2);;All files (*)"
        )
        if path:
            self._load_nd2_from(path)

    def _load_nd2_from(self, path):
        try:
            import nd2
        except ImportError:
            QMessageBox.critical(self, "nd2 not installed",
                "Install with:\n    pip install nd2")
            return
        try:
            self.status_bar.showMessage(f"Reading {os.path.basename(path)} metadata…")
            QApplication.processEvents()
            with nd2.ND2File(path) as f:
                sizes = dict(f.sizes)   # e.g. {'T':37,'P':208,'Z':9,'C':3,'Y':1022,'X':1024}
                dims  = list(f.sizes.keys())
                try:
                    cal = f.voxel_size()
                    self.spin_dz.setValue(round(cal.z, 4))
                    self.spin_dy.setValue(round(cal.y, 4))
                    self.spin_dx.setValue(round(cal.x, 4))
                except Exception:
                    pass
                # Identify the phase-contrast channel (OrganoID segmentation input).
                names = []
                try:
                    names = [c.channel.name for c in f.metadata.channels]
                except Exception:
                    pass
                self._channel_names = names
                self._phase_channel = guess_phase_channel(names)
                # Per-timepoint acquisition time (hours) from the TimeLoop period.
                try:
                    th = []
                    for lp in f.experiment:
                        if type(lp).__name__ == "TimeLoop":
                            per_ms = float(lp.parameters.periodMs)
                            cnt = int(getattr(lp, "count", sizes.get("T", 1)))
                            th = [round(i * per_ms / 3_600_000.0, 4) for i in range(cnt)]
                            break
                    self._time_hours = th
                except Exception:
                    self._time_hours = []
                if names:
                    self.status_bar.showMessage(
                        f"Channels {names} — segmenting on '{names[self._phase_channel]}'")

            self._loaded_path = path
            self._loaded_kind = "nd2"
            if sizes.get("P", 1) > 1:
                # Multi-position acquisition — far too large to hold all at once.
                # Load one position at a time, lazily, via dask.
                self._nd2_path   = path
                self._nd2_stem   = Path(path).stem
                self._nd2_dims   = dims
                self._pos_axis   = dims.index("P")
                self.n_positions = int(sizes["P"])
                self._setup_positions()
            else:
                self._reset_positions()
                self.status_bar.showMessage(f"Loading {os.path.basename(path)}…")
                QApplication.processEvents()
                with nd2.ND2File(path) as f:
                    data = f.asarray()
                self._ingest_volume_array(data, sizes, Path(path).stem)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    # ── Multi-position handling ───────────────────────────────────────────────

    def _reset_positions(self):
        """Return to single-dataset mode (hide the Positions panel)."""
        self._nd2_path   = None
        self._nd2_dims   = []
        self._pos_axis   = None
        self.n_positions = 1
        self.current_p   = 0
        self.grp_positions.setVisible(False)

    def _setup_positions(self):
        """Configure the Positions panel for a freshly-opened multi-position ND2."""
        self.grp_positions.setVisible(True)
        self.spin_pos.blockSignals(True)
        self.spin_pos.setRange(0, self.n_positions - 1)
        self.spin_pos.setValue(0)
        self.spin_pos.blockSignals(False)
        self.current_p = 0
        self._populate_position_list()
        self._load_position(0)

    def _populate_position_list(self):
        self.list_positions.blockSignals(True)
        self.list_positions.clear()
        for p in range(self.n_positions):
            it = QListWidgetItem(f"Position {p + 1}")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            it.setData(Qt.ItemDataRole.UserRole, p)
            self.list_positions.addItem(it)
        self.list_positions.blockSignals(False)

    def _selected_positions(self):
        out = []
        for i in range(self.list_positions.count()):
            it = self.list_positions.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(int(it.data(Qt.ItemDataRole.UserRole)))
        return out

    def _select_all_positions(self):
        for i in range(self.list_positions.count()):
            self.list_positions.item(i).setCheckState(Qt.CheckState.Checked)

    def _select_no_positions(self):
        for i in range(self.list_positions.count()):
            self.list_positions.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _read_position_array(self, p):
        """Lazily read one position from the ND2 as a canonical (T,Z,C,Y,X) array."""
        import nd2
        with nd2.ND2File(self._nd2_path) as f:
            darr = f.to_dask()
            idx = [slice(None)] * darr.ndim
            idx[self._pos_axis] = p
            sub = np.asarray(darr[tuple(idx)])
        # sub axes are _nd2_dims minus 'P'; reorder to canonical T,Z,C,Y,X
        dims_noP  = [d for d in self._nd2_dims if d != "P"]
        canonical = [d for d in ["T", "Z", "C", "Y", "X"] if d in dims_noP]
        order     = [dims_noP.index(d) for d in canonical]
        sub       = np.transpose(sub, order)
        sizes     = {d: sub.shape[i] for i, d in enumerate(canonical)}
        return sub, sizes

    def _load_position(self, p):
        """Load a single position into the working set (interactive view)."""
        if self._nd2_path is None:
            return
        self.current_p = p
        self.lbl_pos.setText(f"{p + 1} / {self.n_positions}")
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        self.status_bar.showMessage(f"Loading position {p + 1} (this can take ~30 s)…")
        QApplication.processEvents()
        try:
            sub, sizes = self._read_position_array(p)
            self._ingest_volume_array(sub, sizes, f"{self._nd2_stem}_P{p + 1:04d}")
        except Exception as e:
            QMessageBox.critical(self, "Position load error", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _on_pos_view_changed(self, p):
        self._load_position(int(p))

    # ── Position batch run ────────────────────────────────────────────────────

    def _parse_image_positions(self, analyzed):
        """
        Parse the 'Save JPGs for' field into "all", or a set of position indices.
        Accepts: '' (none), 'all', '3', '3,7,10', or ranges like '3-6'.
        Only positions that are also being analyzed are kept.
        """
        txt = self.edit_img_positions.text().strip().lower()
        if not txt:
            return set()
        if txt == "all":
            return "all"
        wanted = set()
        for part in txt.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                try:
                    a, b = part.split("-")
                    wanted.update(range(int(a), int(b) + 1))
                except ValueError:
                    continue
            else:
                try:
                    wanted.add(int(part))
                except ValueError:
                    continue
        return wanted & set(analyzed)

    def _run_positions(self):
        if self._nd2_path is None:
            QMessageBox.warning(self, "No positions",
                "Load a multi-position ND2 file first.")
            return
        positions = self._selected_positions()
        if not positions:
            QMessageBox.warning(self, "No positions selected",
                "Check at least one position to analyze.")
            return

        seg_params = self._get_seg_params()
        if seg_params.get("model") == "organoid_id" and not seg_params.get("oid_weights_path"):
            QMessageBox.warning(self, "OrganoID weights",
                "Select the OrganoID .npz weights (Segment tab) before running.")
            return

        # Which positions also get JPG images (default: none — metrics only)
        img_positions = self._parse_image_positions(positions)

        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder for results")
        if not out_dir:
            return
        out_root = os.path.join(out_dir, self._nd2_stem or "experiment")

        if img_positions == "all":
            img_note = f"mask/overlay JPGs for ALL {len(positions)} positions"
        elif img_positions:
            img_note = f"JPGs for position(s): {sorted(img_positions)}"
        else:
            img_note = "no JPG images (metrics Excel only)"
        reply = QMessageBox.question(
            self, "Run position batch",
            f"Process {len(positions)} position(s) with model "
            f"'{seg_params.get('model')}'?\n\n"
            f"A single combined workbook ({self._nd2_stem or 'experiment'}_metrics.xlsx) "
            f"will be written for the whole experiment, plus\n{img_note}.\n\n"
            f"Output folder: {out_root}\\\n\n"
            "This can take a long time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        voxel  = (self.spin_dz.value(), self.spin_dy.value(), self.spin_dx.value())
        flat_method = FLAT_METHODS[self.combo_flat.currentIndex()]

        self._batch_worker = PositionBatchWorker(
            positions, self._nd2_path, self._nd2_dims, self._pos_axis, out_root,
            seg_params, self.spin_iou.value(), self.spin_memory.value(), voxel,
            flat_method=flat_method, flat_sigma=self.spin_flat_sigma.value(),
            image_positions=img_positions, experiment=(self._nd2_stem or "experiment"),
            rois=self._rois)
        self._batch_worker.phase_channel = self._phase_channel
        self._batch_worker.time_hours = getattr(self, "_time_hours", [])
        self._batch_worker.channel_names = list(getattr(self, "_channel_names", []))
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.error.connect(self._on_batch_error)

        self.btn_run_positions.setEnabled(False)
        self.spin_pos.setEnabled(False)
        self.batch_progress.setValue(0)
        self.batch_progress.setVisible(True)
        self.lbl_batch.setVisible(True)
        self.lbl_batch.setText("Starting…")
        self.status_bar.showMessage("Batch processing positions…")
        self._batch_worker.start()

    def _on_batch_progress(self, pct, msg):
        self.batch_progress.setValue(pct)
        self.lbl_batch.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_batch_finished(self, out_root):
        self.batch_progress.setVisible(False)
        self.lbl_batch.setVisible(False)
        self.btn_run_positions.setEnabled(True)
        self.spin_pos.setEnabled(True)
        QMessageBox.information(self, "Batch complete",
            f"Results written under:\n{out_root}")
        self.status_bar.showMessage(f"Batch complete — {out_root}")

    def _on_batch_error(self, msg):
        self.batch_progress.setVisible(False)
        self.lbl_batch.setVisible(False)
        self.btn_run_positions.setEnabled(True)
        self.spin_pos.setEnabled(True)
        QMessageBox.critical(self, "Batch error", msg)

    def _load_tiff_stack(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load TIFF Z-stack", "", "TIFF files (*.tif *.tiff)"
        )
        if path:
            self._load_tiff_from(path)

    def _load_tiff_from(self, path):
        try:
            self.status_bar.showMessage(f"Loading {os.path.basename(path)}…")
            QApplication.processEvents()
            with tifffile.TiffFile(path) as tif:
                data = tif.asarray()
                axes = ""
                try:
                    axes = tif.series[0].axes.upper()
                except Exception:
                    pass
            sizes = self._infer_sizes(data, axes)
            self._reset_positions()
            self._loaded_path = path
            self._loaded_kind = "tiff"
            self._ingest_volume_array(data, sizes, Path(path).stem)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _infer_sizes(self, data, axes):
        """Build a sizes dict from a TIFF array and optional axes string."""
        if axes and len(axes) == data.ndim:
            return {a: data.shape[i] for i, a in enumerate(axes)}
        # Heuristic
        if data.ndim == 2:   return {"Y": data.shape[0], "X": data.shape[1]}
        if data.ndim == 3:   return {"Z": data.shape[0], "Y": data.shape[1], "X": data.shape[2]}
        if data.ndim == 4:   return {"Z": data.shape[0], "C": data.shape[1], "Y": data.shape[2], "X": data.shape[3]}
        if data.ndim == 5:   return {"T": data.shape[0], "Z": data.shape[1], "C": data.shape[2], "Y": data.shape[3], "X": data.shape[4]}
        return {}

    def _ingest_volume_array(self, data, sizes, stem):
        """Reshape raw array into {t: (Z,C,H,W)} volumes and reset state."""
        # Normalise axis order to T, Z, C, Y, X
        # Common ND2 layouts: TZCYX, TZYX, TCYX, ZYX, TCYX, CYX, YX
        T = sizes.get("T", 1)
        Z = sizes.get("Z", 1)
        C = sizes.get("C", 1)

        arr = data.astype(np.float32)

        # Reshape to (T, Z, C, H, W)
        if arr.ndim == 2:
            arr = arr[None, None, None]          # → (1,1,1,H,W)
        elif arr.ndim == 3:
            if "T" not in sizes and "Z" not in sizes:
                arr = arr[None, None]            # (C,H,W) → (1,1,C,H,W) if C small
            elif "Z" in sizes:
                arr = arr[None, :, None]         # (Z,H,W) → (1,Z,1,H,W)
        elif arr.ndim == 4:
            if "T" in sizes and "Z" in sizes:
                arr = arr[:, :, None]            # (T,Z,H,W) → (T,Z,1,H,W)
            elif "Z" in sizes and "C" in sizes:
                arr = arr[None]                  # (Z,C,H,W) → (1,Z,C,H,W)
            elif "T" in sizes and "C" in sizes:
                arr = arr[:, None]               # (T,C,H,W) → (T,1,C,H,W)
        # arr should now be (T, Z, C, H, W) or close

        # Final guarantee: ensure 5D
        while arr.ndim < 5:
            arr = arr[None]

        T_actual = arr.shape[0]
        Z_actual = arr.shape[1]

        self.volumes = {}
        self.masks   = {}
        self.labels  = {}
        self._corr_volumes = {}   # drop any correction preview from a prior volume
        for t in range(T_actual):
            self.volumes[t] = arr[t]                           # (Z,C,H,W)
            self.masks[t]   = np.zeros((Z_actual, arr.shape[3], arr.shape[4]), dtype=np.int32)
            self.labels[t]  = {}

        self.n_timepoints = T_actual
        self.n_z          = Z_actual
        self.n_channels   = arr.shape[2]
        self.source_name  = stem

        # Per-channel global intensity range across ALL timepoints (and Z), so the
        # contrast/display normalization is consistent frame-to-frame.
        self._chan_ranges = []
        for c in range(self.n_channels):
            cmin = min(float(v[:, c].min()) for v in self.volumes.values())
            cmax = max(float(v[:, c].max()) for v in self.volumes.values())
            self._chan_ranges.append((cmin, cmax))
        self.current_t    = 0
        self.current_z    = Z_actual // 2

        # Enable only the contrast sliders for channels that exist (first 3 shown),
        # and label them with the real channel name + display color.
        for c, s in enumerate(self.contrast_sliders):
            active = c < self.n_channels
            s.setEnabled(active)
            self.contrast_labels[c].setEnabled(active)
            if active:
                name = (self._channel_names[c]
                        if c < len(self._channel_names) else f"Ch {c + 1}")
                hint = "gray" if c == self._phase_channel else ["R", "G", "B"][c % 3]
                self.contrast_labels[c].setText(f"{name} ({hint})")
            s.blockSignals(True)
            s.setValue(50)
            s.blockSignals(False)

        # Update sliders
        self.slider_t.setMaximum(max(0, T_actual - 1))
        self.slider_t.setValue(0)
        self.slider_z.setMaximum(max(0, Z_actual - 1))
        self.slider_z.setValue(self.current_z)
        self.spin_t_start.setMaximum(T_actual - 1)
        self.spin_t_end.setMaximum(T_actual - 1)
        self.spin_t_end.setValue(T_actual - 1)

        self._build_z_strip_buttons()
        self._refresh_canvas()
        self._populate_metric_combo()
        self._update_metric_plot()
        self.lbl_source.setText(
            f"{stem}\n{T_actual} timepoints × {Z_actual} Z slices\n"
            f"{arr.shape[4]}×{arr.shape[3]} px  ·  {arr.shape[2]} ch"
        )
        self.status_bar.showMessage(
            f"Loaded: {stem}  —  {T_actual}T × {Z_actual}Z, "
            f"{arr.shape[4]}×{arr.shape[3]} px"
        )

    def _build_z_strip_buttons(self):
        for btn in self._zstrip_buttons:
            btn.setParent(None)
        self._zstrip_buttons.clear()

        for t in range(self.n_timepoints):
            btn = QPushButton(str(t))
            btn.setFixedSize(60, 90)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, tt=t: self.slider_t.setValue(tt))
            self._zstrip_layout.addWidget(btn)
            self._zstrip_buttons.append(btn)

        self._update_z_strip_thumbnails()

    def _update_z_strip_thumbnails(self):
        if not self.volumes:
            return
        mid_z = self.n_z // 2
        for t, btn in enumerate(self._zstrip_buttons):
            if t not in self.volumes:
                continue
            vol = self.volumes[t]          # (Z,C,H,W)
            slc = vol[mid_z]               # middle Z slice
            ranges = self._chan_ranges if self._chan_ranges else None
            thumb = normalize_slice(slc, ranges=ranges, phase_channel=self._phase_channel)
            H, W = thumb.shape[:2]
            scale = min(60 / W, 90 / H)
            tw, th = int(W * scale), int(H * scale)
            rgb = np.ascontiguousarray(thumb)
            qimg = QImage(rgb.data, W, H, 3 * W, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg).scaled(
                tw, th, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            btn.setIcon(QIcon(pm))
            btn.setText(f"T{t}")
            btn.setChecked(t == self.current_t)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _on_t_changed(self, t):
        self.current_t = t
        self.lbl_t.setText(f"T: {t + 1}/{self.n_timepoints}")
        for i, btn in enumerate(self._zstrip_buttons):
            btn.setChecked(i == t)
        self._refresh_canvas()
        self._refresh_organoid_list()
        self._update_metric_plot()   # move the current-timepoint highlight

    def _on_z_changed(self, z):
        self.current_z = z
        self.lbl_z.setText(f"Z: {z + 1}/{self.n_z}")
        self._refresh_canvas()

    def _toggle_roi_mode(self, on):
        self.canvas.roi_mode = on
        self.canvas.setCursor(QCursor(
            Qt.CursorShape.PointingHandCursor if on else Qt.CursorShape.CrossCursor))
        self.status_bar.showMessage(
            "ROI pencil ON — left-drag a loop to select a region; only organoids "
            "fully inside it are kept (per position)."
            if on else "ROI pencil off.")

    def _on_roi_changed(self):
        """Persist the canvas ROI for the current position."""
        self._rois[self.current_p] = self.canvas.roi_mask
        if self.canvas.roi_mask is not None:
            n = int(self.canvas.roi_mask.sum())
            self.status_bar.showMessage(
                f"ROI updated for position {self.current_p + 1}: keeping organoids "
                f"fully within {n} px.")

    def _clear_roi(self):
        self._rois.pop(self.current_p, None)
        self.canvas.clear_roi()
        self.status_bar.showMessage(f"ROI cleared for position {self.current_p + 1}.")

    def _toggle_masks(self, checked):
        self.canvas.show_masks = checked
        self.canvas.update()

    def _toggle_ids(self, checked):
        self.canvas.show_ids = checked
        self.canvas.update()

    def _refresh_canvas(self):
        t, z = self.current_t, self.current_z
        if t not in self.volumes:
            return

        # Show the correction-applied volume for timepoints that were segmented with
        # a correction active, so the phase-contrast view reflects what was segmented.
        corrected = t in self._corr_volumes
        vol = self._corr_volumes[t] if corrected else self.volumes[t]   # (Z, C, H, W)
        if self.chk_max_proj.isChecked():
            slc = vol.max(axis=0)      # (C, H, W)
            msk = self.masks.get(t, np.zeros(vol.shape[1:], dtype=np.int32)).max(axis=0)
        else:
            slc = vol[z]               # (C, H, W)
            msk = self.masks.get(t, np.zeros(vol.shape[1:], dtype=np.int32))[z]

        gains  = [s.value() / 50.0 for s in self.contrast_sliders]
        # Corrected volumes have a different intensity range than the raw stack, so
        # derive display ranges from the corrected data rather than the cached raw ones.
        if corrected:
            ranges = [(float(vol[:, c].min()), float(vol[:, c].max()))
                      for c in range(vol.shape[1])]
        else:
            ranges = self._chan_ranges if self._chan_ranges else None
        img = normalize_slice(slc, gains=gains, ranges=ranges,
                              phase_channel=self._phase_channel)
        if self.canvas.image is None:
            self.canvas._fit_to_window.__func__

        by_track = self.chk_color_by_track.isChecked()
        color_override = {}
        if by_track:
            for oid, ann in self.labels.get(t, {}).items():
                if ann.track_id >= 1:
                    color_override[oid] = self._track_color(ann.track_id)

        self.canvas.roi_mask = self._rois.get(self.current_p)   # per-position ROI
        self.canvas.set_data(img, msk, self.labels.get(t, {}),
                             color_override, label_by_track=by_track)

        if self.canvas.image is None or \
           self.canvas.image.shape != img.shape:
            self.canvas.image = img
            self.canvas._fit_to_window()

    # ── Segmentation ──────────────────────────────────────────────────────────

    def _get_seg_params(self):
        fast_mode = self.radio_fast.isChecked()
        return {
            "model":             "organoid_id",
            "gpu":               self.combo_device.currentIndex() > 0,
            "min_size":          self.spin_min_vol.value(),
            "stitch_threshold":  self.spin_stitch.value(),
            "do_3d":             not fast_mode,          # Precise (per-Z + stitch) vs Fast (max-Z)
            "separate_contours": self.chk_watershed.isChecked(),
            "multiscale":        self.chk_multiscale.isChecked(),
            "oid_weights_path":  self._oid_weights_path,
            "phase_channel":     getattr(self, "_phase_channel", None),
            "roi":               self._rois.get(self.current_p),
            "illum_method":      ILLUM_METHODS[self.combo_illum.currentIndex()],
            "illum_radius":      self.spin_illum_radius.value(),
        }

    def _on_model_changed(self):
        is_oid = "OrganoID" in self.combo_model.currentText()
        self._oid_weights_row_widget.setVisible(is_oid)

    def _browse_oid_weights(self):
        # A fine-tuned OrganoID model is a TensorFlow SavedModel *directory*.
        path = QFileDialog.getExistingDirectory(
            self, "Select OrganoID model folder (SavedModel)",
            self._oid_weights_path or "")
        if path:
            self._oid_weights_path = path
            self.lbl_oid_weights.setText(Path(path).name)
            self.lbl_oid_weights.setToolTip(path)

    def _run_seg_frame(self):
        t = self.current_t
        if t not in self.volumes:
            QMessageBox.warning(self, "No data", "Load a file first.")
            return
        self._run_segmentation({t: self.volumes[t]})

    def _run_seg_all(self):
        if not self.volumes:
            QMessageBox.warning(self, "No data", "Load a file first.")
            return
        t0 = self.spin_t_start.value()
        t1 = self.spin_t_end.value()
        subset = {t: self.volumes[t] for t in range(t0, t1 + 1) if t in self.volumes}
        self._run_segmentation(subset)

    def _apply_seg_corrections(self, volumes_subset, params):
        """Return {t: corrected (Z,C,H,W)} with the selected flat-field and
        illumination corrections applied to the PHASE (segmentation) channel only —
        fluorescence channels are left untouched, so intensity metrics are unchanged.
        This is the exact image that gets segmented, so it can also be shown on the
        canvas. Passthrough copy when both are 'none' (or there is no phase channel)."""
        flat_method  = FLAT_METHODS[self.combo_flat.currentIndex()]
        flat_sigma   = self.spin_flat_sigma.value()
        illum_method = params.get("illum_method", "none")
        illum_radius = int(params.get("illum_radius", 50))
        pc           = params.get("phase_channel", None)

        if pc is None or (flat_method == "none" and illum_method == "none"):
            return {t: v.astype(np.float32).copy() for t, v in volumes_subset.items()}

        # Flat-field corrector fitted on the PHASE channel only (single-channel).
        fc = None
        if flat_method != "none":
            fc = FlatfieldCorrector(flat_method, flat_sigma)
            if fc.method == "basic":
                imgs = np.stack([v[z, pc] for v in volumes_subset.values()
                                 for z in range(v.shape[0])])
                if len(imgs) > 64:
                    sel = np.linspace(0, len(imgs) - 1, 64).astype(int)
                    imgs = imgs[sel]
                try:
                    fc.fit({0: imgs})           # channel key 0 = phase sub-volume
                except Exception:
                    fc.method = "none"

        out = {}
        for t, vol in volumes_subset.items():
            v = vol.astype(np.float32).copy()
            if pc < v.shape[1]:
                if fc is not None:
                    sub = v[:, [pc]]            # (Z, 1, H, W) phase-only sub-volume
                    v[:, pc] = fc.apply_volume(sub)[:, 0]
                if illum_method != "none":
                    for z in range(v.shape[0]):
                        v[z, pc] = correct_illumination_2d(v[z, pc], illum_method, illum_radius)
            out[t] = v
        return out

    def _run_segmentation(self, volumes_subset):
        params = self._get_seg_params()

        # Apply the selected corrections up front so the SAME corrected image is
        # both segmented and displayed. Corrections are then disabled on the worker
        # (it would otherwise re-apply the illumination step).
        self._corr_volumes = self._apply_seg_corrections(volumes_subset, params)
        seg_input = self._corr_volumes
        params = dict(params, illum_method="none")

        self._seg_start = __import__("time").monotonic()
        self.seg_progress.setValue(0)
        self.seg_progress.setVisible(True)
        self.lbl_seg_stage.setText("Starting…")
        self.lbl_seg_stage.setVisible(True)
        self.lbl_seg_time.setText("Elapsed: 0s")
        self.lbl_seg_time.setVisible(True)
        self.btn_seg_frame.setEnabled(False)
        self.btn_seg_all.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._seg_timer.start()

        self._seg_worker = Seg3DWorker(seg_input, params)
        self._seg_worker.progress.connect(self._on_seg_progress)
        self._seg_worker.finished.connect(self._on_seg_finished)
        self._seg_worker.error.connect(self._on_seg_error)
        self._seg_worker.cancelled.connect(self._on_seg_cancelled)
        self._seg_worker.start()

    def _on_seg_progress(self, pct, msg):
        self.seg_progress.setValue(pct)
        self.lbl_seg_stage.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_seg_finished(self, results):
        self._seg_timer.stop()
        self._seg_ui_idle()
        vox_size = (self.spin_dz.value(), self.spin_dy.value(), self.spin_dx.value())
        for t, mask_3d in results.items():
            self.masks[t] = mask_3d
            # Intensity metrics always use the RAW volume — the flat-field and
            # illumination corrections only touch the phase channel (not measured).
            metrics = compute_3d_metrics(mask_3d, vox_size, volume=self.volumes.get(t),
                                         phase_channel=self._phase_channel)
            for oid, m in metrics.items():
                if oid not in self.labels.get(t, {}):
                    ann = OrgAnnotation()
                    ann.org_id  = oid
                    ann.metrics = m
                    if t not in self.labels:
                        self.labels[t] = {}
                    self.labels[t][oid] = ann
                else:
                    self.labels[t][oid].metrics = m
        self._refresh_canvas()
        self._refresh_organoid_list()
        self._populate_metric_combo()
        self._update_metric_plot()
        n_total = sum(len(np.unique(m)) - 1 for m in results.values())
        self.status_bar.showMessage(f"Segmentation complete — {n_total} organoids found.")

    def _on_seg_error(self, msg):
        self._seg_timer.stop()
        self._seg_ui_idle()
        QMessageBox.critical(self, "Segmentation error", msg)

    def _on_seg_cancelled(self):
        self._seg_timer.stop()
        self._seg_ui_idle()
        self.status_bar.showMessage("Segmentation cancelled.")

    def _seg_ui_idle(self):
        self.seg_progress.setVisible(False)
        self.lbl_seg_stage.setVisible(False)
        self.lbl_seg_time.setVisible(False)
        self.btn_seg_frame.setEnabled(True)
        self.btn_seg_all.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _stop_seg(self):
        if self._seg_worker and self._seg_worker.isRunning():
            self._seg_worker.request_stop()

    def _update_seg_time(self):
        if self._seg_start is None:
            return
        import time
        elapsed = time.monotonic() - self._seg_start
        m, s = divmod(int(elapsed), 60)
        self.lbl_seg_time.setText(f"Elapsed: {m}m {s:02d}s")

    # ── Annotation ────────────────────────────────────────────────────────────

    def _on_canvas_select(self, oid):
        self.canvas.selected = oid
        self.canvas.update()
        if oid == 0:
            return
        # Sync list widget
        for i in range(self.org_list.count()):
            item = self.org_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == oid:
                self.org_list.blockSignals(True)
                self.org_list.setCurrentRow(i)
                self.org_list.blockSignals(False)
                break
        self._populate_annotation_fields(oid)
        self._show_metrics(oid)
        self._update_metric_plot()

    def _on_list_select(self, row):
        item = self.org_list.item(row)
        if item is None:
            return
        oid = item.data(Qt.ItemDataRole.UserRole)
        self.canvas.selected = oid
        self.canvas.update()
        self._populate_annotation_fields(oid)
        self._show_metrics(oid)
        self._update_metric_plot()

    def _populate_annotation_fields(self, oid):
        ann = self.labels.get(self.current_t, {}).get(oid)
        if ann is None:
            return
        idx = self.combo_type.findText(ann.org_type)
        self.combo_type.setCurrentIndex(max(0, idx))
        idx2 = self.combo_state.findText(ann.state)
        self.combo_state.setCurrentIndex(max(0, idx2))
        self.edit_notes.setText(ann.notes)

    def _apply_label(self):
        oid = self.canvas.selected
        if oid == 0:
            return
        t = self.current_t
        if t not in self.labels or oid not in self.labels[t]:
            return
        ann = self.labels[t][oid]
        ann.org_type = self.combo_type.currentText()
        ann.state    = self.combo_state.currentText()
        ann.notes    = self.edit_notes.text()
        self._refresh_organoid_list()
        self.canvas.update()

    def _delete_organoid(self):
        oid = self.canvas.selected
        if oid == 0:
            return
        t = self.current_t
        if t in self.masks:
            self.masks[t][self.masks[t] == oid] = 0
        if t in self.labels:
            self.labels[t].pop(oid, None)
        self.canvas.selected = 0
        self._refresh_canvas()
        self._refresh_organoid_list()

    def _relabel_organoids(self):
        t = self.current_t
        if t not in self.masks:
            return
        mask = self.masks[t]
        old_ids = sorted(np.unique(mask))
        old_ids = [i for i in old_ids if i != 0]
        new_mask = np.zeros_like(mask)
        new_labels = {}
        for new_id, old_id in enumerate(old_ids, start=1):
            new_mask[mask == old_id] = new_id
            ann = self.labels.get(t, {}).get(old_id, OrgAnnotation())
            ann.org_id = new_id
            new_labels[new_id] = ann
        self.masks[t] = new_mask
        self.labels[t] = new_labels
        self.canvas.selected = 0
        self._refresh_canvas()
        self._refresh_organoid_list()
        self._update_metric_plot()
        self.status_bar.showMessage(f"Relabelled {len(old_ids)} organoids.")

    # ── Tracking ──────────────────────────────────────────────────────────────

    # Deterministic color palette for track IDs
    _TRACK_PALETTE = [
        QColor(230,  25,  75), QColor( 60, 180,  75), QColor(255, 225,  25),
        QColor(  0, 130, 200), QColor(245, 130,  48), QColor(145,  30, 180),
        QColor( 70, 240, 240), QColor(240,  50, 230), QColor(210, 245,  60),
        QColor(250, 190, 212), QColor(  0, 128, 128), QColor(220, 190, 255),
        QColor(170, 110,  40), QColor(255, 250, 200), QColor(128,   0,   0),
        QColor(170, 255, 195), QColor(128, 128,   0), QColor(255, 215, 180),
        QColor(  0,   0, 128), QColor(128, 128, 128),
    ]

    def _track_color(self, track_id):
        if track_id < 1:
            return TYPE_COLORS["Unlabelled"]
        return self._TRACK_PALETTE[(track_id - 1) % len(self._TRACK_PALETTE)]

    def _run_tracking(self):
        segmented = [t for t in self.masks if np.any(self.masks[t] != 0)]
        if not segmented:
            QMessageBox.warning(self, "No masks",
                "Run segmentation on at least two timepoints first.")
            return
        if len(segmented) < 2:
            QMessageBox.information(self, "Single timepoint",
                "Only one timepoint has been segmented.\n"
                "Track IDs will be assigned but no cross-T linking is possible.")

        self.btn_track.setEnabled(False)
        self.btn_clear_tracks.setEnabled(False)
        self.track_progress.setValue(0)
        self.track_progress.setVisible(True)
        self.lbl_track_stage.setText("Starting…")
        self.lbl_track_stage.setVisible(True)
        self.status_bar.showMessage("Tracking…")

        self._track_worker = TrackWorker(
            self.masks,
            iou_thresh = self.spin_iou.value(),
            memory     = self.spin_memory.value(),
        )
        self._track_worker.progress.connect(self._on_track_progress)
        self._track_worker.finished.connect(self._on_track_finished)
        self._track_worker.error.connect(self._on_track_error)
        self._track_worker.start()

    def _on_track_progress(self, pct, msg):
        self.track_progress.setValue(pct)
        self.lbl_track_stage.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_track_finished(self, track_maps):
        self.track_progress.setVisible(False)
        self.lbl_track_stage.setVisible(False)
        self.btn_track.setEnabled(True)
        self.btn_clear_tracks.setEnabled(True)

        n_assigned = 0
        for t, tmap in track_maps.items():
            for oid, tid in tmap.items():
                if t not in self.labels:
                    self.labels[t] = {}
                if oid not in self.labels[t]:
                    ann = OrgAnnotation()
                    ann.org_id = oid
                    self.labels[t][oid] = ann
                self.labels[t][oid].track_id = tid
                n_assigned += 1

        n_tracks = len({tid for tmap in track_maps.values() for tid in tmap.values()})
        n_tp     = len(track_maps)
        n_total  = sum(len(m) for m in track_maps.values())
        n_linked = n_total - n_tracks  # organoids that re-used a prior track_id
        # Auto-enable track coloring so the user can see results immediately
        self.chk_color_by_track.blockSignals(True)
        self.chk_color_by_track.setChecked(True)
        self.chk_color_by_track.blockSignals(False)
        self.status_bar.showMessage(
            f"Tracking complete — {n_tracks} unique tracks, "
            f"{n_linked} cross-frame links across {n_tp} timepoints.")
        self._refresh_organoid_list()
        self._refresh_canvas()   # update canvas color_override with new track_ids
        self._update_metric_plot()   # now the series can follow the track over time

    def _on_track_error(self, msg):
        self.track_progress.setVisible(False)
        self.lbl_track_stage.setVisible(False)
        self.btn_track.setEnabled(True)
        self.btn_clear_tracks.setEnabled(True)
        QMessageBox.critical(self, "Tracking error", msg)

    def _clear_tracks(self):
        for t_labs in self.labels.values():
            for ann in t_labs.values():
                ann.track_id = -1
        self._refresh_organoid_list()
        self._refresh_canvas()
        self._update_metric_plot()
        self.status_bar.showMessage("Track IDs cleared.")

    def _refresh_organoid_list(self):
        t = self.current_t
        self.org_list.blockSignals(True)
        self.org_list.clear()

        mask = self.masks.get(t)
        if mask is None:
            self.org_list.blockSignals(False)
            return

        ids = sorted(int(i) for i in np.unique(mask) if i != 0)
        by_track = self.chk_color_by_track.isChecked()
        labelled = 0
        for oid in ids:
            ann = self.labels.get(t, {}).get(oid)
            otype    = ann.org_type if ann else ""
            state    = ann.state    if ann else ""
            vol      = ann.metrics.get("volume_vox", "?") if ann else "?"
            track_id = ann.track_id if ann else -1
            tid_str  = f"  T{track_id}" if track_id >= 1 else ""
            text     = f"#{oid}{tid_str}  {otype or '—'}  [{state}]  {vol}vox"
            item     = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, oid)
            if by_track and track_id >= 1:
                col = self._track_color(track_id)
            else:
                col = TYPE_COLORS.get(otype, TYPE_COLORS["Unlabelled"])
            item.setBackground(QBrush(col))
            if oid == self.canvas.selected:
                item.setSelected(True)
            self.org_list.addItem(item)
            if otype:
                labelled += 1

        self.lbl_n_total.setText(str(len(ids)))
        self.lbl_n_labelled.setText(str(labelled))
        self.org_list.blockSignals(False)

    def _intensity_for(self, t, oid):
        """Recompute the full metric set for one organoid from the stored volume +
        mask (used to backfill annotations loaded from an older project). Returns {}
        if unavailable."""
        vol  = self.volumes.get(t)
        mask = self.masks.get(t)
        if vol is None or mask is None or vol.ndim != 4:
            return {}
        if mask.shape != (vol.shape[0], vol.shape[2], vol.shape[3]):
            return {}
        vox_size = (self.spin_dz.value(), self.spin_dy.value(), self.spin_dx.value())
        mets = compute_3d_metrics(mask, vox_size, volume=vol,
                                  phase_channel=getattr(self, "_phase_channel", None))
        return mets.get(int(oid), {})

    def _ensure_intensity(self, t, oid, ann):
        """Backfill intensity metrics on demand if the annotation lacks them."""
        if ann is None or any(k.endswith("_vol") for k in ann.metrics):
            return
        extra = self._intensity_for(t, oid)
        if extra:
            ann.metrics.update(extra)

    def _show_metrics(self, oid):
        ann = self.labels.get(self.current_t, {}).get(oid)
        self.metrics_table.setRowCount(0)
        if ann is None or not ann.metrics:
            return
        # Backfill intensity/sphericity if missing, then show EVERY metric the
        # organoid has — the same set that is written to the Excel file.
        self._ensure_intensity(self.current_t, oid, ann)
        tid = ann.track_id if ann.track_id >= 1 else "—"
        rows = [("Track ID", tid)]
        for k in self._ordered_metric_keys(set(ann.metrics.keys())):
            rows.append((self._metric_label(k), ann.metrics.get(k, "—")))
        self.metrics_table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.metrics_table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.metrics_table.setItem(r, 1, QTableWidgetItem(str(v)))

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_csv(self):
        import csv
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"{self.source_name or 'organoids'}_annotations.csv",
            "CSV files (*.csv)"
        )
        if not path:
            return
        # Data-driven: the metric columns are exactly those on the organoids
        # (== the Excel metric set). Backfill first so all keys are present.
        present = set()
        for t in sorted(self.labels.keys()):
            for oid, ann in self.labels[t].items():
                self._ensure_intensity(t, oid, ann)
                present.update(ann.metrics.keys())
        metric_keys = self._ordered_metric_keys(present)
        header = (["experiment", "XY_position", "track_id", "timepoint",
                   "org_type", "state"] + metric_keys + ["notes"])
        rows = []
        for t in sorted(self.labels.keys()):
            for oid, ann in self.labels[t].items():
                m = ann.metrics
                tid = ann.track_id if ann.track_id >= 1 else oid
                row = [self.source_name, self.current_p + 1, tid, t + 1,
                       ann.org_type, ann.state]
                row += [m.get(k, "") for k in metric_keys]
                row.append(ann.notes)
                rows.append(row)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        QMessageBox.information(self, "Export complete",
            f"✅ Exported {len(rows)} organoid annotations.\n\n{path}")

    def _export_hdf5(self):
        try:
            import h5py
        except ImportError:
            QMessageBox.critical(self, "h5py not installed",
                "Install with:\n    pip install h5py")
            return
        if not self.volumes:
            QMessageBox.warning(self, "No data", "Load data first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export HDF5",
            f"{self.source_name or 'organoids'}.h5",
            "HDF5 files (*.h5 *.hdf5)"
        )
        if not out_path:
            return

        try:
            with h5py.File(out_path, "w") as f:
                meta = f.create_group("metadata")
                meta.attrs["source"]      = self.source_name
                meta.attrs["n_timepoints"] = self.n_timepoints
                meta.attrs["n_z"]          = self.n_z
                meta.attrs["voxel_size_um"] = [
                    self.spin_dz.value(), self.spin_dy.value(), self.spin_dx.value()
                ]

                tg = f.create_group("timepoints")
                for t in sorted(self.volumes.keys()):
                    g = tg.create_group(f"{t:04d}")
                    g.create_dataset("volume", data=self.volumes[t],
                                     compression="gzip", compression_opts=4)
                    if t in self.masks:
                        g.create_dataset("mask_3d", data=self.masks[t],
                                         compression="gzip", compression_opts=4)
                    labs = self.labels.get(t, {})
                    if labs:
                        # Data-driven annotation table: id/type/state + every metric
                        # currently on the organoids (== the Excel metric set).
                        present = set()
                        for oid, ann in labs.items():
                            self._ensure_intensity(t, oid, ann)
                            present.update(ann.metrics.keys())
                        metric_keys = self._ordered_metric_keys(present)
                        dt = np.dtype([
                            ("org_id",   np.int32),
                            ("track_id", np.int32),
                            ("org_type", "S32"),
                            ("state",    "S16"),
                        ] + [(k, np.float32) for k in metric_keys])
                        rows = []
                        for oid, ann in labs.items():
                            m = ann.metrics
                            base = (int(oid), int(ann.track_id),
                                    ann.org_type.encode(), ann.state.encode())
                            base += tuple(float(m.get(k, 0) or 0) for k in metric_keys)
                            rows.append(base)
                        g.create_dataset("annotations",
                                         data=np.array(rows, dtype=dt))
                    QApplication.processEvents()

        except Exception as e:
            QMessageBox.critical(self, "HDF5 export failed", str(e))
            return

        QMessageBox.information(self, "HDF5 export complete",
            f"✅ Saved {len(self.volumes)} timepoints.\n\n"
            f"Schema:\n"
            f"  timepoints/NNNN/volume    — (Z,C,H,W) float32\n"
            f"  timepoints/NNNN/mask_3d   — (Z,H,W) int32\n"
            f"  timepoints/NNNN/annotations — organoid table\n\n{out_path}")

    # ── Project save / load ───────────────────────────────────────────────────

    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project",
            f"{self.source_name or 'organoid_project'}.orgproj",
            "Organoid Project (*.orgproj)"
        )
        if not path:
            return

        try:
            # Labels → JSON string stored as a 1-element string array
            meta = {
                "version":     VERSION,
                "source_name": self.source_name,
                "voxel_size":  [self.spin_dz.value(),
                                self.spin_dy.value(),
                                self.spin_dx.value()],
                "labels": {
                    str(t): {str(oid): ann.to_dict()
                             for oid, ann in labs.items()}
                    for t, labs in self.labels.items()
                },
            }
            save_dict = {
                "labels_json": np.array([json.dumps(meta)])
            }

            # Masks — only save timepoints that have been segmented
            n_saved = 0
            for t, mask in self.masks.items():
                if np.any(mask != 0):
                    save_dict[f"mask_{t:05d}"] = mask
                    n_saved += 1

            # ROIs — per-position include masks (pencil tool)
            for p, roi in self._rois.items():
                if roi is not None and np.any(roi):
                    save_dict[f"roi_{p:04d}"] = roi.astype(np.uint8)

            np.savez_compressed(path, **save_dict)
            self.status_bar.showMessage(
                f"Project saved — {n_saved} mask(s) + annotations → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "",
            "Organoid Project (*.orgproj);;All files (*)"
        )
        if not path:
            return

        try:
            npz = np.load(path, allow_pickle=False)

            # Restore labels
            labels_json = npz["labels_json"][0]
            meta = json.loads(labels_json)

            self.source_name = meta.get("source_name", "")
            vox = meta.get("voxel_size", [1.0, 0.5, 0.5])
            self.spin_dz.setValue(vox[0])
            self.spin_dy.setValue(vox[1])
            self.spin_dx.setValue(vox[2])

            raw_labels = meta.get("labels", {})
            loaded_labels = {
                int(t): {int(oid): OrgAnnotation.from_dict(d)
                         for oid, d in labs.items()}
                for t, labs in raw_labels.items()
            }

            # Restore masks + ROIs
            loaded_masks = {}
            for key in npz.files:
                if key.startswith("mask_"):
                    t = int(key[5:])
                    loaded_masks[t] = npz[key]
                elif key.startswith("roi_"):
                    p = int(key[4:])
                    self._rois[p] = npz[key].astype(bool)

            # Merge into existing data (preserve volumes if already loaded)
            for t, labs in loaded_labels.items():
                self.labels[t] = labs
            for t, mask in loaded_masks.items():
                self.masks[t] = mask
                # If no volume exists for this timepoint, create a placeholder
                if t not in self.volumes and mask.ndim == 3:
                    Z, H, W = mask.shape
                    self.volumes[t] = np.zeros((Z, 1, H, W), dtype=np.float32)
                    if self.n_z == 0:
                        self.n_z = Z
                        self.n_timepoints = max(self.n_timepoints, t + 1)
                        self.slider_z.setMaximum(max(0, Z - 1))
                        self.slider_z.setValue(Z // 2)
                        self.current_z = Z // 2

            # Infer channel count from restored intensity metrics so the metric
            # table/plot expose them even when no raw volume is loaded.
            import re as _re
            max_ch = self.n_channels
            for labs in loaded_labels.values():
                for ann in labs.values():
                    for k in ann.metrics:
                        mm = _re.match(r"int_sum_ch(\d+)$", k)
                        if mm:
                            max_ch = max(max_ch, int(mm.group(1)))
            self.n_channels = max_ch

            self._refresh_organoid_list()
            self._refresh_canvas()
            self._populate_metric_combo()
            self._update_metric_plot()
            n_masks = len(loaded_masks)
            n_anns  = sum(len(v) for v in loaded_labels.values())
            self.status_bar.showMessage(
                f"Project loaded — {n_masks} mask(s), {n_anns} annotations  |  {path}")

        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = OrgAnnotator()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
