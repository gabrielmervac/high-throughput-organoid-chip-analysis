"""=====================================================================
 ORGANOID METRICS  --  edit this file to add / change measurements.
=====================================================================

HOW IT WORKS
------------
A metric is just a function that receives a context object and returns a
dict of {column_name: value}. Register it with @plane_metric or @volume_metric:

    @plane_metric
    def my_metric(ctx: PlaneContext) -> dict:
        return {"my_column": some_value}

  * PLANE metrics run once per organoid per Z-plane  (ctx.mask is 2-D).
  * VOLUME metrics run once per organoid over the whole 3-D object
    (ctx.mask is 3-D, Z x Y x X).

Both contexts give you:
  ctx.mask            boolean mask of THIS organoid
  ctx.intensity       {channel_name: image}  (2-D for plane, 3-D for volume)
  ctx.voxel           {"x":um, "y":um, "z":um}
  ctx.label           the organoid / track id

To ADD a measurement: write a function and decorate it. That's the only change.
To REMOVE one: delete or comment out its function.
The Excel exporter picks up whatever columns these functions return.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import skimage.measure

# ---------------------------------------------------------------------
# Registry machinery (you normally don't need to touch this).
# ---------------------------------------------------------------------
PLANE_METRICS: List[Callable] = []
VOLUME_METRICS: List[Callable] = []


def plane_metric(fn: Callable) -> Callable:
    PLANE_METRICS.append(fn)
    return fn


def volume_metric(fn: Callable) -> Callable:
    VOLUME_METRICS.append(fn)
    return fn


@dataclass
class PlaneContext:
    mask: np.ndarray                    # bool, 2-D (Y, X)
    intensity: Dict[str, np.ndarray]    # channel -> 2-D image
    voxel: Dict[str, float]
    label: int
    background: Dict[str, float] = None  # per-channel background to subtract


@dataclass
class VolumeContext:
    mask: np.ndarray                    # bool, 3-D (Z, Y, X)
    intensity: Dict[str, np.ndarray]    # channel -> 3-D image
    voxel: Dict[str, float]
    label: int
    background: Dict[str, float] = None  # per-channel background to subtract
    phase: str = None                    # phase channel name (for best-focus)


def _best_focus_plane(mask3d: np.ndarray, phase_stack) -> int:
    """Sharpest Z plane over the organoid footprint (variance of Laplacian of the
    phase image). Falls back to the largest-area plane if no image is given."""
    Z = mask3d.shape[0]
    fp = mask3d.any(axis=0)
    if not fp.any():
        return 0
    if phase_stack is None:
        return int(np.argmax([int(mask3d[z].sum()) for z in range(Z)]))
    from scipy.ndimage import laplace
    ys, xs = np.where(fp)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fpc = fp[y0:y1, x0:x1]
    best_z, best = 0, -1.0
    stack = np.asarray(phase_stack)
    for z in range(Z):
        vals = laplace(stack[z, y0:y1, x0:x1].astype(np.float32))[fpc]
        v = float(vals.var()) if vals.size else 0.0
        if v > best:
            best, best_z = v, z
    return int(best_z)


def compute_plane(ctx: PlaneContext) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for fn in PLANE_METRICS:
        out.update(fn(ctx))
    return out


def compute_volume(ctx: VolumeContext) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for fn in VOLUME_METRICS:
        out.update(fn(ctx))
    return out


# =====================================================================
#  PLANE-SCOPE METRICS  (per organoid, per Z-plane)
# =====================================================================
@plane_metric
def plane_morphology(ctx: PlaneContext) -> dict:
    """Basic 2-D shape descriptors for this organoid on this plane."""
    m = ctx.mask
    area_px = int(m.sum())
    if area_px == 0:
        return {"area_px": 0, "area_um2": 0.0, "perimeter_px": 0.0,
                "equiv_diameter_px": 0.0, "eccentricity": 0.0,
                "solidity": 0.0, "circularity": 0.0}
    props = skimage.measure.regionprops(m.astype(np.uint8))[0]
    perim = float(props.perimeter)
    circ = (4 * np.pi * area_px / (perim ** 2)) if perim > 0 else 0.0
    px_area = ctx.voxel["x"] * ctx.voxel["y"]
    return {
        "area_px": area_px,
        "area_um2": round(area_px * px_area, 4),
        "perimeter_px": round(perim, 3),
        "equiv_diameter_px": round(float(props.equivalent_diameter_area), 3),
        "eccentricity": round(float(props.eccentricity), 4),
        "solidity": round(float(props.solidity), 4),
        "circularity": round(min(circ, 1.0), 4),
    }


def _intensity_columns(intensity, mask, background) -> dict:
    """Background-subtracted (clipped at 0) mean/median/total per channel."""
    bg = background or {}
    out: dict = {}
    for chan, img in intensity.items():
        b = float(bg.get(chan, 0.0))
        vals = np.asarray(img)[mask].astype(np.float32) - b
        np.clip(vals, 0.0, None, out=vals)
        if vals.size == 0:
            out[f"{chan}_mean"] = 0.0
            out[f"{chan}_median"] = 0.0
            out[f"{chan}_total"] = 0.0
        else:
            out[f"{chan}_mean"] = round(float(vals.mean()), 3)
            out[f"{chan}_median"] = round(float(np.median(vals)), 3)
            out[f"{chan}_total"] = float(vals.sum())
        out[f"{chan}_bg"] = round(b, 3)
    return out


@plane_metric
def plane_intensity(ctx: PlaneContext) -> dict:
    """Background-subtracted mean/median/total intensity in this plane, per channel.
    Background = 25th percentile of the whole channel stack (all Z), matching the GUI."""
    return _intensity_columns(ctx.intensity, ctx.mask, ctx.background)


# =====================================================================
#  VOLUME-SCOPE METRICS  (per organoid, whole 3-D object)
# =====================================================================
@volume_metric
def volume_morphology(ctx: VolumeContext) -> dict:
    """3-D size descriptors over all planes the organoid spans."""
    m = ctx.mask
    vox = int(m.sum())
    vx, vy, vz = ctx.voxel["x"], ctx.voxel["y"], ctx.voxel["z"]
    z_planes = int(np.count_nonzero(m.reshape(m.shape[0], -1).any(axis=1)))
    vol_um3 = vox * vx * vy * vz
    # Equivalent-sphere diameter from the physical volume.
    equiv_d = (6.0 * vol_um3 / np.pi) ** (1.0 / 3.0) if vol_um3 > 0 else 0.0
    # Max in-plane footprint area (largest single-plane area).
    max_area = int(max((m[z].sum() for z in range(m.shape[0])), default=0))
    return {
        "volume_vox": vox,
        "volume_um3": round(vol_um3, 3),
        "z_extent_planes": z_planes,
        "max_plane_area_px": max_area,
        "equiv_sphere_diameter_um": round(equiv_d, 3),
    }


@volume_metric
def volume_intensity(ctx: VolumeContext) -> dict:
    """Background-subtracted mean/median/total over the whole 3-D organoid, per channel.
    Background = 25th percentile of the whole channel stack (all Z), matching the GUI."""
    return _intensity_columns(ctx.intensity, ctx.mask, ctx.background)


@volume_metric
def volume_geometry(ctx: VolumeContext) -> dict:
    """Centroid, XY-footprint area/diameter, and the best-focus plane (with its
    area/diameter) for the whole organoid."""
    m = ctx.mask
    zz, yy, xx = np.where(m)
    if zz.size == 0:
        return {}
    vx, vy = ctx.voxel["x"], ctx.voxel["y"]
    fp_px = int(m.any(axis=0).sum())
    area_um2 = fp_px * vx * vy
    diam_um = 2.0 * np.sqrt(area_um2 / np.pi) if area_um2 > 0 else 0.0
    phase_stack = ctx.intensity.get(ctx.phase) if ctx.phase else None
    bf_z = _best_focus_plane(m, phase_stack)
    bf_px = int(m[bf_z].sum())
    bf_area = bf_px * vx * vy
    bf_diam = 2.0 * np.sqrt(bf_area / np.pi) if bf_area > 0 else 0.0
    return {
        "centroid_x": round(float(xx.mean()), 1),
        "centroid_y": round(float(yy.mean()), 1),
        "footprint_area_um2": round(area_um2, 2),
        "footprint_diameter_um": round(diam_um, 2),
        "best_focus_z": int(bf_z) + 1,   # 1-based plane number
        "bestfocus_area_um2": round(bf_area, 2),
        "bestfocus_diameter_um": round(bf_diam, 2),
    }


# Column order helpers (Excel export uses these so new metrics show up too).
def plane_columns(channels: List[str]) -> List[str]:
    dummy = PlaneContext(np.ones((2, 2), bool),
                         {c: np.ones((2, 2)) for c in channels},
                         {"x": 1.0, "y": 1.0, "z": 1.0}, 1)
    return list(compute_plane(dummy).keys())


def volume_columns(channels: List[str]) -> List[str]:
    dummy = VolumeContext(np.ones((2, 2, 2), bool),
                          {c: np.ones((2, 2, 2)) for c in channels},
                          {"x": 1.0, "y": 1.0, "z": 1.0}, 1)
    return list(compute_volume(dummy).keys())
