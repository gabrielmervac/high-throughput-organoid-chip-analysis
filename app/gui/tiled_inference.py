"""Tiled (native-resolution) inference for OrganoID — prototype.

The stock path (_prepare_gray in organoid_id_tf) downscales the WHOLE frame to the
model's 512x512 input. For a very large organoid that discards detail; this module
instead runs the model over overlapping native-resolution 512 tiles and stitches the
belief maps, so no spatial downscaling happens.

    belief = tiled_belief(gray_native, model, meta)          # (H,W) float in [0,1]
    label  = segment_frame_tiled(model, meta, gray_native)   # labeled instance mask

NOTE (diagnostic): tiling RAISES the apparent size of every organoid in the model's
input (no downscale), so it helps only if the failure is lost detail. If big organoids
fail because they are out-of-SCALE vs training, downscaling MORE (see the experiment's
"zoom-out" mode) is the opposite, and correct, lever. Run the comparison before wiring
this into the GUI.
"""
from __future__ import annotations

import numpy as np

from organoid_id_tf import detect, postprocess  # same directory


def _norm_255(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    vmin, vmax = float(g.min()), float(g.max())
    return 255.0 * (g - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(g)


def _tile_starts(length: int, tile: int, step: int):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tiled_belief(gray: np.ndarray, model, meta, tile: int | None = None,
                 overlap: int = 96) -> np.ndarray:
    """Run the model on overlapping native-resolution tiles; stitch belief maps.

    gray : (H, W) native-resolution grayscale (any dtype).
    Returns (H, W) belief map in [0,1] at native resolution.
    """
    tile = tile or int(meta.input_h)
    H, W = gray.shape
    g = _norm_255(gray)                                   # one global normalisation
    step = max(1, tile - overlap)
    ys = _tile_starts(H, tile, step)
    xs = _tile_starts(W, tile, step)

    belief = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    bg = float(np.median(g))
    for y in ys:
        for x in xs:
            yy, xx = min(tile, H - y), min(tile, W - x)
            patch = np.full((tile, tile), bg, np.float32)
            patch[:yy, :xx] = g[y:y + yy, x:x + xx]
            prob = detect(model, patch[np.newaxis, np.newaxis], meta)[0]   # (tile,tile)
            belief[y:y + yy, x:x + xx] += prob[:yy, :xx]
            wsum[y:y + yy, x:x + xx] += 1.0
    return belief / np.maximum(wsum, 1e-6)


def segment_frame_tiled(model, meta, gray: np.ndarray, threshold: float = 0.5,
                        min_area: int = 100, tile: int | None = None,
                        overlap: int = 96, **post_kwargs):
    """Tiled belief -> instance label mask at native resolution.

    min_area is in NATIVE pixels (scale it up from the 512-based default yourself).
    """
    belief = tiled_belief(gray, model, meta, tile, overlap)
    label = postprocess(belief, threshold=threshold, min_area=min_area, **post_kwargs)
    return label, belief
