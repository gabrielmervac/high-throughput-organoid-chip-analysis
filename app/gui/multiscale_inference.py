"""Multi-scale (2-scale) inference for OrganoID — prototype.

Motivation (measured on 260401_10, pos10/frame20, a ~277um organoid):
the stock single-scale path truncates/fragments organoids much larger than the
training distribution (median ~38um). Running the SAME model on a zoomed-OUT copy
(organoid shrunk toward training scale) fills them cleanly, but zooming out loses
small organoids. So we run BOTH and merge:

  * fine pass   : stock (native -> 512)          -> good for small/normal organoids
  * coarse pass : native -> `coarse_target`, padded to 512 -> good for BIG organoids

Merge rule: trust the COARSE pass only for LARGE blobs (diameter >= large_min_diam_px),
and take everything else from the FINE pass (fine pixels inside a kept coarse-large
object are dropped so a big organoid isn't double-counted / re-fragmented).

    label, dbg = multiscale_label(model, meta, gray_native)
"""
from __future__ import annotations

import numpy as np
import skimage.transform
import skimage.measure
from scipy import ndimage

from organoid_id_tf import _prepare_gray, detect, postprocess


def _resize(a, hw, order=1):
    return skimage.transform.resize(a, hw, order=order, preserve_range=True,
                                    anti_aliasing=(order > 0)).astype(np.float32)


def _norm255(gray):
    g = gray.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    return 255.0 * (g - lo) / (hi - lo) if hi > lo else np.zeros_like(g)


def fine_belief(gray, model, meta):
    """Stock path: whole frame -> 512 -> model -> belief upsampled to native."""
    H, W = gray.shape
    b = detect(model, _prepare_gray(gray, meta.input_h, meta.input_w), meta)[0]
    return _resize(b, (H, W))


def coarse_belief(gray, model, meta, coarse_target=256):
    """Zoom-out: frame -> coarse_target (<512), centre-padded to 512 -> belief -> native."""
    H, W = gray.shape
    g = _norm255(gray)
    sh, sw = round(H * coarse_target / max(H, W)), round(W * coarse_target / max(H, W))
    small = _resize(g, (sh, sw))
    canvas = np.full((meta.input_h, meta.input_w), float(np.median(g)), np.float32)
    oy, ox = (meta.input_h - sh) // 2, (meta.input_w - sw) // 2
    canvas[oy:oy + sh, ox:ox + sw] = small
    prob = detect(model, canvas[np.newaxis, np.newaxis], meta)[0]
    return _resize(prob[oy:oy + sh, ox:ox + sw], (H, W))


def multiscale_label(model, meta, gray, threshold=0.5, coarse_target=256,
                     large_min_diam_px=200, min_area=None, **post_kwargs):
    """Return a merged instance-label mask (native res) and a debug dict."""
    H, W = gray.shape
    if min_area is None:
        min_area = int(100 * (max(H, W) / meta.input_h) ** 2)

    bf = fine_belief(gray, model, meta)
    bc = coarse_belief(gray, model, meta, coarse_target)
    label_fine = postprocess(bf, threshold=threshold, min_area=min_area, **post_kwargs)
    label_coarse = postprocess(bc, threshold=threshold, min_area=min_area, **post_kwargs)

    merged = np.zeros((H, W), np.int32)
    nid = 0
    kept_big = np.zeros((H, W), bool)
    for rp in skimage.measure.regionprops(label_coarse):
        if rp.equivalent_diameter >= large_min_diam_px:
            nid += 1
            coords = rp.coords
            merged[coords[:, 0], coords[:, 1]] = nid
            kept_big[coords[:, 0], coords[:, 1]] = True

    # add each FINE instance as-is (preserving its watershed separation), UNLESS it
    # lies mostly inside a kept big organoid (then it's a fragment of it -> drop).
    for rp in skimage.measure.regionprops(label_fine):
        coords = rp.coords
        inside_big = kept_big[coords[:, 0], coords[:, 1]].mean()
        if inside_big > 0.5:
            continue
        nid += 1
        merged[coords[:, 0], coords[:, 1]] = nid

    dbg = {"belief_fine": bf, "belief_coarse": bc,
           "label_fine": label_fine, "label_coarse": label_coarse,
           "n_big_from_coarse": int(kept_big.any()) and int(merged.max())}
    return merged, dbg
