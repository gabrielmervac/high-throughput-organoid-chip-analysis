"""
organoid_id_tf.py — TensorFlow backend for OrganoID segmentation.

Drop-in replacement for organoid_id_torch.py that runs the NATIVE OrganoID
TensorFlow/Keras model (our fine-tuned SavedModel, or a .tflite) instead of the
PyTorch port. The post-processing (watershed contour separation, cleanup),
cross-Z stitching, 2-D->3-D expansion and image preparation are identical to the
reference implementation — only the forward pass differs.

Public API used by organoid_annotator.py:
    model, meta = load_tf_model(path)          # path = SavedModel dir or .tflite
    mask_3d = segment_volume(model, meta, vol, ...)
    mask_3d = segment_volume_per_z(model, meta, vol, ...)
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # OrganoID is Keras 2 — must run legacy
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Model metadata + loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrganoIDMeta:
    input_h: int
    input_w: int
    is_tflite: bool = False


def load_tf_model(path: str | Path):
    """Load a fine-tuned OrganoID model.

    path : a Keras SavedModel directory (e.g. models/organoid_finetuned_BEST)
           or a .tflite file.
    Returns (model, meta).
    """
    import tensorflow as tf
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".tflite":
        interp = tf.lite.Interpreter(model_path=str(p))
        interp.allocate_tensors()
        h, w = interp.get_input_details()[0]["shape"][1:3]
        return interp, OrganoIDMeta(int(h), int(w), is_tflite=True)
    model = tf.keras.models.load_model(str(p))
    shape = model.inputs[0].shape
    h = int(shape[1]) if shape[1] is not None else 512
    w = int(shape[2]) if shape[2] is not None else 512
    return model, OrganoIDMeta(h, w, is_tflite=False)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing (identical to OrganoID's PrepareImagesForModel)
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_gray(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """(H,W) grayscale -> (1, 1, h, w) float32, min-max normalised to [0,255]."""
    from PIL import Image as PILImage
    arr = image
    if arr.dtype != np.uint8:
        vmin, vmax = float(arr.min()), float(arr.max())
        arr = (((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
               if vmax > vmin else np.zeros(arr.shape, np.uint8))
    pil = PILImage.fromarray(arr, mode="L").resize((w, h), PILImage.Resampling.BILINEAR)
    out = np.asarray(pil, dtype=np.float32)
    vmin2, vmax2 = out.min(), out.max()
    if vmax2 > vmin2:
        out = 255.0 * (out - vmin2) / (vmax2 - vmin2)
    return out[np.newaxis, np.newaxis]


# ─────────────────────────────────────────────────────────────────────────────
# Inference (TensorFlow)
# ─────────────────────────────────────────────────────────────────────────────

def detect(model, prepared: np.ndarray, meta: OrganoIDMeta) -> np.ndarray:
    """prepared: (N,1,H,W) float32 -> (N,H,W) float32 belief maps in [0,1]."""
    x = np.transpose(prepared, (0, 2, 3, 1)).astype(np.float32)   # (N,H,W,1)
    if meta.is_tflite:
        import tensorflow as tf  # noqa
        inp = model.get_input_details()[0]
        outd = model.get_output_details()[0]
        model.resize_tensor_input(inp["index"], x.shape, strict=False)
        model.allocate_tensors()
        model.set_tensor(inp["index"], x)
        model.invoke()
        out = model.get_tensor(outd["index"])
    else:
        out = model.predict(x, verbose=0)
    return out[..., 0]


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing (mirrors OrganoID Core/Identification.py)
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(prob_map: np.ndarray,
                threshold: float = 0.5,
                min_area: int = 100,
                fill_holes: bool = True,
                remove_border: bool = False,
                separate_contours: bool = True,
                edge_sigma: float = 2.0,
                edge_low: float = 0.005,
                edge_high: float = 0.05) -> np.ndarray:
    import skimage.morphology
    import skimage.measure
    import skimage.filters
    import skimage.segmentation

    foreground = skimage.morphology.binary_opening(prob_map >= threshold)

    if separate_contours:
        smooth = skimage.filters.gaussian(prob_map, edge_sigma)
        edges_raw = skimage.filters.sobel(prob_map)
        edges_smth = skimage.filters.gaussian(edges_raw, edge_sigma)
        edges_bool = skimage.filters.apply_hysteresis_threshold(edges_smth, edge_low, edge_high)
        edges_bool = edges_bool & foreground
        centers = foreground & ~edges_bool
        basins = skimage.measure.label(centers)
        labeled = skimage.segmentation.watershed(-smooth, basins, mask=foreground)
        unsplit = foreground & (labeled == 0)
        ul = skimage.measure.label(unsplit)
        if ul.max() > 0:
            ul[ul > 0] += labeled.max()
        labeled = labeled + ul
    else:
        labeled = skimage.measure.label(foreground)

    cleaned = np.zeros_like(labeled, dtype=np.int32)
    H, W = labeled.shape
    for rp in skimage.measure.regionprops(labeled):
        if rp.area < min_area:
            continue
        coords = np.array(rp.coords)
        if remove_border and (
            0 in coords[:, 0] or H - 1 in coords[:, 0] or
            0 in coords[:, 1] or W - 1 in coords[:, 1]
        ):
            continue
        r0, c0, r1, c1 = rp.bbox
        src = rp.image_filled if fill_holes else rp.image
        cleaned[r0:r1, c0:c1] = np.where(src, rp.label, cleaned[r0:r1, c0:c1])
    return cleaned.astype(np.int32)


def _stitch_z(slice_masks: list, iou_threshold: float = 0.25) -> list:
    if not slice_masks:
        return slice_masks
    result = [slice_masks[0].copy()]
    next_id = int(slice_masks[0].max()) + 1
    for z in range(1, len(slice_masks)):
        prev = result[z - 1]
        curr = slice_masks[z].copy()
        prev_ids = [l for l in np.unique(prev) if l != 0]
        curr_ids = [l for l in np.unique(curr) if l != 0]
        if not prev_ids or not curr_ids:
            new = np.zeros_like(curr)
            for cid in curr_ids:
                new[curr == cid] = next_id
                next_id += 1
            result.append(new)
            continue
        prev_masks = {pid: prev == pid for pid in prev_ids}
        curr_masks = {cid: curr == cid for cid in curr_ids}
        iou_pairs = []
        for cid in curr_ids:
            cm = curr_masks[cid]
            for pid in prev_ids:
                pm = prev_masks[pid]
                inter = float(np.logical_and(cm, pm).sum())
                if inter == 0.0:
                    continue
                iou = inter / np.logical_or(cm, pm).sum()
                if iou >= iou_threshold:
                    iou_pairs.append((iou, cid, pid))
        iou_pairs.sort(reverse=True)
        remap, assigned_curr, assigned_prev = {}, set(), set()
        for iou, cid, pid in iou_pairs:
            if cid in assigned_curr or pid in assigned_prev:
                continue
            remap[cid] = pid
            assigned_curr.add(cid)
            assigned_prev.add(pid)
        for cid in curr_ids:
            if cid not in remap:
                remap[cid] = next_id
                next_id += 1
        new = np.zeros_like(curr)
        for cid, nid in remap.items():
            new[curr == cid] = nid
        result.append(new)
    return result


def _expand_to_3d(mask_2d: np.ndarray, gray_vol: np.ndarray) -> np.ndarray:
    Z, H, W = gray_vol.shape
    mask_3d = np.zeros((Z, H, W), dtype=np.int32)
    vmin, vmax = gray_vol.min(), gray_vol.max()
    vol_norm = (gray_vol - vmin) / (max(vmax - vmin, 1e-8))
    fg_vals = vol_norm[vol_norm > 0.01]
    z_thr = np.percentile(fg_vals, 20) if len(fg_vals) > 0 else 0.05
    for oid in np.unique(mask_2d):
        if oid == 0:
            continue
        xy = mask_2d == oid
        covered = False
        for z in range(Z):
            if vol_norm[z][xy].mean() >= z_thr:
                mask_3d[z][xy] = oid
                covered = True
        if not covered:
            mask_3d[Z // 2][xy] = oid
    return mask_3d


# ─────────────────────────────────────────────────────────────────────────────
# 3-D integration
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray(vol: np.ndarray, phase_channel: int | None = None) -> np.ndarray:
    """(Z,C,H,W)->(Z,H,W): use the phase channel when known, else channel mean."""
    if vol.ndim == 4:
        if phase_channel is not None and 0 <= phase_channel < vol.shape[1]:
            return vol[:, phase_channel].astype(np.float32)
        return vol.mean(axis=1).astype(np.float32)
    return vol.astype(np.float32)


def _select_organoids_in_roi(mask_3d: np.ndarray, roi):
    """Keep only organoids that lie ENTIRELY within the ROI.

    The model is run on the untouched full frame (homogeneous segmentation, no
    boundary artifacts); the ROI is applied afterwards purely as a *selection*
    of true organoids. Any label with at least one voxel outside the ROI —
    whether fully outside or straddling the boundary — is dropped in full, so
    partially-clipped objects never contribute wrong area/volume metrics.
    """
    if roi is None or roi.shape != mask_3d.shape[1:]:
        return mask_3d
    # Labels that touch any out-of-ROI pixel (in any Z slice) are not fully
    # contained and must be removed entirely.
    outside_labels = np.unique(mask_3d[:, ~roi])
    outside_labels = outside_labels[outside_labels != 0]
    if outside_labels.size:
        mask_3d = mask_3d.copy()
        mask_3d[np.isin(mask_3d, outside_labels)] = 0
    return mask_3d


def _segment_2d(model, meta: OrganoIDMeta, gray2d: np.ndarray,
                threshold: float, min_area: int, fill_holes: bool,
                remove_border: bool, separate_contours: bool,
                multiscale: bool, coarse_target: int, large_min_diam_px: int) -> np.ndarray:
    """Segment one 2-D grayscale frame -> instance-label mask.

    Single-scale: whole frame -> 512 -> model -> postprocess (label at 512).
    Multi-scale : fine (stock) + coarse (zoomed-out) passes merged, so organoids
                  much larger than the training data are recovered instead of
                  fragmented (label at the frame's native resolution). `min_area`
                  is given at model (512) scale and rescaled to native here.
    """
    if not multiscale:
        inp = _prepare_gray(gray2d, meta.input_h, meta.input_w)
        prob = detect(model, inp, meta)[0]
        return postprocess(prob, threshold, min_area, fill_holes,
                           remove_border, separate_contours)
    from multiscale_inference import multiscale_label
    H, W = gray2d.shape
    ms_min_area = int(min_area * (max(H, W) / meta.input_h) ** 2)
    label, _ = multiscale_label(model, meta, gray2d, threshold=threshold,
                                coarse_target=coarse_target,
                                large_min_diam_px=large_min_diam_px,
                                min_area=ms_min_area, fill_holes=fill_holes,
                                remove_border=remove_border,
                                separate_contours=separate_contours)
    return label


def segment_volume(model, meta: OrganoIDMeta, vol: np.ndarray,
                   threshold: float = 0.5, min_area: int = 100,
                   fill_holes: bool = True, remove_border: bool = False,
                   separate_contours: bool = True,
                   phase_channel: int | None = None, roi=None,
                   multiscale: bool = False, coarse_target: int = 256,
                   large_min_diam_px: int = 200) -> np.ndarray:
    """Max-Z projection -> single 2-D inference -> expand to 3-D by intensity.

    roi : optional (H, W) bool include-mask. The full frame is segmented; only
          organoids lying entirely within the ROI are kept.
    multiscale : add a zoomed-out pass to recover organoids larger than training.
    """
    gray = _to_gray(vol, phase_channel)
    Z, H, W = gray.shape
    proj = gray.max(axis=0)
    label_2d = _segment_2d(model, meta, proj, threshold, min_area, fill_holes,
                           remove_border, separate_contours,
                           multiscale, coarse_target, large_min_diam_px)
    if label_2d.shape != (H, W):
        from PIL import Image as PILImage
        label_2d = np.array(PILImage.fromarray(label_2d).resize(
            (W, H), PILImage.Resampling.NEAREST), dtype=np.int32)
    return _select_organoids_in_roi(_expand_to_3d(label_2d, gray), roi)


def segment_volume_per_z(model, meta: OrganoIDMeta, vol: np.ndarray,
                         threshold: float = 0.5, min_area: int = 100,
                         fill_holes: bool = True, remove_border: bool = False,
                         separate_contours: bool = True,
                         stitch_threshold: float = 0.25,
                         phase_channel: int | None = None, roi=None,
                         multiscale: bool = False, coarse_target: int = 256,
                         large_min_diam_px: int = 200) -> np.ndarray:
    """One 2-D inference per Z slice, then stitch consistent IDs across Z.

    roi : optional (H, W) bool include-mask. The full frame is segmented; only
          organoids lying entirely within the ROI are kept.
    multiscale : add a zoomed-out pass per slice to recover very large organoids.
    """
    gray = _to_gray(vol, phase_channel)
    Z, H, W = gray.shape
    h, w = meta.input_h, meta.input_w
    slice_masks = []
    if multiscale:
        for z in range(Z):
            slice_masks.append(_segment_2d(
                model, meta, gray[z], threshold, min_area, fill_holes,
                remove_border, separate_contours, True, coarse_target,
                large_min_diam_px))            # already native (H, W)
    else:
        batch = np.concatenate([_prepare_gray(gray[z], h, w) for z in range(Z)], axis=0)
        prob_maps = detect(model, batch, meta)                # (Z,h,w)
        for z in range(Z):
            label = postprocess(prob_maps[z], threshold, min_area, fill_holes,
                                remove_border, separate_contours)
            if label.shape != (H, W):
                from PIL import Image as PILImage
                label = np.array(PILImage.fromarray(label).resize(
                    (W, H), PILImage.Resampling.NEAREST), dtype=np.int32)
            slice_masks.append(label)
    stitched = _stitch_z(slice_masks, iou_threshold=stitch_threshold)
    return _select_organoids_in_roi(np.stack(stitched, axis=0).astype(np.int32), roi)
