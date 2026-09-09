"""Link per-plane organoid masks across Z into one shared 3D identity.

Each Z-plane is segmented independently (giving per-plane instance labels); this
module stitches them so that a single physical organoid carries ONE integer id
across every plane it appears in. Matching is greedy highest-IoU-first between
adjacent planes -- the same approach the prior app used in `_stitch_z`.

Result: a (Z, H, W) uint16 array whose non-zero ids are consistent across Z.
A "3D organoid" is then simply "all voxels equal to id k".
"""
from __future__ import annotations

import numpy as np


def _iou_matrix(prev: np.ndarray, curr: np.ndarray):
    """IoU between every (prev-label, curr-label) pair that overlaps.

    Returns (pairs, prev_labels, curr_labels) where pairs is a list of
    (iou, prev_label, curr_label) for overlapping pairs only.
    """
    prev_labels = np.unique(prev)
    prev_labels = prev_labels[prev_labels != 0]
    curr_labels = np.unique(curr)
    curr_labels = curr_labels[curr_labels != 0]
    if prev_labels.size == 0 or curr_labels.size == 0:
        return [], prev_labels, curr_labels

    prev_area = {int(l): int(np.count_nonzero(prev == l)) for l in prev_labels}
    curr_area = {int(l): int(np.count_nonzero(curr == l)) for l in curr_labels}

    # Consider only pixels where both planes are foreground.
    both = (prev != 0) & (curr != 0)
    if not np.any(both):
        return [], prev_labels, curr_labels
    pv = prev[both].astype(np.int64)
    cv = curr[both].astype(np.int64)
    # Encode (prev,curr) pairs into a single key and count co-occurrences.
    key = pv * (int(curr.max()) + 1) + cv
    uniq, counts = np.unique(key, return_counts=True)
    mult = int(curr.max()) + 1

    pairs = []
    for k, inter in zip(uniq, counts):
        pl = int(k // mult)
        cl = int(k % mult)
        union = prev_area[pl] + curr_area[cl] - int(inter)
        if union > 0:
            pairs.append((int(inter) / union, pl, cl))
    pairs.sort(reverse=True)   # highest IoU first
    return pairs, prev_labels, curr_labels


def link_z(labels_zyx: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    """Relabel a (Z, H, W) stack so ids are consistent across planes."""
    labels_zyx = np.asarray(labels_zyx)
    Z = labels_zyx.shape[0]
    out = np.zeros_like(labels_zyx, dtype=np.uint16)

    next_id = 1
    # Plane 0: assign fresh global ids.
    prev_map: dict[int, int] = {}
    for l in np.unique(labels_zyx[0]):
        if l == 0:
            continue
        out[0][labels_zyx[0] == l] = next_id
        prev_map[int(l)] = next_id
        next_id += 1

    for z in range(1, Z):
        prev = out[z - 1]                       # already-global ids
        curr = labels_zyx[z]                     # per-plane local ids
        pairs, _, curr_labels = _iou_matrix(prev, curr)

        used_prev: set[int] = set()
        used_curr: set[int] = set()
        assign: dict[int, int] = {}              # curr local label -> global id
        for iou, gid_prev, cl in pairs:
            if iou < iou_threshold:
                break
            if gid_prev in used_prev or cl in used_curr:
                continue
            assign[cl] = gid_prev
            used_prev.add(gid_prev)
            used_curr.add(cl)

        for cl in curr_labels:
            cl = int(cl)
            gid = assign.get(cl)
            if gid is None:
                gid = next_id
                next_id += 1
            out[z][curr == cl] = gid

    return out


def volume_projection(labels_zyx: np.ndarray) -> np.ndarray:
    """Flatten a Z-linked stack to a 2-D (H, W) label image (the XY footprint of
    each 3D organoid), used as input to time-tracking. Assumes ids are already
    consistent across Z (from link_z)."""
    labels_zyx = np.asarray(labels_zyx)
    out = np.zeros(labels_zyx.shape[1:], dtype=np.uint16)
    for z in range(labels_zyx.shape[0]):
        out = np.where(labels_zyx[z] != 0, labels_zyx[z], out)
    return out
