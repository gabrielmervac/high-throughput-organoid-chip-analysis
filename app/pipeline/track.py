"""Track 3D organoids across time within a single XY position.

Uses OrganoID's native tracker (Core.Tracking.Track = overlap cost + Hungarian
assignment) exactly as its own pipeline does: `Track(stack, 1, Inverse(Overlap))`.

We track on the 2-D XY footprint of each timepoint's Z-linked stack, then relabel
the full 3-D stacks so every plane of an organoid carries its persistent
`track_id` across all timepoints.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from Core.Tracking import Track, Inverse, Overlap
from zlink import volume_projection


def track_position(stacks_by_t: Dict[int, np.ndarray],
                   track_lost_cutoff: int = 10) -> Dict[int, np.ndarray]:
    """Track organoids over time for one XY position.

    stacks_by_t: {t_index: (Z, H, W) Z-linked label stack}.
    Returns {t_index: (Z, H, W) stack relabeled with persistent track ids}.
    """
    ts = sorted(stacks_by_t.keys())
    if not ts:
        return {}

    # 2-D footprint per timepoint, in time order.
    projections = np.stack([volume_projection(stacks_by_t[t]) for t in ts], axis=0)

    # OrganoID temporal tracking (same call as Core.RunPipeline).
    tracked_proj = Track(projections, 1, Inverse(Overlap), trackLostCutoff=track_lost_cutoff)

    # Recover, per timepoint, the map {footprint_id -> track_id} and apply it to
    # the full 3-D stack.
    out: Dict[int, np.ndarray] = {}
    for i, t in enumerate(ts):
        before = projections[i]
        after = tracked_proj[i]
        mapping: Dict[int, int] = {}
        for lbl in np.unique(before):
            if lbl == 0:
                continue
            vals = after[before == lbl]
            vals = vals[vals != 0]
            if vals.size:
                # All pixels of one footprint map to a single track id.
                mapping[int(lbl)] = int(np.bincount(vals).argmax())

        src = stacks_by_t[t]
        relabeled = np.zeros_like(src, dtype=np.uint16)
        for old_id, tid in mapping.items():
            relabeled[src == old_id] = tid
        out[t] = relabeled
    return out
