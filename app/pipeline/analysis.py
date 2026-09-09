"""Orchestrate the full per-position analysis:

    phase z-stack  --segment-->  per-plane labels
                   --link_z--->  3D-identity labels (per timepoint)
                   --track----->  persistent track ids across time

and turn the resulting masks + fluorescence intensities into flat records
(one per organoid-plane and one per organoid-volume) ready for Excel export.

Kept UI-agnostic so it can run headless or be driven by the GUI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from nd2_reader import ND2Reader
from segment import SegParams, segment_frames
from zlink import link_z
from track import track_position
import metrics as M


@dataclass
class PositionResult:
    experiment: str
    position: int
    channels: List[str]
    phase_channel: int
    voxel: Dict[str, float]
    time_hours: List[float]
    timepoints: List[int]
    # {t: (Z, H, W) uint16 labels with persistent track ids}
    masks: Dict[int, np.ndarray] = field(default_factory=dict)


def analyze_position(reader: ND2Reader, model, position: int,
                     timepoints: Optional[List[int]] = None,
                     seg_params: Optional[SegParams] = None,
                     iou_threshold: float = 0.3,
                     track_lost_cutoff: int = 10,
                     do_track: bool = True,
                     progress: Optional[Callable[[str, float], None]] = None
                     ) -> PositionResult:
    info = reader.info
    timepoints = timepoints if timepoints is not None else list(range(info.nT))
    seg_params = seg_params or SegParams()

    def report(msg, frac):
        if progress:
            progress(msg, frac)

    linked_by_t: Dict[int, np.ndarray] = {}
    for i, t in enumerate(timepoints):
        report(f"Segmenting t={t} (pos {position})", i / max(len(timepoints), 1))
        phase = reader.phase_zstack(t, position)            # (Z, Y, X) uint16
        per_plane = segment_frames(model, phase, seg_params, out_hw=phase.shape[1:3])
        linked_by_t[t] = link_z(per_plane, iou_threshold)

    if do_track and len(linked_by_t) > 1:
        report(f"Tracking pos {position} over time", 0.95)
        tracked = track_position(linked_by_t, track_lost_cutoff)
    else:
        # No tracking: keep per-timepoint 3D-linked ids (identities not linked over time).
        tracked = linked_by_t

    return PositionResult(
        experiment=info.experiment, position=position,
        channels=info.channel_names, phase_channel=reader.phase_channel,
        voxel=info.voxel_um, time_hours=info.time_hours,
        timepoints=timepoints, masks=tracked,
    )


def build_records(reader: ND2Reader, result: PositionResult,
                  annotations: Optional[dict] = None,
                  include_excluded: bool = False) -> List[dict]:
    """Flatten a PositionResult into Excel rows.

    One row per (timepoint, track, Z-plane) plus one VOLUME row per (timepoint,
    track). Every row carries Experiment / XY position / Z position / Time keys.
    `annotations` is {(t, track_id): {"tag":.., "state":.., "notes":.., "include":bool}}.
    """
    annotations = annotations or {}
    channels = result.channels
    voxel = result.voxel
    rows: List[dict] = []

    for t in result.timepoints:
        if t not in result.masks:
            continue
        labels_zyx = result.masks[t]                        # (Z, H, W) track ids
        track_ids = np.unique(labels_zyx)
        track_ids = track_ids[track_ids != 0]
        if track_ids.size == 0:
            continue

        # Fluorescence + phase intensity stacks for this (t, p): {channel: (Z,H,W)}.
        intensity3d = {name: reader.zstack(t, result.position, c).astype(np.float32)
                       for c, name in enumerate(channels)}
        # Per-channel background = 25th percentile of ALL pixels/Z (matches the GUI).
        backgrounds = {name: float(np.percentile(intensity3d[name], 25))
                       for name in channels}
        phase_name = (channels[result.phase_channel]
                      if 0 <= result.phase_channel < len(channels) else None)

        t_hours = result.time_hours[t] if t < len(result.time_hours) else float(t)

        for tid in track_ids:
            tid = int(tid)
            ann = annotations.get((t, tid), {})
            included = bool(ann.get("include", True))
            if not included and not include_excluded:
                continue
            base = {
                "Experiment": result.experiment,
                "XY_position": result.position,
                "Time_index": t,
                "Time_hours": t_hours,
                "track_id": tid,
                "tag": ann.get("tag", ""),
                "state": ann.get("state", ""),
                "include": included,
                "notes": ann.get("notes", ""),
            }
            mask3d = labels_zyx == tid

            # --- per-plane rows ---
            for z in range(mask3d.shape[0]):
                m2 = mask3d[z]
                if not m2.any():
                    continue
                pctx = M.PlaneContext(
                    mask=m2,
                    intensity={c: intensity3d[c][z] for c in channels},
                    voxel=voxel, label=tid, background=backgrounds)
                row = dict(base)
                row["Z_position"] = z + 1        # 1-based plane number
                row.update(M.compute_plane(pctx))
                rows.append(row)

            # --- whole-volume row ---
            vctx = M.VolumeContext(mask=mask3d, intensity=intensity3d,
                                   voxel=voxel, label=tid,
                                   background=backgrounds, phase=phase_name)
            vrow = dict(base)
            vrow["Z_position"] = "VOLUME"
            vrow.update(M.compute_volume(vctx))
            rows.append(vrow)

    return rows
