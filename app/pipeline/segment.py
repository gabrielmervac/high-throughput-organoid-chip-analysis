"""Segment phase-contrast frames into labeled organoid instances.

Thin wrapper over OrganoID's own inference + post-processing (Core.Model.Detect,
Core.Identification.{DetectEdges,SeparateContours,Label,Cleanup}) -- the model
and its algorithms are used unchanged. We add only:
  * numpy<->PIL glue so we can feed ND2 frames (not files), and
  * resizing instance labels back to the original frame resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import skimage.transform

from Core.Model import LoadFullModel, LoadLiteModel, Detect, PrepareImagesForModel
from Core.Identification import DetectEdges, SeparateContours, Label, Cleanup


@dataclass
class SegParams:
    threshold: float = 0.5        # foreground belief threshold
    edge_sigma: float = 2.0       # gaussian sigma for edge detection / watershed smoothing
    edge_min: float = 0.005       # hysteresis low
    edge_max: float = 0.03        # hysteresis high
    min_area: int = 25            # minimum organoid area (px) at model resolution
    fill_holes: bool = True
    remove_border: bool = False
    separate_contours: bool = True  # watershed-split touching organoids
    batch_size: int = 16


def load_model(model_path: str | Path):
    """Load a fine-tuned Keras SavedModel dir or a .tflite file."""
    p = Path(model_path)
    return LoadLiteModel(p) if p.is_file() else LoadFullModel(p)


def _frames_to_prepared(model, frames: np.ndarray):
    """Normalize (N, Y, X) frames to the model input size/scale, exactly as
    OrganoID does for images loaded from disk."""
    pil = [Image.fromarray(np.asarray(f)) for f in frames]
    return PrepareImagesForModel(pil, model, verbose=False)


def belief_maps(model, frames: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Run the network -> (N, h, w) belief maps at model resolution."""
    prepared = _frames_to_prepared(model, frames)
    return Detect(model, prepared, batchSize=batch_size)


def _resize_labels(labels: np.ndarray, out_hw) -> np.ndarray:
    """Nearest-neighbour resize of an instance-label image to (H, W)."""
    if labels.shape == tuple(out_hw):
        return labels
    resized = skimage.transform.resize(
        labels, out_hw, order=0, preserve_range=True, anti_aliasing=False)
    return resized.astype(np.uint16)


def segment_frames(model, frames: np.ndarray, params: SegParams | None = None,
                   out_hw=None) -> np.ndarray:
    """Segment (N, Y, X) frames -> (N, H, W) labeled instance masks.

    out_hw defaults to the input frame resolution.
    """
    params = params or SegParams()
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames[None]
    if out_hw is None:
        out_hw = frames.shape[1:3]

    beliefs = belief_maps(model, frames, params.batch_size)

    if params.separate_contours:
        edges = DetectEdges(beliefs, params.edge_sigma, params.edge_min,
                            params.edge_max, params.threshold)
        labeled = SeparateContours(beliefs, edges, params.threshold, params.edge_sigma)
    else:
        labeled = Label(beliefs, params.threshold)

    labeled = Cleanup(labeled, params.min_area, params.remove_border, params.fill_holes)

    out = np.zeros((labeled.shape[0],) + tuple(out_hw), dtype=np.uint16)
    for i in range(labeled.shape[0]):
        out[i] = _resize_labels(labeled[i], out_hw)
    return out


if __name__ == "__main__":
    import sys, _env  # noqa
    from nd2_reader import ND2Reader
    model = load_model(sys.argv[1])   # e.g. models/organoid_finetuned_BEST
    nd2_path = sys.argv[2] if len(sys.argv) > 2 else r"D:\Yiyu\Experiments\260401_10.nd2"
    with ND2Reader(nd2_path) as r:
        stack = r.phase_zstack(0, 0)          # (Z, Y, X)
        labels = segment_frames(model, stack)
        for z in range(labels.shape[0]):
            print(f"z={z}: {labels[z].max()} organoids")
