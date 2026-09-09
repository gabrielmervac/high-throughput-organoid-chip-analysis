"""Format-agnostic volume source for the organoid-analysis pipeline.

The whole analysis pipeline (segmentation -> Z-linking -> 3-D IoU tracking ->
metrics) only needs two things from a microscopy file:

  1. metadata  -- axis sizes (T, P/XY, Z, C, Y, X), channel names, voxel size,
                  per-timepoint acquisition time, and which channel is phase; and
  2. per-position pixels  -- a lazy read returning ``{t: (Z, C, Y, X)}`` for one
                  XY position, in canonical (T, Z, C, Y, X) order.

This module abstracts those two things behind :class:`VolumeSource` so the exact
same downstream code runs on either:

* **Nikon ND2** (:class:`ND2Source`) -- the format used for the paper data; the
  reading logic is byte-for-byte the same as the original ``organoid_annotator``
  /``batch_analyze`` ND2 path (``nd2`` + ``dask``, lazy per position), and
* **TIFF** (:class:`TiffSource`) -- OME-TIFF or ImageJ hyperstacks, plus plain
  multi-dimensional TIFF stacks when the axis order is given explicitly.

Use :func:`open_source` to open a path without caring which backend applies::

    from io.volume_source import open_source
    src = open_source("experiment.ome.tif", voxel_xy=0.65, voxel_z=5.0)
    print(src.info.sizes, src.info.channel_names)
    volumes = src.read_position(0)          # {t: (Z, C, Y, X)}  float32

TIFF axis / position convention (primary, documented path)
----------------------------------------------------------
* A single OME-TIFF or ImageJ hyperstack whose axes are discoverable from
  metadata is read directly. Recognised axes are T (time), Z (focus), C
  (channel), Y, X.
* **XY positions** are taken from the TIFF *series*: a file with several series
  is treated as one position per series (the OME-TIFF convention for multi-point
  acquisitions). A single-series file is a single-position dataset.
* For a plain TIFF stack whose axes are not in metadata, pass ``axes=`` (e.g.
  ``"TZCYX"``, ``"TCYX"``, ``"ZYX"``) to declare the order explicitly.

Calibration (voxel size / timing) is read from metadata when present
(OME ``PhysicalSize*`` / ImageJ ``spacing``+resolution / ND2 voxel size) and can
always be overridden with the ``voxel_xy``, ``voxel_z`` and ``time_step_h``
keywords; anything still unknown defaults to 1.0 with a warning.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Canonical axis order the whole pipeline expects for a single position.
CANONICAL = ["T", "Z", "C", "Y", "X"]

# Heuristics for identifying the phase-contrast / brightfield channel (the
# segmentation input). Kept identical to the original pipeline.
PHASE_HINTS = ("ph", "phase", "bf", "brightfield", "trans", "dia")


def guess_phase_channel(names: Sequence[str]) -> int:
    """Index of the phase-contrast / brightfield channel among ``names``.

    Matches by name hint; falls back to the last channel (phase is commonly
    acquired last). Identical behaviour to ``organoid_annotator.guess_phase_channel``.
    """
    for i, n in enumerate(names):
        if any(h in str(n).lower() for h in PHASE_HINTS):
            return i
    return max(len(names) - 1, 0)


@dataclass
class SourceInfo:
    """Format-independent description of an opened dataset."""

    path: Path
    experiment: str                       # file stem, used as the "Experiment" key
    sizes: Dict[str, int]                 # {'T':..,'P':..,'Z':..,'C':..,'Y':..,'X':..}
    dims: List[str]                       # native axis order, e.g. ['T','P','Z','C','Y','X']
    channel_names: List[str]              # e.g. ['FITC', 'Cy3', 'Ph20X']
    voxel_zyx: tuple                       # (dz, dy, dx) in microns, GUI convention
    time_hours: List[float] = field(default_factory=list)  # acquisition time per T (hours)
    phase_channel: int = 0                # index into channel_names

    @property
    def nT(self) -> int: return int(self.sizes.get("T", 1))
    @property
    def nP(self) -> int: return int(self.sizes.get("P", 1))
    @property
    def nZ(self) -> int: return int(self.sizes.get("Z", 1))
    @property
    def nC(self) -> int: return int(self.sizes.get("C", 1))
    @property
    def height(self) -> int: return int(self.sizes["Y"])
    @property
    def width(self) -> int: return int(self.sizes["X"])


class VolumeSource(ABC):
    """A microscopy dataset, opened for lazy per-position reading."""

    info: SourceInfo

    @abstractmethod
    def read_position(self, p: int) -> Dict[int, np.ndarray]:
        """Return ``{t: (Z, C, Y, X)}`` float32 for XY position ``p`` (0-based)."""

    # Convenience accessors mirroring the modular pipeline's ND2Reader.
    def zstack(self, t: int, p: int, c: int) -> np.ndarray:
        """(Z, Y, X) stack for one channel at (t, p)."""
        return self.read_position(p)[t][:, c]

    def phase_zstack(self, t: int, p: int) -> np.ndarray:
        return self.zstack(t, p, self.info.phase_channel)

    def close(self) -> None:  # pragma: no cover - backends override if needed
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ── helpers ────────────────────────────────────────────────────────────────

def _canonicalize(arr: np.ndarray, src_axes: Sequence[str]) -> np.ndarray:
    """Reorder ``arr`` (whose axes are ``src_axes``, P already removed) into full
    (T, Z, C, Y, X), inserting size-1 axes for anything missing.

    This is exactly the transform the original ND2 per-position reader applied.
    """
    src_axes = [a for a in src_axes if a != "P"]
    present = [a for a in CANONICAL if a in src_axes]
    order = [src_axes.index(a) for a in present]
    out = np.transpose(arr, order).astype(np.float32)
    for ax_i, ax in enumerate(CANONICAL):
        if ax not in present:
            out = np.expand_dims(out, ax_i)
    return out


def _apply_overrides(voxel_zyx, time_hours, nT,
                     voxel_xy, voxel_z, time_step_h):
    """Apply CLI calibration overrides on top of metadata, warning on defaults."""
    dz, dy, dx = voxel_zyx
    if voxel_xy is not None:
        dx = dy = float(voxel_xy)
    if voxel_z is not None:
        dz = float(voxel_z)
    if (dx == 1.0 and dy == 1.0) and voxel_xy is None:
        warnings.warn("XY voxel size unknown; defaulting to 1.0 um/px. "
                      "Pass voxel_xy=<um/px> for correct areas/diameters.")
    if dz == 1.0 and voxel_z is None:
        warnings.warn("Z voxel size unknown; defaulting to 1.0 um. "
                      "Pass voxel_z=<um> for correct volumes.")
    if time_step_h is not None:
        time_hours = [round(i * float(time_step_h), 4) for i in range(nT)]
    if not time_hours:
        time_hours = [float(i) for i in range(nT)]
    return (dz, dy, dx), time_hours


# ── ND2 backend (unchanged reading logic) ────────────────────────────────────

class ND2Source(VolumeSource):
    """Nikon ND2 backend. Reading logic identical to the original paper pipeline."""

    def __init__(self, path: str | Path, *, voxel_xy=None, voxel_z=None,
                 time_step_h=None, phase_channel=None, phase_name=None):
        import nd2
        self.path = Path(path)
        with nd2.ND2File(str(self.path)) as f:
            sizes = dict(f.sizes)
            dims = list(f.sizes.keys())
            try:
                names = [c.channel.name for c in f.metadata.channels]
            except Exception:
                names = [f"C{i}" for i in range(sizes.get("C", 1))]
            if not names:
                names = [f"C{i}" for i in range(sizes.get("C", 1))]
            try:
                vs = f.voxel_size()
                voxel = (float(vs.z), float(vs.y), float(vs.x))   # (dz, dy, dx)
            except Exception:
                voxel = (1.0, 1.0, 1.0)
            time_hours: List[float] = []
            try:
                for lp in f.experiment:
                    if type(lp).__name__ == "TimeLoop":
                        per_ms = float(lp.parameters.periodMs)
                        cnt = int(getattr(lp, "count", sizes.get("T", 1)))
                        time_hours = [round(i * per_ms / 3_600_000.0, 4) for i in range(cnt)]
                        break
            except Exception:
                pass

        self.dims = dims
        self.pos_axis = dims.index("P") if "P" in dims else None
        voxel, time_hours = _apply_overrides(voxel, time_hours, int(sizes.get("T", 1)),
                                             voxel_xy, voxel_z, time_step_h)
        pc = _resolve_phase(names, phase_channel, phase_name)
        self.info = SourceInfo(
            path=self.path, experiment=self.path.stem, sizes=sizes, dims=dims,
            channel_names=names, voxel_zyx=voxel, time_hours=time_hours,
            phase_channel=pc)

    def read_position(self, p: int) -> Dict[int, np.ndarray]:
        import nd2
        with nd2.ND2File(str(self.path)) as f:
            darr = f.to_dask()
            idx = [slice(None)] * darr.ndim
            if self.pos_axis is not None:
                idx[self.pos_axis] = p
            sub = np.asarray(darr[tuple(idx)])
        sub = _canonicalize(sub, self.dims)                  # -> (T, Z, C, Y, X)
        return {t: sub[t] for t in range(sub.shape[0])}


# ── TIFF backend (OME-TIFF / ImageJ / explicit-axes) ─────────────────────────

class TiffSource(VolumeSource):
    """TIFF backend: OME-TIFF, ImageJ hyperstacks, or plain stacks via ``axes=``.

    XY positions are the TIFF *series* (one position per series). Pass ``axes``
    to declare the per-series axis order when it is not in metadata.
    """

    def __init__(self, path: str | Path, *, axes: Optional[str] = None,
                 voxel_xy=None, voxel_z=None, time_step_h=None,
                 channel_names: Optional[Sequence[str]] = None,
                 phase_channel=None, phase_name=None):
        import tifffile
        self.path = Path(path)
        self._tf = tifffile.TiffFile(str(self.path))
        self._series = self._tf.series
        if not self._series:
            raise ValueError(f"{self.path.name}: no image series found")

        s0 = self._series[0]
        self._axes = (axes or s0.axes or "").upper()
        # tifffile uses 'S' (sample) for interleaved RGB-like channels; treat as C.
        self._axes = self._axes.replace("S", "C").replace("I", "T").replace("Q", "T")
        if "Y" not in self._axes or "X" not in self._axes:
            raise ValueError(
                f"{self.path.name}: could not determine axes (got {s0.axes!r}). "
                f"Pass axes=, e.g. axes='TZCYX'.")

        shape = s0.shape
        if len(self._axes) != len(shape):
            raise ValueError(
                f"{self.path.name}: axes {self._axes!r} do not match shape {shape}. "
                f"Pass a matching axes= string.")
        amap = {ax: n for ax, n in zip(self._axes, shape)}

        nP = len(self._series)
        sizes = {
            "T": int(amap.get("T", 1)), "P": nP,
            "Z": int(amap.get("Z", 1)), "C": int(amap.get("C", 1)),
            "Y": int(amap["Y"]), "X": int(amap["X"]),
        }
        # Native dims list (P first, matching ND2's typical ['T','P','Z','C','Y','X']).
        dims = ["P"] + [a for a in CANONICAL if a in self._axes]

        names = list(channel_names) if channel_names else self._read_channel_names(sizes["C"])
        voxel = self._read_voxel()                            # (dz, dy, dx) from metadata
        time_hours = self._read_time_hours(sizes["T"])

        voxel, time_hours = _apply_overrides(voxel, time_hours, sizes["T"],
                                             voxel_xy, voxel_z, time_step_h)
        pc = _resolve_phase(names, phase_channel, phase_name)
        self.info = SourceInfo(
            path=self.path, experiment=self.path.stem, sizes=sizes, dims=dims,
            channel_names=names, voxel_zyx=voxel, time_hours=time_hours,
            phase_channel=pc)

    # -- metadata helpers -------------------------------------------------
    def _read_channel_names(self, nC: int) -> List[str]:
        # OME channel names.
        try:
            import xml.etree.ElementTree as ET
            ome = self._tf.ome_metadata
            if ome:
                root = ET.fromstring(ome)
                ns = {"ome": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
                chans = root.findall(".//ome:Channel", ns) if ns else root.findall(".//Channel")
                got = [c.get("Name") or c.get("ID") or f"C{i}" for i, c in enumerate(chans)]
                got = [g for g in got if g]
                if len(got) >= nC:
                    return got[:nC]
        except Exception:
            pass
        return [f"C{i}" for i in range(nC)]

    def _read_voxel(self) -> tuple:
        dz = dy = dx = 1.0
        # ImageJ hyperstack: spacing (z, um) + XY resolution tags.
        ij = getattr(self._tf, "imagej_metadata", None)
        if ij:
            dz = float(ij.get("spacing", dz) or dz)
            try:
                page = self._tf.pages[0]
                xres = page.tags.get("XResolution")
                yres = page.tags.get("YResolution")
                unit = ij.get("unit", "")
                if xres and xres.value[0]:
                    dx = xres.value[1] / xres.value[0]
                if yres and yres.value[0]:
                    dy = yres.value[1] / yres.value[0]
                if unit not in ("micron", "um", "\\u00b5m", "µm", ""):
                    # Only micron calibration is trusted; otherwise leave to overrides.
                    dx = dy = 1.0
            except Exception:
                pass
        # OME PhysicalSize*.
        try:
            import xml.etree.ElementTree as ET
            ome = self._tf.ome_metadata
            if ome:
                root = ET.fromstring(ome)
                pix = root.find(".//{*}Pixels")
                if pix is not None:
                    dx = float(pix.get("PhysicalSizeX", dx) or dx)
                    dy = float(pix.get("PhysicalSizeY", dy) or dy)
                    dz = float(pix.get("PhysicalSizeZ", dz) or dz)
        except Exception:
            pass
        return (dz, dy, dx)

    def _read_time_hours(self, nT: int) -> List[float]:
        # ImageJ frame interval (seconds).
        ij = getattr(self._tf, "imagej_metadata", None)
        if ij and ij.get("finterval"):
            step_h = float(ij["finterval"]) / 3600.0
            return [round(i * step_h, 4) for i in range(nT)]
        return []

    # -- pixels -----------------------------------------------------------
    def read_position(self, p: int) -> Dict[int, np.ndarray]:
        arr = self._series[p].asarray()                       # native axis order
        sub = _canonicalize(arr, list(self._axes))            # -> (T, Z, C, Y, X)
        return {t: sub[t] for t in range(sub.shape[0])}

    def close(self) -> None:
        try:
            self._tf.close()
        except Exception:
            pass


def _resolve_phase(names, phase_channel, phase_name) -> int:
    if phase_channel is not None:
        return int(phase_channel)
    if phase_name is not None:
        for i, n in enumerate(names):
            if str(phase_name).lower() in str(n).lower():
                return i
    return guess_phase_channel(names)


# ── factory ──────────────────────────────────────────────────────────────

def open_source(path: str | Path, **kw) -> VolumeSource:
    """Open ``path`` with the backend matching its extension.

    Keyword overrides (all optional): ``axes`` (TIFF only), ``voxel_xy``,
    ``voxel_z``, ``time_step_h``, ``channel_names`` (TIFF only),
    ``phase_channel``, ``phase_name``.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".nd2":
        for k in ("axes", "channel_names"):
            kw.pop(k, None)
        return ND2Source(path, **kw)
    if ext in (".tif", ".tiff"):
        return TiffSource(path, **kw)
    raise ValueError(f"Unsupported file type: {ext!r} ({path.name}). "
                     f"Supported: .nd2, .tif/.tiff.")


if __name__ == "__main__":
    import sys
    src = open_source(sys.argv[1])
    i = src.info
    print("experiment:", i.experiment)
    print("sizes:", i.sizes, "dims:", i.dims)
    print("channels:", i.channel_names, "| phase idx:", i.phase_channel)
    print("voxel (dz,dy,dx) um:", i.voxel_zyx)
    print("time_hours[:5]:", i.time_hours[:5])
    vol = src.read_position(0)
    t0 = next(iter(vol))
    print("pos0 t%d volume:" % t0, vol[t0].shape, vol[t0].dtype)
    src.close()
