"""Lazy reader for Nikon ND2 files, dimension-aware (T, P/XY, Z, C).

Wraps the `nd2` package. Exposes the axis sizes, per-channel names, voxel size,
per-timepoint acquisition time, and a lazy per-(T, P, Z, C) frame fetch so we
never load the whole (often multi-GB) file into memory.

The prior PyTorch app ignored the P (multi-XY-position) axis; this reader treats
P as a first-class dimension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import nd2


@dataclass
class ND2Info:
    path: Path
    experiment: str                     # file stem, used as the Excel "Experiment" key
    sizes: Dict[str, int]               # e.g. {'T':37,'P':10,'Z':9,'C':3,'Y':1022,'X':1024}
    channel_names: List[str]            # e.g. ['FITC', 'Cy3', 'Ph20X']
    voxel_um: Dict[str, float]          # {'x':.., 'y':.., 'z':..}
    time_hours: List[float] = field(default_factory=list)  # acquisition time per T, in hours

    @property
    def nT(self) -> int: return self.sizes.get("T", 1)
    @property
    def nP(self) -> int: return self.sizes.get("P", 1)
    @property
    def nZ(self) -> int: return self.sizes.get("Z", 1)
    @property
    def nC(self) -> int: return self.sizes.get("C", 1)
    @property
    def height(self) -> int: return self.sizes["Y"]
    @property
    def width(self) -> int: return self.sizes["X"]


# Heuristics for identifying channels by name.
PHASE_HINTS = ("ph", "phase", "bf", "brightfield", "trans", "dia")


def _guess_phase_channel(names: List[str]) -> int:
    """Index of the phase-contrast / brightfield channel (segmentation input)."""
    for i, n in enumerate(names):
        low = n.lower()
        if any(h in low for h in PHASE_HINTS):
            return i
    # Fallback: last channel (phase is commonly acquired last).
    return len(names) - 1


class ND2Reader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = nd2.ND2File(str(self.path))
        self._dask = self._f.to_dask()                       # lazy, dims follow sizes order
        self._axes = list(self._f.sizes.keys())              # ordered axis labels
        self.info = self._build_info()
        self.phase_channel = _guess_phase_channel(self.info.channel_names)

    # -- metadata ---------------------------------------------------------
    def _build_info(self) -> ND2Info:
        sizes = dict(self._f.sizes)
        names: List[str] = []
        try:
            for c in self._f.metadata.channels:
                names.append(c.channel.name)
        except Exception:
            names = [f"C{i}" for i in range(sizes.get("C", 1))]
        if not names:
            names = [f"C{i}" for i in range(sizes.get("C", 1))]

        try:
            vs = self._f.voxel_size()
            voxel = {"x": float(vs.x), "y": float(vs.y), "z": float(vs.z)}
        except Exception:
            voxel = {"x": 1.0, "y": 1.0, "z": 1.0}

        time_hours: List[float] = []
        try:
            for lp in self._f.experiment:
                if type(lp).__name__ == "TimeLoop":
                    period_ms = float(lp.parameters.periodMs)
                    n = int(getattr(lp, "count", sizes.get("T", 1)))
                    time_hours = [round(i * period_ms / 3_600_000.0, 4) for i in range(n)]
                    break
        except Exception:
            pass
        if not time_hours:
            time_hours = [float(i) for i in range(sizes.get("T", 1))]

        return ND2Info(path=self.path, experiment=self.path.stem, sizes=sizes,
                       channel_names=names, voxel_um=voxel, time_hours=time_hours)

    # -- frame access -----------------------------------------------------
    def _index(self, t: int, p: int, z: int, c: int):
        """Build an indexing tuple matching the dask array's axis order."""
        pos = {"T": t, "P": p, "Z": z, "C": c}
        idx = []
        for ax in self._axes:
            if ax in ("Y", "X"):
                idx.append(slice(None))
            else:
                idx.append(pos.get(ax, 0))
        return tuple(idx)

    def frame(self, t: int = 0, p: int = 0, z: int = 0, c: int = 0) -> np.ndarray:
        """Return a single 2-D (Y, X) frame as a numpy array (native dtype, uint16)."""
        return np.asarray(self._dask[self._index(t, p, z, c)])

    def zstack(self, t: int, p: int, c: int) -> np.ndarray:
        """Return the (Z, Y, X) stack for one channel at (t, p)."""
        return np.stack([self.frame(t, p, z, c) for z in range(self.info.nZ)], axis=0)

    def phase_zstack(self, t: int, p: int) -> np.ndarray:
        """(Z, Y, X) stack of the phase-contrast channel (segmentation input)."""
        return self.zstack(t, p, self.phase_channel)

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    import sys
    r = ND2Reader(sys.argv[1] if len(sys.argv) > 1 else r"D:\Yiyu\Experiments\260401_10.nd2")
    print("experiment:", r.info.experiment)
    print("sizes:", r.info.sizes)
    print("channels:", r.info.channel_names, "| phase idx:", r.phase_channel)
    print("voxel um:", r.info.voxel_um)
    print("time_hours[:5]:", r.info.time_hours[:5])
    f = r.frame(0, 0, r.info.nZ // 2, r.phase_channel)
    print("mid phase frame:", f.shape, f.dtype, "min", f.min(), "max", f.max())
    r.close()
