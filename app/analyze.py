"""analyze.py — format-agnostic, headless organoid analysis (ND2 or TIFF).

Runs the exact paper pipeline — per-Z OrganoID segmentation, greedy-IoU
Z-stitching for 3-D identity, greedy 3-D volumetric-IoU time tracking with
gap-closing, split-detection merging, inclusion filters and the 3-D metric
suite — on **any** ND2 or TIFF dataset, by reading pixels through the
:mod:`io.volume_source` abstraction. It reuses ``organoid_annotator``'s own
building blocks (Seg3DWorker, IoUTracker, PositionBatchWorker, compute_3d_metrics),
so results are identical to the GUI's "Run selected positions -> disk".

The run is crash-safe: metrics are written incrementally (append-only CSV as the
source of truth + an atomically rebuilt XLSX after every position), finished
positions are skipped on resume, and the next position is prefetched while the
current one segments.

Examples
--------
Nikon ND2 (as used for the paper)::

    python analyze.py experiment.nd2 --model models/organoid_finetuned_orgaug_v2_BEST \\
        --out results/experiment

OME-TIFF / ImageJ hyperstack, giving calibration explicitly::

    python analyze.py experiment.ome.tif --model models/organoid_finetuned_orgaug_v2_BEST \\
        --out results/experiment --voxel-xy 0.65 --voxel-z 5 --time-step 4

Plain multi-dimensional TIFF stack with an explicit axis order::

    python analyze.py stack.tif --axes TZCYX --voxel-xy 0.65 --voxel-z 5 --time-step 4 \\
        --model models/organoid_finetuned_orgaug_v2_BEST --out results/stack
"""
from __future__ import annotations

# OrganoID is Keras 2 -> legacy mode; set before any TF import. Qt runs headless.
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import queue
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for sub in (_HERE / "gui", _HERE / "io", _HERE):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

from PyQt6.QtWidgets import QApplication          # noqa: E402
import organoid_annotator as OA                    # noqa: E402
from volume_source import open_source              # noqa: E402


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── ROI loading (optional; keep-fully-inside selection) ─────────────────────

def load_rois(roi_dir, expected_hw) -> dict:
    """{position_index (0-based): bool (H,W) keep-mask} from a folder of NNN.tif.

    Filenames are NNN.tif (1-based position); white (>0) = keep-region. Positions
    without a tif are analysed over the full frame.
    """
    rois: dict = {}
    if not roi_dir:
        return rois
    folder = Path(roi_dir)
    if not folder.is_dir():
        log(f"  (ROI dir {folder} not found; every position uses the full frame)")
        return rois
    from PIL import Image
    for tif in sorted(folder.glob("*.tif")):
        try:
            p = int(tif.stem) - 1
        except ValueError:
            log(f"  ! skipping ROI with non-numeric name: {tif.name}")
            continue
        arr = np.array(Image.open(tif))
        if arr.ndim != 2:
            arr = arr[..., 0]
        if arr.shape != tuple(expected_hw):
            log(f"  ! {tif.name}: shape {arr.shape} != frame {tuple(expected_hw)}; skipping")
            continue
        rois[p] = arr > 0
    log(f"  loaded {len(rois)} ROI mask(s) from {folder.name}")
    return rois


# ── incremental workbook writer ─────────────────────────────────────────────

def build_rename(channel_names, phase_channel):
    rename = {}
    for c, name in enumerate(channel_names or []):
        if c == phase_channel:
            continue
        ci = c + 1
        nm = re.sub(r"\W+", "", str(name)) or f"ch{ci}"
        rename[f"int_sum_ch{ci}_maxarea"]    = f"{nm}_maxarea_sum"
        rename[f"int_mean_ch{ci}_maxarea"]   = f"{nm}_maxarea_mean"
        rename[f"int_median_ch{ci}_maxarea"] = f"{nm}_maxarea_median"
        rename[f"int_sum_ch{ci}_vol"]        = f"{nm}_vol_sum"
        rename[f"int_mean_ch{ci}_vol"]       = f"{nm}_vol_mean"
        rename[f"int_median_ch{ci}_vol"]     = f"{nm}_vol_median"
        rename[f"int_bg_ch{ci}"]             = f"{nm}_bg"
    return rename


def append_and_rebuild(rows, rename, csv_path: Path, xlsx_path: Path):
    if rows:
        df_new = pd.DataFrame(rows).rename(columns=rename)
        header = not csv_path.exists()
        df_new.to_csv(csv_path, mode="a", header=header, index=False)
    if not csv_path.exists():
        return
    full = pd.read_csv(csv_path)
    sort_cols = [c for c in ("track_id", "timepoint") if c in full.columns]
    if sort_cols:
        full = full.sort_values(sort_cols).reset_index(drop=True)
    tmp = xlsx_path.with_suffix(".xlsx.tmp")
    try:
        full.to_excel(tmp, index=False)
        os.replace(tmp, xlsx_path)
    except Exception as e:
        log(f"  ! could not write xlsx ({e}); CSV is up to date at {csv_path.name}")


def done_positions_from_csv(csv_path: Path):
    if not csv_path.exists():
        return set(), 1
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c in ("XY_position", "track_id"))
    except Exception:
        return set(), 1
    done = set(int(v) - 1 for v in df.get("XY_position", pd.Series([])).unique())
    next_id = (int(df["track_id"].max()) + 1) if ("track_id" in df and len(df)) else 1
    return done, next_id


# ── main analysis ───────────────────────────────────────────────────────────

def run(args):
    in_path = Path(args.input)
    if not in_path.exists():
        log(f"FATAL: input not found: {in_path}")
        sys.exit(1)
    if not os.path.isdir(args.model):
        log(f"FATAL: model folder not found: {args.model}")
        sys.exit(1)

    src = open_source(
        in_path, axes=args.axes, voxel_xy=args.voxel_xy, voxel_z=args.voxel_z,
        time_step_h=args.time_step, channel_names=args.channels,
        phase_channel=args.phase_channel, phase_name=args.phase_name)
    info = src.info
    stem = info.experiment
    hw = (info.height, info.width)
    log(f"==== {stem} ({in_path.suffix.lower()}) ====")
    log(f"  P={info.nP} T={info.nT} Z={info.nZ} C={info.nC} channels={info.channel_names} "
        f"phase='{info.channel_names[info.phase_channel]}' voxel(z,y,x)={info.voxel_zyx}")

    rois = load_rois(args.roi_dir, hw)

    out_dir = Path(args.out) if args.out else Path.cwd() / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / f"{stem}_metrics_incremental.csv"
    xlsx_path = out_dir / f"{stem}_metrics.xlsx"

    positions = list(range(info.nP))
    if args.positions is not None:
        positions = [p for p in positions if p in set(args.positions)]
    elif args.limit_positions is not None:
        positions = positions[:args.limit_positions]

    done, next_gid = done_positions_from_csv(csv_path)
    todo = [p for p in positions if p not in done]
    if done:
        log(f"  resume: {len(done)} position(s) already written; {len(todo)} remaining")

    seg_params = {
        "model": "organoid_id", "gpu": False,
        "min_size": args.min_size, "stitch_threshold": args.stitch_threshold,
        "do_3d": True, "separate_contours": args.watershed,
        "multiscale": args.multiscale, "oid_weights_path": args.model,
        "phase_channel": info.phase_channel, "roi": None,
        "illum_method": "none", "illum_radius": 50,
    }

    worker = OA.PositionBatchWorker(
        positions, str(in_path), info.dims,
        (info.dims.index("P") if "P" in info.dims else 0), str(out_dir),
        dict(seg_params), args.iou, args.memory, info.voxel_zyx,
        flat_method="none", image_positions=("all" if args.jpgs else set()),
        experiment=stem, rois=rois, source=src)
    worker.phase_channel = info.phase_channel
    worker.time_hours    = info.time_hours
    worker.channel_names = info.channel_names
    worker._next_global_id = next_gid

    rename = build_rename(info.channel_names, info.phase_channel)

    # Prefetch: read the next position while the current one segments.
    _READ_FAIL = object()
    read_q: "queue.Queue" = queue.Queue(maxsize=1)

    def _producer():
        for p in todo:
            try:
                read_q.put((p, worker._read_position(p)))
            except Exception:
                read_q.put((p, (_READ_FAIL, traceback.format_exc())))
        read_q.put(None)

    reader = threading.Thread(target=_producer, name="prefetch", daemon=True)
    reader.start()

    processed = 0
    while True:
        item = read_q.get()
        if item is None:
            break
        p, volumes = item
        processed += 1
        t0 = time.time()
        has_roi = "ROI" if p in rois else "full frame"
        log(f"  position {p + 1}/{info.nP} ({has_roi}) [{processed}/{len(todo)} this run] …")
        try:
            if isinstance(volumes, tuple) and volumes and volumes[0] is _READ_FAIL:
                raise RuntimeError(f"read failed:\n{volumes[1]}")
            sp = dict(seg_params, roi=rois.get(p))
            seg = OA.Seg3DWorker(volumes, sp)
            errs = []
            seg.error.connect(lambda m: errs.append(m))
            masks = seg.segment_now()
            if errs:
                raise RuntimeError(errs[0])
            track_maps = OA.IoUTracker(masks, args.iou, args.memory).run()
            worker._merge_split_detections(masks, track_maps)
            rows = worker._write_position(p, volumes, masks, track_maps) or []
            append_and_rebuild(rows, rename, csv_path, xlsx_path)
            log(f"    -> {len(rows)} row(s) in {time.time() - t0:.1f}s "
                f"(cumulative track ids up to {worker._next_global_id - 1})")
            del volumes, masks, track_maps
        except Exception:
            log(f"  !! position {p + 1} FAILED — leaving it for a later resume:\n"
                + traceback.format_exc())
            continue

    reader.join(timeout=5)
    for d in out_dir.glob("position_*"):
        try:
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except Exception:
            pass
    src.close()
    log(f"==== {stem} done: workbook -> {xlsx_path} ====")


def build_parser():
    ap = argparse.ArgumentParser(
        description="Headless organoid analysis for any ND2 or TIFF dataset.")
    ap.add_argument("input", help="path to an .nd2 or .tif/.tiff dataset")
    ap.add_argument("--model", required=True, help="TensorFlow SavedModel folder")
    ap.add_argument("--out", default=None, help="output folder (default: ./<stem>)")
    ap.add_argument("--roi-dir", default=None,
                    help="folder of NNN.tif keep-region masks (1-based position)")
    # input structure / calibration
    ap.add_argument("--axes", default=None,
                    help="TIFF axis order when not in metadata, e.g. TZCYX")
    ap.add_argument("--voxel-xy", type=float, default=None, help="XY pixel size (um/px)")
    ap.add_argument("--voxel-z", type=float, default=None, help="Z step (um)")
    ap.add_argument("--time-step", type=float, default=None,
                    help="time between frames (hours)")
    ap.add_argument("--phase-channel", type=int, default=None,
                    help="0-based index of the phase/brightfield channel")
    ap.add_argument("--phase-name", default=None,
                    help="substring identifying the phase channel by name")
    ap.add_argument("--channels", nargs="*", default=None,
                    help="channel names in order (used for TIFF and for intensity "
                         "column names, e.g. --channels FITC Cy3 Phase)")
    # segmentation / tracking parameters (paper defaults)
    ap.add_argument("--min-size", type=int, default=25)
    ap.add_argument("--stitch-threshold", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--memory", type=int, default=2)
    ap.add_argument("--watershed", action="store_true", help="split touching organoids")
    ap.add_argument("--multiscale", action="store_true",
                    help="extra zoomed-out pass to recover very large organoids")
    ap.add_argument("--jpgs", action="store_true", help="also write mask/overlay JPGs")
    # selection
    ap.add_argument("--positions", type=int, nargs="*", default=None,
                    help="0-based positions to restrict to")
    ap.add_argument("--limit-positions", type=int, default=None,
                    help="only the first N positions (smoke test)")
    return ap


def main():
    args = build_parser().parse_args()
    _app = QApplication.instance() or QApplication(sys.argv)   # noqa: F841 (Qt machinery)
    run(args)
    log("ALL DONE.")


if __name__ == "__main__":
    main()
