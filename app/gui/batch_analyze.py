"""
batch_analyze.py — headless, crash-safe batch analysis of the Paper Data ND2 files.

Reproduces the GUI's "Analyze positions" pipeline EXACTLY (same segmentation,
tracking, metrics and Excel columns) by reusing organoid_annotator's own building
blocks — PositionBatchWorker helpers, Seg3DWorker, IoUTracker, compute_3d_metrics —
but drives the per-position loop here so we can:

  * read each ND2 position lazily over the network (never download the whole file),
  * write the metrics workbook incrementally (after every position), so a crash
    loses at most the position in progress, and
  * RESUME automatically: positions already present in the on-disk CSV are skipped.

ROI handling (new "selection" behaviour): the whole frame is segmented and only
organoids lying ENTIRELY within the ROI are kept. ROIs live in
    <DATA>/Mask/<stem>_Mask/NNN.tif   (white = keep-region, NNN = 1-based position)
A position with no matching tif is analysed over the full field (no ROI).

Fixed settings (match the user's GUI configuration):
    model            organoid_finetuned_orgaug_v2_BEST  (TensorFlow SavedModel)
    illumination     none (radius 50 px, unused)
    flat-field       none
    mode             per-Z slices + stitch (do_3d=True)
    multiscale       OFF
    everything else  GUI defaults (min_size 25, stitch 0.5, watershed off,
                     threshold 0.5, IoU 0.30, gap-closing 2)

Usage (from app/gui, using the project venv):
    python batch_analyze.py                       # all 5 files, all positions
    python batch_analyze.py --files 260303_24     # one file
    python batch_analyze.py --limit-positions 1 --max-timepoints 3   # quick smoke test
"""
from __future__ import annotations

# TensorFlow / Keras: OrganoID is Keras 2 — must run legacy. Set before any TF import.
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Run Qt without a display (we only need QThread/QObject machinery, no windows).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# organoid_annotator lives next to this file.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from PyQt6.QtWidgets import QApplication          # noqa: E402
import organoid_annotator as OA                    # noqa: E402


# ── Fixed configuration ───────────────────────────────────────────────────────
# This driver is the WORKED EXAMPLE that produced the paper data; DATA_DIR /
# OUT_ROOT point at the (separately deposited) paper dataset and are meant to be
# edited or overridden via environment variables. For general use on any single
# ND2 or TIFF file, prefer ``python app/analyze.py`` instead.

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR   = Path(os.environ.get("PAPER_DATA_DIR", r"Z:\Gabriel\YIYU\Paper Data"))
MASK_DIR   = DATA_DIR / "Mask"
MODEL_PATH = os.environ.get(
    "ORGANOID_MODEL",
    str(_REPO_ROOT / "models" / "organoid_finetuned_orgaug_v2_BEST"))
OUT_ROOT   = Path(os.environ.get("PAPER_OUT_DIR", _REPO_ROOT / "results"))

ALL_FILES  = ["260303_24", "260401", "260601", "260609", "260617"]

# Segmentation parameters — the user's GUI settings. `roi` is filled per position.
BASE_SEG_PARAMS = {
    "model":             "organoid_id",
    "gpu":               False,          # unused by the OrganoID/TF path (TF auto-selects)
    "min_size":          25,             # GUI default
    "stitch_threshold":  0.5,            # GUI default
    "do_3d":             True,           # Per-Z slices + stitch
    "separate_contours": False,          # watershed OFF (GUI default)
    "multiscale":        False,          # Multi-scale detection OFF
    "oid_weights_path":  MODEL_PATH,
    "phase_channel":     None,           # set per file
    "roi":               None,           # set per position
    "illum_method":      "none",         # Illumination correction OFF
    "illum_radius":      50,             # px (unused while illum_method == "none")
}
IOU_THRESH  = 0.30    # GUI default
MEMORY      = 2       # GUI default gap-closing
FLAT_METHOD = "none"  # Flat-field: none


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── ROI loading ───────────────────────────────────────────────────────────────

def load_rois_for(stem: str, expected_hw) -> dict:
    """Return {position_index (0-based): bool (H,W) keep-mask} from <stem>_Mask.

    Filenames are NNN.tif (1-based position). White (>0) = keep-region. Positions
    without a tif are absent from the dict → analysed over the full frame.
    """
    folder = MASK_DIR / f"{stem}_Mask"
    rois: dict = {}
    if not folder.is_dir():
        log(f"  (no mask folder {folder.name}; every position uses the full frame)")
        return rois
    from PIL import Image
    for tif in sorted(folder.glob("*.tif")):
        try:
            p = int(tif.stem) - 1          # NNN (1-based) -> 0-based position index
        except ValueError:
            log(f"  ! skipping mask with non-numeric name: {tif.name}")
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


# ── ND2 metadata (mirror organoid_annotator._load_nd2_from) ────────────────────

def read_nd2_meta(path: Path):
    import nd2
    with nd2.ND2File(str(path)) as f:
        sizes = dict(f.sizes)
        dims  = list(f.sizes.keys())
        try:
            names = [c.channel.name for c in f.metadata.channels]
        except Exception:
            names = [f"C{i}" for i in range(sizes.get("C", 1))]
        try:
            vs = f.voxel_size()
            voxel = (float(vs.z), float(vs.y), float(vs.x))   # (dz, dy, dx) as the GUI uses
        except Exception:
            voxel = (1.0, 0.5, 0.5)
        time_hours = []
        try:
            for lp in f.experiment:
                if type(lp).__name__ == "TimeLoop":
                    per_ms = float(lp.parameters.periodMs)
                    cnt = int(getattr(lp, "count", sizes.get("T", 1)))
                    time_hours = [round(i * per_ms / 3_600_000.0, 4) for i in range(cnt)]
                    break
        except Exception:
            pass
    phase_channel = OA.guess_phase_channel(names)
    return sizes, dims, names, voxel, time_hours, phase_channel


# ── Lazy per-position read (with optional timepoint cap for smoke tests) ────────

def read_position(worker, p, t_max=None):
    """{t: (Z,C,H,W)} for position p, read lazily. t_max limits timepoints (smoke)."""
    if t_max is None:
        return worker._read_position(p)      # exact GUI code path
    import nd2
    with nd2.ND2File(worker.nd2_path) as f:
        darr = f.to_dask()
        idx = [slice(None)] * darr.ndim
        idx[worker.pos_axis] = p
        idx[worker.dims.index("T")] = slice(0, t_max)
        sub = np.asarray(darr[tuple(idx)])
    dims_noP  = [d for d in worker.dims if d != "P"]
    canonical = [d for d in ["T", "Z", "C", "Y", "X"] if d in dims_noP]
    order     = [dims_noP.index(d) for d in canonical]
    sub       = np.transpose(sub, order).astype(np.float32)
    for ax_i, ax in enumerate(["T", "Z", "C", "Y", "X"]):
        if ax not in canonical:
            sub = np.expand_dims(sub, ax_i)
    return {t: sub[t] for t in range(sub.shape[0])}


# ── Incremental workbook writer (mirrors PositionBatchWorker._write_workbook) ───

def build_rename(channel_names, phase_channel):
    import re
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
    """Append this position's rows to the durable CSV, then rebuild the xlsx.

    The CSV is the crash-safe source of truth (append-only, flushed each position);
    the xlsx is a convenience view rebuilt from the full CSV after every position.
    """
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
        os.replace(tmp, xlsx_path)          # atomic swap
    except Exception as e:
        log(f"  ! could not write xlsx ({e}); CSV is up to date at {csv_path.name}")


def done_positions_from_csv(csv_path: Path):
    """(set of 0-based positions already written, next experiment-wide track id)."""
    if not csv_path.exists():
        return set(), 1
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c in ("XY_position", "track_id"))
    except Exception:
        return set(), 1
    done = set(int(v) - 1 for v in df.get("XY_position", pd.Series([])).unique())
    next_id = (int(df["track_id"].max()) + 1) if ("track_id" in df and len(df)) else 1
    return done, next_id


# ── Per-file processing ─────────────────────────────────────────────────────────

def process_file(stem: str, positions_filter=None, t_max=None):
    path = DATA_DIR / f"{stem}.nd2"
    if not path.exists():
        log(f"SKIP {stem}: file not found ({path})")
        return

    log(f"==== {stem} ====")
    sizes, dims, names, voxel, time_hours, phase_channel = read_nd2_meta(path)
    nP = int(sizes.get("P", 1))
    hw = (sizes["Y"], sizes["X"])
    log(f"  P={nP} T={sizes.get('T')} Z={sizes.get('Z')} C={sizes.get('C')} "
        f"channels={names} phase='{names[phase_channel]}' voxel(z,y,x)={voxel}")

    rois = load_rois_for(stem, hw)

    out_dir = OUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / f"{stem}_metrics_incremental.csv"
    xlsx_path = out_dir / f"{stem}_metrics.xlsx"

    positions = list(range(nP))
    if positions_filter is not None:
        positions = [p for p in positions if p in positions_filter]

    done, next_gid = done_positions_from_csv(csv_path)
    todo = [p for p in positions if p not in done]
    if done:
        log(f"  resume: {len(done)} position(s) already written; {len(todo)} remaining")

    # A worker instance purely to reuse its exact helper methods.
    worker = OA.PositionBatchWorker(
        positions, str(path), dims, dims.index("P"), str(out_dir),
        dict(BASE_SEG_PARAMS), IOU_THRESH, MEMORY, voxel,
        flat_method=FLAT_METHOD, image_positions=set(),   # metrics only, no JPGs
        experiment=stem, rois=rois)
    worker.phase_channel = phase_channel
    worker.time_hours    = time_hours
    worker.channel_names = names
    worker._next_global_id = next_gid

    rename = build_rename(names, phase_channel)

    # ── Prefetch pipeline ───────────────────────────────────────────────────────
    # Network reads (~38 s/pos) and GPU segmentation (~98 s/pos) are otherwise
    # serial. A background thread reads the NEXT position while the main thread
    # segments the current one, hiding the read behind compute (no local copy /
    # extra disk needed). Queue size 1 → at most 2 volumes (~8 GB) in RAM at once.
    _READ_FAIL = object()
    read_q: "queue.Queue" = queue.Queue(maxsize=1)

    def _producer():
        for p in todo:
            try:
                read_q.put((p, read_position(worker, p, t_max=t_max)))
            except Exception:
                read_q.put((p, (_READ_FAIL, traceback.format_exc())))
        read_q.put(None)   # sentinel: no more positions

    reader = threading.Thread(target=_producer, name="nd2-prefetch", daemon=True)
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
        log(f"  position {p + 1}/{nP} ({has_roi}) [{processed}/{len(todo)} this run] …")
        try:
            if isinstance(volumes, tuple) and volumes and volumes[0] is _READ_FAIL:
                raise RuntimeError(f"read failed:\n{volumes[1]}")

            seg_params = dict(BASE_SEG_PARAMS, roi=rois.get(p))
            seg = OA.Seg3DWorker(volumes, seg_params)
            errs = []
            seg.error.connect(lambda m: errs.append(m))
            masks = seg.segment_now()
            if errs:
                raise RuntimeError(errs[0])

            track_maps = OA.IoUTracker(masks, IOU_THRESH, MEMORY).run()
            worker._merge_split_detections(masks, track_maps)
            rows = worker._write_position(p, volumes, masks, track_maps) or []

            append_and_rebuild(rows, rename, csv_path, xlsx_path)
            log(f"    -> {len(rows)} row(s) in {time.time() - t0:.1f}s "
                f"(cumulative track ids up to {worker._next_global_id - 1})")
            del volumes, masks, track_maps
        except Exception:
            log(f"  !! position {p + 1} FAILED — leaving it for a later resume:\n"
                + traceback.format_exc())
            # Keep going: other positions are independent.
            continue

    reader.join(timeout=5)

    # Tidy the empty position_#### folders _write_position creates in metrics-only mode.
    for d in out_dir.glob("position_*"):
        try:
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except Exception:
            pass

    log(f"==== {stem} done: workbook -> {xlsx_path} ====")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Batch-analyze the Paper Data ND2 files.")
    ap.add_argument("--files", nargs="*", default=ALL_FILES,
                    help=f"file stems to process (default: {ALL_FILES})")
    ap.add_argument("--positions", type=int, nargs="*", default=None,
                    help="0-based position indices to restrict to (default: all)")
    ap.add_argument("--limit-positions", type=int, default=None,
                    help="process only the first N positions of each file (smoke test)")
    ap.add_argument("--max-timepoints", type=int, default=None,
                    help="cap timepoints per position (smoke test only; NOT for real runs)")
    args = ap.parse_args()

    if not os.path.isdir(MODEL_PATH):
        log(f"FATAL: model folder not found: {MODEL_PATH}")
        sys.exit(1)

    _app = QApplication.instance() or QApplication(sys.argv)   # noqa: F841 (Qt machinery)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"Output root: {OUT_ROOT}")
    if args.max_timepoints:
        log(f"*** SMOKE MODE: capping timepoints at {args.max_timepoints} ***")

    for stem in args.files:
        pf = None
        if args.positions is not None:
            pf = set(args.positions)
        elif args.limit_positions is not None:
            pf = set(range(args.limit_positions))
        try:
            process_file(stem, positions_filter=pf, t_max=args.max_timepoints)
        except Exception:
            log(f"FILE {stem} crashed:\n" + traceback.format_exc())

    log("ALL DONE.")


if __name__ == "__main__":
    main()
