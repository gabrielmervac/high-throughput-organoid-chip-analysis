# Organoid ND2 Analyzer (TensorFlow / OrganoID)

Fine-tunes the native-TensorFlow **OrganoID** U-Net on labeled data and provides a
PySide6 GUI to segment, Z-link, track, measure, annotate, and export organoids from
Nikon **ND2** files.

## Layout
```
organoid_tf/        cloned OrganoID repo (UNCHANGED; model + Core/ reused as-is)
dataset/            local clean copy of ChipData (training/validation/testing)
models/             fine-tuned model(s)  ->  organoid_finetuned_BEST/
outputs/            Excel exports + evaluation reports
app/
  _env.py           forces legacy Keras 2 + sys.path (import FIRST everywhere)
  stage_dataset.py  copy ChipData -> dataset/ (skips Thumbs.db, verifies pairs)
  train_finetune.py fine-tune (reuses Core.Model.TrainModel) + eval on testing split
  run_headless.py   ND2 -> Excel with no GUI (batch / verification)
  pipeline/
    nd2_reader.py   lazy ND2 access (T, P/XY, Z, C), voxel size, timing
    segment.py      OrganoID inference + watershed post-processing
    zlink.py        greedy-IoU linking of masks across Z (3D identity)
    track.py        OrganoID overlap+Hungarian tracking across time
    metrics.py      *** EDIT THIS to change measurements ***
    analysis.py     orchestration + record building
    excel_export.py keyed long-format workbook + summary sheets
  gui/
    organoid_annotator.py  the application (PyQt6) — run this
    organoid_id_tf.py      TensorFlow segmentation backend (drop-in for the old torch one)
run_gui.bat         launch the GUI
run_train.bat       stage data + fine-tune + evaluate
```

## Environment
Windows, Python 3.12, virtualenv in `.venv`. TensorFlow 2.17 + `tf-keras` 2.17 run
in **legacy Keras 2 mode** (`TF_USE_LEGACY_KERAS=1`) so OrganoID's model code is used
unchanged. Compute is **CPU** (TensorFlow has no GPU on native Windows, and the RTX
5070 Ti / Blackwell is newer than TF's CUDA build). The model is small, so this is fine.

## Usage
1. **Fine-tune** (once): `run_train.bat`  → writes `models/organoid_finetuned_BEST/`
   and `outputs/organoid_finetuned_eval.json` (Dice/IoU on the held-out testing split).
2. **Analyze**: `run_gui.bat` — the **Organoid Annotator** (adapted from the lab's
   mature PyQt6 tool; CellPose/StarDist removed, wired to the fine-tuned TF model).
   - *Load / View tab*: **Load ND2**. For a multi-position file, a **Positions** panel
     appears (view one position, or check several for batch). Navigate **T / Z**, per-channel
     contrast, **Max projection**; voxel size auto-filled from the ND2.
   - *ROI pencil* (canvas toolbar **✏ Draw ROI**): left-drag a loop to mark the region to
     **include**; draw several loops to add more. Only the ROI is segmented/tracked, and it
     applies to **every timepoint** of that XY position (draw it once on the first frame). The
     excluded area is dimmed; **Clear ROI** removes it. ROIs are per-position and saved in the project.
   - *Segment tab*: model defaults to `models/organoid_finetuned_BEST`. Choose **Per-Z
     slices + stitch** (segment every plane, link across Z — default) or **Max-Z projection**.
     "Segment Current Timepoint" / "Segment All Timepoints" run in the background.
     Segmentation uses the **phase channel** (auto-detected by name).
     **Multi-scale detection** (checkbox, default OFF): also runs a zoomed-out pass so
     organoids *much larger* than the training data (which the normal pass truncates or
     fragments) are recovered as one solid object; normal/small organoids still come from
     the standard pass. Composes with either 3-D mode; ~2× segmentation time. Turn it on
     for fields with very large organoids.
   - *Track & Export tab*: **Track All Timepoints** (3-D IoU overlap + gap-closing);
     "Color by track ID". Export **CSV / HDF5**; **Save / Load Project** (`.orgproj`).
   - *Annotate tab*: per-organoid Type / State / Notes, Delete, Relabel.
   - *Metrics tab*: per-organoid 3-D metrics table + metric-over-time plot.
   - **Run selected positions → disk**: streams each chosen position (load → segment →
     track → metrics), writing `<out>/<experiment>/position_####/metrics.xlsx` (+ optional JPGs).
3. **Batch/headless (alt pipeline)**: `.venv\Scripts\python app\run_headless.py MODEL ND2 [--positions ..] [--timepoints ..]`
   uses the separate per-Z pipeline in `app/pipeline/` and writes the long-format keyed workbook.

## Excel output
"Run selected positions → disk" writes a **single combined workbook** for the whole experiment:
`<out>/<experiment>_metrics.xlsx` (not one file per position). Rows are **sorted by organoid id**,
and organoid ids (`track_id`) are **relabeled 1..N across the entire experiment** (the same ids
shown on the overlay JPGs). Intensity columns use the **ND2 channel names** (e.g. `FITC_vol_mean`,
`Cy3_bg`). One row per organoid per timepoint, in this column order:

1. `experiment` · 2. `XY_position` (**1-based**) · 3. `track_id` (organoid identity, consistent
across time) · 4. `timepoint` (**1-based**) · 5. `time_hours`
6–8. `centroid_x/y/z` (px) · 9. `z_extent` (number of occupied planes, counts from 1) ·
10. `best_z` (plane of **maximum cross-section**)
11–13. `area_px` / `area_um2` / `diameter_um` (measured **on the max-area plane**)
14–15. `volume_vox` / `volume_um3`
16–21. channel 1 (e.g. FITC): `<name>_maxarea_{sum,mean,median}` then `<name>_vol_{sum,mean,median}`
22–27. channel 2 (e.g. Cy3): same
28–29. `<ch1>_bg` / `<ch2>_bg`   (column names use the ND2 channel names)

Intensities are **background-subtracted** (background = 25th percentile of that channel over all Z,
clipped at 0). Phase channel (ch3) intensity is **not** measured. `_maxarea` = pixels on the
max-area plane only; `_vol` = the whole 3-D organoid. The GUI metrics table and plot show this
same set.

**Discard filters** (an organoid must pass ALL to appear in the Excel and saved masks):
diameter > 40 µm · z_extent ≥ 2 planes · the track appears (as a valid detection) in > 2 frames ·
the track is present in the first OR last frame.

**Watershed toggle** (Segment tab, default OFF): when off, touching organoids stay merged
(avoids over-fragmentation); when on, a watershed step splits them.

**Batch JPGs** (optional, per "Save JPGs for"): saved at **0.5×** the original resolution, showing
only the kept organoids. `masks/` = **binary PNG** (white = organoid); `overlay/` = phase image
with the colored mask and a **large track-id label**. Filenames are 1-based (`T001_Z01.jpg`).

## Changing what is measured
Open **`app/pipeline/metrics.py`**. Each metric is a small function returning
`{column: value}`, registered with `@plane_metric` (per Z-plane) or `@volume_metric`
(whole 3-D object). Add a function to add a column; delete it to remove one. The Excel
exporter picks up whatever columns these functions return — no other change needed.
