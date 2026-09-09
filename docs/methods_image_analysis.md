# Organoid segmentation, 3-D linking and tracking — Methods (for reporting)

This document describes, exactly as implemented in the code, the models, training
procedure, and image-analysis pipeline used to segment, reconstruct in 3-D, track and
measure organoids from the Nikon ND2 time-lapse Z-stacks (the "Paper Data" experiments
`260303_24`, `260401`, `260601`, `260609`, `260617`).

All source is in `D:\Yiyu\Organoid ID Finetuned`. File/line references are given so every
statement can be traced back to code.

---

## 1. Software and base network

Segmentation is built on **OrganoID** (Matthews et al.), a U-Net trained for label-free
brightfield/phase-contrast organoid segmentation. We used OrganoID's **native TensorFlow /
Keras** implementation *unmodified* for the network and its forward pass; the repository is
vendored under `organoid_tf/` and its `Core/` code is imported as-is.

- Framework: TensorFlow 2.17 + `tf-keras` 2.17 run in **legacy Keras-2 mode**
  (`TF_USE_LEGACY_KERAS=1`) so OrganoID's original model code runs unchanged
  (`app/_env.py`).
- Compute: **CPU only** (the installed TensorFlow has no GPU build compatible with the
  workstation's RTX 5070 Ti / Blackwell). The U-Net is small, so CPU inference is practical.
- Hardware: Intel i5-13600K (14 physical / 20 logical cores), 64 GB RAM.

**Network architecture** (`organoid_tf/Core/Model.py`, `BuildModel`): a 5-level U-Net.
Input 512 × 512, 3-channel (grayscale replicated); contracting path of 5 stages with
`3×3` convolutions (ELU activation, He-normal init, "same" padding), 2×2 max-pooling
between stages, and per-stage dropout scaled by depth (`dropoutRate·i`); symmetric
expanding path with `2×2` transposed convolutions and skip-concatenations; final `1×1`
convolution with **sigmoid** activation producing a per-pixel foreground probability
("belief") map. Base hyperparameters: `firstLayerFilterCount = 8`, `dropoutRate = 0.125`,
input size `512×512` (OrganoID's published `TrainableModel` defaults;
`organoid_tf/CommandLine/Train.py`).

---

## 2. Training data

Annotated **phase-contrast** frames of on-chip organoids (the lab "ChipData" set), staged
locally as paired 8-bit PNG image/mask pairs and split into disjoint sets
(`app/stage_dataset.py`):

| Split | Pairs | Use |
|---|---|---|
| training | 806 | fine-tuning |
| validation | 208 | early-stopping monitor |
| testing | 208 | held-out evaluation (never seen in training) |

Masks are binary (foreground = organoid). Every image is min–max normalized to 8-bit at
load, so absolute brightness/contrast are removed before the network sees the frame
(`Core.Model.PrepareImagesForModel`; mirrored in `organoid_id_tf._prepare_gray`).

**Large-organoid supplement.** Because the ChipData organoids are mostly small
(median ≈ 38 µm) and the network fragmented organoids much larger than the training
distribution, we added a dedicated set of **36 large-organoid** annotated frames (16-bit
phase + binary mask), staged and 16→8-bit min–max converted to match the training PNGs
(`app/stage_large_organoids.py`; 35 usable image/mask pairs after pairing).

---

## 3. Data augmentation

Augmentation expands **only the training split**; validation and testing always stay raw.

The training data were augmented with **OrganoID's stock recipe, unchanged**
(`app/augment_data.py --recipe organoid`, OrganoID's own `Core.DataPreparation.AugmentImages`):
random rotation ±20°, horizontal/vertical flips, zoom (factor 0.7), shear ±20°, elastic
(random) distortion, skew, and resize to 512×512. This is **purely geometric** — the
identical transform is applied to image and mask.

- ChipData training set → **2806** pairs (806 originals + 2000 augmentations).
- Large-organoid frames → **735** pairs (35 originals + 700 augmentations), same recipe.

---

## 4. Fine-tuning procedure

Fine-tuning reuses OrganoID's own `Core.Model.TrainModel` unchanged; our launcher
(`app/train_finetune.py`) only adds robust image↔mask pairing, a fine-tuning learning
rate, and held-out evaluation. Training settings (all runs):

- Optimizer **Adam**, learning rate **1×10⁻⁴** (`-LR`).
- Loss **binary cross-entropy** (`Core.Model.TrainModel`).
- Batch size **8**; up to **100** epochs.
- **Early stopping** on validation loss, patience **10**, `restore_best_weights=True`
  (the saved `_BEST` model = the weights from the best validation epoch).
- Data shuffled each epoch; validation = the raw 208-pair split.

### 4.1 Model lineage

The segmentation model was produced by two successive fine-tuning stages, both starting
from OrganoID's published `TrainableModel`. The final model used for all Paper Data
analysis is **`organoid_finetuned_orgaug_v2_BEST`**.

| Stage | Init from | Training set | Pairs | Epochs run (best) |
|---|---|---|---|---|
| Stage 1 | OrganoID base | stock-augmented ChipData (`training_augmented_organoid`) | 2806 | 88, best 78 |
| **Stage 2 (final model used)** | Stage-1 model | ChipData + large-organoid (`training_v2`) | **3541** | 20, best 10 |

`training_v2` = the stock-augmented ChipData set (2806) **+** the large-organoid augmented
set (735) = **3541** pairs. The network was first fine-tuned from base OrganoID on the
stock-augmented chip organoids (Stage 1), then continued on the combined set that
additionally included the large-organoid frames to recover big organoids (Stage 2, the
final model).

### 4.2 Held-out evaluation of the final model (`testing` split, n = 208)

Per-image semantic Dice and IoU at threshold 0.5, plus size-stratified **per-object** IoU
(each ground-truth object vs its best-overlapping prediction; size bucket = fraction of
frame area, "large" ≥ 3 %). Values written to `outputs/organoid_finetuned_orgaug_v2_eval.json`.

| Dice | IoU | per-object IoU (all / small / medium / large) |
|---|---|---|
| 0.828 | 0.717 | 0.482 / 0.375 / 0.742 / 0.816 |

---

## 5. Inference pipeline (per frame)

Implemented in `app/gui/organoid_id_tf.py`; the GUI and the headless batch call the exact
same functions.

1. **Pre-processing** (`_prepare_gray`): take the phase channel, resize to 512×512
   (bilinear), min–max normalize to [0,255]. Identical to OrganoID's
   `PrepareImagesForModel`.
2. **Forward pass** (`detect`): the U-Net produces a [0,1] belief map. Unchanged OrganoID
   model.
3. **Post-processing** (`postprocess`, mirrors OrganoID `Core/Identification.py`):
   threshold the belief map at **0.5**, morphological binary opening, hole-filling,
   connected-component labelling, and removal of objects below a minimum area of
   `min_size = 25` px (at model scale). Touching organoids were kept merged.

---

## 6. 2-D → 3-D reconstruction across Z (the key adaptation)

OrganoID was developed for **2-D** images. Our data are **Z-stacks**, so we segment in 2-D
and reconstruct 3-D identity ourselves. The mode used for the Paper Data is **per-Z +
stitch** (`segment_volume_per_z` in `organoid_id_tf.py`):

1. **Segment every Z-plane independently** with the pipeline of §5, giving one instance-
   label mask per plane (planes are batched through the network together for speed).
2. **Link labels across Z by greedy IoU stitching** (`_stitch_z`): planes are processed
   from bottom to top. For each object in plane *z*, its 2-D IoU with every object in
   plane *z−1* is computed; candidate pairs with **IoU ≥ `stitch_threshold`** are greedily
   matched in descending-IoU order (each plane-*z* object and each plane-*(z−1)* object
   matched at most once). A matched object **inherits the ID** of the plane below;
   unmatched objects **start a new ID**. The result is a 3-D label volume in which one
   physical organoid carries a single consistent label through all the planes it spans.
   - For the Paper Data, `stitch_threshold = 0.5`.

This per-Z-segment-then-IoU-stitch step is our extension; OrganoID itself has no Z / 3-D
handling.

### Region-of-interest (ROI) as a *post-hoc selection*
Where an experiment defined an ROI (`Mask/<stem>_Mask/NNN.tif`, white = keep-region,
`NNN` = 1-based position), we **segment the whole frame** and then keep only organoids that
lie **entirely** within the ROI; any labelled object with even one voxel outside the ROI
(fully out, or straddling the border) is discarded in full (`_select_organoids_in_roi`).
Applying the ROI as a selection after full-frame segmentation (rather than masking the
input beforehand) keeps segmentation homogeneous across the frame, avoids boundary
artifacts at the ROI edge, and never produces partially-clipped area/volume. Positions with
no ROI file are analysed over the full field.

### Illumination
No illumination or flat-field correction was applied; segmentation ran on the raw
min–max-normalized phase channel.

---

## 7. Tracking organoids over time

Time-linking is done by our **3-D IoU tracker with gap-closing**
(`IoUTracker`, `app/gui/organoid_annotator.py`), applied to the per-timepoint 3-D label
volumes:

- Frames are processed in time order; every object in the first frame seeds a track.
- For each object at time *t*, its 3-D voxel region is intersected with the objects of the
  most recent previous frame(s); it inherits the track of the previous object with the
  **highest volumetric IoU**, provided **IoU ≥ `iou_thresh`**; otherwise it starts a new
  track. IoU uses pre-computed per-frame voxel counts (intersection from the previous
  labels lying under the new region; union = |new| + |old| − |inter|).
- **Gap closing (`memory`)**: if no qualifying match exists in the immediately preceding
  frame, up to `memory` earlier frames are checked (nearest first), so a track survives a
  few missed detections.
- **Splits are allowed** (two new objects may inherit the same track), matching the
  reference behaviour.
- Paper Data settings: `iou_thresh = 0.30`, `memory = 2`.

**Relationship to OrganoID's tracker.** OrganoID's original tracker
(`organoid_tf/Core/Tracking.py`) links 2-D detections between consecutive frames by
**Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) on an overlap cost with a
fixed cost-of-non-assignment and a lost-track cutoff. We replaced this with the greedy
**3-D volumetric-IoU** tracker above because (i) our objects are 3-D volumes, not 2-D
regions, and (ii) greedy IoU with an explicit threshold and gap-closing is more robust than
global-assignment centroid/overlap linking for large, crowded, slowly-drifting organoids,
where global assignment tends to swap identities between neighbours. The original Hungarian
tracker remains in the vendored repo but was **not** used for this analysis.

### Post-tracking cleanup and inclusion filters
- **Merge split detections** (`_merge_split_detections`): within a timepoint, all
  detections that the tracker assigned to the same track id are merged into one labelled
  object (fixes double-counting of a fragmented organoid → one row per organoid per
  timepoint).
- **Inclusion filters** — an organoid must pass **all** of the following to appear in the
  output (`PositionBatchWorker._write_position`):
  1. diameter (on its maximum-area plane) **> 40 µm**;
  2. Z-extent **≥ 2 planes**;
  3. its track is present (as a valid detection) in **> 2 timepoints**; and
  4. its track appears in the **first or last** timepoint.
- **Identity relabelling**: surviving tracks are relabelled to experiment-wide ids 1..N
  (assigned in order, continuous across all positions of an experiment).

---

## 8. Quantification (per organoid, per timepoint)

Computed in `compute_3d_metrics` (`organoid_annotator.py`). Voxel size is read from the
ND2 metadata. For each organoid:

- **Geometry**: centroid (x,y,z); `z_extent` (number of occupied planes); `best_z` (plane
  of maximum cross-section); on that max-area plane, `area_px`, `area_um2`, and
  `diameter_um` = 2·√(area/π); `volume_vox`; and **`volume_um3`** computed by
  **trapezoidal integration of cross-sectional area along Z** between the first and last
  occupied planes (integrating the k−1 gaps rather than summing k slabs avoids the
  half-slab overhang of a naïve voxel count).
- **Fluorescence intensities** (all non-phase channels), reported both on the max-area
  plane (`_maxarea`) and over the whole 3-D object (`_vol`), as sum / mean / median. Every
  intensity is **background-subtracted**: background = **25th percentile** of that channel
  over background (non-object) voxels, clipped at 0 (`channel_backgrounds`,
  `BG_PERCENTILE = 25`). The phase channel intensity is not measured. Column names use the
  ND2 channel names.

Output is one row per organoid per timepoint (long format), plus `time_hours` from the ND2
per-frame timing.

---

## 9. Batch analysis of the Paper Data (exact configuration used)

Headless, crash-safe driver `app/gui/batch_analyze.py` reproduces the GUI "Analyze
positions" pipeline exactly by reusing the same building blocks (`Seg3DWorker`,
`IoUTracker`, `compute_3d_metrics`, `PositionBatchWorker` helpers). ND2 files were read
**lazily over the network** (per position, via `nd2` + dask — never downloading whole
files), with a one-position prefetch thread overlapping network read with CPU compute.
Results were written **incrementally** (append-only CSV as source of truth + atomically
rebuilt XLSX after every position) so a crash loses at most the position in progress, and
finished positions are **skipped on resume**.

**Fixed settings for all five files** (`BASE_SEG_PARAMS`):

| Parameter | Value |
|---|---|
| Model | `organoid_finetuned_orgaug_v2_BEST` (TF SavedModel) |
| Mode | per-Z slices + stitch (`do_3d = True`) |
| Belief threshold | 0.5 |
| Minimum object area | 25 px (model scale) |
| Z-stitch IoU threshold | 0.50 |
| Time-tracking IoU threshold | 0.30 |
| Gap-closing memory | 2 frames |
| ROI | per-position `Mask/<stem>_Mask/NNN.tif`, keep-fully-inside (else full frame) |
| Inclusion filters | diameter > 40 µm; z_extent ≥ 2; track in > 2 frames; track in first or last frame |

Runs were parallelized simply by launching several **single-file** batch instances
concurrently (disjoint files, separate outputs) rather than in-process multiprocessing, so
results are deterministic. Deliverables: `D:\Yiyu\Paper_Data_Analysis\<stem>\<stem>_metrics.xlsx`
(+ `_metrics_incremental.csv`). Totals across the five experiments: 2,357 tracked
organoids, 51,439 organoid-timepoint rows.

---

## 10. Reproducibility pointers

- Training: `run_train.bat` / `run_augment_train.bat` → `app/train_finetune.py`
  (`--init-model`, `--train-dir`, `-E/-B/-LR/-P`). Augmentation seed defaults to 0
  (`app/augment_data.py --seed`).
- Evaluation JSONs: `outputs/organoid_*_eval.json`. Training logs:
  `outputs/train_*_log.txt`.
- Inference/tracking/metrics: `app/gui/organoid_id_tf.py`, `app/gui/organoid_annotator.py`.
- Batch: `run_batch_analysis.bat` → `app/gui/batch_analyze.py`.
- The vendored, unmodified OrganoID network/forward-pass/post-processing reference is under
  `organoid_tf/Core/` (`Model.py`, `Identification.py`, `Tracking.py`).

### What is OrganoID (unchanged) vs. what we added
- **Unchanged from OrganoID:** U-Net architecture, training routine (Adam + BCE + early
  stopping), image preprocessing (min–max 512×512), belief-map forward pass, and the 2-D
  post-processing (threshold, opening, size filtering).
- **Our additions/changes:** a large-organoid training supplement (augmented with the same
  stock OrganoID recipe); two-stage fine-tuning on chip + large-organoid data; **per-Z
  segmentation with greedy-IoU Z-stitching for 3-D identity**; **greedy 3-D-IoU time tracker
  with gap-closing** (replacing OrganoID's Hungarian 2-D tracker); ROI-as-post-hoc-selection;
  split-detection merging; inclusion filters; and the 3-D metric suite (trapezoidal volume,
  background-subtracted per-channel intensities).
