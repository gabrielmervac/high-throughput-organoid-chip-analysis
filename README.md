# High-throughput organoid-chip image analysis

Image-analysis code for the study **"High-throughput microfluidic platform
resolves schedule-dependent drug responses in patient-derived colorectal
organoids."**

The platform is a two-layer, valve-based PDMS chip with **208 individually
addressable, matrix-compatible chambers**. Each chamber sustains matrix-embedded
patient-derived organoids for a week and can be programmed with its own drug
combination, concentration and **delivery schedule** (before / with / after a
chemotherapy backbone) drawn from up to 16 inputs. This repository contains the
computational pipeline that reads the resulting time-lapse microscopy and
quantifies organoid **growth** and **viability** to read out drug response.

The pipeline fine-tunes the [OrganoID](https://github.com/jono-m/OrganoID) U-Net
(Matthews *et al.*, *PLoS Comput. Biol.* 2022) for on-chip, matrix-embedded
organoids and adds 3-D reconstruction and tracking suited to confocal-style
Z-stacks. **It runs on any Nikon ND2 or TIFF (OME-TIFF / ImageJ hyperstack)
time-lapse Z-stack.**

> **Scope.** This repo is the image-analysis half of the study. The microfluidic
> device-control code (MATLAB) and the downstream statistics/figure code
> (normalization, two-way ANOVA, phase-plane trajectories, heatmaps) are
> maintained separately. Raw microscopy and the training dataset are too large
> for git and will be deposited on a data repository (e.g. Zenodo); this repo
> ships the code, the final fine-tuned model, and the per-organoid results.

---

## What's in here

```
app/
  analyze.py            general headless analyzer — ANY .nd2 or .tif/.tiff  ← start here
  io/volume_source.py   format-agnostic reader (ND2 + OME-TIFF/ImageJ/plain TIFF)
  train_finetune.py     two-stage fine-tuning of OrganoID (reuses Core.Model.TrainModel)
  augment_data.py       OrganoID's stock geometric augmentation recipe
  stage_dataset.py      stage annotated image/mask pairs into dataset/
  stage_large_organoids.py   stage the large-organoid supplement
  run_headless.py       alternate modular pipeline (pipeline/, OrganoID Hungarian tracker)
  smoke_test.py         confirm TF + legacy Keras loads OrganoID unchanged
  gui/
    organoid_annotator.py  interactive PyQt6 annotator (segment/track/measure/review)
    organoid_id_tf.py      TensorFlow segmentation backend
    batch_analyze.py       paper-data batch driver (kept as a worked example)
  pipeline/             modular reference pipeline (segment / zlink / track / metrics)
models/
  organoid_finetuned_orgaug_v2_BEST/   final fine-tuned model used for all results
outputs/
  organoid_finetuned_orgaug_v2_eval.json   held-out Dice/IoU for the final model
results/                per-experiment per-organoid metrics workbooks (source data)
docs/
  methods_image_analysis.md   the Methods, traced to code
  pipeline_reference.md       detailed pipeline / GUI reference
organoid_tf/            OrganoID upstream — a git SUBMODULE (see Install)
```

## Install

```bash
git clone --recurse-submodules https://github.com/gabrielmervac/high-throughput-organoid-chip-analysis
cd high-throughput-organoid-chip-analysis
# if you cloned without --recurse-submodules:
git submodule update --init

python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

`organoid_tf/` is the unmodified OrganoID repository, referenced as a submodule
(its architecture, training routine, preprocessing and post-processing are reused
as-is). TensorFlow runs on **CPU** here and OrganoID's Keras-2 model is run in
legacy mode — the entry points set `TF_USE_LEGACY_KERAS=1` for you.

## Quickstart — analyze a dataset

The general analyzer runs the exact published pipeline (per-Z OrganoID
segmentation → greedy-IoU Z-stitching for 3-D identity → greedy 3-D
volumetric-IoU tracking with gap-closing → split-merge + inclusion filters →
3-D growth/viability metrics) and writes one long-format workbook.

**Nikon ND2** (calibration and channel names come from the file):

```bash
python app/analyze.py experiment.nd2 \
    --model models/organoid_finetuned_orgaug_v2_BEST \
    --out results/experiment
```

**OME-TIFF / ImageJ hyperstack** (axes auto-detected; give calibration if the
file lacks it):

```bash
python app/analyze.py experiment.ome.tif \
    --model models/organoid_finetuned_orgaug_v2_BEST \
    --out results/experiment \
    --voxel-xy 0.65 --voxel-z 5 --time-step 4 \
    --channels FITC Cy3 Phase
```

**Plain multi-dimensional TIFF** (declare the axis order yourself):

```bash
python app/analyze.py stack.tif --axes TZCYX \
    --model models/organoid_finetuned_orgaug_v2_BEST \
    --out results/stack --voxel-xy 0.65 --voxel-z 5 --time-step 4
```

The run is **crash-safe and resumable**: metrics stream to an append-only CSV
(source of truth) with an atomically rebuilt `.xlsx` after every position, and
finished positions are skipped on resume. `python app/analyze.py -h` lists all
options (segmentation thresholds, tracking IoU/memory, ROI folder, watershed,
multi-scale, JPG export, position selection).

### Input structure & calibration

The pipeline needs axes **T** (time), **Z** (focus), **C** (channel), **Y**, **X**,
and one XY-**position** dimension. For TIFF, positions are the TIFF *series*
(one position per series — the OME multi-point convention); single-series files
are single-position. The **phase-contrast/brightfield channel** is the
segmentation input — auto-detected by name, or set it with `--phase-channel` /
`--phase-name`. Voxel size and frame interval are read from ND2 or from
OME/ImageJ metadata when present, and can always be overridden with
`--voxel-xy` (µm/px), `--voxel-z` (µm) and `--time-step` (hours).

## Interactive GUI

```bash
python app/gui/organoid_annotator.py          # or run_gui.bat on Windows
```

Load an ND2/TIFF, draw include-ROIs, segment (per-Z + stitch, optional
multi-scale for very large organoids), track over time, review/relabel
organoids, inspect the 3-D metrics table and metric-over-time plots, and export
CSV/HDF5 or run selected positions to disk. See
[`docs/pipeline_reference.md`](docs/pipeline_reference.md).

## The model

`models/organoid_finetuned_orgaug_v2_BEST` is the OrganoID U-Net after two-stage
fine-tuning (chip organoids, then chip + a large-organoid supplement). Held-out
test performance (n = 208): **Dice 0.83, IoU 0.72**
(`outputs/organoid_finetuned_orgaug_v2_eval.json`). To reproduce training see
[`docs/methods_image_analysis.md`](docs/methods_image_analysis.md) and
`app/train_finetune.py` (`run_train.bat` / `run_augment_train.bat`); this needs
the training dataset (deposited separately).

## Results

`results/<experiment>/<experiment>_metrics.xlsx` — one row per organoid per
timepoint: identity, geometry (centroid, Z-extent, best focus plane, area,
diameter, trapezoidal volume) and background-subtracted per-channel fluorescence
(live/dead), with column names from the acquisition channels. These are the
source data underlying the growth and viability readouts in the manuscript.

## Citing

Please cite the associated manuscript (see `CITATION.cff`) **and** the OrganoID
paper this builds on (see `NOTICE`).

## Acknowledgements

During the creation of this code we used Anthropic's Claude as an AI coding
assistant. All code was reviewed and validated by the authors.

## License

Original code in this repository is released under the **MIT License**
(`LICENSE`). The upstream OrganoID code under `organoid_tf/` is a submodule and
remains under the terms of its own repository. See `NOTICE` for attribution and
a summary of what is reused unchanged versus added here.
