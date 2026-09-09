"""Offline training-data augmentation.

READ-ONLY SOURCE: the raw dataset on Z:\\Yiyu_Zhang\\ChipData is never touched.
`stage_dataset.py` first copies it to D:\\...\\dataset\\ (local, writable); this
script reads that local copy and writes the augmented set back under D:\\...\\dataset\\.
It refuses to write anywhere on the read-only source.

Two recipes (choose with --recipe):

  organoid  -- OrganoID's OWN recipe, UNCHANGED (Core.DataPreparation.AugmentImages):
               rotate +/-20, H/V flip, zoom_random(0.7), shear +/-20, elastic
               distortion, skew, resize 512. Purely geometric. Use as the A/B baseline.

  enhanced  -- (default) same geometry backbone but improved for the big-organoid
               problem, in two stages:
                 1. GEOMETRY via Augmentor (image + mask together): rotate, flips,
                    RANGED magnify zoom(1.0..1.8) instead of the fixed 1.2x crop-zoom
                    (this manufactures big-organoid views), milder shear/skew/distortion.
                 2. PHOTOMETRY as a mask-free post-pass on the IMAGES ONLY: a synthetic
                    low-frequency illumination field + gamma + noise + occasional blur.
               Why a separate photometric pass: Augmentor applies every op to the mask
               too (Operations.py RandomBrightness), which would corrupt binary labels.
               Why these specific photometric ops: the model min-max normalises every
               image at load, so GLOBAL brightness/contrast is undone before the network
               sees it -- only NONLINEAR (gamma) and SPATIAL (illumination field, blur,
               noise) changes survive and actually build robustness.

  --size-weight (default on for enhanced): oversample source frames that contain a
               large organoid, so rare big organoids appear more often in the augmented
               set. Uses the masks you already have; zero extra disk (in-memory).

It expands ONLY the training split. Validation and testing stay raw.

Usage:
    python app/augment_data.py                          # enhanced, 2000, size-weighted
    python app/augment_data.py --recipe organoid        # stock OrganoID baseline
    python app/augment_data.py --count 3000 --no-photometric
Then:
    python app/train_finetune.py --train-dir dataset/training_augmented
"""
import _env  # noqa: F401  (sets TF_USE_LEGACY_KERAS + puts organoid_tf on sys.path)

import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

DATASET = _env.DATASET_DIR
SRC_READONLY = Path(os.environ.get("CHIPDATA_SRC", r"Z:\Yiyu_Zhang\ChipData"))


# ── helpers ──────────────────────────────────────────────────────────────────

def _reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _guard_output(out: Path):
    """Never write to the read-only source share."""
    out_res = out.resolve()
    if out_res == SRC_READONLY or SRC_READONLY.resolve() in out_res.parents:
        raise SystemExit(f"Refusing to write inside the read-only source: {SRC_READONLY}")


def _rearrange_output(out: Path):
    """Move Augmentor's flat output into images/ and segmentations/ subfolders,
    renaming each pair to its shared uuid. Mirrors Core.DataPreparation.AugmentImages."""
    out_img, out_seg = out / "images", out / "segmentations"
    files = [p for p in out.iterdir() if p.is_file()]
    seg_files = [p for p in files if re.match("_groundtruth", p.stem)]
    img_files = [p for p in files if p not in seg_files]
    for f in seg_files:
        f.rename(out_seg / re.sub(".*_", "", f.name))
    for f in img_files:
        f.rename(out_img / re.sub(".*_", "", f.name))


# ── enhanced geometry (Augmentor) ────────────────────────────────────────────

def _build_enhanced_pipeline(in_img: Path, in_seg: Path, out: Path):
    import Augmentor
    p = Augmentor.Pipeline(source_directory=str(in_img.resolve()),
                           output_directory=str(out.resolve()))
    p.set_save_format("auto")
    p.ground_truth(str(in_seg.resolve()))

    p.rotate(probability=0.9, max_left_rotation=20, max_right_rotation=20)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    # Ranged MAGNIFY (>=1): synthesises views where organoids fill more of the field.
    # (min_factor kept >=1.0 on purpose: factor<1 pads with black and creates fake edges.)
    p.zoom(probability=0.75, min_factor=1.0, max_factor=1.8)
    # Milder than the stock recipe -- heavy shear/skew/distortion warps boundaries.
    p.shear(probability=0.4, max_shear_left=12, max_shear_right=12)
    p.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=2)
    p.skew(probability=0.3, magnitude=0.2)
    p.resize(1, 512, 512)
    return p


def _apply_size_weighting(pipeline, frac_thresh: float, max_extra: int):
    """Duplicate (in memory) the sampling slots of frames whose largest organoid
    covers >= frac_thresh of the frame, so big organoids are sampled more often."""
    try:
        from scipy import ndimage
    except Exception:
        print("size-weight: scipy unavailable -- skipping (uniform sampling).")
        return
    base = list(pipeline.augmentor_images)
    extra, big = [], 0
    for ai in base:
        gt = getattr(ai, "ground_truth", None)
        if not gt:
            continue
        try:
            m = np.asarray(Image.open(gt).convert("1"))
        except Exception:
            continue
        lbl, n = ndimage.label(m)
        if n == 0:
            continue
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        max_frac = counts.max() / m.size
        if max_frac >= frac_thresh:
            big += 1
            k = min(max_extra, int(max_frac // frac_thresh))
            extra.extend([ai] * k)
    pipeline.augmentor_images = base + extra
    print(f"size-weight: {big}/{len(base)} frames have a large organoid "
          f"(>= {frac_thresh:.0%} of frame); added {len(extra)} extra sampling slots "
          f"(pool now {len(pipeline.augmentor_images)}).")


# ── photometric post-pass (images only) ──────────────────────────────────────

def _illum_field(h: int, w: int, rng, max_amp: float) -> np.ndarray:
    """Smooth low-frequency multiplicative field in a random direction -- a synthetic
    illumination gradient. Being SPATIAL, it survives the model's per-image min-max
    normalisation (unlike a global brightness change)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= max(w - 1, 1)
    yy /= max(h - 1, 1)
    ang = rng.uniform(0, 2 * np.pi)
    ramp = np.cos(ang) * xx + np.sin(ang) * yy
    ramp -= ramp.min()
    ramp /= (ramp.max() + 1e-6)
    amp = rng.uniform(0, max_amp)
    return (1.0 + amp * (ramp - 0.5) * 2.0).astype(np.float32)


def _photometric_pass(img_dir: Path, seed: int, max_amp: float = 0.5):
    rng = np.random.default_rng(seed)
    n = 0
    for p in sorted(img_dir.glob("*.png")):
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        h, w = arr.shape
        arr = arr * _illum_field(h, w, rng, max_amp)          # spatial shading
        mx = float(arr.max()) or 1.0                          # gamma (nonlinear)
        arr = mx * np.power(np.clip(arr, 0, mx) / mx, rng.uniform(0.6, 1.7))
        if rng.random() < 0.5:                                # noise
            arr = arr + rng.normal(0, rng.uniform(2, 8), arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr, "L")
        if rng.random() < 0.3:                                # occasional blur
            im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.3)))
        im.save(p)
        n += 1
    print(f"photometric: jittered {n} augmented images (masks untouched).")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Augment the OrganoID training split.")
    ap.add_argument("--input", default=str(DATASET / "training"),
                    help="Split dir with images/ and segmentations/ (default: dataset/training).")
    ap.add_argument("--out", default=str(DATASET / "training_augmented"),
                    help="Output dir (default: dataset/training_augmented).")
    ap.add_argument("--recipe", choices=["organoid", "enhanced"], default="enhanced",
                    help="'organoid' = stock OrganoID recipe; 'enhanced' = improved (default).")
    ap.add_argument("-N", "--count", type=int, default=2000,
                    help="Number of augmented samples to generate (default: 2000).")
    ap.add_argument("--no-originals", action="store_true",
                    help="Do NOT copy the original pairs in (augmented-only).")
    ap.add_argument("--size-weight", action=argparse.BooleanOptionalAction, default=True,
                    help="Oversample frames containing a large organoid (enhanced only).")
    ap.add_argument("--big-frac", type=float, default=0.03,
                    help="A frame is 'big-organoid' if its largest object covers this "
                         "fraction of the frame (default: 0.03).")
    ap.add_argument("--max-extra", type=int, default=3,
                    help="Max extra sampling slots per big-organoid frame (default: 3).")
    ap.add_argument("--photometric", action=argparse.BooleanOptionalAction, default=True,
                    help="Apply the images-only photometric post-pass (enhanced only).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    inp, out = Path(args.input), Path(args.out)
    _guard_output(out)
    in_img, in_seg = inp / "images", inp / "segmentations"
    for d in (in_img, in_seg):
        if not d.is_dir():
            raise SystemExit(f"Missing required directory: {d}")

    n_orig = len(list(in_img.glob("*.png")))
    print(f"Recipe: {args.recipe} | source training pairs: {n_orig} ({inp})")

    _reset_dir(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "segmentations").mkdir(parents=True, exist_ok=True)

    if args.recipe == "organoid":
        from Core.DataPreparation import AugmentImages
        print(f"Generating {args.count} augmented samples (stock OrganoID recipe)...")
        AugmentImages(in_img, in_seg, out, args.count)
    else:
        print(f"Generating {args.count} augmented samples (enhanced recipe)...")
        pipe = _build_enhanced_pipeline(in_img, in_seg, out)
        if args.size_weight:
            _apply_size_weighting(pipe, args.big_frac, args.max_extra)
        pipe.sample(args.count)
        _rearrange_output(out)
        if args.photometric:
            _photometric_pass(out / "images", args.seed + 1)

    if not args.no_originals:
        print("Copying original pairs in (kept pristine -- no photometric)...")
        seg_names = {p.name for p in in_seg.glob("*.png")}
        for img in in_img.glob("*.png"):
            if img.name in seg_names:
                shutil.copy2(img, out / "images" / img.name)
                shutil.copy2(in_seg / img.name, out / "segmentations" / img.name)

    n_img = len(list((out / "images").glob("*.png")))
    n_seg = len(list((out / "segmentations").glob("*.png")))
    print(f"\nDone. Training set at {out}: {n_img} images / {n_seg} segmentations")
    if n_img != n_seg:
        print("WARNING: image/segmentation counts differ -- check pairing before training.")


if __name__ == "__main__":
    main()
