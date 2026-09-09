"""Phase 1 -- fine-tune the OrganoID model on the staged ChipData set.

This reuses OrganoID's own model code UNCHANGED (Core.Model.TrainModel, the same
routine its `train` CLI calls). The only things this launcher adds are:
  * robust, sorted image<->mask pairing (avoids iterdir ordering surprises),
  * a fine-tuning learning rate, and
  * a held-out evaluation on the `testing/` split (Dice + IoU) written to outputs/.

The model architecture is never modified -- we load TrainableModel and continue
training its weights.
"""
import _env  # noqa: F401  (sets TF_USE_LEGACY_KERAS + sys.path)

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from Core.Model import (LoadFullModel, TrainModel, GroundTruth, Detect,
                        PrepareImagesForModel, PrepareSegmentationsForModel)

DATASET = _env.DATASET_DIR
MODELS = _env.MODELS_DIR
OUTPUTS = _env.OUTPUTS_DIR
BASE_MODEL = _env.ORGANOID_REPO / "TrainableModel"


def paired_ground_truth_dir(img_dir: Path, seg_dir: Path):
    imgs = sorted(img_dir.glob("*.png"), key=lambda p: p.name)
    segs = {p.name: p for p in seg_dir.glob("*.png")}
    pairs = []
    for img in imgs:
        seg = segs.get(img.name)
        if seg is not None:
            pairs.append(GroundTruth(img, seg))
    return pairs


def paired_ground_truth(split: str):
    return paired_ground_truth_dir(DATASET / split / "images", DATASET / split / "segmentations")


# Size buckets by fraction of the frame area occupied by a GROUND-TRUTH organoid.
# "large" uses the same 3% threshold as augment_data.py's size-weighting, so the
# metric reports exactly the population that augmentation is meant to help.
SIZE_BUCKETS = [("small", 0.0, 0.005), ("medium", 0.005, 0.03), ("large", 0.03, 1.01)]


def _per_object_records(pred_bool, gt_bool):
    """For each GROUND-TRUTH organoid: (area_fraction, instance_IoU, coverage).

    instance_IoU  -- IoU of that GT object vs the best-overlapping predicted blob
                     (0 if the model predicted nothing there = a full miss).
    coverage      -- fraction of the GT object's pixels the model called foreground;
                     this is the number that drops when the belief map holes/truncates
                     a big organoid.
    """
    from scipy import ndimage
    gt_lbl, ngt = ndimage.label(gt_bool)
    pred_lbl, _ = ndimage.label(pred_bool)
    total = gt_bool.size
    out = []
    for gid in range(1, ngt + 1):
        gt_obj = gt_lbl == gid
        area = int(gt_obj.sum())
        if area == 0:
            continue
        overlap = pred_lbl[gt_obj]
        overlap = overlap[overlap > 0]
        if overlap.size == 0:
            iou, cov = 0.0, 0.0
        else:
            best = int(np.bincount(overlap).argmax())
            pred_obj = pred_lbl == best
            inter = int(np.count_nonzero(gt_obj & pred_obj))
            union = int(np.count_nonzero(gt_obj | pred_obj))
            iou = inter / union if union else 0.0
            cov = inter / area
        out.append((area / total, iou, cov))
    return out


def _bucket_per_object(records):
    def agg(sel):
        if not sel:
            return {"n": 0, "mean_iou": None, "mean_coverage": None}
        return {"n": len(sel),
                "mean_iou": float(np.mean([r[1] for r in sel])),
                "mean_coverage": float(np.mean([r[2] for r in sel]))}
    out = {name: agg([r for r in records if lo <= r[0] < hi])
           for name, lo, hi in SIZE_BUCKETS}
    out["all"] = agg(records)
    return out


def evaluate(model, split: str, threshold: float = 0.5):
    """Per-image semantic Dice + IoU, AND size-stratified per-object IoU/coverage."""
    img_dir = DATASET / split / "images"
    seg_dir = DATASET / split / "segmentations"
    names = sorted(p.name for p in img_dir.glob("*.png"))
    dices, ious = [], []
    obj_records = []
    for name in names:
        pi = Image.open(img_dir / name)
        si = Image.open(seg_dir / name)
        pred = Detect(model, PrepareImagesForModel([pi], model, verbose=False)) >= threshold
        true = PrepareSegmentationsForModel([si], model).astype(bool)
        p = pred[0].astype(bool)
        t = true[0].astype(bool)
        inter = np.count_nonzero(p & t)
        union = np.count_nonzero(p | t)
        psum = np.count_nonzero(p) + np.count_nonzero(t)
        ious.append(inter / union if union else 1.0)
        dices.append(2 * inter / psum if psum else 1.0)
        obj_records.extend(_per_object_records(p, t))
    return {
        "split": split, "n": len(names),
        "mean_dice": float(np.mean(dices)), "mean_iou": float(np.mean(ious)),
        "median_dice": float(np.median(dices)), "median_iou": float(np.median(ious)),
        "per_object": _bucket_per_object(obj_records),
    }


def main():
    ap = argparse.ArgumentParser(description="Fine-tune OrganoID on ChipData")
    ap.add_argument("--name", default="organoid_finetuned")
    ap.add_argument("--train-dir", default=None,
                    help="Training split dir with images/ and segmentations/ "
                         "(default: dataset/training). Point this at dataset/training_augmented "
                         "to train on augmented data. Validation/testing always stay raw.")
    ap.add_argument("--init-model", default=None,
                    help="Model dir to START fine-tuning FROM (default: the OrganoID "
                         "TrainableModel base). Set to a fine-tuned model to CONTINUE training it.")
    ap.add_argument("-E", "--epochs", type=int, default=100)
    ap.add_argument("-B", "--batch", type=int, default=8)
    ap.add_argument("-LR", "--lr", type=float, default=1e-4,
                    help="Fine-tuning learning rate (lower than from-scratch).")
    ap.add_argument("-P", "--patience", type=int, default=10)
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training; just evaluate an existing model.")
    ap.add_argument("--model", default=None,
                    help="Model dir to evaluate (default: the fine-tuned _BEST).")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    best_path = MODELS / (args.name + "_BEST")

    if not args.eval_only:
        if args.train_dir:
            train_dir = Path(args.train_dir)
            training = paired_ground_truth_dir(train_dir / "images", train_dir / "segmentations")
            print(f"Training from: {train_dir}")
        else:
            training = paired_ground_truth("training")
        validation = paired_ground_truth("validation")
        print(f"Training pairs: {len(training)} | Validation pairs: {len(validation)}")

        init_path = Path(args.init_model) if args.init_model else BASE_MODEL
        print(f"Loading init model: {init_path}")
        model = LoadFullModel(init_path)

        # TrainModel saves final (best-restored) weights to saveDirectory/(name+"_BEST").
        TrainModel(model, args.lr, args.patience, args.epochs, args.batch,
                   training, validation, MODELS, args.name,
                   saveLite=False, saveAll=False)
        print(f"Saved fine-tuned model to: {best_path}")

    # Evaluate.
    eval_model_path = Path(args.model) if args.model else best_path
    print(f"Evaluating: {eval_model_path}")
    model = LoadFullModel(eval_model_path)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": str(eval_model_path),
        "results": [evaluate(model, s) for s in ("validation", "testing")],
    }
    report_path = OUTPUTS / f"{args.name}_eval.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["results"], indent=2))

    # Highlight the size-stratified per-object numbers -- this is the big-organoid signal.
    print("\nPer-object IoU / coverage by organoid size (the big-organoid check):")
    for res in report["results"]:
        po = res["per_object"]
        print(f"  [{res['split']}]")
        for b in ("small", "medium", "large", "all"):
            s = po[b]
            if s["n"]:
                print(f"    {b:<7} n={s['n']:<5} IoU={s['mean_iou']:.3f}  "
                      f"coverage={s['mean_coverage']:.3f}")
            else:
                print(f"    {b:<7} n=0")
    print(f"\nWrote evaluation report: {report_path}")


if __name__ == "__main__":
    main()
