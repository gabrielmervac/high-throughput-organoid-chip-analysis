"""Stage a clean local copy of the ChipData training set.

Copies only paired .png files (skips Thumbs.db and any unpaired file) from the
network share into a local folder, so that:
  * OrganoID's GroundTruth stem-matching never trips over junk files, and
  * training reads from fast local disk instead of the Z: network share.

Run once before fine-tuning. Safe to re-run (it clears the destination first).
"""
import os
from pathlib import Path
import shutil

# The annotated ChipData set is deposited separately (too large for git). Point
# CHIPDATA_SRC at your local copy; DST defaults to this repo's dataset/ folder.
SRC = Path(os.environ.get("CHIPDATA_SRC", r"Z:\Yiyu_Zhang\ChipData"))
DST = Path(os.environ.get("ORGANOID_DATASET_DIR",
                          Path(__file__).resolve().parent.parent / "dataset"))
SPLITS = ["training", "validation", "testing"]


def stage():
    for split in SPLITS:
        img_src = SRC / split / "images"
        seg_src = SRC / split / "segmentations"
        if not img_src.is_dir() or not seg_src.is_dir():
            print(f"[skip] {split}: missing images/ or segmentations/")
            continue

        img_names = {p.name for p in img_src.glob("*.png")}
        seg_names = {p.name for p in seg_src.glob("*.png")}
        paired = sorted(img_names & seg_names)
        img_only = img_names - seg_names
        seg_only = seg_names - img_names

        img_dst = DST / split / "images"
        seg_dst = DST / split / "segmentations"
        for d in (img_dst, seg_dst):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        for name in paired:
            shutil.copy2(img_src / name, img_dst / name)
            shutil.copy2(seg_src / name, seg_dst / name)

        print(f"[{split}] copied {len(paired)} pairs "
              f"(image-only skipped: {len(img_only)}, seg-only skipped: {len(seg_only)})")

    print(f"\nStaged dataset at: {DST}")


if __name__ == "__main__":
    stage()
