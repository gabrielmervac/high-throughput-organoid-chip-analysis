"""Stage the extra large-organoid image/mask pairs into the dataset.

Source (read-only): D:\\Yiyu\\Organoid ID Finetuned\\Large Organoid\\
    Large Organoid Phase Contrast\\*.png   (16-bit phase)
    Large Organoid Mask\\*.png             (binary mask, 0/255)

Writes 8-bit-normalised, name-matched pairs to dataset/large_organoid/{images,segmentations}
so they can be augmented (stock OrganoID recipe) and mixed into training. The phase is
converted 16-bit -> 8-bit by per-image min-max (what the model's loader does anyway),
matching the existing 8-bit training PNGs so Augmentor handles them identically.
"""
import _env  # noqa: F401
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# Large-organoid supplement (deposited separately). Override with LARGE_ORGANOID_SRC.
SRC = Path(os.environ.get("LARGE_ORGANOID_SRC",
                          _env.PROJECT_ROOT / "Large Organoid"))
PHASE = SRC / "Large Organoid Phase Contrast"
MASK = SRC / "Large Organoid Mask"
DST = _env.DATASET_DIR / "large_organoid"


def main():
    img_out, seg_out = DST / "images", DST / "segmentations"
    for d in (img_out, seg_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in PHASE.glob("*.png"))
    n = 0
    for name in names:
        mpath = MASK / name
        if not mpath.exists():
            print(f"[skip] no mask for {name}")
            continue
        # phase: 16-bit -> 8-bit min-max, mode L
        arr = np.asarray(Image.open(PHASE / name))
        lo, hi = float(arr.min()), float(arr.max())
        u8 = (((arr - lo) / (hi - lo) * 255).astype(np.uint8)
              if hi > lo else np.zeros(arr.shape, np.uint8))
        Image.fromarray(u8, "L").save(img_out / name)
        # mask: force binary L (0/255)
        m = np.asarray(Image.open(mpath).convert("L"))
        Image.fromarray(np.where(m > 127, 255, 0).astype(np.uint8), "L").save(seg_out / name)
        n += 1

    print(f"Staged {n} large-organoid pairs -> {DST}")


if __name__ == "__main__":
    main()
