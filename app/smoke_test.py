"""Phase 0 smoke test: confirm the pinned TF + legacy Keras loads OrganoID's
TrainableModel unchanged and can run inference on a staged ChipData image."""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent / "organoid_tf"  # repo_root/organoid_tf
sys.path.insert(0, str(REPO))

import tensorflow as tf
print("TF version:", tf.__version__)
try:
    import tf_keras
    print("tf_keras version:", tf_keras.__version__)
except Exception as e:
    print("tf_keras import:", e)
print("Keras from tf.keras:", tf.keras.__version__ if hasattr(tf.keras, "__version__") else "?")
gpus = tf.config.list_physical_devices("GPU")
print("GPUs visible to TF:", gpus)

from Core.Model import LoadFullModel, LoadLiteModel, InputSize, Detect

model = LoadFullModel(REPO / "TrainableModel")
try:
    in_shape = model.inputs[0].shape
except Exception:
    in_shape = "unknown"
print("TrainableModel input shape:", in_shape)
print("InputSize():", InputSize(model))

# Run inference on one staged ChipData image.
from PIL import Image
img_dir = Path(r"D:\Yiyu\Organoid ID Finetuned\dataset\training\images")
sample = sorted(img_dir.glob("*.png"))[0]
print("Sample:", sample.name)
from Core.Model import PrepareImagesForModel
prepared = PrepareImagesForModel([Image.open(sample)], model, verbose=False)
print("Prepared shape:", prepared.shape, prepared.dtype)
belief = Detect(model, prepared)
print("Belief map:", belief.shape, "min", round(float(belief.min()), 4),
      "max", round(float(belief.max()), 4), "mean", round(float(belief.mean()), 4))
print("SMOKE TEST OK")
