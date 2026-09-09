"""Shared environment setup — import this FIRST in every entry point.

Forces legacy Keras 2 (so OrganoID's tf.keras code runs unchanged) and puts the
cloned OrganoID repo on sys.path so `from Core.X import ...` works.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
from pathlib import Path

# Repo root = parent of this app/ package, so the tree is relocatable.
# Override any path with an environment variable if your layout differs.
import os as _os

PROJECT_ROOT = Path(_os.environ.get("ORGANOID_PROJECT_ROOT",
                                    Path(__file__).resolve().parent.parent))
ORGANOID_REPO = Path(_os.environ.get("ORGANOID_REPO", PROJECT_ROOT / "organoid_tf"))
MODELS_DIR = Path(_os.environ.get("ORGANOID_MODELS_DIR", PROJECT_ROOT / "models"))
OUTPUTS_DIR = Path(_os.environ.get("ORGANOID_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
DATASET_DIR = Path(_os.environ.get("ORGANOID_DATASET_DIR", PROJECT_ROOT / "dataset"))

if str(ORGANOID_REPO) not in sys.path:
    sys.path.insert(0, str(ORGANOID_REPO))
