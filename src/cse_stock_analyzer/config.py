"""Project paths and environment-backed defaults."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("CSE_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
MODEL_PATH = Path(
    os.getenv("CSE_MODEL_PATH", PROJECT_ROOT / "models" / "cse_next_day_regressor.json")
)
METADATA_PATH = Path(
    os.getenv("CSE_METADATA_PATH", PROJECT_ROOT / "models" / "cse_next_day_regressor.metadata.json")
)
DEFAULT_PERIOD = os.getenv("CSE_DATA_PERIOD", "5y")

# Bars dated after this are excluded from training by default. Yahoo's CSE feed
# forward-fills ghost rows (Volume=0) past this date; the volume filter removes
# them, but the cutoff keeps any residual dead bars out of training. Clear with
# CSE_TRAIN_END= (empty) or --train-end once the feed is live again.
DEFAULT_TRAIN_END = os.getenv("CSE_TRAIN_END", "2026-01-31") or None
