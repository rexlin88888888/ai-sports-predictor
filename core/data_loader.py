from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("sports_predictor")


def read_csv_checked(path: Path, required_columns: set[str]) -> pd.DataFrame | None:
    if not path.exists():
        LOGGER.warning("WARNING: missing data file %s", path)
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("WARNING: could not read %s: %s", path, exc)
        return None
    missing = required_columns - set(frame.columns)
    if missing:
        LOGGER.warning("WARNING: %s missing required columns: %s", path, ", ".join(sorted(missing)))
        return None
    return frame

