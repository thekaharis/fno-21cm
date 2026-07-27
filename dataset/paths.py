"""Canonical dataset locations.

Datasets live outside the repository, under the work directory's ``data/``:

    <work>/data/data/        raw 21cmFAST lightcone HDF5 files
    <work>/data/compressed/  derived caches (trainset.h5, zre_*.h5, band sets)

Nothing large belongs in the project root.  Import from here rather than
hard-coding paths, so a relocation is a one-line change.

Override with environment variables when running against a different copy:

    FNO_DATA_ROOT    parent of data/ and compressed/   (default <work>/data)
    FNO_LIGHTCONES   raw lightcone directory
    FNO_COMPRESSED   derived-cache directory
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("FNO_DATA_ROOT", PROJECT_ROOT.parent / "data"))
LIGHTCONES = Path(os.environ.get("FNO_LIGHTCONES", DATA_ROOT / "data"))
COMPRESSED = Path(os.environ.get("FNO_COMPRESSED", DATA_ROOT / "compressed"))

# Named caches, so callers never spell the filenames themselves.
TRAINSET = COMPRESSED / "trainset.h5"
ZRE_TARGETS = COMPRESSED / "zre_targets.h5"
ZRE_INPUTS = COMPRESSED / "zre_inputs.h5"


def compressed(name: str) -> Path:
    """Path to a derived cache by filename."""
    return COMPRESSED / name


__all__ = [
    "PROJECT_ROOT", "DATA_ROOT", "LIGHTCONES", "COMPRESSED",
    "TRAINSET", "ZRE_TARGETS", "ZRE_INPUTS", "compressed",
]
