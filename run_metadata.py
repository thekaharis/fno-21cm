"""Persist the data/model contract needed to reproduce a trained run."""

from __future__ import annotations

import json
import os
from pathlib import Path


METADATA_FILENAME = "run_metadata.json"


def write_run_metadata(checkpoint_dir: str | Path, metadata: dict) -> Path:
    """Atomically write JSON metadata next to the run checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / METADATA_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def load_run_metadata(checkpoint_dir: str | Path) -> dict | None:
    path = Path(checkpoint_dir) / METADATA_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def resolve_checkpoint(checkpoint_dir: str | Path) -> Path:
    """Resolve explicit, best/final, and legacy checkpoint names."""
    explicit = os.environ.get("CHECKPOINT")
    if explicit:
        return Path(explicit)

    checkpoint_dir = Path(checkpoint_dir)
    kind = os.environ.get("CHECKPOINT_KIND", "best").strip().lower()
    if kind not in {"best", "final"}:
        raise ValueError("CHECKPOINT_KIND must be 'best' or 'final'")

    preferred = checkpoint_dir / f"{kind}_model_state_dict.pt"
    if preferred.exists():
        return preferred

    legacy = checkpoint_dir / "model_state_dict.pt"
    return legacy if legacy.exists() else preferred
