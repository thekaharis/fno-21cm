"""Resolve the local ``neuralop`` checkout before importing the package."""

from __future__ import annotations

import sys
from pathlib import Path


def prefer_local_neuralop(project_dir: Path | None = None) -> Path | None:
    """Put the first complete local ``neuralop`` checkout on ``sys.path``.

    Returns the selected checkout root, or ``None`` when the installed package
    should be used.
    """
    root = project_dir or Path(__file__).resolve().parent
    candidates = (root / "neuraloperator", root.parent / "neuraloperator", root)
    for candidate in candidates:
        if (candidate / "neuralop" / "__init__.py").is_file():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    return None
