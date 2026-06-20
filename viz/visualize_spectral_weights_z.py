#!/usr/bin/env python3
"""Plot only LOS/Z Fourier-weight diagnostics throughout training."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from modeling import ModelConfig
from util.spectral_weights import HISTORY_FILENAME
from viz.visualize_spectral_weights import (
    load_history,
    plot_cutoff_ratios,
    plot_evolution,
    plot_profiles,
    write_csv,
)


Z_AXIS = ("z",)


def parse_args() -> argparse.Namespace:
    default_checkpoint_dir = str(ModelConfig.from_env().default_checkpoint_dir)
    parser = argparse.ArgumentParser(
        description="Render Z/LOS-only Fourier-weight diagnostics."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path(
            os.environ.get("CHECKPOINT_DIR", default_checkpoint_dir)
        )
        / HISTORY_FILENAME,
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_history(args.history)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (
        Path("figures") / f"spectral-weights-z_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_evolution(
        history,
        output_dir / "spectral_weight_z_evolution.png",
        Z_AXIS,
    )
    plot_profiles(
        history,
        output_dir / "spectral_weight_z_profiles.png",
        Z_AXIS,
    )
    plot_cutoff_ratios(
        history,
        output_dir / "spectral_weight_z_cutoff_ratio.png",
        Z_AXIS,
    )
    write_csv(
        history,
        output_dir / "spectral_weight_z_history.csv",
        Z_AXIS,
    )
    print(f"Wrote Z-only spectral-weight diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
