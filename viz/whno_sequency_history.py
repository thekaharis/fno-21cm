#!/usr/bin/env python3
"""Build a spectral-weight history for Walsh-Hadamard runs, from a checkpoint.

`SpectralWeightHistory` records nothing for a WHNO run: its extractor only
recognises Fourier-style layers, so a hadamard/hadamard model logs
"[spectral-weights] disabled" and writes no .npz.  The information is not
missing though -- `WalshHadamardOperator` stores exactly the same kind of
object the FNO does:

    self.weight = nn.Parameter(randn(channels, channels, *n_modes))

a per-mode channel-mixing tensor.  The only difference is the meaning of the
mode index: Walsh coefficients are ordered by *sequency* (number of sign
changes), which is already monotone from smoothest to roughest, so unlike
Fourier modes they need no centering about zero.

This writes an npz in the same format `viz/visualize_spectral_weights.py`
consumes, so the WHNO profile can be plotted with the same code and read on the
same axes as the U-FNO figures.

Limitation worth stating plainly: a checkpoint holds one epoch, so the
"evolution" figure will show a single trace.  Only a training run can produce
the trajectory -- this recovers the endpoint, not the history.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

AXES = ("x", "y", "z", "shell")
HISTORY_FORMAT_VERSION = 2


def sequency_profiles(weight: torch.Tensor) -> dict[str, np.ndarray]:
    """RMS weight per sequency index, per axis, plus a radial shell profile.

    ``weight`` is (C_in, C_out, m_x, m_y, m_z).  Reducing over the channel pair
    and the two other mode axes gives the marginal a profile plot expects.
    """
    w = weight.detach().float()
    out: dict[str, np.ndarray] = {}
    for axis_index, name in zip((2, 3, 4), ("x", "y", "z")):
        others = tuple(i for i in range(w.ndim) if i != axis_index)
        out[name] = torch.sqrt((w**2).mean(dim=others)).numpy()
    # Shell: RMS over cells whose sequency radius falls in each integer bin.
    mx, my, mz = w.shape[2:]
    gx, gy, gz = np.meshgrid(np.arange(mx), np.arange(my), np.arange(mz),
                             indexing="ij")
    radius = np.sqrt(gx**2 + gy**2 + gz**2)
    energy = (w**2).mean(dim=(0, 1)).numpy()
    n_shell = int(np.ceil(radius.max())) + 1
    shell = np.zeros(n_shell)
    for r in range(n_shell):
        m = (radius >= r) & (radius < r + 1)
        shell[r] = np.sqrt(energy[m].mean()) if m.any() else np.nan
    out["shell"] = shell
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--epoch", type=int, default=-1,
                    help="epoch label to record (cosmetic; a checkpoint has one)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    layers = {
        k[: -len(".weight")]: v
        for k, v in sd.items()
        if k.endswith("spectral.weight") and v.ndim == 5 and min(v.shape[2:]) > 1
    }
    if not layers:
        raise SystemExit(
            f"No Walsh-Hadamard spectral weights in {args.checkpoint}. "
            "Expected keys ending in 'spectral.weight' with 5 dimensions."
        )

    names = list(layers)
    profiles = {n: sequency_profiles(layers[n]) for n in names}
    # Layers can retain different mode counts (local 6x6x12 vs global 16^3),
    # so pad to the longest per axis; NaN is skipped by matplotlib.
    data = {}
    for axis in AXES:
        width = max(profiles[n][axis].size for n in names)
        block = np.full((1, len(names), width), np.nan, dtype=np.float64)
        for i, n in enumerate(names):
            p = profiles[n][axis]
            block[0, i, : p.size] = p
        data[axis] = block

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        epochs=np.array([args.epoch], dtype=np.int64),
        layers=np.array(names),
        format_version=np.array(HISTORY_FORMAT_VERSION),
        **data,
    )
    print(f"wrote {args.out}")
    for n in names:
        widths = {a: int(np.isfinite(data[a][0, names.index(n)]).sum())
                  for a in ("x", "y", "z")}
        print(f"  {n:<34} modes {tuple(layers[n].shape[2:])}  bins {widths}")


if __name__ == "__main__":
    main()
