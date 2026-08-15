#!/usr/bin/env python3
"""Sequency-weight profiles for Walsh-Hadamard operator slots.

`util/spectral_weights.py` only recognises Fourier-family layers: its
`_channel_power` requires a *complex* weight, so a WHNO run logs
"[spectral-weights] disabled" and produces no history. The quantity it reports
is still well defined on the real Walsh weight, and this computes it from a
saved checkpoint.

The Fourier diagnostic asks how learned weight is distributed over |k|, and its
headline number is the **edge/peak ratio** -- weight at the retained boundary
divided by weight at the peak. Low means the operator has collapsed onto low
frequencies and the mode budget is not binding; high means it is using the
whole band and more modes might help.

The Walsh analogue is the same statistic over *sequency*, which counts sign
changes and so plays the role of frequency in this basis. Two differences:

* the weight is real, so power is `w**2` rather than `|w|**2`;
* sequency is non-negative by construction, so a coordinate runs 0 -> m-1 with
  0 the constant (DC) Walsh function -- there is no centring or folding, unlike
  the signed Fourier axes.

Read the axis profiles, not just the shell: the 3-D campaign's finding is that
the LOS axis behaves differently from the transverse ones, and a shell profile
averages exactly that away.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def channel_power(weight: torch.Tensor) -> np.ndarray:
    """Mean squared weight over the two channel axes -> (*modes) grid."""
    if weight.ndim != 5 or torch.is_complex(weight):
        raise ValueError(
            f"expected a real (C_in, C_out, mx, my, mz) weight, got "
            f"shape {tuple(weight.shape)} complex={torch.is_complex(weight)}"
        )
    return weight.detach().float().square().mean(dim=(0, 1)).cpu().numpy()


def profiles(power: np.ndarray) -> dict:
    """Marginal sequency profiles per axis, plus an isotropic shell profile."""
    out = {}
    for axis, name in enumerate(("x", "y", "z")):
        others = tuple(a for a in range(3) if a != axis)
        out[name] = power.mean(axis=others)
    sx, sy, sz = (np.arange(n) for n in power.shape)
    gx, gy, gz = np.meshgrid(sx, sy, sz, indexing="ij")
    shell_index = np.floor(np.sqrt(gx**2 + gy**2 + gz**2)).astype(int)
    n_shells = shell_index.max() + 1
    shell = np.array([
        power[shell_index == s].mean() if (shell_index == s).any() else np.nan
        for s in range(n_shells)
    ])
    out["shell"] = shell
    return out


def edge_peak(profile: np.ndarray) -> float:
    """Weight at the retained edge / weight at the peak."""
    finite = profile[np.isfinite(profile)]
    if finite.size == 0 or finite.max() <= 0:
        return float("nan")
    return float(finite[-1] / finite.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.name or args.checkpoint.parent.name

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    layers = {
        k[: -len(".weight")]: v
        for k, v in sd.items()
        if k.endswith(".weight") and v.ndim == 5 and not torch.is_complex(v)
        and min(v.shape[2:]) > 1          # 1x1x1 convs are not spectral slots
    }
    if not layers:
        raise SystemExit(f"no Walsh-style spectral weights found in {args.checkpoint}")

    rows, curves = [], {}
    for name, w in layers.items():
        p = profiles(channel_power(w))
        curves[name] = {k: v.tolist() for k, v in p.items()}
        rows.append({
            "layer": name,
            "modes": list(w.shape[2:]),
            "edge_peak_x": edge_peak(p["x"]),
            "edge_peak_y": edge_peak(p["y"]),
            "edge_peak_z": edge_peak(p["z"]),
            "edge_peak_shell": edge_peak(p["shell"]),
            "total_power": float(channel_power(w).sum()),
        })

    csv = args.out / f"walsh_weight_profile_{tag}.csv"
    cols = list(rows[0].keys())
    with open(csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    with open(args.out / f"walsh_weight_curves_{tag}.json", "w") as fh:
        json.dump(curves, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, key in zip(axes, ("x", "y", "z", "shell")):
        for name in layers:
            prof = np.asarray(curves[name][key], dtype=float)
            norm = prof / prof.max() if prof.max() > 0 else prof
            ax.plot(np.arange(len(norm)), norm, lw=1.4, label=name.split(".")[-2:][0])
        ax.set_title(f"sequency profile: {key}")
        ax.set_xlabel("sequency" if key != "shell" else "shell radius")
        ax.set_yscale("log")
    axes[0].set_ylabel("normalised mean weight power")
    axes[-1].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Walsh-Hadamard sequency weight profiles -- {tag}")
    fig.tight_layout()
    fig.savefig(args.out / f"walsh_weight_profile_{tag}.png", dpi=140)
    print(f"wrote {csv}")
    for r in rows:
        print("  %-34s modes=%-14s edge/peak x=%.3f y=%.3f z=%.3f shell=%.3f"
              % (r["layer"], r["modes"], r["edge_peak_x"], r["edge_peak_y"],
                 r["edge_peak_z"], r["edge_peak_shell"]))


if __name__ == "__main__":
    main()
