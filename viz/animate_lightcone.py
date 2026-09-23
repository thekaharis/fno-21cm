"""Animate a lightcone prediction: transverse slices stepping through redshift.

Each frame is one transverse slice, truth beside prediction, for every target
field in the prediction file. Walking the frames is walking down the line of
sight, so the whole reionization history plays out in order.

Colour limits are fixed over the whole cone rather than per frame, otherwise
the scale would breathe from frame to frame and quiet slices would look as
structured as active ones. Bounded fields use their physical range; others use
a percentile over the cone.

  python -m viz.animate_lightcone --prediction pred/cone_541.h5 --out-dir figures/.../animations
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

BOX_MPC = 200.0
CMAPS = {"neutral_fraction": "magma", "brightness_temp": "inferno"}
BOUNDS = {"neutral_fraction": (0.0, 1.0)}


def limits(field, truth):
    if field in BOUNDS:
        return BOUNDS[field]
    lo, hi = np.percentile(truth[::4, ::4, ::4], [0.5, 99.5])
    return (float(lo), float(hi)) if hi > lo else (float(truth.min()), float(truth.max()) + 1e-6)


def animate(path, out_dir, max_frames, fps, dpi):
    with h5py.File(path, "r") as f:
        cone = int(f.attrs["cone_id"])
        z = f["target_z"][:]
        fields = list(f["target"])
        truth = {k: f[f"target/{k}"][:].astype(np.float32) for k in fields}
        pred = {k: f[f"prediction/{k}"][:].astype(np.float32) for k in fields}
        units = {k: f[f"target/{k}"].attrs.get("units", "") for k in fields}

    n = len(z)
    step = max(1, int(np.ceil(n / max_frames)))
    frames = range(0, n, step)
    rows = len(fields)
    fig, axes = plt.subplots(rows, 2, figsize=(7.4, 3.75 * rows), squeeze=False)
    images = {}
    for r, field in enumerate(fields):
        v0, v1 = limits(field, truth[field])
        cmap = CMAPS.get(field, "viridis")
        for c, (name, data) in enumerate((("truth", truth), ("prediction", pred))):
            ax = axes[r][c]
            images[(field, c)] = ax.imshow(data[field][:, :, 0].T, origin="lower", cmap=cmap,
                                           vmin=v0, vmax=v1, extent=[0, BOX_MPC, 0, BOX_MPC],
                                           interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=12)
            if c == 0:
                ax.set_ylabel(f"{field}\n200 x 200 Mpc", fontsize=10)
        fig.colorbar(images[(field, 1)], ax=axes[r].tolist(), fraction=0.046,
                     pad=0.02, label=units[field])
    title = fig.suptitle("", fontsize=13)

    def update(j):
        for field in fields:
            images[(field, 0)].set_data(truth[field][:, :, j].T)
            images[(field, 1)].set_data(pred[field][:, :, j].T)
        title.set_text(f"cone {cone}    z = {z[j]:.2f}    slice {j + 1}/{n}")
        return list(images.values()) + [title]

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cone{cone}_lightcone.gif"
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    size = out.stat().st_size / 2 ** 20
    print(f"wrote {out}  ({len(list(frames))} frames of {n} slices, step {step}, {size:.1f} MiB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=320,
                    help="native cones have ~2400 slices; frames are strided down to this")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=72)
    args = ap.parse_args()
    for p in args.prediction:
        animate(p, args.out_dir, args.max_frames, args.fps, args.dpi)


if __name__ == "__main__":
    main()
