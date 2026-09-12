"""Plot a checkpoint bank's bins, sampled candidates, and orthonormal modes.

python -m viz.learned_waveforms --checkpoint best_model_state_dict.pt \
    --bank fno.encoder0.spectral.bank --shape 16 16 --out waveforms.png

All branches/axes, without loading a dataset:
python -m viz.learned_waveforms --checkpoint-dir checkpoints/my_run

Older metadata needs --input-shape, e.g. 140 140 256 for a 3-D run ONLY if
that was its actual input shape. Explicit --checkpoint with no --bank/--all
lists bank names, preserving the original single-bank interface.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from learned_waveform_operator import WaveformBank

BRANCHES = ("encoder0", "encoder1", "bottleneck.0", "decoder1", "decoder0")
DOWNSAMPLE = dict(zip(BRANCHES, (1, 2, 4, 2, 1)))
WAVEFORM_NAMES = {"learned_waveform", "waveform", "orthogonal_waveform"}


def branch_name(bank_name):
    match = re.search(r"(?:^|\.)(encoder0|encoder1|bottleneck\.0|decoder1|decoder0)\.spectral\.bank$", bank_name)
    if match is None:
        raise ValueError(f"cannot identify U-Net branch for {bank_name!r}; use --bank/--shape")
    return match.group(1)


def bank_names(state, metadata=None):
    names = {key.split(".tables.")[0] for key in state if ".bank.tables." in key}
    # A fully DC-only bank has no tables. Metadata distinguishes its real
    # mixing weight from a Walsh operator with the same weight shape.
    config = (metadata or {}).get("model_config", {})
    for key, value in state.items():
        if not key.endswith(".spectral.weight") or value.ndim not in (4, 5):
            continue
        if any(n != 1 for n in value.shape[2:]):
            continue
        name = key.removesuffix(".weight") + ".bank"
        try:
            branch = branch_name(name)
        except ValueError:
            continue
        slot = "global_operator" if branch == "bottleneck.0" else "local_operator"
        if config.get(slot) in WAVEFORM_NAMES:
            names.add(name)
    return sorted(names)


def load_bank(state, name, *, condition_limit=1e4):
    weight_key = name.removesuffix(".bank") + ".weight"
    if weight_key not in state:
        raise ValueError(f"unknown bank {name!r}; available: {bank_names(state)}")
    modes = tuple(state[weight_key].shape[2:])
    tables = {key.removeprefix(name + "."): value for key, value in state.items()
              if key.startswith(name + ".tables.")}
    if not tables and any(m != 1 for m in modes):
        raise ValueError(f"missing waveform tables for {name!r}")
    bins = next(iter(tables.values())).numel() if tables else 3
    # Initialization is immediately replaced; avoid altering the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        bank = WaveformBank(len(modes), modes, bins, condition_limit=condition_limit).double()
    bank.load_state_dict(tables, strict=True)
    return bank


@torch.no_grad()
def render_bank(bank, shape, output, *, title="Learned waveform bank", max_modes=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shape = tuple(shape)
    if max_modes < 1:
        raise ValueError("max_modes must be positive")
    bases = bank.materialize_transform(shape, device="cpu", dtype=torch.float64)
    fig, axes = plt.subplots(bank.ndim, 3, figsize=(13, 3.2 * bank.ndim),
                             squeeze=False, layout="constrained")
    arrays = {}
    for axis, (size, modes, basis) in enumerate(zip(shape, bank.n_modes, bases)):
        row = axes[axis]
        if str(axis) in bank.tables:
            table = bank.tables[str(axis)].detach().cpu().double()
            values = table.numpy()
            row[0].stairs(values, np.linspace(0, 1, bank.bins + 1))
            raw = bank.sampled_candidates(table, size, modes).numpy()
            for j in range(min(max_modes, modes - 1)):
                row[1].plot(np.arange(size) / size, raw[:, j],
                            label=f"k={1 + j // 2}, phase={(j % 2) / 4:g}")
            row[1].legend(fontsize=7, ncols=2)
            arrays[f"axis{axis}_bins"] = values
            arrays[f"axis{axis}_candidates"] = raw
        else:
            row[0].text(.5, .5, "DC only — no learned table", ha="center")
        for j in range(min(max_modes, modes)):
            row[2].plot(np.arange(size) / size, basis[:, j].numpy(),
                        label="DC" if j == 0 else f"Mode {j}")
        row[2].legend(fontsize=7, ncols=2)
        arrays[f"axis{axis}_orthonormal"] = basis.numpy()
        for panel in row:
            panel.set_xlabel("Fraction of axis / period")
            panel.grid(alpha=.2)
        axis_label = ("X", "Y", "Z / LOS")[axis]
        row[0].set_title(f"{axis_label}: raw bin amplitudes")
        row[1].set_title(f"Sampled candidates, N={size} (first {min(max_modes, modes - 1)})")
        row[2].set_title(f"Orthonormal modes incl. DC (first {min(max_modes, modes)})")
    fig.suptitle(title)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    np.savez(output.with_suffix(".npz"), **arrays)


def bank_geometry(state, metadata, *, input_shape=None):
    """Resolve actual operator grids without guessing a simulation resolution."""
    config = metadata.get("model_config", {})
    if not config:
        raise ValueError("all-bank plotting needs run_metadata.json with model_config; use --metadata or --bank/--shape")
    names = bank_names(state, metadata)
    if not names:
        raise ValueError("no learned waveform banks in checkpoint")
    saved_shape = metadata.get("input_features", {}).get("spatial_shape")
    shape = input_shape if input_shape is not None else saved_shape
    ndim = len(state[names[0].removesuffix(".bank") + ".weight"].shape) - 2
    if shape is not None:
        shape = tuple(int(n) for n in shape)
        if len(shape) != ndim or any(n < 4 for n in shape):
            raise ValueError(f"input shape must contain {ndim} dimensions >=4 for this two-level U-Net")
    result = []
    for name in sorted(names, key=lambda n: BRANCHES.index(branch_name(n))):
        branch = branch_name(name)
        local = branch != "bottleneck.0"
        windowed = local and config.get("local_windowed") is not False
        if windowed:
            grid = config.get("localfno_window")
            if grid is None or len(grid) != ndim or any(int(n) < 1 for n in grid):
                raise ValueError("metadata must record a valid localfno_window")
            grid = tuple(int(n) for n in grid)
        else:
            if shape is None:
                raise ValueError("metadata has no input spatial_shape; supply --input-shape from the training run")
            grid = tuple(n // DOWNSAMPLE[branch] for n in shape)
        result.append({"bank": name, "branch": branch, "shape": grid,
                       "downsample": DOWNSAMPLE[branch], "windowed": windowed,
                       "shared_blocks": ["bottleneck.0", "bottleneck.1"] if not local else [branch]})
    return result


@torch.no_grad()
def render_all(state, metadata, output_dir, *, input_shape=None, max_modes=6, checkpoint=None):
    """Branch-by-axis overview, detailed banks, and portable numerical exports."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if max_modes < 1:
        raise ValueError("max_modes must be positive")
    geometry = bank_geometry(state, metadata, input_shape=input_shape)
    condition_limit = float(metadata["model_config"].get("waveform_condition_limit", 1e4))
    banks = [load_bank(state, entry["bank"], condition_limit=condition_limit) for entry in geometry]
    ndim = banks[0].ndim
    # Validate every requested grid before creating a partially rendered report.
    bases = [b.materialize_transform(g["shape"], device="cpu", dtype=torch.float64)
             for b, g in zip(banks, geometry)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task = metadata.get("task", "3d" if ndim == 3 else "2d")
    title = f"Learned waveforms — {task} — {ndim} spatial axes"
    figures = [plt.subplots(len(banks), ndim, figsize=(5 * ndim, 2.65 * len(banks)),
                            squeeze=False, layout="constrained") for _ in range(2)]
    for r, (bank, entry, us) in enumerate(zip(banks, geometry, bases)):
        label = "Bottleneck (blocks 0 + 1 share bank)" if entry["branch"] == "bottleneck.0" else entry["branch"]
        for a, (n, u) in enumerate(zip(entry["shape"], us)):
            raw_panel, mode_panel = [axes[r, a] for _, axes in figures]
            if str(a) in bank.tables:
                raw_panel.stairs(bank.tables[str(a)].numpy(), np.linspace(0, 1, bank.bins + 1))
            else:
                raw_panel.text(.5, .5, "DC only", ha="center", va="center")
            count = min(max_modes, u.shape[1])
            for j in range(count):
                curve = u[:, j].numpy()
                # Offsets and per-curve display scaling keep shapes legible;
                # NPZ exports retain the actual orthonormal amplitudes.
                mode_panel.plot(np.arange(n) / n, j + .38 * curve / np.abs(curve).max(),
                                color=f"C{j % 10}")
            mode_panel.set_yticks(range(count), ["DC"] + [str(j) for j in range(1, count)])
            mode_panel.set_ylim(-.6, count - .4)
            for panel in (raw_panel, mode_panel):
                panel.set_title(f"{label}\n{('X', 'Y', 'Z / LOS')[a]} · N={n} · level spacing ×{entry['downsample']}", fontsize=9)
                panel.set_xlabel("Fraction of one period / axis", fontsize=8)
                panel.grid(alpha=.15)
                panel.tick_params(labelsize=8)
        stem = entry["branch"].replace(".", "_")
        entry["figure"] = f"{stem}.png"
        entry["arrays"] = f"{stem}.npz"
        render_bank(bank, entry["shape"], output_dir / entry["figure"], title=label, max_modes=max_modes)
    for (fig, _), filename, subtitle in zip(figures, ("overview_bins.png", "overview_modes.png"),
                                          ("Raw learned bin amplitudes", "Orthonormal mode shapes · unit peak and vertical offsets for display")):
        fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
    manifest = {"checkpoint": str(checkpoint) if checkpoint else None, "task": task,
                "input_shape": input_shape if input_shape is not None else metadata.get("input_features", {}).get("spatial_shape"),
                "spatial_dimensions": ndim, "max_display_modes": max_modes,
                "overviews": ["overview_bins.png", "overview_modes.png"], "banks": geometry}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-dir", type=Path, help="Render all banks using this run's metadata")
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bank", help="Exact checkpoint bank prefix")
    mode.add_argument("--all", action="store_true", help="Render all branches and spatial axes")
    mode.add_argument("--list", action="store_true", help="List bank names without rendering")
    parser.add_argument("--metadata", type=Path, help="Defaults to run_metadata.json beside checkpoint")
    parser.add_argument("--input-shape", type=int, nargs="+", help="Original input spatial shape for old metadata or resolution inspection")
    parser.add_argument("--shape", type=int, nargs="+", help="Actual spatial shape of this bank")
    parser.add_argument("--out", type=Path, default=Path("waveforms.png"))
    parser.add_argument("--out-dir", type=Path, help="All-bank report directory")
    parser.add_argument("--max-modes", type=int, default=6)
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.checkpoint_dir / f"{args.checkpoint_kind}_model_state_dict.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata_path = args.metadata or checkpoint.parent / "run_metadata.json"
    if args.metadata and not metadata_path.is_file():
        parser.error(f"metadata file does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if args.list or (not args.bank and not args.all and args.checkpoint_dir is None):
        print("\n".join(bank_names(state, metadata)) or "No learned waveform banks in checkpoint")
        return
    try:
        if args.bank:
            if args.shape is None:
                parser.error("--shape is required when plotting a bank")
            bank = load_bank(state, args.bank, condition_limit=float(metadata.get("model_config", {}).get("waveform_condition_limit", 1e4)))
            render_bank(bank, args.shape, args.out, title=args.bank, max_modes=args.max_modes)
            print(f"Wrote {args.out} and {args.out.with_suffix('.npz')}")
        else:
            out = args.out_dir or Path("figures") / "shared" / "diagnostics" / "waveforms" / checkpoint.parent.name / checkpoint.stem
            manifest = render_all(state, metadata, out, input_shape=args.input_shape,
                                  max_modes=args.max_modes, checkpoint=checkpoint.resolve())
            print(f"Wrote {len(manifest['banks'])} bank plots, two overviews and numerical exports to {out}")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
