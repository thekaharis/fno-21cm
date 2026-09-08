"""Plot a checkpoint bank's bins, sampled candidates, and orthonormal modes.

python -m viz.learned_waveforms --checkpoint best_model_state_dict.pt \
    --bank fno.encoder0.spectral.bank --shape 16 16 --out waveforms.png

Omit --bank to list the bank names stored in a checkpoint. Supply the actual
operator grid (patch extent locally; downsampled whole-field extent globally).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from learned_waveform_operator import WaveformBank


def bank_names(state):
    return sorted({key.split(".tables.")[0] for key in state if ".bank.tables." in key})


def load_bank(state, name):
    weight_key = name.removesuffix(".bank") + ".weight"
    if name not in bank_names(state) or weight_key not in state:
        raise ValueError(f"unknown bank {name!r}; available: {bank_names(state)}")
    modes = tuple(state[weight_key].shape[2:])
    tables = {key.removeprefix(name + "."): value for key, value in state.items()
              if key.startswith(name + ".tables.")}
    bins = next(iter(tables.values())).numel()
    # Initialization is immediately replaced; avoid altering the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        bank = WaveformBank(len(modes), modes, bins).double()
    bank.load_state_dict(tables, strict=True)
    return bank


@torch.no_grad()
def render_bank(bank, shape, output, *, title="Learned waveform bank", max_modes=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shape = tuple(shape)
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
            row[1].plot(np.arange(size) / size, raw[:, :max_modes])
            arrays[f"axis{axis}_bins"] = values
            arrays[f"axis{axis}_candidates"] = raw
        else:
            row[0].text(.5, .5, "DC only — no learned table", ha="center")
        row[2].plot(np.arange(size) / size, basis[:, :max_modes].numpy())
        arrays[f"axis{axis}_orthonormal"] = basis.numpy()
        for panel in row:
            panel.set_xlabel("Fraction of axis / period")
            panel.grid(alpha=.2)
        row[0].set_title(f"Axis {axis}: raw bin amplitudes")
        row[1].set_title(f"Sampled candidates, N={size} (first {min(max_modes, modes - 1)})")
        row[2].set_title(f"Orthonormal modes incl. DC (first {min(max_modes, modes)})")
    fig.suptitle(title)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    np.savez(output.with_suffix(".npz"), **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", help="Exact checkpoint bank prefix; omit to list banks")
    parser.add_argument("--shape", type=int, nargs="+", help="Actual spatial shape of this bank")
    parser.add_argument("--out", type=Path, default=Path("waveforms.png"))
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.bank is None:
        print("\n".join(bank_names(state)) or "No learned waveform banks in checkpoint")
        return
    if args.shape is None:
        parser.error("--shape is required when plotting a bank")
    render_bank(load_bank(state, args.bank), args.shape, args.out, title=args.bank)
    print(f"Wrote {args.out} and {args.out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
