"""Export sampled residual frequency couplings from a checkpoint, without data.

python -m viz.frequency_mixing --checkpoint-dir RUN --out-dir REPORT
NPZ matrices have output coefficients on rows and input coefficients on columns.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modeling import ModelConfig, build_model
from spectral_mixing_operator import FrequencyMixingOperator


@torch.no_grad()
def export_mixer(op, output, *, max_modes=64):
    if max_modes < 1:
        raise ValueError("max_modes must be positive")
    mixer = op.mixer
    indices = torch.linspace(0, mixer.count - 1, min(max_modes, mixer.count),
                             device=mixer.coordinates.device).round().long().unique()
    if mixer.backend == "factorized":
        coords = mixer.coordinates[indices]
        a = mixer._factor(mixer.synthesis_net, coords).flatten(0, 1)
        p = mixer._factor(mixer.analysis_net, coords).flatten(0, 1)
        matrix = a @ p.T
    else:
        flat = (torch.arange(mixer.channels, device=indices.device)[:, None] * mixer.count
                + indices[None, :]).flatten()
        matrix = mixer.dense_matrix()[flat][:, flat]
    summary = {key: value.item() if isinstance(value, torch.Tensor) else value
               for key, value in op.diagnostics().items()}
    summary.update(backend=mixer.backend, fft_modes=list(op.n_modes),
                   real_mode_counts=list(op.basis.counts), sampled_modes=len(indices),
                   interpretation="Residual only; sampled singular values are not the full operator spectrum.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"),
                        residual_matrix=matrix.cpu().numpy(),
                        mode_indices=indices.cpu().numpy(),
                        coordinates=mixer.coordinates[indices].cpu().numpy(),
                        sampled_singular_values=torch.linalg.svdvals(matrix).cpu().numpy())
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-kind", choices=("best", "final", "initial"), default="best")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-modes", type=int, default=64)
    args = parser.parse_args()
    metadata = json.loads((args.checkpoint_dir / "run_metadata.json").read_text())
    raw = torch.load(args.checkpoint_dir / f"{args.checkpoint_kind}_model_state_dict.pt",
                     map_location="cpu", weights_only=True)
    state = {k.removeprefix("module.").removeprefix("fno."): v for k, v in raw.items()}
    if "base.lifting.weight" in state:
        # ContrastComposed's output map does not enter spectral diagnostics.
        state = {k.removeprefix("base."): v for k, v in state.items() if k.startswith("base.")}
    config = ModelConfig.from_dict(metadata["model_config"])
    model = build_model(config, in_channels=state["lifting.weight"].shape[1])
    model.load_state_dict(state, strict=True)
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, FrequencyMixingOperator):
            export_mixer(module, args.out_dir / name.replace(".", "_"), max_modes=args.max_modes)
            count += 1
    if not count:
        parser.error("checkpoint contains no frequency_mixing operators")
    print(f"Exported {count} frequency mixers to {args.out_dir}")


if __name__ == "__main__":
    main()
