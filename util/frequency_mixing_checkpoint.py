"""Explicit weight-only migration of a local/global Fourier checkpoint.

python -m util.frequency_mixing_checkpoint --checkpoint-dir SOURCE --out-dir NEW
The new directory contains weights and updated metadata, never optimizer state.
"""
import argparse
import json
from pathlib import Path

import torch

from modeling import ModelConfig
from spectral_mixing_operator import FrequencyMixingOperator


def convert_state(state, config, *, slots="global"):
    """Return a new state/config without modifying the caller's objects."""
    if slots not in {"global", "both"}:
        raise ValueError("slots must be global or both")
    if not config.is_local_global or config.global_operator != "fourier":
        raise ValueError("migration requires a local/global model with a Fourier global slot")
    if slots == "both" and config.local_operator != "fourier":
        raise ValueError("both-slot migration requires a Fourier local slot")
    cfg = config.to_dict()
    cfg.update(kind="localop", global_operator="frequency_mixing")
    if slots == "both":
        cfg["local_operator"] = "frequency_mixing"
    target = ModelConfig.from_dict(cfg)
    output = dict(state)
    count = 0
    for key in state:
        if not key.endswith(".spectral.weights1"):
            continue
        prefix = key.removesuffix("weights1")
        is_global = ".bottleneck." in "." + prefix
        if slots == "global" and not is_global:
            continue
        weight = state[key]
        modes = tuple(weight.shape[2:])
        expected_modes = target.modes if is_global else target.localfno_modes
        if modes != expected_modes or len(modes) != target.ndim or weight.shape[0] != weight.shape[1]:
            raise ValueError(f"checkpoint/metadata mismatch at {prefix}")
        options = target._slot_kwargs("frequency_mixing", local=not is_global)
        op = FrequencyMixingOperator(weight.shape[0], target.ndim, modes, **options)
        op.to(device=weight.device, dtype=weight.real.dtype)
        old = {name: state[prefix + name] for name in
               (f"weights{i + 1}" for i in range(2 ** (target.ndim - 1)))}
        op.load_fourier_weights(old)
        for name in old:
            del output[prefix + name]
        output.update({prefix + name: value for name, value in op.state_dict().items()})
        count += 1
    expected_count = 2 if slots == "global" else 6
    if count != expected_count:
        raise ValueError(f"expected {expected_count} Fourier blocks, found {count}")
    return output, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--slots", choices=("global", "both"), default="global")
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()
    metadata = json.loads((args.checkpoint_dir / "run_metadata.json").read_text())
    config = dict(metadata["model_config"], frequency_mixing_rank=args.rank)
    source = args.checkpoint_dir / f"{args.checkpoint_kind}_model_state_dict.pt"
    state, config = convert_state(torch.load(source, map_location="cpu", weights_only=True),
                                  ModelConfig.from_dict(config), slots=args.slots)
    # Refuse to overwrite any prior run, including source==destination.
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.save(state, args.out_dir / "initial_model_state_dict.pt")
    metadata = {"model_config": config.to_dict(), "migration": {
        "source": str(source.resolve()), "slots": args.slots, "weight_only": True,
        "instruction": "Use INIT_CHECKPOINT with a fresh optimizer; not RESUME_DIR."}}
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.out_dir / "initial_model_state_dict.pt")


if __name__ == "__main__":
    main()
