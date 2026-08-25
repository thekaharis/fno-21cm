"""Parameter count and inference speed across every localop operator pairing.

Covers the full local x global matrix over {fourier, wavelet, hadamard} plus
siren_fourier and cnn on the diagonal, and the two non-local baselines (plain
FNO, U-FNO), all at the 2-D x_HI configuration the sweeps used.

Timing is forward-only under ``inference_mode`` on a fixed input, after warmup,
with CUDA synchronised around each repeat -- otherwise the asynchronous launch
queue makes everything look equally fast. Reported as median over repeats,
since the mean is skewed by occasional scheduler hiccups.

Run: sbatch slurm/bench_operator_variants.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.neuralop_setup import prefer_local_neuralop

prefer_local_neuralop()

import torch
import dataclasses

# These were never imported here: the script predates the pipeline
# refactor and reached them through the now-deleted `fno_21cm` module.
from modeling import ModelConfig, build_model

IN_CHANNELS = 13          # density + 1/(1+z) + 11 cosmological parameters
RESOLUTION = 140
TAGS = {"fourier": "fno", "wavelet": "wno", "hadamard": "whno",
        "siren_fourier": "sfno", "siren_hadamard": "swhno", "cnn": "cnn"}

# (label, MODEL_KIND, LOCAL_OPERATOR, GLOBAL_OPERATOR)
VARIANTS = [("FNO (plain)", "fno", None, None),
            ("U-FNO", "ufno", None, None)]
# The matrix grew from 3x3 to 4x4: siren_hadamard (SIREN-generated Walsh
# weights) is now a first-class slot operator, and it is the global slot of the
# strongest new cells. cnn in the local slot IS the U-FNO's U-Net path, so the
# cnn/* row isolates the global basis at fixed local path.
REFERENCE_KINDS = ("fno", "ufno", "sirenfno")   # sized outside the sweep
REFERENCE_MODES = (32, 32)
REFERENCE_HIDDEN = 64

MATRIX = ("fourier", "wavelet", "hadamard", "siren_hadamard")
for loc in MATRIX:
    for glob in MATRIX:
        VARIANTS.append((f"local {TAGS[loc]} / global {TAGS[glob]}",
                         "localop", loc, glob))
VARIANTS += [("local sirenfno / global sirenfno", "localop",
              "siren_fourier", "siren_fourier"),
             ("local sfno / global swhno", "localop",
              "siren_fourier", "siren_hadamard"),
             ("local cnn / global fno", "localop", "cnn", "fourier"),
             ("local cnn / global whno", "localop", "cnn", "hadamard"),
             ("local cnn / global swhno", "localop", "cnn", "siren_hadamard")]

BASE_ENV = {
    "N_MODES_X": "32", "N_MODES_Y": "32", "HIDDEN_CHANNELS": "64",
    "N_LAYERS": "4", "UFNO_WIDTH": "32", "UFNO_NORM": "batchnorm",
    "LOCALFNO_BASE_WIDTH": "32", "LOCALFNO_WINDOW_X": "16",
    "LOCALFNO_WINDOW_Y": "16", "LOCALFNO_MODES_X": "6",
    "LOCALFNO_MODES_Y": "6", "LOCALFNO_GLOBAL_MODES_X": "16",
    "LOCALFNO_GLOBAL_MODES_Y": "16", "LOCALFNO_SPECTRAL_RANK": "16",
    "LOCALFNO_PATCH_CHUNK_SIZE": "32", "LOCALWNO_LEVELS": "2",
    "INPUT_FEATURES": "density_z_params",
}


def build(kind, local_op, global_op):
    """Build one variant from the environment.

    This used to `import fno_21cm; importlib.reload(...)` because that module
    read its config at import time. ModelConfig.from_env() does that now, and
    the 2-D module was removed in the pipeline refactor -- the stale import
    made every variant fail with ModuleNotFoundError while the run still
    reported success and wrote an empty JSON over the previous results.
    """
    os.environ.update(BASE_ENV)
    os.environ["MODEL_KIND"] = kind
    if local_op:
        os.environ["LOCAL_OPERATOR"] = local_op
        os.environ["GLOBAL_OPERATOR"] = global_op
    else:
        os.environ.pop("LOCAL_OPERATOR", None)
        os.environ.pop("GLOBAL_OPERATOR", None)
    cfg = ModelConfig.from_env(ndim=2)
    # The whole-volume reference architectures are sized independently of the
    # local/global sweep, and BASE_ENV can no longer express that:
    #   * modes -- ModelConfig.axes() takes the FIRST prefix that is set, and
    #     BASE_ENV sets LOCALFNO_GLOBAL_MODES_* (for the sweep), which silently
    #     shadows N_MODES_* and left FNO/U-FNO at 16 modes instead of 32.
    #   * hidden_channels -- never read from the environment at all; it is a
    #     dataclass default, so HIDDEN_CHANNELS=64 was a no-op.
    # Set them on the config directly so the reference points are what they
    # claim to be, and so a future env rename cannot quietly resize them again.
    if kind in REFERENCE_KINDS:
        cfg = dataclasses.replace(cfg, modes=REFERENCE_MODES,
                                  hidden_channels=REFERENCE_HIDDEN)
    return build_model(cfg, IN_CHANNELS), cfg, cfg.checkpoint_tag


def bench(model, batch, repeats, device):
    model = model.to(device).eval()
    x = torch.randn(batch, IN_CHANNELS, RESOLUTION, RESOLUTION, device=device)
    with torch.inference_mode():
        for _ in range(3):                       # warmup: cudnn autotune, alloc
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    return statistics.median(times), peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="figures/operator_variant_benchmark.json")
    args = ap.parse_args()

    if args.device == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"input: batch {args.batch} x {IN_CHANNELS} x {RESOLUTION}^2, "
          f"median of {args.repeats} forward passes\n")
    print(f"{'variant':<34} {'params':>10} {'real*':>10} {'ms/batch':>9} "
          f"{'slices/s':>9} {'peak MiB':>9} {'vs FNO':>7}")
    print("-" * 97)

    rows, ref = [], None
    for label, kind, loc, glob in VARIANTS:
        try:
            model, cfg, _desc = build(kind, loc, glob)
        except Exception as exc:                                # noqa: BLE001
            print(f"{label:<34} build failed: {type(exc).__name__}: "
                  f"{str(exc)[:34]}")
            continue
        # Two conventions, because they differ by a lot here and the training
        # logs use the second: torch's numel counts a complex weight once,
        # neuralop's count_model_params counts real and imaginary separately.
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_real = sum(p.numel() * (2 if p.is_complex() else 1)
                     for p in model.parameters() if p.requires_grad)
        try:
            dt, peak = bench(model, args.batch, args.repeats, args.device)
        except Exception as exc:                                # noqa: BLE001
            print(f"{label:<34} {n:10,} bench failed: {str(exc)[:30]}")
            continue
        ref = ref or dt
        print(f"{label:<34} {n:10,} {n_real:10,} {dt*1e3:9.2f} "
              f"{args.batch/dt:9.1f} {peak:9.1f} {dt/ref:6.2f}x")
        rows.append({"variant": label, "kind": kind, "local": loc,
                     "global": glob, "params": n, "params_real": n_real, "ms_per_batch": dt * 1e3,
                     "slices_per_s": args.batch / dt, "peak_mib": peak,
                     "relative_to_fno": dt / ref})
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Never overwrite good results with a failed sweep. A stale import once
    # made all 23 variants fail, and the run still wrote an empty JSON over the
    # only copy of the previous benchmark, which was untracked by git.
    if not rows:
        print(f"\nALL {len(VARIANTS)} VARIANTS FAILED -- refusing to write "
              f"{args.json}; the existing file is left untouched.")
        raise SystemExit(1)

    os.makedirs("figures", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"batch": args.batch, "repeats": args.repeats,
                   "resolution": RESOLUTION, "in_channels": IN_CHANNELS,
                   "device": (torch.cuda.get_device_name(0)
                              if args.device == "cuda" else "cpu"),
                   "rows": rows}, f, indent=2)
    print("\n* real: complex weights counted as two parameters, the convention "
          "neuralop's\n  count_model_params uses -- this is what the training "
          "logs report.")
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
