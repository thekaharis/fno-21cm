# Native LOS window training

`fno_multifield.py` supports `--sampling contiguous` and
`--sampling coarse_context`, for density → neutral fraction as well as other
disjoint field mappings. The default `full` mode and existing resampled caches
remain supported. The legacy `fno_21cm_3d.py` entry point retains its old recipe;
use the multi-field entry point for these new sampling modes.

## Prepare native sources once

```bash
python fno_multifield.py prepare --data "$RAW_LIGHTCONES" --native-los \
  --fields density,neutral_fraction,brightness_temp,los_velocity \
  --conditioning z_params --split-seed 42 \
  --out experiments/los_windows/preparation.json
```

This reads the original `/lightcone/<field>` arrays, with
`lightcone_redshifts` and `lightcone_distances` in comoving Mpc. Native LOS
lengths may differ between cones. All selected fields must share each cone's
grid, the transverse dimensions must match, and the native distance spacing
must be uniform within each cone. Reversed axes are consistently reordered to
increasing redshift. No LOS interpolation, slice skipping or resampling occurs.

Native preparation uses the **full coverage of each source**. Do not combine it
with `--n-z`, `--target-z`, `--z-min`, `--z-max`, or `--cache`; these combinations
are rejected. A 256-slice resampled cache cannot be used to recover native cells.
Only the requested fields need exist, so prepare just `density,neutral_fraction`
for a two-field study. Raw reads use HDF5 hyperslabs; performance still depends
on source chunking/compression. No duplicated cache of overlapping windows is
created. Preparation scans training data once in 256-slice blocks to compute
statistics, which can take substantial I/O on the complete source ensemble.

The artifact stores simulation splits before windows are generated, source
fingerprints, conditioning and train-only normalization. Reuse the same native
preparation for both sampling modes. Cone IDs follow the sorted raw file list,
as in the existing raw multi-field reader; preserve the exact file population
when comparing studies. Windows from a simulation never cross split boundaries.

## Contiguous native windows

```bash
python fno_multifield.py train \
  --preparation experiments/los_windows/preparation.json \
  --inputs density --targets neutral_fraction \
  --sampling contiguous --window-size 256 --window-halo 32 \
  --windows-per-cone 8 --epochs 20 --batch-size 1 --workers 4 --device cuda \
  --run-dir checkpoints/los_windows/contiguous_seed42
```

Each sample preserves the full transverse plane and 256 consecutive native
LOS cells. There are eight random draws per training cone per epoch; centers
are sampled uniformly among that cone's native cells. The sampling seed is the
training `--seed`; draws change with epoch and are independent of DataLoader
worker count. More draws provide more coverage, not more independent simulations.
One epoch therefore has a different compute cost than in full-cone training.

The central 192 slices contribute to normalized MSE. The 32 slices on each side
provide context. At real cone boundaries, edge values are replicated; synthetic
cells and halos are excluded from the objective. Very short cones are supported
through the same padding and validity mask. Keep window sizes compatible with
the selected backbone's downsampling and retained Fourier modes.

Inputs include selected normalized fields, actual per-cell `1/(1+z)`, optional
normalized simulation parameters, relative LOS distance in fixed units of
1000 comoving Mpc, and a native-validity channel. Absolute redshift conditioning
(`z` or `z_params`) is required. Distance coordinates are centered on each
window with the original physical spacing. Absolute transverse coordinates are
not added; the selected backbone may still add its standard local grid embedding.

## Fine window plus coarse surroundings

Use the same preparation and training flags, changing the mode and run directory:

```bash
python fno_multifield.py train \
  --preparation experiments/los_windows/preparation.json \
  --inputs density --targets neutral_fraction \
  --sampling coarse_context --window-size 256 --window-halo 32 \
  --windows-per-cone 8 --context-factor 4 --context-xy 4 --context-features 8 \
  --epochs 20 --batch-size 1 --workers 4 --device cuda \
  --run-dir checkpoints/los_windows/coarse_context_seed42
```

The fine input and supervised target remain at native resolution. A second
input covers a centered 1024-slice surrounding region, using **only selected
input fields**, never targets. Box averaging over 4 × 4 × 4 cells low-pass
filters and decimates it to 35 × 35 × 256 for a native 140 × 140 plane. This is
a simple box filter, not an ideal spectral antialiasing filter. Transverse
pooling must divide the native dimensions. Coordinates and native-validity
fractions are pooled on the same grid; simulation parameters are broadcast
after pooling to avoid allocating huge constant arrays.

A small trainable encoder uses periodic transverse padding and replicated LOS
padding. Its features are sampled at fine voxel positions in the central
region; an additional spatially pooled summary lets distant surroundings
influence every fine prediction. Both feature sets enter the fine backbone.
`--context-features 8` adds 16 feature channels before its lifting layer.
The entire encoder and backbone train jointly from scratch. This is a wider
surrounding-region branch, not a full-lightcone branch; increasing
`--context-factor` increases its physical reach and CPU read volume.

## Evaluation, export and sweeps

Validation and test evaluate every native voxel of every selected cone. Input
windows overlap through their halos; disjoint central predictions are stitched
in order, with the final partial core and both endpoints retained exactly once.
No blending is performed, so this does not guarantee absence of seam artifacts.
Only individual windows live on the GPU; the assembled prediction lives on CPU.
Metrics accumulate in slabs on CPU, including the existing transverse spectra.
They do not yet add cylindrical LOS spectra or seam-specific diagnostics.

```bash
python fno_multifield.py evaluate --checkpoint checkpoints/los_windows/coarse_context_seed42/best.pt \
  --split test --out native_test_metrics.json --device cuda
python fno_multifield.py predict --checkpoint checkpoints/los_windows/coarse_context_seed42/best.pt \
  --cone-id 0 --out native_prediction.h5 --device cuda
```

Checkpoints record the complete sampling configuration, input-channel contract,
model configuration and preparation. Reload reconstructs the correct context
encoder automatically. Exports contain full native targets and predictions in
physical units, increasing `target_z`, aligned `lightcone_distances`, and sampling
metadata. They never include padded cells.

Use the supplied `experiments/los_windows/contiguous.json` or
`experiments/los_windows/coarse_context.json` with the existing sweep planner:

```bash
python -m util.multifield_experiments plan \
  --preparation experiments/los_windows/preparation.json \
  --fields density,neutral_fraction,brightness_temp,los_velocity --stage pairwise \
  --sampling-config experiments/los_windows/coarse_context.json \
  --out experiments/los_windows/coarse_plan.json
```

The existing `slurm/train_multifield.sbatch` runner forwards those settings.
Plan hashes, paired-baseline matching and aggregate grouping include sampling
settings, so different window recipes are not silently paired or pooled.
No cluster jobs are submitted by preparation or planning.

These modes retain the multi-field normalized-MSE recipe and one process/GPU
per run. Compare modes at matched training budgets and inspect seams and
large-scale context sensitivity before selecting window/halo sizes. GPU memory,
native HDF5 throughput and scientific accuracy need measurement on real cones.
