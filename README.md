# fno-21cm

A Fourier Neural Operator (FNO) surrogate for the **density → neutral
fraction** map of 21cmFAST reionization lightcones. Part of a master's thesis
on Effective Field Theory for reionization simulations.

Two pipelines live side by side:

- **v2 — 2-D, per-slice.** Takes one matter-density slice (140 × 140,
  200 Mpc box) at a fixed redshift and predicts `x_HI` on the same grid.
- **v3 — 3-D, full lightcone.** Takes an entire lightcone cube, interpolates
  the LOS axis down to a fixed 256-cell grid, and predicts the whole `x_HI`
  cube in a single forward pass. The default input carries density, explicit
  `1/(1+z)`, and the 11 sampled simulation parameters. `INPUT_FEATURES`
  selects controlled conditioning ablations.

## Repository layout

### Top-level layout

```
.
├── fno_21cm.py, fno_21cm_3d.py    # training entry points (v2, v3)
├── modeling.py, losses.py         # shared model factory and Trainer adapters
├── siren_fno_3d.py                # 3-D SirenFNO architecture
├── models_ufno.py, ufno.py        # U-FNO network architecture
├── dataset/                       # readers, PyTorch datasets, cache builders
│   ├── dataset.py, dataset_3d.py
│   ├── loader.py, lightcone_params.py
│   └── build_trainset.py, build_cubes.py
├── viz/                           # prediction and spectral-weight plots
│   ├── visualize.py, visualize_3d.py
│   ├── visualize_3d_detailed.py
│   └── visualize_spectral_weights.py, visualize_spectral_weights_z.py
├── util/                          # metadata, diagnostics, setup helpers
│   ├── neuralop_setup.py, run_metadata.py
│   └── metrics_21cm.py, spectral_weights.py
├── slurm/                         # all sbatch scripts (cluster)
├── tests/                         # unit and integration tests
├── figures/                       # all generated plots
├── data/                          # raw lightcone .h5 files (gitignored)
├── checkpoints/, checkpoints_3d/  # trained models (gitignored)
└── neuraloperator/                # vendored third-party lib (gitignored)
```

### 2-D pipeline (v2)
| File | Purpose |
|------|---------|
| `fno_21cm.py` | 2-D training entry point. |
| `dataset/dataset.py` | `LightconeSliceDataset` / `SliceCache` — per-redshift 2-D slices. |
| `dataset/build_trainset.py` | One-time pass: extract K slices/cone into a compact `trainset.h5`. |
| `viz/visualize.py` | Loads a 2-D checkpoint and plots true vs predicted `x_HI` + scatter into `figures/`. |
| `figures/comparison_*.png`, `figures/scatter_*.png` | Example outputs from the v2 run. |

### 3-D pipeline (v3)
| File | Purpose |
|------|---------|
| `fno_21cm_3d.py` | 3-D training entry point (full lightcone in / full cube out). Auto-detects the cube cache; falls back to streaming. |
| `dataset/dataset_3d.py` | `LightconeCubeDataset` (streamed) and `LightconeCubeCache` (pre-computed) — both expose the same one-cube-per-index interface. |
| `dataset/build_cubes.py` | One-time pass: pre-interpolate every lightcone to a fixed z-grid; writes `cubes_3d.h5`. ~10x faster training reads. |
| `viz/visualize_3d.py` | Loads a 3-D checkpoint and its run metadata; renders image comparisons plus global-history, power-spectrum, Fourier-correlation, and bubble-size diagnostics. |
| `viz/visualize_spectral_weights.py` | Plots per-layer Fourier-weight magnitudes over training epochs, selected-epoch profiles, and high-mode/low-mode cutoff ratios. |
| `viz/visualize_spectral_weights_z.py` | Compact version that renders only the LOS/Z modes and writes a Z-only CSV. |

### SLURM scripts (`slurm/`)
| File | Purpose |
|------|---------|
| `slurm/train.sbatch` | Single-GPU training (H200 default; change `--gres` for A30/A100). |
| `slurm/train_h200_4gpu.sbatch` | 4-GPU DDP training on the H200 node (4 × H200 NVL, NVLink). |
| `slurm/train_sirenfno_h200_4gpu.sbatch` | 4-GPU H200 DDP training for SirenFNO with explicit SIREN defaults and a separate `checkpoints_3d_sirenfno/` output directory. |
| `slurm/train_ufno_h200_4gpu.sbatch` | 4-GPU DDP training of the **U-FNO v1** (3 FNO + 3 U-Fourier blocks; BatchNorm + SyncBN; modes (16,16,16); 0.5/0.5 L²/H¹). |
| `slurm/train_ufno_v2_h200_4gpu.sbatch` | 4-GPU DDP training of the **U-FNO v2** "A+B+C bundle" — asymmetric Z modes (16,16,32), GroupNorm in the U-Net path, H¹-weighted loss `0.3·L² + 0.7·H¹`. Writes to `./checkpoints_3d_ufno_v2/`. |
| `slurm/train_ufno_v3_anisoz_h200_4gpu.sbatch` | **U-FNO v3 / option D** — anisotropic Z U-Net: stride=(2,2,4) on the outermost stage, doubling LOS receptive field. Inherits v2 overrides. Writes to `./checkpoints_3d_ufno_v3_anisoz/`. |
| `slurm/train_ufno_v3_globalres_h200_4gpu.sbatch` | **U-FNO v3 / option E** — global-pooling residual added to each U-Net path (gives cone-level context to the local-feature path). Composable with v3-anisoz or v3-los1d via env-var. Writes to `./checkpoints_3d_ufno_v3_globalres/`. |
| `slurm/train_ufno_v3_los1d_h200_4gpu.sbatch` | **U-FNO v3 / option F** — replaces the 3-D U-Net with a stack of 1-D LOS-only Conv3d layers (kernel `(1,1,7)`, 4 layers; 25-cell receptive field). Spectral path keeps doing the transverse work. Writes to `./checkpoints_3d_ufno_v3_los1d/`. |
| `slurm/viz.sbatch` | Render PNGs from the latest plain-FNO checkpoint in `./checkpoints_3d/` (4 cones per split, evenly-spaced z; 1 GPU, 30 min). |
| `slurm/viz_ufno.sbatch` | Same, for the U-FNO checkpoint in `./checkpoints_3d_ufno/`. |
| `slurm/viz_detailed.sbatch` | **Detailed** variant — 16 cones per split, active-z slice picker, and an automatic shared low-z cutoff where global `x_HI` first departs from its settled late-time state. Set `PLOT_Z_MIN` to override the cutoff. FNO checkpoint. |
| `slurm/viz_sirenfno.sbatch` | Standard SirenFNO prediction visualization using the best checkpoint by default. |
| `slurm/viz_sirenfno_detailed.sbatch` | Detailed SirenFNO visualization with 16 cones per split and active-redshift diagnostics. |
| `slurm/viz_ufno_detailed.sbatch` | Same as `viz_detailed.sbatch` but for the U-FNO checkpoint. |
| `slurm/viz_spectral_weights.sbatch` | Render the compact epoch-by-epoch Fourier-weight history written during 3-D training. Set `CHECKPOINT_DIR` for another run. |
| `slurm/viz_spectral_weights_z.sbatch` | Render only Z/LOS spectral-weight diagnostics for a selected checkpoint directory. |
| `slurm/viz_spectral_weights_ufno.sbatch` | Render spectral-weight diagnostics for the basic U-FNO run in `./checkpoints_3d_ufno/`. `CHECKPOINT_DIR` remains overridable for another U-FNO variant. |

The prediction-visualization scripts write into a per-run subfolder under
`figures/` whose
name encodes the model variant, timestamp, and (when running under SLURM)
the job id — e.g. `figures/ufno_20260606-143022_job3965704/`. A
`run_info.txt` is dropped in each folder summarising the config so old
renders are self-explanatory. Successive viz runs never overwrite each
other.
| `slurm/build.sbatch`, `slurm/merge.sbatch` | v2 slice cache build + merge (array job). |
| `slurm/build_cubes.sbatch`, `slurm/build_cubes_merge.sbatch` | v3 cube cache build + merge. |

All sbatch scripts auto-resolve the project root from their own location, so
they can be submitted from anywhere (`sbatch slurm/train.sbatch` from the
project root is the conventional usage).

### Shared
| File | Purpose |
|------|---------|
| `dataset/loader.py` | `LightconeFile` — h5py reader for 21cmFAST `raw_lightcone_v2.0` files (used by both pipelines). |

**Not included in the repo** (see `.gitignore`):
- `data/` — the 21cmFAST lightcone HDF5 files. Provide your own.
- `neuraloperator/` — the third-party library (see below).
- `checkpoints/`, `checkpoints_3d/` — trained model artifacts (regenerated by training).

## Environment

Python 3.12+ with PyTorch. The model depends on the
[`neuraloperator`](https://github.com/neuraloperator/neuraloperator) library
plus a few scientific packages. On a conda system:

```bash
conda install -c conda-forge pytorch numpy scipy h5py matplotlib \
    tensorly tensorly-torch opt-einsum
```

> Note: the import name `tltorch` is provided by the conda package
> **`tensorly-torch`** (there is no `tltorch` package).

### Providing `neuralop`

The training and visualization scripts prefer a **local checkout** of the
library if one is present. They search, in order:

1. `./neuraloperator/` — checkout vendored inside the repo
2. `../neuraloperator/` — checkout sibling to the repo (the
   `project/{data, neuraloperator, fno-21cm}` layout)
3. `./` — `neuralop` dropped straight into the repo root

If none is found, the scripts fall back to an installed `neuralop`. To
use a local checkout next to the repo:

```bash
cd ..                                                    # parent project folder
git clone https://github.com/neuraloperator/neuraloperator
```

Otherwise just `pip install neuraloperator` (or the conda deps above plus the
package) and the code will use the installed copy automatically.

## Usage

### 3-D (full lightcone, current focus)

```bash
# Optional: point at lightcones outside ./data (default fallback).
# slurm/train.sbatch already exports this for the cluster.
export LIGHTCONE_DIR=/path/to/21cmfast_11d_sample_h5_files

# (Recommended) One-time: pre-interpolate every cube into cubes_3d.h5.
# Speeds up training reads ~10x.  On the cluster, use the array job:
#     AID=$(sbatch --parsable slurm/build_cubes.sbatch)
#     sbatch --dependency=afterok:"$AID" slurm/build_cubes_merge.sbatch
# Locally:
python -m dataset.build_cubes --data "$LIGHTCONE_DIR" --out cubes_3d.h5

# If the cache exists at ./cubes_3d.h5 (or $CUBES_CACHE), training and
# visualization use it automatically; otherwise they stream raw lightcones.
python fno_21cm_3d.py

# Visualize the best-validation checkpoint (set CHECKPOINT_KIND=final for
# the final epoch instead).
python -m viz.visualize_3d

# Plot Fourier weight evolution from initialization through every epoch.
python -m viz.visualize_spectral_weights

# Plot only the Z/LOS Fourier modes.
python -m viz.visualize_spectral_weights_z
```

Select the 3-D architecture with `MODEL_KIND`:

```bash
MODEL_KIND=fno python fno_21cm_3d.py
MODEL_KIND=ufno python fno_21cm_3d.py
MODEL_KIND=sirenfno python fno_21cm_3d.py
```

On the four-GPU H200 job:

```bash
sbatch slurm/train_sirenfno_h200_4gpu.sbatch
sbatch slurm/viz_sirenfno.sbatch
sbatch slurm/viz_sirenfno_detailed.sbatch
```

The SirenFNO defaults to retained modes `(16,16,16)`, four residual layers,
width 32, SIREN hidden width 64, 16 Fourier features, and eight replicated
cells of non-periodic LOS padding. Its spectral weights are generated only
for the four retained signed X/Y FFT quadrants rather than for the complete
lightcone FFT grid. Set retained modes with `N_MODES_X/Y/Z`; override the
SIREN-specific settings with `SIREN_HIDDEN_DIM`,
`SIREN_OMEGA`, `SIREN_N_HIDDEN`, `SIREN_FEATURE_DIM`, `SIREN_FF_SIGMA`,
`SIREN_LEARNABLE_FF`, `SIREN_PADDING_X/Y/Z`, and `SIREN_MLP_DROPOUT`.

For controlled repeated runs, keep `SPLIT_SEED=42` unchanged and vary
`RUN_SEED`. This changes model initialization and training order while using
the same train/validation/test cones:

```bash
RUN_SEED=41 CHECKPOINT_DIR=checkpoints_3d_ufno_z32_seed41 python fno_21cm_3d.py
RUN_SEED=42 CHECKPOINT_DIR=checkpoints_3d_ufno_z32_seed42 python fno_21cm_3d.py
RUN_SEED=43 CHECKPOINT_DIR=checkpoints_3d_ufno_z32_seed43 python fno_21cm_3d.py
```

Set `DETERMINISTIC_RUN=true` when bitwise repeatability is more important
than maximum training throughput. Every seed must use its own
`CHECKPOINT_DIR`.

Each lightcone is interpolated along the LOS axis from its native ~2340 cells
down to `N_Z = 256` (configurable) so a full cube fits on an A30 (24 GB) at
`BATCH_SIZE = 1`. `INPUT_FEATURES` accepts `density`, `params`, `density_z`,
or `density_z_params` (default). Parameter statistics are fitted only on
training cones and stored in `run_metadata.json`, so cached and raw loading
use the same held-out transformation. The FNO's grid positional embedding
adds normalized (x, y, grid-z) coordinates. The cache is sampled uniformly
in redshift, so grid-z is normalized redshift rather than comoving distance.

Each training run writes `best_model_state_dict.pt` (lowest globally reduced
`val_l2`), `final_model_state_dict.pt`, `run_metadata.json`, and a compact
`spectral_weight_history.npz`. The latter stores channel-aggregated RMS
complex-weight magnitudes for every Fourier layer at initialization and after
every epoch. `viz/visualize_spectral_weights.py` turns it into mode/epoch heatmaps,
selected-epoch profiles, a high-mode/low-mode cutoff ratio, and a CSV export.
The transverse axes fold positive and negative frequencies into absolute
mode index; the LOS axis follows the non-negative real-FFT convention. Thus,
for NeuralOperator `n_modes=(16,16,16)`, the plots show absolute transverse
indices `0..8` and LOS indices `0..8`, rather than 16 distinct positive
wavenumbers. The Wen et al. U-FNO implementation retains separate positive
and negative transverse slices, so `modes=(16,16,16)` spans absolute
transverse indices `0..16` and LOS indices `0..15`.
Visualization
defaults to the best checkpoint and reproduces the recorded model and input
configuration. Its `physical_metrics.json` includes global `x_HI(z)`,
active-window errors, isotropic power spectra, Fourier cross-correlation, and
ionized-region size summaries. It also reports X/Y transpose consistency and
transverse edge-versus-interior residuals for boundary diagnostics.

Key hyperparameters at the top of `fno_21cm_3d.py`:
`N_MODES = (16, 16, 16)`, `HIDDEN_CHANNELS = 32`, `N_LAYERS = 4`,
`BATCH_SIZE = 1`, `LEARNING_RATE = 5e-4`, `N_EPOCHS = 100`.

Loss: `0.5 * absL2 + 0.5 * absH1` with `d=3`. H1 uses periodic finite
differences in the transverse X/Y plane and centered differences on interior
LOS cells. The value term still covers the complete cube, but the unrelated
`z=5` and `z=25` endpoints are excluded from the LOS derivative term.

For U-FNO, training starts with the L2 term and linearly introduces H1 over
five epochs. Its default base learning rate is `1e-4` and gradients are clipped
to norm `1.0`, preventing the output sigmoid from collapsing to all zero.
Override these safeguards with `UFNO_LEARNING_RATE`,
`UFNO_H1_WARMUP_EPOCHS`, and `UFNO_GRAD_CLIP_NORM`. Metrics include
`val/test_pred_mean`, `pred_std`, and saturation fractions so output collapse
is visible after the first epoch.

### 2-D (legacy, kept for comparison)

```bash
# 1. Build the compact slice cache (once)
python -m dataset.build_trainset --data ./data --out trainset.h5

# 2. Train (expects trainset.h5 in the project root)
python fno_21cm.py

# 3. Visualize predictions from the latest 2-D checkpoint
python -m viz.visualize
```

Key hyperparameters are constants at the top of `fno_21cm.py`
(`N_MODES`, `HIDDEN_CHANNELS`, `N_LAYERS`, `BATCH_SIZE`, `LEARNING_RATE`, ...).

## Status

The 2-D model recovers large-scale ionization morphology but struggles with
sharp bubble edges and with parameter sets outside the training distribution
— partly a small-dataset / train-val-test split artifact, partly the
single-slice input not uniquely determining `x_HI` (a given 2-D density slice
can correspond to very different ionization states depending on the global
reionization history).

The 3-D version addresses the latter directly: the model sees the entire
cube at once, so the global reionization history is encoded in the input,
and the explicit `1/(1+z)` channel lets the network condition cleanly on
cosmic time. Active work: training on the full ~6600-cone dataset,
parameter sweeps over `n_modes` and `hidden_channels`, and astrophysical
parameter conditioning.
