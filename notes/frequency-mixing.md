# Fourier operator with frequency mixing

Implemented on `codex/frequency-mixing-transform`. The default operator uses
fixed real Fourier coordinates and adds coordinate-generated interactions to
the existing signed-quadrant Fourier multiplier:

    c = U.T @ x
    y = FourierMultiplier(x) + U @ A_theta @ P_theta @ c

The multiplier is computed with FFTs, avoiding an explicit real coefficient
matrix. Real-packed complex weights make FP32/FP64 conversion and AMP safe.
The residual uses separable DC/sine/cosine transforms. The two coordinate
networks emit factors indexed by the complete multidimensional mode tuple and
phase type. Their parameters are independent of spatial resolution. They are
shared across input examples; this is linear mixing, not input-conditioned
attention. Physical-space nonlinearities remain in the surrounding blocks.

Every retained coefficient can affect every other coefficient, including DC.
The additional matrix has rank at most `FREQUENCY_MIXING_RANK`; it is not an
unrestricted dense operator. The transform is fixed and no exponential envelope
or Laplace pole is learned. The original waveform operator remains available.

## Use

Recommended local/global configuration, with mixing at the global bottleneck:

```bash
MODEL_KIND=localop LOCAL_OPERATOR=fourier GLOBAL_OPERATOR=frequency_mixing \
FREQUENCY_MIXING_RANK=32 python fno_21cm_3d.py
```

The same environment works with `fno_xhi2d.py` and `fno_zre.py`.
`LOCAL_OPERATOR=frequency_mixing` enables the windowed local branches too.
Factors are generated once per block forward and reused over its patch chunks.

Cluster runner presets (these commands submit training; implementation has not):

```bash
sbatch --export=ALL,ARCH=fno_fmix,LOSS=plain slurm/train_3d_matrix.sbatch
sbatch --export=ALL,ARCH=fmix_fmix,LOSS=plain slurm/train_3d_matrix.sbatch
```

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `FREQUENCY_MIXING_BACKEND` | `factorized` | Production factors; `dense` or `pairwise` for small references |
| `FREQUENCY_MIXING_RANK` | 32 | Rank of the additional channel/coefficient mapping |
| `FREQUENCY_MIXING_HIDDEN_DIM` | 64 | Width of the coordinate networks |
| `FREQUENCY_MIXING_CHUNK_SIZE` | 1024 | Coordinate generation chunk size |
| `FREQUENCY_MIXING_DENSE_LIMIT` | 4000000 | Maximum entries in any requested dense reference/diagnostic matrix |

These options are saved in `ModelConfig` and run metadata. Mixing rank is
independent of `LOCALFNO_SPECTRAL_RANK`, which reduces feature channels before
the operator. Default patch batch size remains a separate shell setting.

`dense` learns a full additive matrix for small-grid correctness studies.
`pairwise` generates that matrix using a network of both coefficient coordinates.
It avoids an independently learned table but still costs quadratically in modes.
The allocation limit bounds matrix entries, not all generator/backward memory.

## Mode convention and boundaries

`N_MODES_*` and `LOCALFNO_MODES_*` retain the existing signed-quadrant FNO cutoff
convention. They are NOT counts of real columns as in `learned_waveform`.
For FFT cutoffs `(mx,my,mz)`, the new residual retains
`(2*mx+1, 2*my+1, 2*mz-1)` real columns. In 2-D these are
`(2*mx+1, 2*my-1)`. The extra positive signed-axis endpoints complete the real
sine/cosine pairs corresponding to the legacy negative cutoff. Consequently
the residual uses a symmetric completion of the baseline's retained modes.

Each count must fit the spatial axis excluding even-grid Nyquist. For example,
the usual global `(16,16,16)` gives `(33,33,31)` columns, which fit the
`(35,35,64)` bottleneck. Invalid shapes fail explicitly. Existing modes on a
fixed normalized domain keep identical coordinate labels when resolution or
cutoff changes; changing the physical domain is not automatic resolution transfer.

Identity coefficient mixing means projection onto the retained subspace.
The mixer cannot recover dependence on input frequencies discarded by analysis.
Full mixing generally breaks translation equivariance, including transverse
equivariance. No axis restriction is silently applied. Windowing continues to
use the existing periodic transverse and finite longitudinal boundaries.

## Initialization and checkpoints

The synthesis network's final layer is zero initially; the analysis network is
nonzero. Thus the output initially equals the Fourier baseline exactly. The
synthesis output layer receives gradients immediately; earlier synthesis layers
and the analysis network begin learning once it becomes nonzero. A fixed factor
scale is stored with the checkpoint; the Fourier coefficient sum is not averaged
by the number of retained modes.

Resume matching runs normally. Incomplete frequency-mixing loads are rejected
to prevent silent partial loading of only the surrounding U-Net.

To migrate an existing compatible Fourier local/global checkpoint explicitly:

```bash
python -m util.frequency_mixing_checkpoint \
  --checkpoint-dir checkpoints/source_run --out-dir checkpoints/mixing_start \
  --slots global --rank 32
```

`--slots both` converts all six spectral blocks. The tool creates a new directory
with `initial_model_state_dict.pt` and model metadata. Use that file through
`INIT_CHECKPOINT`, with the corresponding operator/rank configuration and the
source run's other architecture/contrast settings. It is a weight-only start
with a new optimizer, never `RESUME_DIR`. Source files are untouched. The layer
API `load_fourier_weights` supports explicit individual-operator migration.

## Diagnostics and resource checks

```bash
python -m viz.frequency_mixing --checkpoint-dir checkpoints/my_run \
  --out-dir figures/my_run/mixing --max-modes 64
python -m util.profile_frequency_mixing --shape 35 35 64 --modes 16 16 16 \
  --device cuda --out figures/mixing_profile.json
```

Diagnostics export JSON summaries and NPZ arrays for sampled **residual**
couplings, coefficient labels, and sampled singular values. Rows are output
channel/mode pairs and columns are input channel/mode pairs. The sampled
singular values are not the full operator spectrum. Full residual Frobenius
norms use factor Gram matrices without constructing a quadratic matrix.

Local CPU checks with PyTorch 2.9.1, C=16 and rank=32 measured approximately:

| Case | Batch | Step including backward + Adam | Whole-process peak RSS |
| --- | --- | --- | --- |
| Global mixing, 35x35x64 | 1 | 0.95 s | 1101 MiB |
| Global Fourier baseline | 1 | 0.14 s | 604 MiB |
| Local mixing, 16x16x32 | 32 | 1.16 s | 606 MiB |

Each timing uses two measured steps after one warmup, with four CPU threads.
These are small CPU measurements, not GPU throughput or full-model memory
claims; the local and baseline profiles ran concurrently and are approximate.
Exact outputs are in `frequency-mixing-profile-*.json` beside this document.
GPU profiling and simulation training remain future work.

The verification suite covers dense spatial oracles and their gradients,
legacy Fourier equivalence, explicit frequency/DC transfer, resolution
consistency, projected spatial multiplication, gradient checks, repeated
optimizer steps, checkpoint migration, diagnostic export, mixed precision,
local/global integration, and two-process distributed gradient equivalence.
