# Learned orthonormal waveform operator

`learned_waveform` (aliases `waveform`, `orthogonal_waveform`) is available in
either slot of `MODEL_KIND=localop`, for all three training entry points.

```bash
MODEL_KIND=localop LOCAL_OPERATOR=learned_waveform GLOBAL_OPERATOR=learned_waveform \
WAVEFORM_LOCAL_BINS=15 WAVEFORM_GLOBAL_BINS=31 WAVEFORM_LR_RATIO=0.1 \
python fno_xhi2d.py
```

Replace the script with `fno_zre.py` or `fno_21cm_3d.py` for those tasks. Existing
data, loss, window, rank, mode-count, checkpoint and distributed switches still
apply. `LOCAL_OPERATOR=learned_waveform GLOBAL_OPERATOR=fourier` is also valid.
No simulation training runs were launched as part of implementation.

## Parameters and mode meaning

Each encoder and decoder branch owns an independent bank with one real mother
table per nonconstant axis. Both bottleneck blocks share the bank registered
under `bottleneck.0.spectral.bank`; their mixing tensors are independent. A
DC-only axis allocates no unused table. Banks are shared across channels and
patches, not across different task models or local branches.

The table consists of B equal bins on one normalized period, with continuous
learnable amplitudes. `WAVEFORM_INIT` selects the initial shape for both local
and global banks; its default is `random`. All tables remain trainable from the
first optimizer step, including named starting shapes. There is no freeze
schedule, learnable frequency, or complex arithmetic.

| `WAVEFORM_INIT` | Starting profile |
| --- | --- |
| `random` | Independent Gaussian bin values per branch/axis; reject a very weak fundamental |
| `smooth_random` | Random sine/cosine harmonic amplitudes decaying as 1/h²; reject a very weak fundamental |
| `sine` | One sinusoidal period; the quarter-period copy supplies cosine |
| `triangle` | Symmetric triangle, with weaker high harmonics than a square |
| `square` | Positive/negative half-period plateaus (zero at a bin center exactly on the discontinuity) |
| `sawtooth` | Linear ramp over one period, followed by a periodic jump |

Named shapes are evaluated at bin centers and mean-centered. The existing
anti-aliasing resampler filters them before QR; plotted effective modes can
therefore differ from the sharp initial table. Seeded random starts are
reproducible and independent between branch/axis tables. Named starts give the
same initial profile at equal bin counts, but the tables train independently.

On an axis of N cells the lowest nonconstant candidate period is N cells.
Candidate dilation k has period N/k, with phase offsets 0 and 1/4 cycle. Physical
periods are these cell counts times the effective spacing at that U-Net level:
local windows at the second encoder/decoder level cover twice the physical
extent of equal-size first-level windows; the bottleneck spans the downsampled
whole field. DC is a separate normalized constant, not a zero-frequency table.

`LOCALFNO_MODES_*` and the existing global mode settings count **retained real
columns**: DC, k=1/phase=0, k=1/phase=1/4, k=2/phase=0, etc. Even counts end
with an unpaired column. Odd counts preserve complete phase pairs. The maximum
is N on odd grids and N-1 on even grids because Nyquist is deliberately excluded.
This differs from an rFFT cutoff: equal numeric mode counts do not match Fourier
parameter budgets or retained subspace dimensions.

| Setting | Default | Meaning |
| --- | --- | --- |
| `WAVEFORM_LOCAL_BINS` | 15 | Odd number >=3 of raw bins per local-axis table |
| `WAVEFORM_GLOBAL_BINS` | 31 | Odd number >=3 of raw bins per global-axis table |
| `WAVEFORM_CONDITION_LIMIT` | 10000 | Maximum normalized candidate singular-value ratio, also enforces minimum singular value >=1/limit |
| `WAVEFORM_LR_RATIO` | 0.1 | Table learning rate divided by base/mixing learning rate |
| `WAVEFORM_INIT` | random | Initial bin profile, chosen from the table above |

Tables receive no weight decay. Other parameters retain the entry point's
configured decay. These settings round-trip in `ModelConfig`/run metadata.
Direct registry construction takes `bins`, `condition_limit`, and `init` as
hyperparameters. `ModelConfig.waveform_init` is recorded in checkpoint metadata
and the training configuration summary. Older metadata defaults to `random`;
loading a checkpoint always replaces the initial bin values with saved values.

## Sampling, transform, and reconstruction

The raw table describes a periodic step function. Its cosine/sine integrals
are computed analytically as a fixed real matrix. Before dilation k is sampled,
only harmonics h with k*h < N/2 and h <= (B-1)/2 are retained. This prevents
aliasing of the sampled candidates, while leaving the bins as learned parameters.
The sharp raw table and its filtered realization need not look identical. More
bins than the observable bandwidth add unidentifiable parameter directions.

Mean-zero candidate columns are normalized and concatenated after DC, then
orthogonalized with reduced QR on the **actual sampled grid**. Signs are made
consistent using the diagonal of R. U contains the orthonormal columns:

    U.T @ U = I
    coefficients = U.T @ input
    mixed[b,o,g,q] = sum_i,p M[i,o,g,p,q] * coefficients[b,i,g,p]
    output = U @ mixed

Here g identifies a tuple of candidate dilations and p/q index the available
phase combinations. On each axis, DC is a singleton and columns (1,2), (3,4),
etc. form pairs. Tensor products yield blocks of up to four phases in 2-D and
eight in 3-D. Each block has a full real channel/phase matrix, including
simultaneous phase changes on multiple axes. Different dilation tuples are not
mixed. After QR these are groups of effective columns, not necessarily literal
phase shifts of one waveform.

The original `weight` tensor stores diagonal phase entries. `phase_weight`
stores only valid off-diagonal entries; DC and unpaired terminal columns have
no invented partners, and there are no unused padded parameters. If there is
no complete pair on any axis, no `phase_weight` parameter is allocated. Cross
entries start at zero, retaining the previous initial output scale, and can
learn immediately. Diagonal initialization is unchanged. A full identity
requires identity channel mixing on the diagonal and zero cross-phase entries.

With sinusoidal tables and sufficient real columns, these blocks can express
the real form of complex Fourier multipliers: [A,-B;B,A] acting on real and
imaginary components (with the corresponding sine/cosine sign convention).
Tests map the actual 2-D/3-D signed-quadrant Fourier layers into these blocks
and compare outputs, input gradients, and Fourier-weight gradients. They
include signed frequencies and the irFFT DC-plane Hermitian completion.
Nyquist remains excluded by the waveform transform. Numeric mode counts are
still real-column counts, **not FNO cutoffs**; choosing `sine` alone does not
automatically match an FNO run's support or initialize its mixing weights.

For the current quadrant FNO cutoffs (m_x,m_y[,m_z]) strictly below Nyquist,
enough columns to cover all signed blocks and their real completion are
(2*m_x+1, 2*m_y-1) in 2-D and
(2*m_x+1, 2*m_y+1, 2*m_z-1) in 3-D. The extra signed-axis pair covers the
negative boundary included by the existing asymmetric slices. This is a
support inclusion rule, not a parameter-budget match; all counts must fit
the waveform shape limits.

Transforms are applied separately along each spatial axis. No dense product-grid
matrix is constructed. QR changes the effective modes into linear combinations
of the candidates; they are no longer necessarily compressed copies of the raw
mother waveform, nor do their indices specify a unique frequency.

Identity mixing is an orthogonal projection onto the retained product space.
It reconstructs arbitrary inputs only for a complete basis (possible on odd
grids with all columns retained). DC and in-span signals round-trip to numerical
precision. No pseudoinverse, ridge bias, learned inverse, or orthogonality loss
is needed. Learned channel mixing itself is unrestricted, and need not preserve
norms or be invertible.

## Checkpoint compatibility

New model/optimizer checkpoints round-trip with phase weights intact. Loading
an older diagonal-only model state automatically supplies zero phase weights,
preserving its predictions exactly. This is a **weight-only warm start**:
an old optimizer state cannot resume unchanged when new phase parameters are
present. Start a fresh optimizer when migrating an old run. Visualization uses
the unchanged diagonal weight shape and saved bank tables, so both formats
remain readable without changing the plotting command.

Full phase blocks increase the spectral mixing parameter count by up to 4×
in 2-D and 8× in 3-D relative to diagonal mixing, with smaller factors when DC
or unpaired columns are present. Check actual parameter counts and GPU memory
when comparing old and new runs; this change does not enforce equal budgets.

## Execution and numerical safeguards

### Waveform-only, alternating, and adaptation-freeze training

All three entry points and waveform SLURM launchers accept these settings.
They are training settings, recorded under `training.waveform_training` in run
metadata; the architecture configuration remains unchanged.

| Setting | Default | Meaning |
| --- | --- | --- |
| `WAVEFORM_TRAINING_MODE` | `joint` | `joint`, `waveform_only`, `kernel_only`, `alternating`, or `joint_then_kernel` |
| `WAVEFORM_ADAPT_EPOCHS` | `25` | Initial joint-training epochs before a permanent kernel phase in `joint_then_kernel`; `0` freezes waveforms from the start |
| `WAVEFORM_PHASE_EPOCHS` | `1` | Consecutive waveform epochs in each alternating cycle |
| `WAVEFORM_KERNEL_EPOCHS` | `5` | Consecutive kernel epochs in each alternating cycle |
| `WAVEFORM_FIRST_PHASE` | `waveform` | First phase of each alternating cycle; alternatively `kernel` |
| `WAVEFORM_KERNEL_SCOPE` | `spectral` | `spectral`: only LWF diagonal and cross-phase mixing weights; `all`: every trainable non-waveform parameter |
| `INIT_CHECKPOINT` | unset | Weight-only start from a matching model; new optimizer and cycle starting at epoch zero |
| `RESUME_DIR` | unset | Continue saved model, optimizer, scheduler, phase schedule and epoch count |

`waveform_only` updates only the bin tables. With default `spectral` scope,
`kernel_only` updates only mixing weights inside learned-waveform operators,
and `alternating` switches between these two groups. Spatial convolutions,
heads, normalization affine parameters, other operator families, and contrast
parameters stay fixed in these modes. `all` broadens the kernel phase to all
non-bin parameters; it has no effect during a waveform phase. Explicitly
frozen parameters are never re-enabled. Joint mode preserves ordinary training.

`joint_then_kernel` trains all eligible parameters for the first
`WAVEFORM_ADAPT_EPOCHS` epochs, then permanently freezes the waveform tables.
Set `WAVEFORM_KERNEL_SCOPE=all` to continue training every non-waveform parameter.
The default `spectral` scope instead continues only LWF mixing weights after
adaptation. The transition uses the existing optimizer and scheduler: kernel
Adam moments and step counts continue, while waveform values and Adam state
stop changing. No warm start, optimizer recreation, or LR restart occurs.
With `WAVEFORM_ADAPT_EPOCHS=25`, zero-based epochs 0–24 are joint and epoch 25
onward is kernel-only (the 26th training epoch). A duration equal to or longer
than the run leaves the whole run in the adaptation phase.

Non-joint modes put the model in evaluation mode during each training forward:
BatchNorm running statistics stay fixed and dropout is disabled in both phases.
This includes both phases of `joint_then_kernel`, so the transition does not
also switch normalization/dropout behavior. For comparisons to ordinary joint
training, account for this policy if the architecture uses BatchNorm or dropout.
Autograd remains enabled. Contrast refitting must be disabled because it would
modify the frozen model outside the optimizer. Task-loss schedules, if enabled,
still follow the overall epoch counter.

The controller masks inactive gradients with `None` **before gradient clipping
and the optimizer step**. Adam/AdamW therefore leave both inactive parameters
and their moment/step state unchanged, even after prior joint training. It does
not merely set their learning rate to zero. Gradients still traverse the whole
network and DDP retains a fixed graph; this favors correct phase switching over
reducing backward compute/memory. The existing cosine LR scheduler continues
once per overall epoch, including inactive phases. `WAVEFORM_LR_RATIO` still
controls the bin LR relative to the base LR.

Example: continue an existing LWF checkpoint with waveform-only updates. Use
the same architecture, input features, data split and preprocessing as its
source run. Here the bin LR ratio is explicitly increased from 0.1 to 1.0 for
the experiment; it is not a tuned recommendation.

```bash
sbatch --export=ALL,INIT_CHECKPOINT=checkpoints/source/final_model_state_dict.pt,WAVEFORM_TRAINING_MODE=waveform_only,WAVEFORM_LR_RATIO=1,N_EPOCHS=20,CHECKPOINT_DIR=checkpoints/source_waveform_only \
  slurm/train_2d_xhi_waveform.sbatch

sbatch --export=ALL,INIT_CHECKPOINT=checkpoints/source/final_model_state_dict.pt,WAVEFORM_TRAINING_MODE=alternating,WAVEFORM_PHASE_EPOCHS=1,WAVEFORM_KERNEL_EPOCHS=5,N_EPOCHS=60,CHECKPOINT_DIR=checkpoints/source_alternating \
  slurm/train_2d_xhi_waveform.sbatch
```

Substitute `train_zre_waveform.sbatch` or `train_3d_waveform.sbatch` for the other
tasks. For a kernel-only control, use `WAVEFORM_TRAINING_MODE=kernel_only`.
Initialization profiles are overridden by saved checkpoint tables. These are
continuations of existing LWF models; this feature does not convert a plain
FNO checkpoint into an LWF architecture.

Example: adapt a square-start basis for 25 epochs, then freeze only the bins
while fitting the rest of the network for the remaining 75 epochs:

```bash
sbatch --export=ALL,WAVEFORM_INIT=square,WAVEFORM_TRAINING_MODE=joint_then_kernel,WAVEFORM_ADAPT_EPOCHS=25,WAVEFORM_KERNEL_SCOPE=all,WAVEFORM_LR_RATIO=1,N_EPOCHS=100,CHECKPOINT_DIR=checkpoints/xhi2d_square_adapt25 \
  slurm/train_2d_xhi_waveform.sbatch
```

Use `WAVEFORM_INIT=sawtooth` and/or `WAVEFORM_ADAPT_EPOCHS=10` for the other
proposed controls, with a distinct checkpoint directory for each run. This
example uses the launcher's normal architecture/data defaults; retain the same
overrides as your comparison runs (including batch size, widths, and base LR).

Use `RESUME_DIR` instead of `INIT_CHECKPOINT` when continuing an interrupted
run with the **same schedule**. `N_EPOCHS` is the total target epoch count, not
the number of additional epochs. The two checkpoint settings are mutually
exclusive. A changed training mode/schedule or a checkpoint from before this
feature needs `INIT_CHECKPOINT` and a fresh optimizer. Legacy joint-mode
resumption remains supported. Resumption follows the model filename in the
training manifest, so final Adam state is not paired with an older best model.
This restores training state; exact stochastic replay still requires the same
data order/RNG state, which the existing checkpoint format does not save.
Adaptation duration is counted from the original run's epoch zero and is
preserved across resume; changing it requires an explicit fresh warm start.
The new `adapt_epochs` metadata key is written only for `joint_then_kernel`,
so existing joint, waveform-only, kernel-only and alternating schedules retain
their previous checkpoint schema.

Every waveform run using these entry points now saves its actual reference
tables to `waveform_initial_tables.pt` beside the metrics file. This is a mapping
from canonical parameter names to unnormalized bin tensors, captured after
warm start and before optimization. They are also stored in optimizer metadata
and preserved across resumption. For an old joint checkpoint without such
metadata, the reference is the resumed model, not its unavailable original
initialization. Do not substitute fresh random draws for these tensors.

The metrics JSONL records `waveform_training_phase`, `waveform_training_cycle`
(zero-based), `waveform_active_parameters`, `waveform_relative_update` (the
aggregate bin change divided by its norm at epoch start),
`waveform_distance_from_initial`, and `waveform_last_grad_norm`. The gradient
norm is the final batch's bin gradient **before masking/clipping**, including
the potential gradient in kernel-only phases; a zero parameter update in those
phases is intentional. These are raw-bin diagnostics; subspace movement still
requires examining the effective transforms.

For custom loops, instantiate `WaveformTrainingController` immediately after
the optimizer and before registering clipping hooks. Pass it to `MetricsTrainer`
as `waveform_training=controller`, or call `begin_epoch(epoch)` and
`prepare_batch()` yourself. Keep the optimizer and parameter groups stable
between phases so their histories can resume.

Prepare each branch's bases once per forward and reuse them across patch chunks.
The shared bottleneck reuses the same bases in both residual blocks. Only fixed
resampling matrices persist across calls; graph-bearing bases do not. All table
parameters exist before optimizer/DDP construction.

Basis construction, checks, QR, spectral mixing and transform contractions run
in float32 outside autocast (float64 for double inputs). The operator returns
the incoming activation dtype. CUDA/CPU use their native linear algebra; MPS
prepares the small bases on CPU and copies them back through differentiable
transfers. CPU and CPU bfloat16-autocast paths were tested locally; GPU/MPS
execution and cluster throughput still require hardware validation.

Check candidate norms before normalization so roundoff from a filtered-out
harmonic cannot become a fake basis direction. Nonfinite candidates, singular
or excessively ill-conditioned banks fail with an axis/shape error before QR.
Orthogonal output columns do not make differentiation through a collapsed raw
dictionary safe. The checks synchronize small scalar decisions with the host;
benchmark this overhead at the intended GPU patch count.

`metrics.jsonl` includes each bank/axis's last-training-forward condition,
minimum singular value, and orthogonality error under `waveform_*` keys. These
are snapshot diagnostics, not epoch maxima. Raw learned tables are stored in
the checkpoint under `*.spectral.bank.tables.<axis>`. New training metadata
records the input spatial shape, so the plotting helper reconstructs all local
and global grids without opening a simulation dataset.

## Waveform visualization

```bash
python -m viz.learned_waveforms --checkpoint-dir checkpoints/checkpoints_2d_xhi_local_lwf_lwf
python -m viz.learned_waveforms --checkpoint-dir checkpoints/my_3d_run --checkpoint-kind final
```

The report contains:

* `overview_bins.png`: one row per independent branch bank, one column per
  spatial axis, showing the learned bin amplitudes.
* `overview_modes.png`: the same branch/axis layout showing orthonormal mode
  shapes, with vertical offsets and unit-peak scaling for legibility.
* One detailed PNG per bank: raw table, filtered sampled candidates labeled by
  dilation/phase, and orthonormal modes including DC. The two bottleneck blocks
  appear as one explicitly shared bank, rather than two independent waveforms.
* One NPZ per bank with the actual, unscaled numerical arrays, plus a JSON
  manifest recording checkpoint, shapes, sharing, and output files.

Default output: `figures/waveforms/<run-directory>/<checkpoint-stem>/`.
Use `--out-dir` to change it and `--max-modes` (default 6) to display more modes.
All retained modes are exported to NPZ regardless of the display limit. Repeating
the same command replaces that snapshot's report; use a distinct output directory
to preserve earlier figures from an evolving best checkpoint.

For older runs without `input_features.spatial_shape`, add `--input-shape` with
the **original training input** dimensions, for example `--input-shape 140 140
256` for a run trained at that 3-D resolution, or `--input-shape 140 140` for
2-D/z_re. The script never assumes these example resolutions. z_re has two
spatial dimensions: its LOS slices are input channels. Local-only waveform
runs with windowed branches need only the saved local-window sizes.

The existing single-bank interface also remains available:

```bash
# List the exact bank prefixes in a saved model:
python -m viz.learned_waveforms --checkpoint /path/to/best_model_state_dict.pt
# Plot one local bank; --shape must match this run's actual local window:
python -m viz.learned_waveforms --checkpoint /path/to/best_model_state_dict.pt \
  --bank fno.encoder0.spectral.bank --shape 16 16 --out figures/waveforms.png
```

Prefixes can include extra wrappers (for example `module.`); use the listed name.
The helper also saves an NPZ containing the raw bins, normalized candidates and
orthonormal matrices. For a bottleneck bank, supply its downsampled whole-field
shape, not the local window shape. These are checkpoint snapshots, not histories.

## SLURM launchers

Submit from the project root, creating the logs directory **before** submission
because SLURM opens log files before the script runs:

```bash
mkdir -p logs
sbatch slurm/train_2d_xhi_waveform.sbatch
sbatch slurm/train_zre_waveform.sbatch
sbatch slurm/train_3d_waveform.sbatch
```

These use the existing cluster environment (`devel/miniforge`, `fno-env`) and
data/cache conventions, with both slots fixed to `learned_waveform`:

| Launcher | Default training | Resources requested | Default checkpoint directory |
| --- | --- | --- | --- |
| `train_2d_xhi_waveform.sbatch` | 20 epochs, batch 8, LR 1e-4 | 1 A100, 8 CPUs, 32G, 8h | `checkpoints/checkpoints_2d_xhi_local_lwf_lwf` |
| `train_zre_waveform.sbatch` | 200 epochs, batch 8, absolute L2, LR 1e-4 | 1 A100, 16 CPUs, 64G, 6h | `checkpoints/checkpoints_zre_local_lwf_lwf_l2` |
| `train_3d_waveform.sbatch` | 20 epochs, fixed batch 1, `LOSS=plain`, LR 1e-4 | 1 A100, 24 CPUs, 200G, 96h | `checkpoints/checkpoints_3d_lwf_lwf_plain` |

All set `WAVEFORM_INIT=random`, table LR ratio 0.1, bins 15/31, condition limit 10000, and gradient
clipping 1.0. Waveform controls, epoch count, and checkpoint directory remain
overridable via `--export`; 2-D batch sizes and LRs are overridable as in their
base launchers. The 3-D trainer uses `LOCALFNO_LEARNING_RATE` and its fixed
batch size of one. These are single-process/single-GPU launchers. Resource
requests follow comparable variants; waveform-specific GPU memory/runtime
have not been benchmarked.

```bash
sbatch --export=ALL,N_EPOCHS=50,WAVEFORM_LOCAL_BINS=31,WAVEFORM_LR_RATIO=0.05,CHECKPOINT_DIR=checkpoints/my_waveform_run \
  slurm/train_2d_xhi_waveform.sbatch
sbatch --export=ALL,LOSS=hybrid,N_EPOCHS=30 \
  slurm/train_3d_waveform.sbatch
sbatch --export=ALL,WAVEFORM_INIT=sine,CHECKPOINT_DIR=checkpoints/lwf_sine \
  slurm/train_2d_xhi_waveform.sbatch
sbatch --export=ALL,WAVEFORM_INIT=smooth_random,CHECKPOINT_DIR=checkpoints/lwf_smooth_random \
  slurm/train_zre_waveform.sbatch
```

3-D delegates to `train_3d_matrix.sbatch`, so its `LOSS` presets, scratch
staging controls and `CONTINUE_FROM` warm start work unchanged. It also accepts
`ARCH=lwf_lwf` directly. The wrappers override stale architecture variables
exported from your shell; other experiment overrides are preserved. Use a new
`CHECKPOINT_DIR` for concurrent runs or changed loss settings.

Plot every branch on the CPU partition after training:

```bash
sbatch --export=ALL,CHECKPOINT_DIR=checkpoints/my_waveform_run \
  slurm/viz_waveforms.sbatch
# For older metadata, export a space-separated shape before submitting:
export WAVEFORM_INPUT_SHAPE="140 140 256"
sbatch --export=ALL,CHECKPOINT_DIR=checkpoints/my_3d_run,CHECKPOINT_KIND=final \
  slurm/viz_waveforms.sbatch
```

`viz_waveforms.sbatch` requests 2 CPUs, 8G and 20 minutes on `compute`; no GPU
or data cache is needed. `WAVEFORM_VIZ_MODES` controls displayed mode count and
`OUT_DIR` controls the report directory. The scripts were syntax-checked and
their delegation tested with stub cluster commands; no jobs were submitted.

Old operator models retain their parameter registration and single optimizer
group. Learned-waveform runs have two groups; resume with their own checkpoint
and recorded configuration. Changing bins/modes/sharing requires a new run or
explicit migration. Fourier checkpoints cannot strictly load into waveform models.

## Scientific limitations and checks

The existing Hann windows, overlap-add normalization, boundary padding, spatial
paths, nonlinearities and task heads remain in place. Orthogonality is a property
of each untapered patch transform; it does not imply that the entire overlapping
window transform or learned network is orthogonal.

Periodic candidates still impose a boundary bias along a finite LOS. This
dictionary is generally not translation-equivariant. One mother per axis plus
dilations is not every possible waveform dictionary, and separable modes do not
directly form spherical atoms. Evaluate shift sensitivity, patch seams and LOS
boundary errors alongside the task's power spectra and bubble/front statistics.
At a new resolution, resampling and QR rebuild an orthonormal transform, but
the retained space can change: this does not prove resolution invariance.

Focused tests cover projection/reconstruction, float64 gradient checks, first-step
table updates, alias filtering, invalid candidates, chunked gradients, shared
bank ownership, mixed operator slots, all three head/dimension configurations,
checkpoint/optimizer continuation, new grids, mixed precision, and two-rank CPU
DDP versus full-batch gradients over two training steps.

Compare frozen random and learned random banks with fixed-basis baselines at
matched real parameter counts. Numerical tests establish implementation
properties, not predictive superiority on simulation data.
