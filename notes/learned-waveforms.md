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
learnable amplitudes. Initialization uses seeded Gaussian values, rejecting
draws with a very weak fundamental. Tables are trained from the first optimizer
step. There is no freeze schedule, learnable frequency, or complex arithmetic.

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

Tables receive no weight decay. Other parameters retain the entry point's
configured decay. These settings round-trip in `ModelConfig`/run metadata.
Direct registry construction takes `bins` and `condition_limit` as hyperparameters.

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
    mixed[b,o,k...] = sum_i weight[i,o,k...] * coefficients[b,i,k...]
    output = U @ mixed

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

## Execution and numerical safeguards

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
the checkpoint under `*.spectral.bank.tables.<axis>`. The plotting helper below
shows both candidate and effective modes at an explicitly supplied grid size.

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
