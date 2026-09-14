# Generalized spectral operator with coefficient interactions

Status: implemented on `codex/frequency-mixing-transform`, 2026-09-14.
The original proposal below is retained as design context. See
[frequency-mixing.md](frequency-mixing.md) for the implemented interface and
validation. Two implementation choices refine the proposal: FFT cutoff counts
are retained for compatibility and expanded into complete real phase pairs;
B0 is evaluated directly with FFTs rather than materializing a real adapter.
No simulation training runs have been launched.

## Recommendation and scope

Use a fixed Fourier analysis/synthesis transform and learn interactions between
its retained coefficients. Keep the transform interface independent of the mixer
so the existing learned-waveform bases can be attached later. The first version
generalizes the operator acting in transform coordinates, not the Fourier
transform itself. It does not prescribe exponential envelopes, Laplace poles,
or a periodic/nonperiodic decomposition.

The kernel is shared across examples; coefficients depend on each input.
Input-conditioned kernels or attention are a separate nonlinear extension.
Keep the existing physical-space activations between residual blocks.

## Current implementation

- `learned_waveform_operator.py`: separable analysis, coefficient mixing, and
  synthesis already have distinct methods.
- Its mixer has channel matrices at individual columns and cross-phase matrices
  within each common dilation group. It cannot connect different groups.
- Learned waveform columns are QR combinations of candidates: their labels are
  not literal Fourier frequencies after learning.
- `tests/test_waveform_phase_mixing.py` already verifies output and gradient
  equivalence to the actual 2-D/3-D Fourier operators on compatible mode sets.
- `operators.py` supplies local/global slots; `modeling.py` supplies shared
  configuration for all three tasks. Local slots process overlapping windows,
  while the global slot processes the bottleneck field.

## Mathematical contract

For an orthonormal retained real basis U, let c = U^T v. Define

    y = U [B0 c + A_theta (P_theta c)].

B0 is the existing Fourier multiplier expressed in real coordinates. Its
channel/phase structure must represent complex Fourier multiplication, rather
than assuming every arbitrary real phase block is translation-equivariant.
The residual term introduces cross-frequency interactions. A zero residual
recovers B0 exactly on the same retained subspace and mode convention.

For M retained product modes and C channels:

    P: r x (C M)
    A: (C M) x r
    B[o,k;i,q] = B0[o,k;i,q] + sum_a A[o,k,a] P[a,i,q].

Generate A and P with coordinate networks using the complete mode tuple,
including real sine/cosine/DC type per axis. Do not allocate independently
learned parameters for every frequency pair. This is a learned separable kernel
in coefficient space, not a learned family of spatial exponential envelopes.

Every retained frequency can affect every other one. The residual has matrix
rank at most r, however: full connectivity does not imply an unrestricted dense
matrix. Increasing r expands the approximation class. Neither a fixed small
rank nor a fixed finite coordinate network guarantees arbitrary operators.
The pre-existing multiplier can retain its existing parameterization; a fully
coordinate-generated multiplier is an alternative, without claiming exact
loading of arbitrary legacy weights into that generator.

The spectral sum is the expansion in a finite-domain Fourier series, not a
numerical approximation of an unbounded continuous Fourier integral. Use a
fixed normalized domain and mode coordinates independent of current grid size
and mode cutoff. Increasing either must not relabel existing modes. Physical
lengths and window extents are separate metadata; resolution changes do not
automatically imply changes of physical domain.

With truncated U, identity mixing gives U U^T, a projection. Mixing cannot
recover dependence on input components already removed by analysis.

## Computational representation

Provide two backends under the same analysis/mixing/synthesis contract:

1. Dense reference for tiny grids: an explicit B, with a strict allocation
   limit. Also allow direct evaluation of a coordinate-pair kernel B_theta(k,q)
   on tiny cases to study a formulation without a fixed low-rank factorization.
2. Production backend: B0 plus coordinate-generated factors A and P, contracted
   in two stages. Never form their outer product on production grids.

Dense storage is O(C^2 M^2). For illustrative C=16 and M=16^3, one real FP32
matrix alone occupies 16 GiB; weights, gradients, and two FP32 Adam moments
would occupy about 64 GiB before activations. Generating pair weights with a
network avoids storing trainable entries but does not remove quadratic compute.

Factor evaluation/storage is O(C M r); application is O(batch C M r), plus
generator, baseline multiplier, and transform costs. Chunk factor generation
and contractions when needed. Reuse factors within one forward graph across
patches; do not retain stale autograd graphs across optimizer steps. Profile
transform cost and backward memory as well as mixer cost. Rank is independent
of the shell's existing channel-projection `spectral_rank`.

## Implementation sequence

1. Add `spectral_mixing_operator.py` with a fixed real Fourier basis provider,
   dense reference mixer, factorized mixer, and transform-agnostic wrapper.
   Use deterministic DC/sine/cosine conventions and complete phase pairs in
   the initial configuration. Exclude even-grid Nyquist initially, explicitly;
   validate odd/even grids and supported mode counts. A direct real basis is
   easiest to verify; add an equivalent FFT backend if profiling warrants it.
2. Reuse or extract separable contraction utilities without changing legacy
   checkpoint paths. Keep the old learned-waveform implementation operational.
   Add a Fourier multiplier adapter using the existing Fourier-equivalence
   fixture as the authority for signs, quadrants, and cutoff conventions.
3. Add a new `frequency_mixing` operator registry entry and tag. Expose mixer
   backend, mixing rank, generator size, and computation chunk size in
   ModelConfig, validation, metadata, and local/global slot settings. Use
   standard training for the fixed basis, not waveform-only schedules.
4. Initialize P nonzero and the final output layer of A to zero. The update is
   initially zero; A's output layer gets task gradients immediately, while P
   and earlier A layers begin receiving them after A becomes nonzero. Verify
   that behavior explicitly. Never zero both factors or also zero a gate.
   Normalize initialization against the configured coefficient count and log
   operator/output scales; avoid silently dividing the mathematical Fourier
   series sum by M when modes change.
5. Support loading an existing compatible multiplier into B0 as an explicit
   weight-only migration, with zero residual and a fresh optimizer. Same-model
   resume restores all state. Reject incompatible mode/basis migrations.
6. Enable either slot. Use the global slot first for integration and profiling
   because it avoids multiplying work across all patches; this does not limit
   the operator definition to the longitudinal axis. Then exercise the windowed
   path and both dimensionalities through the existing entry points and runner.
7. Add diagnostic export for sampled coefficient responses, group-to-group
   transfer, residual/total output norms, and singular values on small or sampled
   cases. Compute low-rank update norms from factors when possible. Do not
   interpret learned transfers as physical periodic/nonperiodic percentages.

## Symmetry and representation boundaries

Unrestricted fixed cross-frequency mixing generally breaks translation
equivariance. A linear operator commuting with translations on a periodic
domain is Fourier diagonal; full mixing and that exact symmetry cannot both
be imposed. This matters for the periodic transverse axes in our data.

Implement the general all-mode version, document the tradeoff, and measure
transverse shift sensitivity. A subsequent symmetry-preserving specialization
can be diagonal in signed transverse frequencies while mixing longitudinal
modes; it requires the correct phase/Hermitian constraints, not merely grouping
real columns by absolute frequency. Do not silently substitute this restriction
for the general operator. Local patch mixing uses relative window coordinates
and does not by itself establish exact global translation equivariance.

The interface can later accept tied or separate learned waveform bases. General
mixing and basis learning have overlapping freedoms: full square invertible
basis changes can be absorbed into a full mixer, whereas changing truncated
subspaces can matter. Coordinate sharing and rank constraints further affect
this equivalence. A learned-basis adapter must use basis labels honestly and
must not present them as exact Fourier frequencies or physical decay rates.

## Required verification before simulation training

- Tiny dense spatial oracle U B U^T: outputs and gradients agree with the
  factored implementation in both 2-D and 3-D.
- A single coefficient at one dilation produces an explicitly prescribed
  different dilation. Include cross-axis and DC/non-DC transfers; no accidental
  constant-preservation constraint unless separately requested.
- Zero residual reproduces the compatible Fourier reference in output and
  gradients with explicit signed-mode and Nyquist conventions.
- Same retained modes on two grids reproduce the same band-limited continuous
  input/output functions after consistent normalization.
- Tiny prescribed spatial multiplication a(x)v(x) is reproduced as the
  retained-subspace projection of its spectral convolution, not as an assertion
  that truncated products have no newly generated high frequencies.
- Initialization learns after successive optimizer steps; gradients are finite
  through factors, contractions, and channel maps. Check float64 derivatives,
  supported mixed precision, and CPU/MPS/CUDA paths as available.
- Configuration and checkpoint round trips, weight-only migration, existing
  relevant tests, windowed/global integration, and a bounded DDP smoke test.
- One representative forward/backward profile at actual local/global shapes,
  with measured time and peak memory. Select default rank from that budget;
  rank 32 is only a provisional starting value, not an established optimum.

When training is subsequently requested, compare the same fixed basis with B0
alone and B0 plus mixing, at recorded mode, parameter, and compute budgets.
Assess transverse translation behavior as well as current field, spectrum,
history, boundary, and bubble metrics. This isolates coefficient interactions
before optionally enabling joint basis learning.

## Background

Kovachki et al., Neural Operator: Learning Maps Between Function Spaces With
Applications to PDEs (JMLR, 2023), develops Fourier and low-rank integral-operator
parameterizations: https://jmlr.org/papers/v24/21-1524.html . The particular
coefficient-space residual design here is a proposal, not a result established
by that paper for reionization data.
