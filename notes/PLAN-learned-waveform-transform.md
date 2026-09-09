# Learned waveform operator: implementation plan

Status: proposed design, 2026-09-09. No model implementation or training changes
are included. This plan refines `PLAN-learned-atom-operator.md`, particularly
its analysis/synthesis rule, initialization, sampling, and branch ownership.
The earlier note is preserved.

The proposed operator learns a discretized mother waveform for each U-Net
branch and spatial axis. Its bin amplitudes start randomly and receive task
gradients from the first optimizer step. Shorter-period copies define a real
function-space dictionary. A matched dual analysis transform computes its
coordinates; learned real channel mixing acts on those coordinates; synthesis
evaluates the resulting waveform expansion in real space.

## 1. Architecture and parameter ownership

Keep the current local/global U-Net, rank projections, spatial paths, task
heads, and normalized Hann overlap-add machinery. Register a new operator
named `learned_waveform` under `MODEL_KIND=localop`.

Interpret a branch as `encoder0`, `encoder1`, `bottleneck`, `decoder1`, or
`decoder0`. Give each branch its own bank; share the bottleneck bank between
its two existing residual blocks, while keeping their mixing weights separate.
Do not tie encoder and decoder banks initially. Expose `bank_scope=block` and
`bank_scope=slot` as later ablations, rather than conflating per-branch learning
with one bank shared by every local block.

Within a bank, use one mother table per spatial axis, shared across channels,
patches, and all wavelengths of that axis. Keep the LOS table independent of
the transverse tables. Train separate models/banks for 3-D x_HI, 2-D x_HI,
and z_re initially. This learns task-specific bases; it does not automatically
select a different basis for each input sample.

Construct and register all tables before optimizer/DDP creation. The model
owns a bank collection and passes the appropriate bank's materialized transform
to each block; each bank has one authoritative state-dict path. Build this
ownership explicitly, instead of registering the same parameter under several
blocks or creating parameters lazily during forward.

## 2. Wavelength, bins, and real modes

For branch b, axis a, let N be the operator's actual axis length and dx its
effective cell spacing. The initial lowest **nonconstant** period is

    lambda_0 = L = N * dx.

For a local branch, L is the patch extent at that level; for the global branch,
it is the whole bottleneck extent. With the current two factor-two pools,
cell spacing is dx, 2 dx, and 4 dx at successive levels. Local patches with
the same voxel count can consequently have different physical periods.
DC is a separate constant function; it has no finite wavelength.

Fix this period and the integer dilation ladder in v1. Store B free real bin
amplitudes theta[j] for equal phase bins [j/B,(j+1)/B) of [0,1). The raw
mother function is piecewise constant:

    psi_theta(t) = theta[floor(B * frac(t))].

This quantizes the waveform's coordinate, not its amplitudes: amplitudes stay
continuous floating-point parameters so ordinary backpropagation applies.
Initialize theta from seeded zero-mean Gaussian noise, once. Do not initialize
at sine waves, reset each step, or freeze the tables during a warmup.

Generate columns at integer k with periods lambda_k = lambda_0/k. Use two
fixed phase offsets, 0 and 1/4 cycle, per k, plus one fixed DC column. This
lets the bank represent two real phase directions without complex tensors.
For a sinusoidal mother these are a sine/cosine pair. For a general mother
they are not necessarily independent or orthogonal; conditioning must be checked.
In particular, a mother dominated by even harmonics can make a pair redundant.

Define M (the existing `modes[a]` value) as the **total number of retained real
columns**, including DC and phase copies. Enumerate DC, (k=1,p=0),
(k=1,p=1/4), (k=2,p=0), ... until M columns are filled. Even M leaves a final
unpaired column, which is permitted and recorded. Prefer odd M in new matched
experiments. Do not interpret M as an rFFT cutoff or impose M <= N/2+1.
The actual constraints are full column rank, M <= N, and the sampled bandwidth
limit. The conservative sampler below excludes Nyquist, so its maximum rank
is N-1 on an even grid, and N on an odd grid.

Start with odd B <= the smallest deployment N for that bank, capped at 31
(e.g. 15 bins on a 16-cell axis). Fix B in checkpoint metadata; it must not
change with evaluation resolution. Increasing bins beyond observable bandwidth
does not necessarily add usable degrees of freedom.

One mother plus dilations is a deliberate restriction: it learns a waveform
family, not every possible dictionary. Independent per-mode tables and more
than one mother per axis are later expressivity ablations. A periodic square
mother is not the entire Walsh family, and dilations alone do not reproduce
translated/localized Haar wavelets.

## 3. Sampling without aliasing, using only real arithmetic

Raw step functions have arbitrarily high harmonics. Neither linear interpolation
nor a smoothness penalty guarantees alias-free compression. Do not simply
gather table entries at increasingly large strides and call that anti-aliasing.

Use a fixed differentiable, scale-dependent low-pass resampling matrix R:

    raw_column_(k,p) = R_(N,B,k,p) @ theta.

A concrete all-real implementation integrates the piecewise-constant table
against cosine/sine functions analytically. For bin centers t_j=(j+1/2)/B:

    c_h = (2/B) sinc(h/B) sum_j theta_j cos(2 pi h t_j)
    s_h = (2/B) sinc(h/B) sum_j theta_j sin(2 pi h t_j)
    column[i] = sum_(h=1..H_k) [c_h cos(2 pi h (k i/N+p))
                              + s_h sin(2 pi h (k i/N+p))]
    H_k = min((B-1)/2, floor((N-1)/(2k))).

Here sinc(x)=sin(pi x)/(pi x); B is odd. This retains only frequencies
k*h < N/2, excludes the special Nyquist case, and gives a directly computable
real matrix R. Reject a requested nonconstant column if H_k < 1. Fourier
functions here are a numerical anti-aliasing filter: theta remains the learned
bin table, and the task coefficients are in the learned waveform dictionary.
No FFT or complex tensor is required anywhere in this operator.

The raw table remains fully flexible at its B bins, but the effective sampled
waveform is necessarily band-limited and scale-dependent. At the shortest
periods, only its fundamental may survive. Sharp bins cannot remain equally
sharp at every dilation on a finite grid. Log both raw and effective waveforms
so this distinction is visible. The low-pass approximation can itself ring
around discontinuities; do not promise the elimination of ringing.

Cache only the fixed R matrices and coordinate/quadrature tensors across
training steps. Derivatives of column with respect to theta are R. Mean
removal and filtering create unidentifiable directions; do not demand nonzero
task gradients in every bin on every minibatch. Check observability of the
combined resampling maps and finite, nontrivial bank gradients instead.

## 4. Matched transforms: the central requirement

Use column notation throughout. For one axis:

    Phi: N x M synthesis matrix (sampled waveform columns)
    Q = I/N: normalized uniform-grid quadrature
    G = Phi.T @ Q @ Phi: M x M Gram matrix.

Set Phi[:,0]=1 exactly. Subtract the sampled mean from every other column and
normalize its Q-norm to one. Reject effectively zero columns at initialization;
use an epsilon floor and diagnostics during training. Keep amplitude in mixing
weights, not extra learnable DC or per-atom amplitude scalars.

For a full-column-rank dictionary, the exact least-squares coordinates are

    A_0 = solve(G, Phi.T @ Q)
    coefficients = A_0 @ x
    reconstructed = Phi @ coefficients.

Thus A_0 Phi = I_M. If M=N, Phi A_0=I_N. If M<N, Phi A_0 is a projection
onto the retained span, not the identity on arbitrary inputs. This is an
unavoidable consequence of truncation; the existing spatial/skip paths carry
information outside that span.

**Proposed training default: a regularized dual**, preserving the meaning of
the learned waveform columns while bounding the solve:

    D = diag(0, 1, ..., 1)
    A_lambda = solve(G + lambda D, Phi.T @ Q)
    coefficients = A_lambda @ x
    y = Phi @ mixed_coefficients.

Start lambda at 1e-4 with normalized columns; it is an experimental default,
not a demonstrated optimum. DC is unpenalized, and sampled mean-zero AC
columns decouple it, so identity mixing preserves constants up to roundoff.
Regularized reconstruction shrinks nonconstant directions: it is not an exact
inverse or exactly idempotent projection. On an AC Gram eigenvalue g, the
coefficient round-trip factor is g/(g+lambda). Report this bias explicitly.
For separable multidimensional transforms, these axis-wise shrinkages compose;
this is not identical to one isotropic ridge solve on the full product basis.

Use a Cholesky-based solve of the small positive-definite regularized system,
with matrix construction and solves outside autocast in float32; float64 for
numerical reference tests. Never explicitly form an inverse. Gradients must
flow through Phi in **both** A_lambda and synthesis and through the solve.
PyTorch recommends solving systems instead of explicitly multiplying by an
inverse ([solve documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html)).

Do not use Phi.T Q for analysis and Phi for synthesis unless orthonormality
has actually been established. Unit column norms and a soft Gram penalty
do not establish it. A separately learned decoder could be useful, but does
not supply a known inverse either.

Two explicit reference alternatives:

* **Exact dual using reduced QR:** factor sqrt(Q) Phi = U T; evaluate
  A_0 = solve_triangular(T, U.T sqrt(Q)). This avoids squaring the condition
  number in normal equations, while synthesis still uses the original Phi.
  Use on well-conditioned banks when exact retained-space reconstruction matters.
* **Orthonormalized effective basis:** use Psi = Q^(-1/2) U, analysis
  Psi.T Q, synthesis Psi. This guarantees orthonormal columns when the input
  bank has full rank, but mixes raw atoms and changes which functions receive
  mode-wise weights. It does not force Fourier, and it no longer preserves
  literal dilated copies as the effective basis columns.

Reduced QR remains dependent on full column rank for valid differentiation;
it does not repair collapsed atoms by itself ([PyTorch QR](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html)).
Avoid a default SVD pseudoinverse with rank-threshold switching and derivative
hazards ([PyTorch pinv](https://docs.pytorch.org/docs/stable/generated/torch.linalg.pinv.html)).

## 5. Real spectral operator and depth

For 3-D, apply the small axis matrices by sequential contractions:

    c[b,i,p,q,r] = sum_(x,y,z) Ax[p,x] Ay[q,y] Az[r,z] u[b,i,x,y,z]
    d[b,o,p,q,r] = sum_i W[i,o,p,q,r] c[b,i,p,q,r]
    v[b,o,x,y,z] = sum_(p,q,r) Phix[x,p] Phiy[y,q] Phiz[z,r] d[b,o,p,q,r].

The 2-D version omits z/r. W and every intermediate are real. Use
W.shape=(C_in,C_out,Mx,My[,Mz]), initially matching the Walsh operator's
channel-mixing convention. Do not build a dense product-grid basis or matrix
over all 3-D voxels. Axis-wise setup costs roughly sum(N M^2 + M^3), while
mixing costs O(batch C_in C_out product(M)); spatial contractions must be
profiled at actual patch counts and ranks.

For v1, use one mix per existing residual block and retain the block's spatial
nonlinearity. The two bottleneck blocks already provide two operator layers.
Multiple linear mixes in unchanged coefficient space with no intervening
nonlinearity collapse to one linear mix. A later transform-once stack must
include real coefficient nonlinearities or mode coupling to add expressivity;
such nonlinearities are basis-dependent and require an explicit ablation.

Mode-wise mixing in an arbitrary dictionary is a spectral analogue, generally
a spatial integral operator, not necessarily a translation-invariant convolution.
Two real phase columns with independent diagonal mixing also do not reproduce
every complex Fourier multiplier; that requires appropriate real 2x2 phase
mixing. Keep claims and baseline comparisons specific to this architecture.

## 6. Training stability and spatial caveats

* Waveforms learn from optimizer step one. Use a separate table group at
  0.1 times the mixing learning rate initially, no table weight decay, and
  the existing gradient clipping. Start W with small **nonzero** random values;
  an exactly zero mix blocks initial task gradients to the bank. An identity
  operator can also be insensitive to a change of complete basis.
* Reject ill-conditioned random initial banks at expected deployment shapes
  with a bounded number of seeded redraws; fail clearly if unsuccessful.
  Suggested starting gates: AC Gram condition below 1e3 at initialization;
  log warning above 1e4 during training. Tune against observed precision and
  reconstruction errors. These thresholds are diagnostics, not guarantees.
* Add a small normalized Gram-coherence penalty, averaged once per unique
  bank/deployment rather than per patch chunk. Start at 1e-4 times the mean
  squared off-diagonal Gram entries; tune relative to task-loss scale. Optionally
  use cyclic table smoothness later, since it biases against sharp waveforms.
  Ridge keeps a solve defined but cannot restore missing dictionary rank.
* Log column norms before normalization, minimum singular value, condition,
  coefficient/output norms, waveform-gradient norms, reconstruction bias, and
  task loss separately. On nonfinite transforms, report bank/axis/shape and fail
  the step clearly; do not silently replace the basis or reinitialize trained
  parameters. In DDP, coordinate any fatal/skip decision across ranks.
* Reuse a materialized transform only inside one forward graph; recompute next
  forward. Never retain graph-bearing or detached learned matrices across
  optimizer steps. Share the same Phi/A within each analysis/mix/synthesis pass.
* A local transform sees an already Hann-tapered patch. Build its dual with the
  ordinary Q above; do not insert a second Hann weighting in the Gram. Preserve
  the current synthesis window and sum-of-window-products denominator. Perfect
  overlap-add reconstruction requires identity patch transforms; truncation or
  ridge bias is not repaired by overlap-add normalization.
* Preserve circular transverse padding and replicate LOS padding for the 3-D
  lightcone. Periodic atoms can be used on a nonperiodic interval, but impose a
  periodic representation bias; padding alone does not remove it. Compare LOS
  boundary errors and, later, a reflected-domain bank with matched transforms.
  Inspect 2-D task axis metadata separately: `models_zre_2d.py` currently uses
  circular/circular padding, which should not be assumed physically valid for
  every potential 2-D lightcone slice.
* Phase-locked learned dictionaries generally lose translation equivariance.
  Test periodic transverse shifts, shifted patch grids, and patch seams. Sharing
  a mother across patches does not restore equivariance. Separable 1-D banks
  also do not directly encode spherical bubbles; rotate/axis-swap diagnostics
  help distinguish anisotropy from task gains.
* Rebuild Phi, A and Q on a new resolution. Normalized quadrature keeps constant
  coefficient scaling stable, but interpolation alone proves no resolution
  invariance. Record whether local patches hold voxel count or physical extent
  fixed, recheck rank/bandwidth, and reject grids too small for saved modes.

## 7. Repository integration, in order

1. New `learned_waveform_operator.py`: `WaveformBank`, fixed real resampler,
   `materialize_transform(spatial_shape, device, dtype)`, differentiable dual,
   `LearnedWaveformOperator`, diagnostics and bank regularization accessor.
   Support 2-D/3-D, arbitrary accepted axis lengths, and explicit shape errors.
2. `operators.py`: registry builder, defaults, validation and aliases; set
   `uses_modes=True`, `rank_projected=True`, `windowed=True`, and identity
   `required_size`. Follow the Walsh contraction pattern, but do not inherit its
   cached fixed basis or transpose-only analysis/synthesis behavior.
3. `local_fno_3d.py` and `models_zre_2d.py`: add bank ownership at model level
   and a shape-aware transform preparation path in both residual block types.
   Existing `materialize_weights(device,dtype)` hooks lack spatial shape and
   cannot be reused unchanged. Prepare at the actual patch size or post-padding
   global size, once per bank/shape/forward, and reuse across patch chunks.
   Preserve the existing hook for SIREN and legacy registration order.
4. `modeling.py`: add the operator tag, `ModelConfig` fields, environment
   parsing, slot/per-branch settings, shape validation, descriptions and saved
   metadata reconstruction. Proposed controls: local/global bin counts, dual
   method, ridge, bank scope, phase scheme, and waveform LR ratio. Keep fixed
   wavelength ladder and the real anti-aliasing rule explicit in metadata.
5. `fno_21cm_3d.py`, `fno_xhi2d.py`, `fno_zre.py`: build deduplicated optimizer
   groups and attach bank regularization to the existing trainer regularizer
   path. That path resets before the model forward and adds its loss afterward;
   collect unique bank terms during forward without storing them across batches.
   Handle wrappers/DDP through a stable model accessor; do not edit vendored
   neuraloperator unnecessarily. Log penalties separately from prediction loss.
6. `training.py`, `util/run_metadata.py` and a new visualization helper: record
   branch/axis identities, bins, nominal and effective bandwidths, physical
   wavelengths, raw/effective tables, seed, solve bias, and numerical diagnostics.
   Do not label atom indices as Fourier frequencies in existing weight plots.
7. Add focused tests below, then document proposed usage:
   `MODEL_KIND=localop LOCAL_OPERATOR=learned_waveform GLOBAL_OPERATOR=learned_waveform`.
   These options do not exist until implementation is complete.

Old operator checkpoints must still rebuild and load exactly. A new waveform
model cannot strictly load a Fourier model as if its transforms were unchanged.
Record architecture/optimizer-group schema versions; changing bins, bank sharing,
or mode counts requires a fresh model or an explicit migration.

## 8. Acceptance checks and experiments

Before training:

* Exact-dual reconstruction of synthesized signals; arbitrary-input projection
  for M<N; coefficient round-trip A Phi=I; full-rank square-bank identity using
  a generic matrix fixture independent of the conservative waveform sampler.
* Regularized reconstruction matches the declared ridge formula, preserves DC,
  and behaves finitely on deliberately near-duplicate columns. Clearly separate
  it from exact inverse tests. Compare QR and Gram exact solves when conditioned.
* Float64 autograd checks through theta -> resampling -> dual -> mixing ->
  synthesis, including finite differences. Verify nonzero bank gradients and
  changes on the first optimizer step under a nondegenerate task/mix.
* Real-only parameters and intermediates; shape/dtype/device behavior on local
  windows and a 35x35x64 global example; odd grids and unsupported small grids.
* Alias-limit checks using known high-harmonic tables; zero columns and collapse
  detection; expected rank after filtering; boundary and phase-shift probes.
* Chunked versus unchunked outputs and bank gradients; identity patch
  overlap-add; checkpoint round-trip; reproducible optimizer resume; DDP gradient
  agreement; no stale graph after two optimizer steps; mixed-precision smoke.
* Model factory and metadata tests for all old registry entries and all three
  task entry points. Shared bottleneck bank has one parameter/optimizer entry
  and receives gradients from both uses.

Experiment order:

1. Small synthetic signals with translated fronts and mixed scales, including
   held-out shifts and widths. An aligned sawtooth alone favors this dictionary
   and is insufficient evidence of task generalization.
2. 2-D x_HI: fixed real basis, frozen random waveform, and learned random waveform;
   local-only, global-only, and both learned; ridge versus exact/orthonormal QR.
3. z_re with its existing mask/loss; then a 3-D feasibility run followed by the
   full scientific comparison if memory and numerical diagnostics pass.
4. Multiple seeds, resolution transfer, phase mixing, and optional independent
   per-mode tables. Learn physical periods only afterward: fractional periods
   break domain periodicity and require resampling/bandwidth rules with usable
   gradients rather than differentiating through a hard harmonic cutoff.

Compare real parameter counts, retained subspace dimension, peak memory and
step time; identical numeric `modes` values do not match complex Fourier budgets.
Use existing task losses/metrics, power spectra, neutral fraction, front/bubble
statistics where applicable, LOS boundaries and seam errors. Select settings
on validation simulations and reserve held-out simulations for final reporting.
Success means improved held-out task metrics with acceptable runtime and stable
transforms, not merely visually interesting learned tables or lower training loss.

## 9. Numerical design check performed for this plan

A small NumPy float64 check used Gaussian bin tables, the real resampler above,
phase-paired normalized columns, and synthesized in-span signals. It was not
a PyTorch/autograd test or a training experiment. Seed: 13; successive draws.

| N / M / B | Gram condition | Transpose-only relative error | Exact dual error | Ridge 1e-4 error |
|---|---:|---:|---:|---:|
| 16 / 6 / 15 | 7.27 | 0.545 | 2.6e-16 | 1.09e-4 |
| 32 / 12 / 31 | 3.77 | 0.400 | 2.8e-16 | 1.61e-4 |
| 35 / 16 / 31 | 60.03 | 1.080 | 8.8e-16 | 1.77e-4 |
| 64 / 16 / 31 | 7.02 | 0.641 | 3.9e-16 | 1.09e-4 |

DC error was below 7e-16 in these examples. These checks illustrate the need
for a dual and the size of ridge bias on these particular banks; they do not
establish conditioning or accuracy during learning. The active Python runtime
did not have PyTorch installed, so framework-level checks remain implementation
acceptance work.
