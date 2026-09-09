# Plan: the learned-atom ("squish") operator

Goal: replace the *fixed* function-space bases of the operator zoo
(`fourier`, `hadamard`, `wavelet`) with an **adaptive basis whose waveforms
are learned**. Each basis function is a periodic discrete function stored as
a tensor ("atom") with a *frequency* and an *amplitude*. Frequency is the
width of the interval the atom lives on: shrinking the interval squishes the
waveform (a sawtooth over a narrower interval has a steeper slope, i.e. a
higher proportionality constant). The atom's values are the parameters.

---

## 1. Formalization

An atom is a triple

```
atom_j = (phi_j, nu_j, a_j)
```

* `phi_j ∈ R^{L_j}` — the learned discrete table, periodic: `phi[· mod L_j]`.
  This is the "function completely discreticized as a tensor on the interval".
* `nu_j` — frequency in **cycles per axis** (equivalently interval width
  `p_j = S / nu_j` cells on an axis of size `S`; in physical units
  `W_j = p_j · dx` Mpc, ~1.43 Mpc/cell on the 200 Mpc / 140-cell grid).
* `a_j` — amplitude.

The atom placed on an axis of `S` cells is the basis row

```
A_j[i] = a_j · interp(phi_j, (i · nu_j · L_j / S) mod L_j),   i = 0..S-1
```

i.e. sample the table at `nu_j` evenly spaced passes over the axis, with
cyclic (wrap-around) interpolation. The step between consecutive grid
samples is `L_j · nu_j / S = L_j / p_j`:

* `L_j < p_j`  → **stretching** (table upsampled onto the grid; always safe),
* `L_j > p_j`  → **squishing** (table subsampled; aliasing risk, §5.1).

### Relation to the existing zoo

Every transform in the registry is "fixed waveform, learned mixing":

| operator | waveform | frequency analogue |
|---|---|---|
| `fourier` | sinusoid | mode index |
| `hadamard` | square wave (±1) | sequency |
| `wavelet` | Haar step | level |
| **`atom` (new)** | **learned table** | **interval width** |

The atom operator is the same machine with the waveform itself as a
parameter. Initialized at a sinusoid it *contains* the Fourier operator;
initialized at a square wave it contains the Walsh operator; initialized at a
sawtooth it spans something no truncated Fourier basis can.

### Why this is the right tool for this thesis

A sawtooth (or any waveform with a sharp front) has Fourier coefficients
~1/k for *all* k: representing one aligned sharp front with a truncated
Fourier basis costs many modes and still rings/blurs. **One atom represents
the whole aligned harmonic comb exactly, for O(L) parameters.** The target
fields are reionization bubbles: neutral regions bounded by sharp ionization
fronts — precisely the non-band-limited content that mode truncation smears
out (the sharpness/blur metrics `width_px`, `peak_grad`, `lowpass` in
`legacy/xhi2d/field_metrics.py` exist because of this). The pitch: *let the
network choose its own waveform, and it should choose something front-like
rather than sinusoidal.*

---

## 2. Design

### 2.1 Operator form (v1): basis-transform, registry-native

Clone the `WalshHadamardOperator` structure exactly — analysis, per-atom
channel mixing, synthesis:

```
coefficients = contract_per_axis(x, A, analysis=True)      # tensordot per axis
mixed        = einsum("bi xyz, io xyz -> bo xyz", coeff, W)  # W: (C_in, C_out, Kx, Ky, Kz)
out          = contract_per_axis(mixed, A, analysis=False)   # transpose of A
```

The only change vs. Walsh: the basis matrix `A` (per axis, shape
`(K, S)`) has **learned rows** built by resampling tables, instead of fixed
Walsh rows. Implementation can subclass `WalshHadamardOperator` and override
`_basis()`, the way `SirenWalshHadamardOperator` overrides `_mixing_weight()`.

Properties that make it drop-in:

* `uses_modes=True` — the existing per-slot mode counts become per-axis atom
  counts `K`; truncation semantics ("keep the smoothest band") becomes
  "keep the first K atoms of the frequency ladder".
* `required_size` = identity — works on *any* spatial size. Unlike Walsh
  (power-of-two) and Haar (divisible), the 35×35×64 bottleneck runs natively
  with no `pad_to_operator_size` detour.
* `rank_projected=True`, `windowed=True` defaults match the other transforms.

Separability: like Fourier/Walsh here, atoms are 1-D and applied per axis
(tables per axis — LOS and transverse axes deserve different waveforms).
Non-separable 2-D/3-D atoms (a radial "bubble atom") are a later extension,
§2.7.

### 2.2 Basis row construction (differentiable, cyclic)

`grid_sample` has no circular padding mode, so do the wrap manually. Cleanest
with an extended table so no modulo is needed in the gather:

```python
# table: (L,), extended: (L+1,) = cat([table, table[:1]])
pos  = (torch.arange(S, ...) * (nu * L / S)) % L      # (S,) sample positions
i0   = pos.floor().long()                              # in [0, L-1]
w    = (pos - i0).unsqueeze(-1)                        # (S, 1) interp weight
row  = (1 - w) * ext[i0] + w * ext[i0 + 1]             # cyclic linear interp
```

Gradients flow into the two gathered table entries per grid point (linear
interp). `interp="cubic"` (4-tap Catmull-Rom) is a flag for smoother
gradients. Rebuild `A` every forward — it is `(K, S)` gather-lERP, negligible
next to the `(C²·K·S)` einsum — so no `_basis_cache` and no DDP broadcast
concerns. Optionally use the block's `materialize_weights` hook
(`local_fno_3d.py:515`) to build the per-size basis once per forward instead
of once per overlap-add patch chunk, exactly as the SIREN operators do.

**DC atom:** `nu = 0` is the constant row; give it a single learnable scalar
per axis (its "table" has length 1).

### 2.3 Frequency ladders and how adaptive to make them

Three levels of adaptivity, in order of risk:

* **v1 (recommended first): fixed ladder.** `nu_j = j`, j = 0..K-1
  (`ATOM_LADDER=linear`) or `nu_j ∈ {0, 1, 2, 4, 8, ...}` (`log` — the
  wavelet-octave flavor). Guaranteed band coverage, no degeneracies, the
  mode-budget semantics of `uses_modes` carry over unchanged. Ablate
  linear vs. log.
* **v2: learnable frequencies.** `nu_j = softplus(theta_j)` (or
  `sigmoid`-scaled into `[1, S/2]`), trained in a low-LR param group.
  Gradients reach `theta` through `pos` — but they are piecewise constant
  under nearest interp and kinked (subgradient) under linear (§5.3). Use
  cubic interp or a soft frequency-mixture if optimizing `nu` gets ugly.
* **v3: conditioned frequencies.** A tiny MLP maps the 11 astro parameters
  (already model inputs) to per-atom frequencies/amplitudes. Physically the
  strongest version — bubble sizes in Mpc depend on the astro parameters, so
  the *basis itself* adapts to the regime. The SIREN operators are the
  in-repo precedent for weight-generating subnetworks. Do this last.

**Phase/translation:** a per-atom phase is just a fractional roll of the
table and is redundant with learning the table values — omit in v1. (Note a
phase-locked atom bank is *not* translation-equivariant, unlike Fourier; see
§5.6 and §2.7.)

**Integer vs. fractional `nu`:** integer `nu` makes the row periodic on the
axis (seamless wrap). Fractional/physical-width parametrization leaves a
seam jump at the axis boundary — acceptable for sawtooth-like atoms (they
have jumps anyway) but worth a flag.

### 2.4 Anti-aliasing budget (the one hard constraint)

If the table carries harmonics up to `B` (cycles per table length), then
after squishing to an on-grid period of `p` cells those harmonics land at
on-grid frequency `B/p` cycles per cell. Aliasing when

```
B / p > 1/2          (Nyquist in cycles per cell)
```

Enforcement options (pick one, in decreasing order of safety):

1. **Band-limited table parametrization:** store the table as
   `phi = IDFT(learned coeffs, keep ≤ B harmonics)`. Smooth by construction,
   C∞ gradients, alias-free whenever `B ≤ p_min/2`. You still "learn the
   values" — just in a basis that enforces the budget. Cheapest insurance.
2. **Native-resolution tables + stretch-only rule:** size `L_j ≈ p_j` at the
   *finest* deployment of that bank; every other deployment stretches
   (`L < p`), and stretching never aliases. Then a smoothness penalty
   `λ Σ_i (phi[i+1] − phi[i])²` is enough soft control.
3. **Raw table + smoothness penalty only:** maximum expressivity, needs the
   penalty tuned and the frequency ladder clamped so `p_j ≥ 2`.

Recipe 2 has a subtlety in this codebase: **each block owns its operator
instance and sees exactly one spatial size** (windowed slots see only their
window size; the global slot only the bottleneck/field), so within a normal
run there is no intra-run squishing — each bank is sized to its own
deployment. Squishing becomes active when:

* **`ATOM_SHARED=1`:** one atom bank shared across all blocks/slots of a
  model (blocks keep only their mixing matrices). The bank is then squished
  onto 16-cell windows, the 64-cell bottleneck, and the 140-cell field in the
  same forward pass. Parameter-efficient, and it makes the thesis story
  literal: *one learned waveform dictionary, deployed at every scale*. The
  aliasing budget must then hold at the **shortest** deployment (the window).
* **Resolution transfer:** evaluating a checkpoint at a different grid (the
  `test_split_resolution.py` concern). Tables resample to any `S` by
  construction; validity requires the budget at the eval resolution.

### 2.5 Normalization, amplitude, regularizers

* **Per-atom L2 normalization of rows** (differentiable, in `forward`):
  `A_j / ||A_j||`. Kills the `(a_j, phi_j)` scale degeneracy and keeps the
  Gram matrix sane. Amplitude then lives in `a_j = softplus(θ_j)` (explicit,
  logged, init 1) and/or the mixing tensor — but not both places at once.
* **Per-atom mixing** `(C_in, C_out, K_per_axis…)` — identical shape and
  init (`1/sqrt(C)`) to the Walsh operator, so parameter budgets compare
  fairly against `fourier`/`hadamard` at the same mode count. Table
  parameters are negligible next to it (`Σ_j L_j ≈ S · ln K` entries).
* **Diversity/orthogonality:** soft penalty on the off-diagonal Gram of the
  stacked rows (`λ_ortho · ||A Aᵀ − I||²_F` on normalized rows) to prevent
  atom collapse (two atoms learning the same waveform). Keep it soft — hard
  orthonormalization would fight the squish structure and re-derive Fourier.
* **Smoothness:** `λ_smooth · Σ ||D phi_j||²` (or the band-limited
  parametrization of §2.4.1).

### 2.6 Amplitude — where it lives

Either (a) explicit per-atom scalar `a_j` (nice for the thesis: amplitude
trajectories per frequency are exactly the "spectral-weight history" style
diagnostic, and `util/`'s mode-weight machinery can be extended to log it),
or (b) folded entirely into the mixing weights. With normalized rows, (a) is
identifiable — recommend (a) + normalized rows.

### 2.7 Follow-up form (B): the equivariant kernel variant

A phase-locked atom bank breaks Fourier's translation equivariance (fine on
the codebase's precedent — `hadamard` already does — but worth an ablation on
the periodic transverse axes). The equivariant alternative: use each squished
atom as a **circular-convolution kernel** of width `p_j` (the atom tiled over
its interval width is the kernel; FFT-conv it), i.e. the layer becomes a
dilated multi-kernel convolution whose kernels are learned squishable
waveforms. That is an FNO spectral layer whose kernel spectrum is *factored
into atoms* instead of free per-mode. Structurally closer to what makes FNO
work; more machinery; do it as variant `atom_conv` after the basis form
works.

---

## 3. Integration steps (ordered)

1. **`atom_operator.py`** (new, top level, style of `wavelet_operator.py`):
   `LearnedAtomOperator(channels, ndim, modes, *, table_factor, interp,
   ladder, learn_freq, smoothness, ortho, shared)` with
   `forward(x) -> (B, C, *spatial)`, per-axis banks
   (`nn.ParameterList` of `(K, L_j)` tables + per-axis DC scalars), mixing
   tensor, cyclic-interp basis builder (§2.2), normalization (§2.5), and an
   `atom_weight_tensors()` diagnostics accessor mirroring
   `wavelet_weight_tensors()` / `walsh_weight_tensors()`.
2. **`operators.py`:** import-free `_build_atom` (lazy, like the others),
   registry entry (§3.1 below), `_validate_atom` (atom count `K ≤ S/2 + 1`
   per axis; `interp`/`ladder` legal; if `shared`, budget holds at the
   smallest slot size), aliases `squish`, `atoms`, `lbo`, `learned_basis`.
   Update the module docstring table.
3. **`modeling.py`:** extend `operator_env_settings()` with
   `ATOM_TABLE_FACTOR`, `ATOM_INTERP`, `ATOM_LADDER`, `ATOM_LEARN_FREQ`,
   `ATOM_SMOOTHNESS`, `ATOM_ORTHO`, `ATOM_SHARED`; add the `atom` branch to
   `slot_hyperparameters()`; record the settings in run metadata like the
   other operators.
4. **`tests/test_atom_operator.py`** (mirror `test_local_wavelet_operator.py`
   + `test_operator_registry.py`):
   * shape preservation 2-D/3-D, any spatial size (incl. 35 — no padding);
   * gradient flows into every table entry (catches dead-entry orbits, §5.2);
   * roll-invariance: cyclically rolling the input by exactly `p_j` cells
     leaves coefficient `j` unchanged;
   * `K = S` with orthonormalization ⇒ transform is invertible (round-trip);
   * aliasing guard: for each atom, effective on-grid bandwidth ≤ 1/2
     cycle/cell at its smallest deployment;
   * registry: `build_operator("atom", …)` validates hyperparameters,
     `required_size` is identity, aliases resolve;
   * shared-bank mode: same tables, different sizes, stretch-only.
5. **Diagnostics:** per-epoch plot of the learned atoms and **their FFTs**
   (the "effective spectrum" — shows e.g. a sawtooth atom's 1/k comb), plus
   `a_j` trajectories. Extend the `util/` spectral-weight history or add a
   small `viz/` helper; these are thesis figures.
6. **Experiment ladder:**
   a. Synthetic 1-D: fit a sawtooth target — atom basis vs. Fourier at
      matched budget (the compressive claim, isolated).
   b. 2-D `fno_xhi2d.py`, `LOCAL_OPERATOR=atom GLOBAL_OPERATOR=atom` vs
      `fourier`/`hadamard`/`wavelet` at matched mode counts — watch the
      sharpness metrics, not just L2.
   c. 3-D `fno_21cm_3d.py` DDP pairing (batch-size-1 safe: GroupNorm, no
      registered buffers — plain-dict/no cache).
   d. Ablations: ladder linear vs. log; `learn_freq` on/off; `ortho` and
      `smoothness` on/off; `shared` on/off; init = sinusoid vs. sawtooth vs.
      random-smooth (the "start at Fourier, learn your way to a front"
      narrative).
   e. Resolution transfer eval via the existing split-resolution machinery.

### 3.1 Registry entry sketch

```python
"atom": OperatorSpec(
    name="atom",
    build=_build_atom,
    defaults={"table_factor": 2.0, "interp": "linear", "ladder": "linear",
              "learn_freq": False, "smoothness": 0.0, "ortho": 0.0,
              "shared": False},
    uses_modes=True,                                   # modes -> atom counts
    rank_projected=True,                               # ablate False later
    windowed=True,
    required_size=lambda size, hp: size,               # any size, no padding
    validate=_validate_atom,
),
```

Env usage: `MODEL_KIND=localop LOCAL_OPERATOR=atom GLOBAL_OPERATOR=atom
python fno_xhi2d.py`, atom counts from the existing mode machinery.

---

## 4. Suggested defaults

| knob | default | note |
|---|---|---|
| ladder | `linear` (`nu_j = j`) | Fourier-comparable; `log` for wavelet flavor |
| table sizing | `L_j = clamp(round(table_factor · S/nu_j), 4, 256)` | native at own deployment; stretch elsewhere |
| interp | `linear` | `cubic` if learning `nu` |
| learn_freq | `False` | enable only with param groups + clamps |
| rows | L2-normalized in forward | amplitude in `a_j` / mixing |
| init | random smooth (low-truncated Fourier, random phase), DC scalar 0 | plus seeded variants for ablations |
| mixing init | `1/sqrt(C)` randn | same as Walsh |
| warmup | freeze tables ~1k steps | random-basis regime first, then unfreeze |
| `ortho`, `smoothness` | 0.0, small (1e-3) if raw tables | 0 if band-limited parametrization |

---

## 5. Error analysis

### 5.1 Numerical / sampling errors

* **Aliasing when squishing (the fundamental one).** Subsampling a table
  with harmonic content `B` onto a period `p < 2B` folds high table
  frequencies onto low grid frequencies — a jagged, wrong waveform, and it
  happens *silently* (loss just gets worse). Guard: the `B/p ≤ 1/2` budget at
  the atom's smallest deployment (§2.4); a unit test that asserts it; the
  band-limited table parametrization makes it impossible by construction.
  A sawtooth table is the worst case (infinite harmonic tail) — band-limit
  or smoothness-penalize it.
* **Dead table entries (gradient orbit coverage).** The grid only reads
  table indices `{floor(i · L/p) mod L}`: if the step `L/p > 1`, entries
  between orbit points get zero gradient forever. With stretch-only sizing
  (`L ≤ p`) coverage is dense. With `ATOM_SHARED`, coarse deployments
  stretch (dense) but the finest deployment subsamples — make the shared
  bank's `L_j` ≤ its shortest deployment period, or average a few sub-phase
  offsets per forward. Test 4 (gradient-into-every-entry) catches this.
* **Interpolation gradient quality.** Nearest interp ⇒ zero gradient to
  unsampled entries; linear ⇒ 2-tap support and kinked (non-smooth)
  gradients — optimizers cope, but frequency learning through `pos` is
  piecewise-constant/kinked (§2.3 v2). Cubic or the Fourier-parametrized
  table (C∞) fixes it at minor cost.
* **Seam/wrap errors.** Non-integer `nu` (or physical-width atoms) produces
  a jump where the axis wraps; integer ladders are seamless. The LOS axis is
  not periodic — respect the existing per-axis `pad_modes` convention
  (circular transverse, replicate LOS) in any windowing/padding interaction;
  the identity `required_size` avoids most of it natively.
* **Gram conditioning.** Fourier/Walsh/Haar are orthonormal (condition
  number 1). Learned rows generally are not: `A Aᵀ` can be near-singular,
  and since synthesis uses the *transpose* (a frame, not the inverse),
  badly-conditioned banks distort and can destabilize training. Mitigate:
  row normalization (default), soft `ortho` penalty, monitor
  `cond(A Aᵀ)` per epoch, small mixing init, warmup freeze.

### 5.2 Optimization degeneracies

* **Scale degeneracy** `(a_j, phi_j) ↔ (c·a_j, phi_j/c)` — resolved by
  normalized rows + amplitude in exactly one place (§2.6). Double-counting
  amplitude (normalized rows *and* free `a_j` *and* per-atom mixing scale)
  creates flat directions; keep it single.
* **Frequency flat directions.** For a smooth table, `nu_j ± ε` spans
  almost the same subspace once mixing adapts ⇒ vanishing gradients toward
  the "right" frequency; the ladder init determines the basin. This is an
  argument for v1 fixed ladders and for reporting frequency drift only as a
  diagnostic, not relying on it.
* **Atom collapse.** Two atoms converging to the same waveform wastes
  budget (rank collapse of the bank). Watch max off-diagonal Gram
  coherence; `ortho` penalty or diversity-encouraging init if it fires.
* **Early instability.** Co-adapting basis + mixing makes output norms
  swing early. Warmup freeze (tables frozen, mixing learns — exactly the
  random-features regime, and statistically reasonable), lower LR on
  tables/`nu` via param groups, then unfreeze.

### 5.3 Integration pitfalls (this codebase specifically)

* **Window Nyquist.** Windowed slots see only their window (16–64 cells):
  atom periods must satisfy `2 ≤ p_j ≤ window size`; per-slot atom counts
  come from the existing mode machinery — keep `K_slot ≤ S_slot/2 + 1`.
  Low-frequency atoms simply cannot be estimated on patches; that's the same
  truncation the Fourier operator already lives with. Also note Hann
  windowing tapers patches — low-`nu` atoms inside a tapered patch are
  modulated (existing behavior for Fourier too; precedent, not a new bug).
* **Rank-projection double bottleneck.** The block sandwiches
  `rank_projected` operators in 1×1 projections; the atom bank already has
  an explicit rank structure (`K` per axis). If capacity chokes, ablate
  `rank_projected=False` (the `cnn` precedent).
* **DDP (3-D pipeline, batch size 1).** Follow the Walsh pattern: no
  registered buffers, no basis cache (rebuild per forward; or key a cache on
  the parameter `_version`), tables as plain `nn.Parameter`s /
  `ParameterList` (DDP flattens them fine). GroupNorm unaffected.
* **Checkpoints / optimizer state.** New `state_dict` keys are additive, so
  old runs load fine; but note the codebase's own warning that optimizer
  state is keyed by parameter *position* — register submodules in a stable
  order (projections, operator, …), and remember that changing `K`, `L_j`,
  or `ladder` breaks resuming an atom run (shapes change).
* **`uses_modes` semantics.** Mode counts become atom counts; validation
  must check `K ≤ S/2 + 1` per axis (the analogue of `_validate_fourier`'s
  FFT-limit check), and `test_operator_registry.py` style tests should
  assert the error messages.

### 5.4 Evaluation / thesis-hygiene errors

* **Unfair baselines.** Compare at matched *mixing-parameter* budget (same
  mode/atom counts) — table params are noise next to `(C²·ΠK)`. Report
  FLOPs: the operator is Walsh-class (dense `(K, S)` contraction per axis +
  einsum), fine at 16–64 windows and the 64³ bottleneck.
* **Equivariance loss.** The basis form is phase-locked: translating the
  input changes coefficients in a way Fourier's do not change. The
  transverse axes are periodic with randomly-placed bubbles, so equivariance
  may matter more than the `hadamard` runs suggested. If 2-D results lag
  Fourier inexplicably, this is suspect #1; form (B) (§2.7) is the fix.
* **Resolution-transfer claims.** "Discretization-invariant operator" only
  holds as well as the resampling approximates the underlying continuous
  atom; linear interp degrades at coarse eval grids, and the aliasing budget
  must be re-checked at eval resolutions. Claim it carefully (interpolation
  order included).
* **Diagnostics mismatch.** The `util/` spectral-weight history is keyed on
  FFT modes; atom "modes" are not frequencies. Map atoms → their FFT (the
  effective spectrum) before reusing those utilities, or the thesis plots
  will silently mislabel.
* **Expressivity risk.** `K` atoms span a `K`-dimensional subspace per
  axis — *the same rank budget* as truncated Fourier, just placed
  differently. If optimization lands badly (bad init, degeneracies above),
  the atom operator is strictly worse than Fourier. Keep the Fourier
  baseline in every table; the synthetic sawtooth experiment (6a) is the
  honest isolation of the win condition.

---

## 6. Literature anchors (for the thesis text)

* **Lifting scheme / learned wavelets** — second-generation wavelets whose
  predict/update filters are learned; closest classical cousin (they keep
  filter banks, though — the squish/resample parametrization of scale is the
  differentiator here).
* **Dictionary learning / K-SVD** — an atom bank with analysis-synthesis is
  a single-layer (unsparsified) dictionary transform; "the dictionary is the
  layer".
* **Learned tight frames** — the `ortho`-penalized, row-normalized variant.
* **Wavelet Neural Operator (Gupta et al. 2021)** — fixed wavelet bases in
  an operator layer; the `wavelet` entry here; the atom operator is its
  adaptive-basis counterpart.
* **SIREN-as-weight-generator (in-repo)** — `siren_fourier`/`siren_hadamard`
  already generate *mixing* weights from coordinates; the atom operator
  instead generates the *basis*, which is the stronger claim.
