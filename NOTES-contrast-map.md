# The (theta, tau) contrast map

A two-parameter monotonic reshaping of a `[0, 1]`-valued prediction, intended
to recover the edge sharpness that pointwise losses systematically destroy.

Status as of 2026-07-27: **closed, negative, and explained.** The map works as
a deblurrer (-9.28% on a purely blurred field) but buys 0.00% on the model at
matched sharpness -- because the model's edges are not blurred, they are hedged
in position, and no pointwise map moves an edge. Sections 5-6 are the evidence;
section 7 is the summary.

---

## 1. Motivation: why pointwise losses blur

If the model is uncertain whether an edge sits at `x0` or `x0 +- delta`, the
L2-optimal prediction is the expectation over that uncertainty -- a ramp of
width `~2*delta`. A sharp edge in the wrong place costs roughly twice the area
of a soft ramp that hedges, so **L2 actively rewards blurring**.

H1 has the same pathology one derivative up: the L2-optimal *gradient* field is
`E[grad]`, which turns a tall narrow ridge into a low wide bump. This explains a
long-standing puzzle in the 3-D runs -- fronts came out 12-14 Mpc against a
3.6 Mpc truth *despite* the loss being 99.4% H1 by contribution.

Measured on the 2-D x_HI task (256 held-out slices, `whno_glob`), truth
transitions are ~0.9 px wide and predictions ~1.7-3.4 px, i.e. ~2-4x too wide.
The local branch (6 modes in a 16 px window) can represent ~1.3-2.7 px
transitions, so the models leave representable sharpness unused: the binding
constraint is the objective, not the bandwidth.

## 2. The map

```
g(x; theta, tau) = [tanh((x - tau)/theta) - tanh(-tau/theta)]
                   / [tanh((1 - tau)/theta) - tanh(-tau/theta)]
```

* `tau` -- the **threshold**, the value the transition is centred on.
* `theta` -- the **smoothness**. `theta -> 0` is a hard step at `tau`;
  `theta -> inf` recovers the identity. Slope at `tau` is `~1/(2*theta)`, so
  `theta` reads directly as a transition width.

**The affine renormalisation is load-bearing.** The originally proposed form,
`theta*x + 2*(1-theta)*tanh(x/theta)`, only behaves at `theta = 1` (where it is
the identity): `tanh(x/theta)` is centred at `x = 0`, not at `1/2`, so as
`theta -> 0` it becomes a step at 0 of height 2. Measured values:

| theta | f(0) | f(0.5) | f(1) |
| --- | --- | --- | --- |
| 1.00 | 0.0000 | 0.5000 | 1.0000 |
| 0.50 | 0.0000 | 1.0116 | 1.4640 |
| 0.05 | 0.0000 | 1.9250 | 1.9500 |

Subtracting the value at 0 and dividing by the span pins `g(0) = 0` and
`g(1) = 1` for **every** `(theta, tau)`, which is what allows `tau` to move off
`1/2` without the prediction leaving `[0, 1]` or the mean level drifting.
Verified: endpoints exact and monotonicity holds across
`theta in {0.05, 0.2, 0.5, 2}` x `tau in {0.3, 0.5, 0.7}`; `theta = THETA_MAX`
reproduces the identity to `6e-4`.

Implementation: `contrast.py` (`apply_contrast`, `ContrastHead`,
`ContrastOutput`, `ContrastComposed`).

## 3. Post-hoc results (closed)

All on the trained `whno_glob` (`xhi2d_whno_glob_lr3e4`, the 2-D x_HI leader:
val L2 0.1060 / test L2 0.1107 at epoch 25). Note the tables below report
**pixel RMSE** on held-out slices, which is a different quantity from the
trainer's `val_l2`, and the slice sets differ between subsections -- compare
within a table, never across.

### 3.1 A global (theta, tau) is worthless

Sweeping a single `theta` shared by all slices, RMSE degrades monotonically;
no value beats the identity.

| theta | RMSE | vs identity | blur_frac | width_px |
| --- | --- | --- | --- | --- |
| TRUTH | -- | -- | 0.1073 | 1.505 |
| identity | **0.14866** | -- | 0.2125 | 3.433 |
| 0.40 | 0.15213 | +2.3% | 0.1385 | 2.260 |
| 0.30 | 0.15595 | +4.9% | 0.1112 | 1.822 |
| 0.20 | 0.16449 | +10.6% | 0.0755 | 1.242 |

This *had* to come out this way: the model already ends in a sigmoid, so it can
represent any monotonic remap of its own logits. If some `g_theta` improved L2,
training would have found it. **Post-hoc pointwise sharpening can never improve
L2 for a converged model** -- a general result, not specific to this
architecture.

The map is still usable as an explicit tradeoff dial: `theta ~ 0.30` lands the
sharpness statistics on truth (blur 0.1112 vs 0.1073) for ~5% RMSE. Legitimate
if the deliverable is summary statistics rather than per-pixel accuracy, but it
must be reported as post-processing, not as a better model.

### 3.2 Per-sample (theta, tau) has real headroom -- and needs both

Fitting parameters per slice, **cross-validated across pixels** (fit on a random
half of each slice, score on the other half, so this is not selection bias):

| variant | RMSE | gain |
| --- | --- | --- |
| identity | 0.15328 | -- |
| per-slice `theta` only (`tau` = 0.5) | 0.15267 | -0.40% |
| per-slice `tau` only (`theta` = global best) | 0.15277 | -0.33% |
| **per-slice both** | **0.14382** | **-6.17%** |

Strongly **superadditive**: ~8x the sum of the parts. The beneficial operation
is *a sharp map placed at a per-slice-specific threshold* -- with `tau` pinned
at 1/2 you can only steepen in the wrong place, and with `theta` near-identity
a near-linear map barely notices its pivot. Fitted values spread widely
(`theta` 0.28-3.21, `tau` 0.10-0.90 at 10-90%), confirming genuine
per-sample heterogeneity.

Held-out (-5.33%) matched or beat in-sample (-5.03%) on a separate 600-slice
run, so selection bias is negligible.

### 3.3 ...but it cannot be realized at inference

R^2 for predicting the fitted parameters, held-out 30%, linear and small MLP:

| features | theta | tau |
| --- | --- | --- |
| pooled prediction stats (6) | 0.26 / 0.28 | 0.01 / -0.06 |
| + value histogram (16) | 0.27 / 0.10 | -0.01 / -0.31 |
| + conditioning (z, 11 params) | 0.35 / -0.58 | **-0.01 / -1.10** |

`theta` is moderately predictable; **`tau` is not predictable from anything
available at inference** -- not the prediction, not its histogram, not even
redshift and the cosmological parameters. Since the gain needs both jointly,
the 6.2% is an **oracle ceiling, not an achievable target**.

A trained `ContrastHead` confirmed this empirically: test RMSE 0.13782 vs
0.13774 identity (+0.05%, i.e. slightly *worse*), with training loss *rising*
0.0171 -> 0.0203 and `tau` pinned at its upper bound. Because the gain is
superadditive, a head with poor `tau` accuracy does not get partial credit --
it applies sharp maps at wrong thresholds, which is the actively harmful
regime.

**Conclusion for the post-hoc route: closed.** Report 6.2% as a measured upper
bound on per-sample output correction, and note the limitation is *information*,
not the functional form.

## 4. End-to-end (done -- negative)

The rationale, stated before the runs, was that this is distinct from the above
and not refuted by it:

* With a **global** learned `(theta, tau)` there is no prediction problem --
  two scalars trained by gradient descent.
* The network can **co-adapt**, shaping its pre-map output to suit the map
  rather than having a finished prediction reshaped afterwards.
* The spectral operators synthesise a **band-limited** field; a pointwise
  nonlinearity applied *after* synthesis creates high-frequency content the
  operator cannot represent. This is genuine added expressiveness relative to
  the linear operator plus a fixed mild sigmoid.

Enabled with `CONTRAST_MODE=off|global|head` in `fno_21cm.py`; initialised at
the identity so a run starts from exactly the baseline. Verified that gradients
reach the contrast parameters *and* all 84 base tensors.

### 4.1 Results

All `whno_glob`, 30 epochs, pure L2, against the **matched** no-contrast pure-L2
baseline `xhi2d_whno_glob_lr3e4`. (Metrics are logged every 5 epochs, so all
figures are epoch 25.)

| job | run | contrast | val L2 | test L2 | vs baseline |
| --- | --- | --- | --- | --- | --- |
| -- | `whno_glob_lr3e4` | none | **0.1060** | **0.1107** | -- |
| 4358634 | `whno_glob_ctrglobal` | global | 0.1062 | 0.1112 | +0.2% / +0.5% |
| 4358635 | `whno_glob_ctrhead` | head | 0.1055 | 0.1117 | -0.5% / +0.9% |
| 4358636 | `whno_glob_ctrglobal_hk` | global + highK | NaN at ~ep 6-10 | -- | crashed |

Val and test disagree in sign for both variants and the spread is a few tenths
of a percent. **No effect.**

### 4.2 The network chose not to use the map

The decisive number is not the RMSE, it is the learned parameters:

| run | learned theta | learned tau | max abs g(x) - x |
| --- | --- | --- | --- |
| initialisation | 4.750 | 0.500 | 0.0007 |
| `ctrglobal` | 4.431 | 0.5044 | **0.0009** |
| `ctrhead` (see 4.3) | 4.441 | 0.5051 | **0.0009** |
| *(for scale)* sharpness-matched | 0.30 | 0.5 | 0.117 |

`theta` was free to move anywhere in `[0.02, 5.0]` and the gradient was verified
to reach it. It moved from 4.75 to 4.43 -- i.e. **nowhere**. The learned map
differs from the identity by at most 9e-4, ~130x less than the setting that
matches truth sharpness. Given a differentiable sharpening dial and 30 epochs,
gradient descent left it at zero.

This is the prediction in the pre-registered version of this note coming true,
and it settles the band-limited argument in section 4's rationale: **that
argument does not hold in practice.** Whatever extra high-k content the
post-synthesis nonlinearity could create, L2 does not want it -- consistent with
section 1 (L2 rewards hedging) and section 3.1 (the model's own sigmoid already
spans these remaps). The map is not a way around the objective; the objective
was always the thing selecting against sharpness.

### 4.3 The `head` run did not test what it was meant to

Caveat, found on inspection of the trained weights: the head **collapsed to a
constant**. Its input-layer weights decayed to ~5e-28 (from ~0.4 at init), so
`net.0` outputs zero regardless of input and the whole MLP degenerates to a bias
chain. Probing it with random, binarised, all-0.1 and all-0.9 fields returns
identical `(theta, tau) = (4.441, 0.5051)` in every case -- the same point the
*global* run reached independently, to three decimals.

The cause is the identity initialisation: `nn.init.zeros_(self.net[-1].weight)`
is the standard trick for starting a module at the identity, but here it zeros
the **only** backward path to the earlier layers. Layer 0's gradient scales as
`W2 * W4`, layer 2's as `W4`, so with `W4` starting at zero the early layers get
essentially no gradient while weight decay keeps acting -- a collapse cascade,
visible in the final magnitudes (`W0` 5e-28 < `W2` 1e-27 << `W4` 5e-2).

So the head run is **not** independent evidence about input-dependent contrast;
it is a second, slower measurement of the global case. It is consistent with
section 3.3 but does not confirm it. Rerunning would need a nonzero (small)
last-layer init, or no weight decay on the head. Given 4.2 -- the global map,
which had no such handicap, also went nowhere -- this is not worth the GPU time
unless the whole approach is revisited under a different loss.

### 4.4 The highK combination is unstable

`ctrglobal_hk` (L2 + 0.12*highK, warmup 5) was healthy at epoch 5
(val_l2 0.1219, val_highk 0.0825 -- the best highK value of any run) and
`avg_loss=nan` by epoch 10; it then died in eval with a CUDA device-side assert
inside `HighKPowerRatio._binned_power`. The saved epoch-9 checkpoint is entirely
finite, so the blowup is after it.

Not isolated. Note that highK alone and contrast alone each trained to epoch 29
without incident, so it is the combination. Plausible mechanism: highK's
`log(P_pred/P_true)` gradient grows without bound as predicted band power falls,
and `d g/d theta ~ 1/theta^2`, so the two can drive each other. The reported
assert location is a red herring -- CUDA errors surface at the next
synchronisation point, and `index[keep]` indexes a validated cached tensor.

## 5. Deriving the parameters from the truth

`tests/probe_contrast_from_truth.py`, 512 held-out test slices. Two questions
the blind fit could not separate.

### 5.1 The physical threshold exists, is predictable, and is worthless

Instead of fitting `(theta, tau)` by minimising RMSE, read them off the truth's
edge geometry: `tau_geom` = median prediction value *at the true boundary*
(where truth crosses 1/2), `theta_geom` = the theta matching the prediction's
transition width to the truth's.

| variant | RMSE | vs identity |
| --- | --- | --- |
| identity | 0.15263 | -- |
| best global (theta=5.00 = ceiling, tau=0.40) | 0.15263 | +0.00% |
| per-slice **geometric** | 0.18853 | **+23.52%** |
| per-slice **blind** fit (RMSE oracle) | 0.14511 | -4.93% |
| blind theta + geometric tau | 0.15390 | +0.83% |
| geometric theta + blind tau | 0.18278 | +19.75% |

`tau_geom` is exactly what section 3.3 said was missing -- it **is** predictable
from what the model can see: corr(`tau_geom`, mean prediction) = **+0.475**,
corr with mean true x_HI = +0.427. But it yields no gain, and the split shows
why: geometric `tau` is nearly harmless (+0.83%), geometric `theta` is what
destroys it (+19.75%).

Decisively, corr(`tau_geom`, `tau_blind`) = only **+0.309**. The `tau` carrying
the -4.93% is *not* the edge-location threshold. `tau_blind` rails to the grid
bounds (10-90%: 0.100-0.900) and correlates with mean prediction at +0.136;
`tau_geom` is tight and sensible (0.283-0.569).

**This closes section 3.3 rather than reopening it.** The useful `tau` is
unpredictable *because it is not the physical one* -- it is a per-slice
bias/calibration nuisance, which a converged model leaves unpredictable almost
by definition: any bias inferable from the input would already have been
removed. The physical threshold is predictable and useless; the useful one is
unpredictable and unphysical.

### 5.2 Projection: the map recovers little even of pure blur

Band-limit the *truth* (Gaussian low-pass, clamped to [0,1] -- ideal cutoffs
ring and the overshoot would contaminate the width statistic) and try to
sharpen it back. Only blur is present, so this is the map's best possible case.

| k_cut | blurred | +global | +per-slice | recovered | width px |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.16139 | 0.15598 | 0.15052 | 6.7% | 14.53 |
| 0.12 | 0.10943 | 0.10159 | 0.09695 | 11.4% | 5.43 |
| **0.20** | **0.07734** | 0.06760 | 0.06232 | **19.4%** | **3.40** |
| *real model* | 0.15263 | 0.15263 | 0.14511 | **4.9%** | **3.21** |
| *truth* | -- | -- | -- | -- | 1.47 |

Two results, both read at matched transition width (3.40 vs 3.21):

* **A pointwise map is a weak deconvolver even when deconvolution is exactly
  right** -- 19.4% in the ideal case. Inherent: blurring moves *where* the
  half-crossing sits, especially on curved boundaries and small bubbles, and no
  monotone remap of values restores position. Sharpening locks in the displaced
  edge.
* **The model's error is mostly not blur.** At equal sharpness, pure
  band-limiting costs 0.077 RMSE and the model costs 0.153. Treating the two
  sources as independent, the non-blur component is
  `sqrt(0.153^2 - 0.077^2) ~ 0.132` -- roughly **86% of the model's error is
  not blur**. (Quadrature assumes independence, which is not exact; read it as
  an estimate.)

## 6. Theta alone, measured against the truth

`tests/probe_contrast_theta_on_truth.py`, same 512 slices, `tau` fixed at 1/2.
The control missing from every earlier sweep: applying the map to a *perfect*
field can only hurt, and that damage `D(theta) = RMSE(g(y), y)` is the price
paid regardless of what the model does.

| theta | D(th) damage to truth | width g(y) | M(th) model | dM | B(th) blurred truth | dB |
| --- | --- | --- | --- | --- | --- | --- |
| 5.000 | 0.00021 | 1.46 | 0.15263 | +0.00000 | 0.07724 | -0.00010 |
| 0.974 | 0.00525 | 1.32 | 0.15286 | +0.00023 | 0.07498 | -0.00236 |
| 0.527 | 0.01550 | 1.05 | 0.15428 | +0.00165 | 0.07163 | -0.00571 |
| **0.350** | 0.02830 | 0.78 | 0.15793 | +0.00530 | **0.07017** | **-0.00717** |
| 0.190 | 0.05266 | 0.40 | 0.17058 | +0.01795 | 0.07698 | -0.00036 |
| 0.020 | 0.08785 | 0.03 | 0.20085 | +0.04822 | 0.10931 | +0.03197 |

Three things:

**The map is nearly free to apply.** At `theta = 1` it costs a perfect field
0.0052 RMSE; even at the sharpness-matched `theta = 0.35` only 0.028. The truth
is already sharp (width 1.47, 11% of pixels in the transition band), so `g`
moves almost nothing -- `g(0) = 0` and `g(1) = 1` exactly. Cost is not why the
map fails.

**The positive control works.** On band-limited truth -- where blur is the only
error -- `theta` alone, with no `tau` at all, improves RMSE
0.07734 -> 0.07017, **-9.28%** at `theta = 0.35`. The map is a functioning
deblurrer.

**On the model it gives +0.00% at every theta.** Best `theta` = 5.000, the
identity end of the grid; monotonically worse below it. Same operation, same
grid, near-identical starting sharpness (3.40 blurred vs 3.21 model) -- it fixes
one and not the other.

That is the cleanest single statement in this note: **the model's sharpness
deficit is not the model's problem.** A field that is blurry *because it was
blurred* is repaired by sharpening; the model's field is equally blurry and is
not repaired at all, so its softness is not blur but hedged edge *positions*.

Two supporting numbers. Over `theta < 1`, corr(`M - ident`, `D`) = **+0.982**
and mean `(M - ident)/D` = 0.346 -- the model's degradation tracks pure
distortion almost perfectly, i.e. the map does nothing model-specific.  And
subtracting the distortion in quadrature (`sqrt(M^2 - D^2)`, the "D-adj" column
in the probe output) still leaves every value **above** identity, so there is no
sharpening benefit hidden underneath the distortion cost.

Incidental but physical: driving `theta -> 0.02` hard-binarises the truth and
costs 0.0879 RMSE. The partial ionisation in the transition band is real
information worth ~58% of the model's entire error -- binarising is not free.

### 6.1 Theta resolved against x_HI -- where the map does work

The pooled sweeps above hide a real effect. `theta` was expected to depend on
ionisation state: early on the field is a few small bubbles in a neutral sea,
late on a few neutral islands in an ionised sea, and those need different
sharpening. Resolving it required more statistics than the training cache
holds -- its sampler down-weights the tails ~50x, leaving only 379 of 3960 test
slices below x_HI = 0.36 -- so `dataset/build_xhi_band.py` builds a cache that
saturates the band from **val + test cones only**: 6292 slices, 566 cones,
~700-1100 per 0.05 bin against 34-68 before.

`theta` fitted per bin on one set of cones, scored on disjoint cones, CI from a
cone-level bootstrap. `edge_t` is truth edge density -- the control that says
whether a bin contains any boundaries to sharpen.

| x_HI | n | nB | edge_t | w_p/w_t | th_pred | identity | +theta | gain | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *0.000-0.005* | *497* | *247* | *0.0001* | *10.65* | *0.161* | *0.06017* | *0.05247* | *-12.79%* | *[-26.4,-4.7]* |
| **0.005-0.010** | 60 | 24 | 0.0038 | 1.72 | 0.289 | 0.13323 | 0.12103 | **-9.16%** | [-15.7,-4.4] |
| **0.010-0.020** | 117 | 62 | 0.0076 | 1.73 | 0.311 | 0.13361 | 0.12321 | **-7.78%** | [-11.6,-4.7] |
| **0.020-0.030** | 133 | 69 | 0.0108 | 2.02 | 0.482 | 0.18623 | 0.18246 | **-2.02%** | [-3.3,-0.9] |
| **0.030-0.050** | 263 | 122 | 0.0162 | 2.14 | 0.482 | 0.17660 | 0.17293 | **-2.08%** | [-3.3,-1.1] |
| 0.050-0.100 | 785 | 407 | 0.0279 | 3.01 | 0.804 | 0.21602 | 0.21565 | -0.17% | [-0.32,-0.02] |
| 0.100-0.200 | 1612 | 807 | 0.0501 | 3.54 | 5.000 | 0.25824 | 0.25825 | +0.00% | [+0.00,+0.01] |
| 0.200-0.360 | 2189 | 1076 | 0.0856 | 3.70 | 5.000 | 0.28232 | 0.28235 | +0.01% | [+0.01,+0.01] |

**There is a genuine, realizable gain at x_HI ~ 0.005-0.05**: 2-9% held out,
every CI clear of zero. It is *not* the degenerate artefact -- those bins carry
edge densities 40-160x the near-empty bin's and width ratios of 1.7-2.1 rather
than 10.65.

The first row stays italicised and excluded from every headline. At
`edge_t = 0.0001` the truth has essentially no boundaries; the model emits faint
spurious structure and a small `theta` squashes it. That is thresholding noise
off a blank field, not sharpening an edge, and it is the same artefact as the
x_HI = 1.000 bin's -42% at the opposite end of reionisation. Any bin whose gain
is not accompanied by real `edge_t` should be read as this.

`theta_pred` rises monotonically with x_HI -- 0.29, 0.31, 0.48, 0.48, 0.80,
identity, identity -- i.e. **sharp -> blurred**, the predicted direction.
Spearman(`theta_pred`, x_HI) = **+0.448** over 0-0.36 and **+0.721** over
0-0.10. Over the full test set it was +0.003: the decile binning averaged it
away entirely.

Two caveats on the size of the prize. Pooled over the whole 0.005-0.36 range
the gain is only **-0.16%**, because the responsive regime is a thin slice of
the data. And the effect is confined to x_HI < 0.05, which is a small part of
any lightcone. This is a targeted correction for very-late-reionisation maps,
not a general improvement.

Why here and nowhere else, consistent with section 6: at x_HI ~ 0.01 the field
is a handful of isolated neutral islands that the model renders as low-contrast
blobs -- correctly located, just under-committed, which is exactly the defect a
pointwise map fixes. By x_HI > 0.1 the field is a dense bubble network where the
errors are *positional*, and there sharpening does nothing at all (+0.00%, CI
width 0.01%, on 800-1100 held-out slices).

## 7. Verdict and open items

**Verdict, amended by 6.1.** The map is not useless -- it has one real regime.
At x_HI ~ 0.005-0.05, very late reionisation, a per-bin `theta(x_HI)` buys a
held-out 2-9% with intervals clear of zero, and `theta` there depends only on
x_HI, which the model estimates from its own mean output. That is realizable.
It is also narrow: pooled over x_HI < 0.36 it is worth -0.16%, and above
x_HI = 0.1 it is exactly zero to within 0.01% on ~1900 held-out slices.

Everywhere else the original verdict stands, and section 6 explains why in one
line: the
map is a working deblurrer (-9.28% on a field whose only defect is blur) that
buys exactly 0.00% on the model, at matched sharpness. **The model's edges are
not blurred, they are hedged** -- soft because their *position* is uncertain,
and no pointwise map moves an edge.

The supporting results: post-hoc, a genuine 6.2% oracle gain that cannot be
realised because the `tau` carrying it is a bias nuisance rather than the
physical threshold (3.3, 5.1). End-to-end, the network declines the map when
free to use it (4.2). The functional form was never the problem.

The map remains useful as an **explicit, reported post-processing dial**
(`theta ~ 0.30` buys truth-matched sharpness statistics for ~5% RMSE, section
3.1) -- but that is a presentation choice, not a better model.

Open:

* Sharpness table for the finished edge-loss runs at matched epoch 29. Ranking
  on L2 at epoch 25: baseline 0.1060 < swd 0.1072 < highk 0.1083 < swd+highk
  0.1089; `edgeonly` (no L2) failed outright at **0.9098**, never beating its
  epoch-0 value -- neither SWD nor highK constrains absolute level or position,
  so an L2 anchor is not optional. All edge variants cost L2, so their case
  rests entirely on sharpness, which is not yet measured at the final epoch.
* Whether any of this transfers to 3-D, where the front-width defect is worse
  (~3.5x vs ~2-4x here).
* If sharpness is still wanted, section 6 says where to spend the effort: the
  loss must penalise *misplaced* edges, not reward *steep* ones. Sharpening is
  measurably the wrong tool -- it fixes blur, and ~86% of the error is not blur
  (5.2). The transport direction (SWD) is the only edge term here that cost
  under 1.2% L2, and it is the only one that can move an edge rather than
  steepen it in place.
