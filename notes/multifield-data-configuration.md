# Initial multi-field data configuration

Coordination date: 2026-09-14. Codex corresponded with Claude through the
Claude Desktop session **3D runs training**. Remote inventory and numerical
checks below were reported by Claude; Codex did not independently read the
remote raw files. This document records the configuration and its evidence,
not a completed cache build or training experiment.

## Initial study configuration

| Setting | Selection |
| --- | --- |
| Fields | `density`, `neutral_fraction`, `brightness_temp`, `los_velocity` |
| Stored variants | Plain `/lightcone/<field>` arrays, consistently across all fields |
| Transverse grid | Preserve the native 140 × 140 grid over 200 Mpc |
| LOS grid | 256 points, linear in redshift, from **5.001 through 24.97** inclusive |
| Resampling | Apply the same linear interpolation to each field; reject out-of-coverage requests |
| Split | Simulation IDs, seed 42; use the pipeline's 80/10/10 train/validation/test default |
| Conditioning | Fixed `z_params` configuration for the initial study |
| Normalization | Density divided by 10; neutral fraction unchanged; brightness temperature and velocity standardized using training cones only |
| Input/target rule | Nonempty, disjoint sets of canonical field identities |
| Deferred field | `tau_21`, to keep the first study at four fields |

Store the actual split lists and normalization in one preparation artifact and
reuse it across every mapping and training seed. A shared sampling-design seed
is not the initial-condition seed. Never split slices from one simulation across
training and evaluation. Regenerating a split from seed 42 alone does not prove
that it matches an earlier implementation; compare the actual ID lists.

Linear redshift sampling does not imply a uniform comoving LOS spacing, and the
redshift-to-distance relation varies with cosmology. Transverse spectra are
appropriate for this representation; a full 3-D FFT would need separate treatment.

## Inventory reported by Claude

- 6,600 raw files named `21cmfast_11d_sample000000.h5` through
  `21cmfast_11d_sample006599.h5`, approximately 7.6 TB in total.
- Producer metadata: 21cmFAST 4.1.1, `raw_lightcone_v2.0`, E-INTEGRAL source model.
- All four requested scalar fields were present in inspected files and aligned
  within each inspected simulation. The four also have `_with_rsds` counterparts.
- Native cubes have shape `(140, 140, nz)`; the exhaustive LOS-axis audit found
  `nz` between 2,074 and 2,919, with 846 distinct lengths.
  Forty-four sampled files had no field-shape disagreements. Those shape checks
  are sampled evidence, not a completed scan of every cube's values.
- The exhaustive ID/seed audit reported 6,600 unique initial-condition seeds and
  contiguous sample IDs 0–6599. Existing two-field caches contain density and
  neutral fraction only, so they cannot supply the other two fields.
- Other quantities, including spin and kinetic temperatures, appear as global
  histories; their presence there does not establish availability as 3-D cubes.

The exhaustive redshift-axis scan found 10 files ending below 25, with the
lowest maximum approximately 24.979489. Every file starts slightly above 5
(at most about `4.4e-8` above it). The chosen 5.001–24.97 interval lies safely
inside all reported axes. The old reader's zero-fill behavior could corrupt
boundary slices on a 5–25 grid. The new multi-field reader already rejects
substantial coverage gaps and clips only within its `1e-6` endpoint tolerance.
Use the explicit revised grid even though the generic CLI default remains 5–25.

Box metadata was checked in 413 files, within-file shape agreement in 44, native
LOS step in 22, and field ranges/velocity physics in only 1–3. These checks do not
establish complete value validity or equal geometry across all files. Reported
brightness-temperature signs/ranges and simple-regression scores must not be
generalized to the full ensemble.

The full four-field cache is about 1.06 TB before compression and metadata.
Compression ratio, build time, user quota and temporary-space requirements still
need measurement. Reported filesystem free space is not a user-quota check.

## Units and velocity handling

Density is a dimensionless overdensity, neutral fraction is dimensionless, and
brightness temperature is in mK. Preserve native velocity numbers during caching.
Claude measured a native velocity standard deviation around `2.3e-17` in a sample;
that small magnitude is not evidence of a constant field.

The current official [21cmFAST RSD API documentation](https://21cmfast.readthedocs.io/en/stable/autoapi/py21cmfast/rsds/)
specifies Mpc/second for LOS velocity and separates velocity-gradient corrections
to optical depth from remapping cells into redshift-space coordinates. The page
served during this review documents a newer development version, not the exact
4.1.1 producer. Claude could not locate the installed producer source or a unit
attribute in the files. Its Fourier linear-theory check across redshifts 6–15
strongly supports a comoving `dx/dt` convention: the fitted coefficient divided
by `f H` was approximately 0.997–0.998 in one simulation, using an assumed
`h=0.6774`. Treat this as empirical corroboration,
not exact-version source verification. If this convention holds, proper peculiar
velocity is `a * dx/dt`, with the usual Mpc-to-km conversion for km/s.

The generic registry retains `source velocity units` because other producers may
rescale exports. Record the above convention and its evidence with this dataset;
do not silently relabel all possible sources as km/s.

This audit exposed a numerical issue in preparation and evaluation. The pipeline
now retains any positive training standard deviation instead of replacing tiny
dimensional scales with 1. Dimensionless evaluation accumulates in normalized
coordinates, then dimensional error metrics convert back to source units. A
regression test checks identical dimensionless metrics for velocity units differing
by a factor of `1e-17`.

Float64 alone does not fix a mixed-scale regression or a dimensional epsilon.
Normalize diagnostic regressors as well: Claude's failed least-squares fit was
already float64, and the solver discarded the tiny raw gradient column until
the regressors were standardized.

Plain and `_with_rsds` fields differ even for density and neutral fraction,
consistent with coordinate remapping. Use one coordinate convention consistently.
Claude subsequently found the producer attribute `include_dvdr_in_tau21=True`,
corroborated by improved brightness-temperature regression when a velocity-gradient
term was included. Thus plain brightness temperature is already corrected for
the optical-depth effect; do not apply that correction a second time. This is
separate from coordinate remapping.

A physical relationship between an input and target is not train/test leakage;
however, mappings with an almost algebraic solution need a physical baseline to
separate that advantage from learning additional information.

Claude's sampled regressions suggest that brightness temperature can be much
easier to recover from the other three fields at some late-time states than
during early heating. This is exploratory, not held-out model performance.
Report metrics by redshift band as well as overall. Fit any empirical coefficients
in a density/neutral-fraction/velocity-gradient baseline on training simulations
only, then evaluate it on the same held-out simulations as the network. Compute
velocity gradients using the correct comoving-distance spacing and convention,
not as derivatives per uniformly sampled redshift pixel.

## Pilot before the full build

Claude proposed roughly 20 cones including the 10 short-coverage files, the
minimum/maximum LOS lengths, and cosmology extremes. This is about 1.6 GB of
uncompressed four-field output; raw inputs and interpolation buffers require
additional space. Exact pilot IDs still need to be saved from the audit.

1. Verify all four shapes, source-axis monotonicity, coverage, box geometry,
   finite values and physical bounds. Check native distance spacing using actual
   distance metadata; do not infer constant distance spacing from redshift steps.
2. Cache on the revised grid and compare with direct raw interpolation. Interpolate
   back only inside the retained interval to quantify information lost by the
   256-point grid. Inspect neutral-fraction fronts and per-field errors by redshift.
   A poor reconstruction would motivate more LOS points before a full build.
3. Record stable simulation IDs and identical train/validation/test lists. The
   current raw reader assigns IDs by sorted file position; full contiguous input
   names match sample IDs, but an arbitrary sparse pilot would reindex them.
   Preserve an explicit original-ID manifest for the pilot and do not reuse its
   preparation artifact for the full dataset. Cached readers support explicit
   noncontiguous `cone_id` values.
4. Fit statistics on pilot training cones only. Confirm finite normalized velocity
   variance, native-unit round trips and reproducible dimensionless metrics.
5. Measure throughput, peak memory and output size; verify available quota before
   the full build. Validate every full-dataset cube during construction, since
   sampled geometry and value checks do not establish full-dataset validity.
6. Validate physical baselines on held-out pilot cones, reporting by redshift band.
   A small or deliberately extreme pilot is for pipeline validation, not the final
   ranking of field combinations.

Build invocation after pilot validation (paths must be set on the data host):

```bash
python fno_multifield.py cache --data "$RAW_LIGHTCONES" \
  --fields density,neutral_fraction,brightness_temp,los_velocity \
  --z-min 5.001 --z-max 24.97 --n-z 256 --out "$MULTIFIELD_CACHE"
python fno_multifield.py prepare --cache "$MULTIFIELD_CACHE" \
  --conditioning z_params --split-seed 42 \
  --out experiments/multifield/preparation.json
```

Status at the end of this exchange: configuration agreed with Claude; local
normalization/metric correction tested (15 multi-field tests passed); no remote
preprocessing, cache construction or training submission performed.
