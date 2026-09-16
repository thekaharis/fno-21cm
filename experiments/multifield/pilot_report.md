# Multi-field pilot validation report

Pilot cache: `data/compressed/multifield_pilot.h5` (20 cones, 1.23 GiB gzip).
Grid as agreed: 4 plain fields, 256 LOS points, z = 5.001 .. 24.97, `z_params`.
Cones: `experiments/multifield/pilot_manifest.json`; axis audit for all 6600
simulations: `experiments/multifield/raw_axis_audit.json`.
Checks implemented in `tests/check_multifield_pilot.py`.

## Passed

| Check | Result |
| --- | --- |
| Raw geometry | 20/20 clean: field shapes agree, z strictly increasing, comoving step 1.428571 Mpc read from distance metadata (not inferred from redshift) |
| Coverage | No zero-filled LOS end planes. All 10 short-`z_max` cones included and intact -- the 5.001/24.97 endpoints do their job |
| Values | All four fields finite; `neutral_fraction` within [0, 1] |
| `cone_id` | 20 unique true sample IDs (260 ... 6461), patched from the stored source paths, not from assumed sort order |
| Split | Reproducible across calls, 16/2/2, zero overlap |
| Statistics | float64 accumulation, training cones only; normalized variance finite and non-degenerate for all four fields (0.33-1.50) |

Build rate 8.6 s/cone; gzip chunked one cone per chunk, ~18% saving.
**Full cache projects to ~400 GiB, not the 1.06 TB estimated before measurement.**

## LOS fidelity -- the pilot's substantive finding

Round-trip relative L2 (native -> n_z grid -> native, inside the retained
interval), typical cone 2299:

Measured on cone 2299 only. **The `neutral_fraction` column of this table is
not representative -- see the correction below.**

| n_z | density | neutral_fraction | brightness_temp | los_velocity |
| ---: | ---: | ---: | ---: | ---: |
| **256** | **1.0016** | 0.0037 | 0.4929 | 0.2984 |
| 512 | 0.8363 | 0.0034 | 0.4584 | 0.1772 |
| 1024 | 0.6221 | 0.0028 | 0.3742 | 0.0939 |
| 2048 | 0.3802 | 0.0020 | 0.2602 | 0.0437 |

Native LOS sampling is ~2400 cells over this interval, so 256 is a ~9.4x
downsample. Density carries most of its variance near the native Nyquist and
retains essentially none of its LOS structure at 256. Narrowing the redshift
range is not a cheap substitute: at fixed n_z = 256, restricting to 5.001-12.0
moves density only from 1.0016 to 0.7699.

**Decision (2026-09-14): keep 256 for this study.** This matches the existing
`cubes_3d.h5` (also n_z = 256 over z 5-25) on which every 3-D result in the
campaign is built, so it is the established grid rather than a regression.

Consequences that must be carried into any write-up:

- Transverse structure is unaffected: the native 140 x 140 grid is preserved.
  Transverse spectra remain meaningful; LOS and full 3-D spectra do not.
- Most of the end-to-end error of a trained model on this cache is the grid
  rather than the model -- quantified in the correction below.

Revisit n_z if the fine LOS structure of any field matters.

## Correction (2026-09-16): the x_HI fidelity figure above is not representative

**What was wrong.** The table reports `neutral_fraction` round-tripping at
0.0037, and that number was used to argue that x_HI is preserved almost exactly
while only density and brightness temperature are degraded. The measurement is
real but the cone is not typical of the ensemble: cone 2299 was selected for the
pilot by matching the median `n_z` and `OMm`, **never by ionization state**. Its
mean x_HI is 0.9996 with standard deviation 0.0020 -- an essentially uniform
neutral field, which round-trips almost perfectly because it has almost no LOS
structure to lose.

On genuinely ionized cones x_HI loses **0.09 to 0.24**, i.e. 25-64x more than
the figure above. The claim that per-field representation is "extremely uneven"
was therefore an artifact of cone selection. It is uneven, but far less so.

**Round-trip bound on four held-out test cones** (`tools_roundtrip_bound.py`,
raw data at 2x transverse stride, results in `roundtrip_bound.json`). Three
quantities, all relative L2:

- `cache` -- native -> 256 grid -> native. What the grid discards, model-free.
- `model` -- prediction vs the *cached* truth, both on the 256 grid. **Not**
  floored by the cache: a perfect model scores zero here.
- `end_to_end` -- prediction lifted back to native vs native truth. What the
  pipeline delivers; bounded below by `cache`.

| cone | mean x_HI | field | cache | model | end_to_end | grid share |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 47 | 0.90 | brightness_temp | 0.3187 | 0.2023 | 0.3222 | 97.9% |
| 67 | 0.99 | brightness_temp | 0.2223 | 0.1289 | 0.2347 | 89.7% |
| 444 | 0.89 | brightness_temp | 0.4743 | 0.3169 | 0.4848 | 95.7% |
| 541 | 0.72 | brightness_temp | 0.3621 | 0.3728 | 0.4541 | 63.6% |
| 47 | 0.90 | neutral_fraction | 0.1776 | 0.1032 | 0.2072 | 73.5% |
| 67 | 0.99 | neutral_fraction | 0.0936 | 0.0213 | 0.0996 | 88.2% |
| 444 | 0.89 | neutral_fraction | 0.2372 | 0.0908 | 0.2486 | 91.0% |
| 541 | 0.72 | neutral_fraction | 0.1501 | 0.1164 | 0.1941 | 59.8% |

(Model column from the epoch-10 snapshot of `mf_cnn_fno`, training still in
progress, so it will improve; the `cache` column will not.)

**What this establishes.**

- `end_to_end` sits essentially on top of `cache` in almost every row. **60-98%
  of the end-to-end error variance is the LOS grid, not the model**, and on
  three of the four cones the brightness-temperature model error is already
  *smaller* than the cache loss. A better architecture cannot recover most of
  this; only more LOS points can.
- **Both** targets are substantially grid-limited, not just brightness
  temperature. The earlier framing -- x_HI faithful, T_b floored -- does not
  survive measurement.
- Cone 541 is the informative exception: the most ionized of the four
  (mean x_HI 0.72), the most real structure, and the only cone where the model
  error meets or exceeds the cache loss (0.373 vs 0.362 for T_b). Where there is
  genuine structure to learn, the model is the limiting factor. High grid shares
  elsewhere partly reflect cones that are close to featureless.

**Superseded statements.** The earlier claim that brightness temperature and
velocity "as targets are floored by the grid" was imprecise in a second way: the
model predicts the *cached* target from *cached* inputs, so the grid does not
floor the model's error against that target at all. What the grid does is
degrade the **inputs** and bound the **end-to-end** error against the native
field. Those are different statements and only the latter two are supported.

**Limits of this measurement.** It captures LOS information loss only, and says
nothing about whether density and velocity determine brightness temperature in
principle. The data audit found that spin temperature -- absent from the four
fields -- leaves between 5% and 90% of T_b unexplained depending on redshift, so
part of the `model` column may be genuinely unpredictable from these inputs
rather than a modelling shortfall.

**Selection lesson for future pilots.** Choosing pilot cones by geometry and
cosmology extremes missed both of the problems that actually mattered: the
non-finite brightness-temperature voxels that aborted the first full build, and
the ionization-state dependence of round-trip fidelity. A pilot set should span
the **target field's own dynamic range**, not only the metadata.

## Heavy tail in brightness temperature

Cone 5984 reaches 25,575 mK in the **raw** lightcone (z = 7.5, `L_X` = 41.9) and
8,897 mK after resampling, against a median of order 10 mK. This is genuine
simulator output, not an interpolation artifact.

Standardizing `brightness_temp` on training cones will therefore have its scale
set by a small number of extreme voxels. Robust statistics or an explicit,
recorded clip is worth considering; whichever is chosen must be stored in the
preparation artifact rather than recomputed.

## Non-finite values and the exclusion decision (2026-09-14)

The first full build (job 4923961) aborted after 10 minutes at cone 72:

    ValueError: invalid/undefined values in brightness_temp, cone 72;
    sparse or sentinel-valued fields require an explicit mask policy

The pilot did not catch this. It selected geometry and cosmology extremes as
agreed, which are not value pathologies, and its finiteness check sampled
`[:, ::4, ::4, :]` -- a 9-voxel defect is invisible to a 1-in-16 subsample.
Both are fixed: any future pilot should include known-defective cones, and the
check must read full resolution.

An exhaustive full-resolution scan of all 6600 simulations followed
(`slurm/scan_nonfinite.sbatch`, 16-way array, ~1 h; results merged into
`experiments/multifield/nonfinite_report.json`):

| Field | Cones affected | Voxels |
| --- | ---: | ---: |
| density | 0 | 0 |
| neutral_fraction | 0 | 0 |
| **brightness_temp** | **33 (0.50%)** | 242 |
| los_velocity | 0 | 0 |

1-30 bad voxels per affected cone (median 8), out of 2.8e11 voxels scanned.
The defect is confined entirely to `brightness_temp`.

**Decision: exclude all 33 simulations uniformly.** The cache is built from
**6567** cones. Excluding per-mapping instead would have kept the affected cones
for mappings that do not use brightness temperature, but the field combinations
would then no longer share an identical simulation set -- confounding exactly
the comparison this study exists to make. Repair by neighbour interpolation was
rejected as silently altering simulator output.

The exclusion is implemented as a staged symlink view at
`data/_multifield_stage`, not by modifying the raw data. The excluded IDs are in
`nonfinite_report.json` and must be carried into the preparation artifact.

Because the reader assigns `cone_id` by sorted file position, the built cache
will carry 0..6566 rather than true sample IDs; it is patched afterwards from
the stored `source_description` paths, as was done for the pilot.

## Full cache built (2026-09-15)

`data/compressed/multifield_z256.h5` -- job 4929874, **12h 05m**, **398 GiB**
(6.6 s/cone). Four fields x (6567, 140, 140, 256) float32, plus `params`,
`cone_id`, `target_z` on 256 points spanning 5.00100-24.97000.

Size landed at 398 GiB against the 1.06 TB quoted before measurement; gzip at
one cone per chunk accounts for most of the difference.

`cone_id` was patched from the stored `source_description` paths to true
21cmFAST sample IDs, asserted before writing: 6567 unique, zero intersection
with the 33 excluded, and 6567 + 33 = 6600. It now runs 0..6598 with gaps at the
excluded simulations rather than a contiguous 0..6566, which would have
misidentified every cone after the first exclusion. The file also carries
`cone_id_source`, `excluded_sample_ids` and `exclusion_reason` in its attributes,
so it is self-describing without the JSON reports.

Verification against the raw lightcones, 12 randomly chosen cones
re-interpolated independently and compared value-by-value:

- worst relative discrepancy **5.5e-08**, i.e. float32 round-off (eps ~1.2e-07).
- the row -> sample_id offsets grow with row index exactly as the exclusions
  accumulate (row 108 -> 109, row 5994 -> 6022), which is the check that the
  identity mapping is right rather than merely self-consistent.
- finiteness and x_HI bounds on 40 further random cones: 0 problems.

Remaining before training: `fno_multifield.py prepare --split-seed 42`. The
exclusion list must be carried into that artifact, and the brightness-temperature
tail (cone 5984 reaches 25,575 mK in the raw lightcone) needs an explicit,
recorded normalization decision -- robust statistics or a stored clip -- before
the preparation artifact is fixed.
