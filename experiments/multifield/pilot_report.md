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

- Per-field representation is extremely uneven. `neutral_fraction` is reproduced
  to 0.4%; `density` is not reproduced at all along the LOS. A field-to-field
  comparison across this cache is therefore **not** a like-for-like comparison of
  how learnable the fields are -- part of any ranking is the cache.
- `brightness_temp` and `los_velocity` as **targets** are floored by the grid, at
  roughly 49% and 30% relative error respectively. Achievable error is bounded by
  the cache, not the model.
- Transverse structure is unaffected: the native 140 x 140 grid is preserved.
  Transverse spectra remain meaningful; LOS and full 3-D spectra do not.

Revisit n_z if brightness temperature or velocity becomes a primary target.

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
