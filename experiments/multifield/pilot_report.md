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
