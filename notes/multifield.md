# Modular 21cmFAST field experiments

For the actual dataset inventory, unit evidence and pilot configuration discussed
with Claude, see [Initial multi-field data configuration](multifield-data-configuration.md).

For native-resolution contiguous LOS windows and fine windows with coarse
surrounding inputs, see [Native LOS window training](los-window-sampling.md).
Both use this entry point and support the same disjoint field mappings.

The first implementation supports arbitrary **disjoint sets of registered,
aligned scalar 3-D fields**, with a fixed backbone architecture and fresh weights
for each mapping. It does not freeze a density → x_HI checkpoint. Every selected
target contributes gradients to the shared layers.

The existing `fno_21cm_3d.py`, 2-D and z_re entry points retain their historical
defaults and checkpoint layouts. `fno_multifield.py` is a separate, deliberately
controlled experiment recipe: per-field normalized MSE, Adam, cosine scheduling,
and validation-based checkpoint selection. Its scores are **not** directly
comparable to the old L2/H1/contrast training recipe without rerunning a matched
single-target baseline.

## Supported field contract

`dataset/fields.py` defines field identities, aliases, units, normalization and
physical bounds. Initial sweep fields are `density`, `neutral_fraction`,
`brightness_temp` and `los_velocity`. A field always has the same normalization
whether it is an input or a target. The aliases `x_HI`, `xhi` and `xH_box` all
resolve to `neutral_fraction`, so aliases cannot bypass the overlap check.
Equivalent targets such as `1-x_HI` are not registered as independent fields.

Raw files use `/lightcone/<field>` with `(X,Y,Z)` arrays and
`/lightcone/lightcone_redshifts`. Cached files use root-level `<field>` arrays
with `(cones,X,Y,Z)`, `cone_id`, `target_z` and optional `params`. Scalar vector
components may be registered separately. A complete custom registry can be
supplied to `prepare`/`cache` using `--registry registry.json`; start with
`FieldRegistry().to_dict()` and add explicit field definitions. Units must match
the stored simulation convention. Velocity is labeled `source velocity units`
until the producer's unit convention is verified and recorded in the registry.
No implicit unit conversion or real/redshift-space remapping is performed.

The reader checks field availability, grid shape, cone IDs, redshift coverage,
parameter schema and finite/valid values. Raw LOS grids may increase or decrease;
all fields are interpolated consistently to the increasing target grid. Requested
redshifts outside any source cone are rejected, rather than filled with zeros.
Different spatial geometries must be aligned before entering this pipeline.

The broader registry includes thermal and ionization-history fields, but their
presence in the registry does not mean they exist in your files or are valid
under every simulation configuration. Undefined/sentinel-valued fields (such as
`z_reion=-1` in cells not yet ionized) are rejected. A domain-specific mask policy
is still needed before using those incomplete fields. Lightcone-fitted 2-D z_re
maps are not substituted for simulator-local 3-D reionization redshifts.

## Prepare the data once

Use the configured data directories from `dataset/paths.py`; large caches belong
outside the repository. Paths below are shell variables set for your machine.

```bash
MULTIFIELD_CACHE=/path/to/compressed/cubes_multifield.h5
RAW_LIGHTCONES=/path/to/raw/lightcones
```

If a cache already contains all four fields, use it directly. Otherwise build
an additional cache from raw files (the old two-field cache cannot supply absent
fields):

```bash
python fno_multifield.py cache --data "$RAW_LIGHTCONES" \
  --fields density,neutral_fraction,brightness_temp,los_velocity \
  --z-min 5.001 --z-max 24.97 --n-z 256 --out "$MULTIFIELD_CACHE"
```

Select a redshift range covered by **every** cone, or pass an increasing `.npy`
grid with `--target-z`. Caching is currently serial and writes one field at a
time for each cone. It validates every cube, stops on failures, writes to a
temporary file, and publishes the cache only after successful completion.
Do not pass this cache to the old two-field shard merger.

Create a preparation artifact containing the common split and training statistics:

```bash
python fno_multifield.py prepare --cache "$MULTIFIELD_CACHE" \
  --conditioning z_params --split-seed 42 \
  --out experiments/multifield/preparation.json
```

Alternatively, `prepare --data "$RAW_LIGHTCONES" ...` trains from raw lightcones
without caching. Preparation needs at least three cones and at least two fields.
Physical cone IDs, rather than row order, define the splits. Noncontiguous IDs
are supported. Reuse **the same preparation file** for every mapping and training
seed. The split seed and training seed are independent.

Statistics are streamed from training cones only. Density keeps the existing
`density/10` convention; fractions stay in `[0,1]`; other fields use a training
mean and standard deviation. Thus the combined loss has explicit field scaling,
but is not automatically equal in gradient magnitude across fields. For a study
that standardizes density too, set its registry normalization to `standard` and
prepare a new study. Constant fields use scale 1. Raw training means/stds are
also saved for baseline evaluation.

Small native values are not treated as constant: any positive training standard
deviation is retained, including velocity scales around `1e-16`. Only an exactly
zero standard deviation uses scale 1. Correlations, normalized errors and spectral
ratios are accumulated in normalized coordinates, then dimensional error metrics
are converted back to source units. This keeps evaluation invariant to a change
of velocity units.

Conditioning is `none`, `z`, `params`, or `z_params` (default). It is fixed by the
preparation artifact. Source paths, sizes, modification times, grids and IDs are
fingerprinted; edits or relocation require a new preparation artifact. This is
a provenance check, not a full byte-content checksum of hundreds of GB of data.

## Train one mapping

```bash
python fno_multifield.py train \
  --preparation experiments/multifield/preparation.json \
  --inputs density --targets neutral_fraction,brightness_temp \
  --model-settings '{"kind":"localop","ndim":3}' \
  --epochs 20 --seed 42 --device cuda \
  --run-dir checkpoints/multifield/density_to_xhi_tb_seed42
```

Input and target order are explicit and recorded. The final projection produces
one independent affine output per field from shared hidden features. Only bounded
fraction channels use a sigmoid. Continuous fields use linear outputs in their
normalized representation; no x_HI contrast map, BCE or bubble loss is applied to
velocity or temperature channels. All existing architecture families use the
same shared `build_model` factory; U-FNO now supports multiple output channels.

The new entry point accepts a `ModelConfig` JSON object, independently of old
architecture environment variables. Use a complete saved config for exact
comparisons. The default is the local/global Fourier architecture; mode counts
must still fit the chosen grid and backbone downsampling, as in existing runs.
Default training is one process on one GPU (or CPU), without mixed precision.

Default loss weights are equal and the weighted sum is divided by the sum of
weights. Override with, for example,
`--loss-weights '{"neutral_fraction":2,"brightness_temp":1}'`.
The best checkpoint minimizes the weighted validation objective. For a primary
target experiment, `--monitor neutral_fraction` selects by that field's validation
MSE instead. No test result is used for checkpoint selection.

Outputs:

- `run_metadata.json`: full model, mapping, preparation, recipe and completion status.
- `metrics.jsonl`: per-epoch training objective and per-field validation metrics.
- `best.pt`, `final.pt`: weights plus complete reconstruction metadata.
- `test_metrics.json`: best-checkpoint test metrics, computed once after training.

Per-field metrics include physical RMSE/MAE, mean bias, normalized MSE, Pearson
correlation and skill relative to a constant **training-mean** predictor. Final
evaluation also reports transverse power ratios and Fourier cross-correlation in
cycles per pixel. Spectra remove each slice's transverse mean and never Fourier
transform the evolving LOS; they do not replace coeval 3-D power spectra. Degenerate
correlations and empty spectral bins are JSON null. `--spectral-bins 0` disables
spectra. Compare each target against its matched single-target run, not raw errors
between fields with different units/scales.

Existing run directories are not overwritten. Failed runs retain their metadata
and completed epoch checkpoints. Exact optimizer/scheduler resume, frozen-backbone
transfer, target-specific morphology diagnostics and arbitrary-input masked
universal models are subsequent extensions; this entry point does not claim to
implement them yet. A fresh attempt needs a new run directory.

## Plan and run the staged study

```bash
python -m util.multifield_experiments plan \
  --preparation experiments/multifield/preparation.json \
  --stage pairwise --seeds 42 43 44 --epochs 20 \
  --out experiments/multifield/pairwise.json
```

`--model-config config.json` fixes the architecture for a plan. Stages are
**cumulative**: `pairwise` has 12 mappings, `single-target` has 28, and `all` has
50. With three seeds these produce 36, 84 and 150 jobs. Reuse the same run root
when expanding the stage: identical jobs get identical directories and can be
skipped when complete. Planning writes a manifest; it never submits training.

```bash
python -m util.multifield_experiments run \
  --plan experiments/multifield/pairwise.json --index 0 --device cuda
```

On the cluster, create `logs` before submission because SLURM opens log files
before the script starts. Adapt resource requests to measured multi-field memory
use; single-target memory estimates are not multi-field measurements.

```bash
mkdir -p logs
sbatch --array=0-35 \
  --export=ALL,MULTIFIELD_PLAN="$PWD/experiments/multifield/pairwise.json" \
  slurm/train_multifield.sbatch
```

Build and prepare the data on the same host where jobs will run: manifests and
preparation artifacts deliberately store resolved source paths.

```bash
python -m util.multifield_experiments summarize \
  --plan experiments/multifield/all.json --out figures/multifield/study_summary
```

The summary includes per-seed rows, means and sample standard deviations, missing
job indices, and paired auxiliary gain `1 - MSE_multi / MSE_single` for matching
inputs, target, seed, preparation, architecture and training recipe. Positive
gain means improvement. Missing baselines remain blank. Results selected by the
combined validation objective describe that selection protocol, not an oracle
checkpoint chosen independently for each target.

## Reload and export

```bash
python fno_multifield.py evaluate --checkpoint /path/to/best.pt \
  --split test --out /path/to/reevaluated.json
python fno_multifield.py predict --checkpoint /path/to/best.pt \
  --cone-id 0 --out /path/to/prediction.h5
```

Exports contain physical-unit `prediction/<field>` and `target/<field>` arrays,
the redshift grid, units and mapping. This first export command operates on cones
in the recorded source dataset; it is an evaluation/export path, not yet a
deployment API for previously unseen files.
