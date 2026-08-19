# legacy/

Code that is no longer part of the active pipeline but is kept **runnable**,
because checkpoints produced by it still exist on the cluster.

Nothing in the main tree imports from here at module level. The one deliberate
link is `modeling.build_model`, which dispatches retired architecture kinds
to `legacy/arch/` lazily — so an old `run_metadata.json` rebuilds its model with
no edits, and the retired code costs nothing until something asks for it.

| Folder | Contents |
|---|---|
| `arch/` | Retired 3-D architectures: standalone `SirenFNO3d`, and the plain `neuralop` FNO. Reached via `modeling.build_model` for `kind="sirenfno"` / `"fno"`. |
| `xhi2d/` | Analysis left over from the 2-D x_HI work: the contrast/sharpness study behind `NOTES-contrast-map.md`, and comparison plots. The pipeline itself was promoted back to the main tree. |
| `probes/` | One-off measurement scripts. They were never tests despite living in `tests/`; none of them assert anything. |
| `slurm/` | Job scripts for all of the above, plus superseded per-variant 3-D jobs that `slurm/train_3d_matrix.sbatch` replaced. |
| `legacy/lploss.py` | Unused; kept only because it predates the vendored `neuralop` copy. |

The 2-D x_HI training pipeline itself lives in the main tree again (`fno_xhi2d.py`).

## Running inference on an old checkpoint

Nothing special is required. The renderer reads the architecture from the run's
own metadata:

```bash
CHECKPOINT_DIR=checkpoints/checkpoints_3d_sirenfno_m64_stable sbatch slurm/viz_localop.sbatch
```

## Running legacy code directly

Everything resolves the project root from its own location, so both forms work
from anywhere:

```bash
python -m fno_xhi2d
python legacy/probes/probe_contrast_theta.py --bins decile
```

## What changed when this folder was created

- The 2-D slice dataset moved to `dataset/slices.py`; its `make_file_split` was
  a byte-identical copy of the 3-D one and is now re-exported from
  `dataset/dataset_3d.py`. The two pipelines are only comparable while they hold
  out the same cones, so that duplicate was a correctness risk, not just noise.
- The SirenFNO module was split: the SIREN weight generators the operator
  registry needs stayed in `siren.py`, the standalone architecture moved to
  `legacy/arch/siren_fno_3d.py`.
- The probes used to hardcode `/pfs/10/work/hd_id260-fno_training/fno-21cm` as
  the project root. They now resolve it relative to themselves and run locally.
