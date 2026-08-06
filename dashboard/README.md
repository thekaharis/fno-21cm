# Training metrics dashboard

Runs are grouped into independent tabs by the `task` field in
`run_metadata.json`: `3d` for full x_HI lightcones, `2d` for transverse x_HI
slices, and `zre` for reionization-redshift maps.

Live-updating dashboard for comparing `metrics.jsonl` files across runs.
Python stdlib only — no installs, no internet needed from the cluster.

## Start (on a Binac login node)

```bash
python3 /pfs/10/work/hd_id260-fno_training/fno-21cm/dashboard/serve.py
```

## View (from your laptop)

```bash
ssh -L 8080:localhost:8080 hd_id260@login.binac2.uni-tuebingen.de
# then open http://localhost:8080
```

**Load-balancer gotcha:** `login.binac2` drops each SSH session on a random
login node (login01–03), and the dashboard is only reachable on the node it
runs on. So the server and your tunnel/forwarder must be on the SAME node:

- With **VS Code**: start `serve.py` in the *integrated terminal* (same node
  as VS Code's port forwarder) and use the Ports panel to forward 8080.
- With a **plain tunnel**: after login run `hostname`; if the server runs on
  another node, chain a hop: `ssh -L 8080:localhost:8080 login0X`.
- SSH between login nodes is passwordless, so
  `ssh login0X 'ss -tln | grep 8080'` finds where a server is running.

## What it shows

- Every run under `fno-21cm/checkpoints/`, `fno-21cm/checkpoints_*/` and
  `fno-21cm/checkpoint-archive/*/` that has a `metrics.jsonl`, labeled with
  model config from `run_metadata.json`.
- Runs are grouped into pages by training target (tabs in the header:
  21cm 3-D lightcones vs z_re maps). The tag comes from `task` in
  `run_metadata.json`, falling back to the directory name; tabs appear
  automatically once runs of more than one target exist. z_re runs write
  `metrics.jsonl` since the `ZreLoggingTrainer` addition to `fno_zre.py` —
  older z_re runs (before that) have no metrics file and stay invisible.
- A **LIVE** badge on runs whose `metrics.jsonl` was written within ~2 epochs.
- A **Live runs** status panel with two progress bars per running job:
  overall training progress (epochs done + current-epoch fraction, out of
  `training/epochs` from `run_metadata.json`) and within-epoch progress with
  phase, batch count, ETA, and throughput. Within-epoch progress is parsed
  from the `[train i/N] ... ETA x min` lines in the newest `logs/*.out` whose
  header names the run's checkpoint dir — no training-code changes needed.
  A ⚠ appears if the log has been silent for >10 min (possible stall).
- Click runs to overlay them; click metric chips to add charts. Hover a chart
  for per-run values at an epoch. "log y" toggles log scale.
- **clip to live epoch** cuts all curves at the live run's current epoch, so a
  finished baseline is compared only up to where the live run has gotten.
- Summary table: best `val_l2_rel`/`test_l2_rel`, last `train_err`, grad norm,
  min/epoch, samples/s, last write time.
- Auto-refreshes every 30 s (configurable in the header).

## Comparing against an arbitrary checkpoint directory

Either paste the directory path (or its `metrics.jsonl` path) into the **Add**
box in the UI, or start the server with:

```bash
python3 serve.py --add /some/other/place/checkpoints_xyz
```

Added paths persist in `dashboard/extra_runs.json` across restarts; remove one
with the ✕ next to its name.

## Options

| Flag     | Default                | Meaning                                   |
|----------|------------------------|--------------------------------------------|
| `--port` | 8080                   | HTTP port                                  |
| `--host` | 127.0.0.1              | Bind address (keep localhost + SSH tunnel on shared login nodes) |
| `--root` | parent of this dir     | Project root scanned for run directories   |
| `--add`  | —                      | Extra run directory (repeatable)           |
