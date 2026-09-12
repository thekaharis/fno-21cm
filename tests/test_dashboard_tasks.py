from __future__ import annotations

import json

from dashboard.serve import find_progress, read_task


def test_dashboard_recognizes_explicit_2d_task(tmp_path) -> None:
    run = tmp_path / "arbitrary-name"
    run.mkdir()
    (run / "run_metadata.json").write_text(json.dumps({"task": "2d"}))
    assert read_task(run, run.name) == "2d"


def test_dashboard_falls_back_to_2d_directory_name(tmp_path) -> None:
    run = tmp_path / "checkpoints_2d_xhi_localwno"
    run.mkdir()
    assert read_task(run, run.name) == "2d"


def test_dashboard_keeps_matched_log_without_batch_progress(tmp_path) -> None:
    run = tmp_path / "checkpoints" / "xhi2d_localwno_lr2e4_e30"
    run.mkdir(parents=True)
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "localwno-xhi2d-4333442.out"
    log.write_text(
        "CHECKPOINT_DIR: checkpoints/2d_xhi/localwno/xhi2d_localwno_lr2e4_e30\n"
        "[0] time=722.23, avg_loss=0.2258, train_err=7.2263\n"
    )

    progress = find_progress(run, tmp_path, metrics_mtime=0)

    assert progress is not None
    assert progress["log"] == "logs/localwno-xhi2d-4333442.out"
    assert progress["job_id"] == 4333442
    assert progress["phase"] is None
    assert progress["epoch_frac"] == 0.0


def test_dashboard_parses_batch_progress_from_matched_log(tmp_path) -> None:
    run = tmp_path / "checkpoints" / "checkpoints_3d_localwno"
    run.mkdir(parents=True)
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "localwno3d-123456.out"
    log.write_text(
        "CHECKPOINT_DIR: ./checkpoints/3d_xhi/localwno/checkpoints_3d_localwno\n"
        "    [train 25/100] 2.50 samples/s (2.50 batches/s, bs=1) "
        " elapsed 10.0s ETA 0.5 min\n"
    )

    progress = find_progress(run, tmp_path, metrics_mtime=0)

    assert progress is not None
    assert progress["log"] == "logs/localwno3d-123456.out"
    assert progress["job_id"] == 123456
    assert progress["phase"] == "train"
    assert progress["done"] == 25
    assert progress["total"] == 100
    assert progress["epoch_frac"] == 0.2375
