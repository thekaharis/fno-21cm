from __future__ import annotations

from unittest.mock import patch

from run_metadata import load_run_metadata, resolve_checkpoint, write_run_metadata


def test_metadata_round_trip_and_checkpoint_selection(tmp_path):
    metadata = {"training": {"best_epoch": 4}, "input_features": {"name": "density"}}
    write_run_metadata(tmp_path, metadata)
    assert load_run_metadata(tmp_path) == metadata

    best = tmp_path / "best_model_state_dict.pt"
    final = tmp_path / "final_model_state_dict.pt"
    best.touch()
    final.touch()
    with patch.dict("os.environ", {"CHECKPOINT_KIND": "best"}, clear=True):
        assert resolve_checkpoint(tmp_path) == best
    with patch.dict("os.environ", {"CHECKPOINT_KIND": "final"}, clear=True):
        assert resolve_checkpoint(tmp_path) == final
