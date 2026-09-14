from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import json
import sys

import h5py
import numpy as np
import pytest
import torch

from dataset.fields import FOUR_FIELDS, FieldMapping, FieldRegistry, enumerate_mappings
from dataset.lightcone_params import PARAM_NAMES
from dataset.multifield import MultiFieldDataset
from modeling import ModelConfig, build_model
from multifield_model import MultiFieldModel, weighted_objective
from util.multifield_metrics import FieldMetrics


def write_cache(path, ids=(0, 1, 2, 3, 4), shape=(8, 8, 8)):
    rng = np.random.default_rng(12)
    shape = (len(ids), *shape)
    density = rng.uniform(-0.5, 1, shape).astype(np.float32)
    fields = {"density": density,
              "neutral_fraction": (density > 0).astype(np.float32),
              "brightness_temp": (10*density - 3).astype(np.float32),
              "los_velocity": (20*density - 5).astype(np.float32)}
    with h5py.File(path, "w") as f:
        for name, values in fields.items():
            f.create_dataset(name, data=values)
        f.create_dataset("cone_id", data=ids)
        f.create_dataset("target_z", data=np.linspace(5, 8, shape[-1]))
        f.create_dataset("params", data=np.arange(len(ids)*len(PARAM_NAMES)).reshape(len(ids), -1))
        f.attrs["param_names"] = np.asarray(PARAM_NAMES, dtype="S")
    return fields


def small_config(kind="localop"):
    return ModelConfig(kind=kind, ndim=3, modes=(1, 1, 2), hidden_channels=4,
                       n_layers=1, localfno_base_width=4, localfno_spectral_rank=2,
                       localfno_window=(4, 4, 4), localfno_modes=(2, 2, 2),
                       ufno_width=4, ufno_norm="groupnorm")


def all_mapping():
    return FieldMapping.create("density", FOUR_FIELDS[1:])


def test_mapping_aliases_overlap_and_stage_counts():
    with pytest.raises(ValueError, match="disjoint"):
        FieldMapping.create("neutral_fraction", "x_HI")
    with pytest.raises(ValueError, match="duplicate"):
        FieldMapping.create("density,matter_density", "xhi")
    with pytest.raises(ValueError, match="unknown"):
        FieldMapping.create("density", "ionized_fraction")
    stages = [enumerate_mappings(stage=stage) for stage in ("pairwise", "single-target", "all")]
    assert [len(s) for s in stages] == [12, 28, 50]
    assert set(stages[0]) < set(stages[1]) < set(stages[2])
    assert len({m.slug for m in stages[2]}) == 50
    assert all(set(m.inputs).isdisjoint(m.targets) for m in stages[2])


def test_preparation_uses_training_only_and_reuses_stats_for_every_role(tmp_path):
    path = tmp_path / "cache.h5"
    values = write_cache(path)
    dataset = MultiFieldDataset(all_mapping(), cache=path)
    prep = dataset.prepare()
    training = prep["split"]["train"]
    assert prep["normalization"]["brightness_temp"]["offset"] == pytest.approx(
        values["brightness_temp"][training].mean(), abs=1e-6)
    assert prep["normalization"]["density"]["scale"] == 10
    assert prep["normalization"]["neutral_fraction"]["scale"] == 1
    installed = dataset.install_preparation(json.loads(json.dumps(prep)))
    assert set(installed) == {"train", "val", "test"}
    reverse = MultiFieldDataset(FieldMapping.create("brightness_temp", "density"), cache=path)
    reverse.install_preparation(prep)
    assert torch.equal(dataset[0]["y"][1], reverse[0]["x"][0])
    assert torch.equal(dataset[0]["x"][0], reverse[0]["y"][0])
    assert dataset[0]["x"].shape[0] == 13
    assert dataset[0]["y"].shape[0] == 3
    dataset.close()
    reverse.close()


def test_noncontiguous_cone_ids_and_reordered_rows_have_identical_splits(tmp_path):
    path = tmp_path / "cache.h5"
    write_cache(path, ids=(90, 4, 17, 6, 100))
    dataset = MultiFieldDataset(all_mapping(), cache=path)
    first = dataset.prepare()
    dataset.close()
    with h5py.File(path, "r+") as f:
        for key in (*FOUR_FIELDS, "cone_id", "params"):
            f[key][:] = f[key][:][::-1]
    dataset = MultiFieldDataset(all_mapping(), cache=path)
    second = dataset.prepare()
    assert first["split"] == second["split"]
    for name in FOUR_FIELDS:
        assert first["normalization"][name] == pytest.approx(second["normalization"][name])
    dataset.close()


def test_raw_descending_grid_matches_cache_and_does_not_zero_fill(tmp_path):
    from fno_multifield import cache_fields
    cache = tmp_path / "cache.h5"
    values = write_cache(cache)
    grid = np.linspace(5, 8, 8)
    files = []
    for row in range(5):
        path = tmp_path / f"raw{row}.h5"
        with h5py.File(path, "w") as f:
            group = f.create_group("lightcone")
            for name, field in values.items():
                group.create_dataset(name, data=field[row, ..., ::-1])
            group.create_dataset("lightcone_redshifts", data=grid[::-1])
            group = f.create_group("params")
            group.create_dataset("names", data=np.asarray(PARAM_NAMES, dtype="S"))
            group.create_dataset("values", data=np.arange(row*11, (row+1)*11))
        files.append(path)
    raw = MultiFieldDataset(all_mapping(), files=files, target_z=grid)
    cached = MultiFieldDataset(all_mapping(), cache=cache)
    raw.install_preparation(raw.prepare())
    cached.install_preparation(cached.prepare())
    for row in range(5):
        for key in ("x", "y"):
            assert torch.allclose(raw[row][key], cached[row][key], atol=1e-6)
    with pytest.raises(ValueError, match="outside"):
        MultiFieldDataset(all_mapping(), files=files, target_z=np.linspace(4, 8, 8))
    raw.close()
    cached.close()
    output = tmp_path / "rebuilt.h5"
    cache_fields(Namespace(out=output, registry=None, fields=",".join(FOUR_FIELDS),
        conditioning="z_params", cache=None, data=tmp_path, glob="raw*.h5",
        target_z=None, z_min=5, z_max=8, n_z=8))
    with h5py.File(output, "r") as f:
        for name in FOUR_FIELDS:
            assert np.allclose(f[name][:], values[name])
        assert f["params"].shape == (5, 11)


def test_bad_source_and_preparation_fail_before_training(tmp_path):
    path = tmp_path / "cache.h5"
    write_cache(path)
    dataset = MultiFieldDataset(all_mapping(), cache=path)
    prep = dataset.prepare()
    bad = deepcopy(prep)
    bad["split"]["train"].append(bad["split"]["test"][0])
    with pytest.raises(ValueError, match="overlaps"):
        dataset.install_preparation(bad)
    bad = deepcopy(prep)
    bad["normalization"]["los_velocity"]["scale"] = 0
    with pytest.raises(ValueError, match="normalization"):
        dataset.install_preparation(bad)
    dataset.close()
    with h5py.File(path, "r+") as f:
        del f["los_velocity"]
    with pytest.raises(ValueError, match="stored field"):
        MultiFieldDataset(all_mapping(), cache=path)
    dataset = MultiFieldDataset(FieldMapping(), cache=path)
    with pytest.raises(ValueError, match="source changed"):
        dataset.install_preparation(prep)
    dataset.close()


@pytest.mark.parametrize("kind", ["localop", "ufno", "fno", "sirenfno"])
def test_architectures_support_multiple_outputs_and_gradients(kind):
    torch.manual_seed(5)
    config = small_config(kind)
    mapping = FieldMapping.create("density", "neutral_fraction,los_velocity", "none")
    model = MultiFieldModel(config, 1, mapping)
    x = torch.randn(1, 1, 8, 8, 8)
    output = model(x)
    assert output.shape == (1, 2, 8, 8, 8)
    assert ((output[:, 0] >= 0) & (output[:, 0] <= 1)).all()
    loss = weighted_objective(output, torch.zeros_like(output), torch.ones(2))
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_signed_field_is_not_sigmoid_clipped():
    mapping = FieldMapping.create("density", "neutral_fraction,brightness_temp", "none")
    model = MultiFieldModel(small_config(), 1, mapping)
    with torch.no_grad():
        model.backbone.projection[-1].weight.zero_()
        model.backbone.projection[-1].bias[:] = torch.tensor([0.0, -7.0])
    result = model(torch.zeros(1, 1, 8, 8, 8))
    assert torch.all(result[:, 0] == 0.5)
    assert torch.all(result[:, 1] == -7)


def test_default_factory_keeps_historical_checkpoint_and_initialization():
    config = small_config()
    torch.manual_seed(9)
    old = build_model(config, 2)
    torch.manual_seed(9)
    explicit = build_model(config, 2, out_channels=1, output_sigmoid=True)
    assert old.state_dict().keys() == explicit.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])


def test_physical_metrics_spectra_and_zero_variance_baseline():
    normalization = {"los_velocity": {"offset": 5, "scale": 2, "train_mean": 5}}
    metrics = FieldMetrics(("los_velocity",), normalization, spectral_bins=4)
    y = torch.randn(2, 1, 8, 8, 8)
    metrics.update(y+1, y)
    result = metrics.result()["los_velocity"]
    assert result["rmse"] == pytest.approx(2)
    assert result["normalized_mse"] == pytest.approx(1)
    assert result["mean_bias"] == pytest.approx(2)
    assert result["pearson_r"] == pytest.approx(1)
    assert result["transverse_spectrum"]["power_ratio"] == pytest.approx([1]*4)
    assert result["transverse_spectrum"]["cross_correlation"] == pytest.approx([1]*4)
    constant = FieldMetrics(("los_velocity",), normalization)
    constant.update(torch.zeros_like(y), torch.zeros_like(y))
    assert constant.result()["los_velocity"]["mse_skill_vs_train_mean"] is None
    assert constant.result()["los_velocity"]["pearson_r"] is None


def test_cli_training_checkpoint_export_and_sweep(tmp_path):
    """Real CLI run on small cubes; reload and export reproduce predictions."""
    from fno_multifield import main, restore
    from util.multifield_experiments import plan, summarize
    from unittest.mock import patch

    path = tmp_path / "cache.h5"
    write_cache(path)
    preparation = tmp_path / "preparation.json"
    def cli(*args):
        with patch.object(sys, "argv", ["fno_multifield.py", *map(str, args)]):
            main()
    cli("prepare", "--cache", path, "--out", preparation)
    run_dir = tmp_path / "run"
    cli("train", "--preparation", preparation, "--targets", "neutral_fraction,los_velocity",
        "--run-dir", run_dir, "--epochs", 2, "--spectral-bins", 4,
        "--model-settings", json.dumps(small_config().to_dict()), "--device", "cpu")
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["training"]["backbone_training"] == "from_scratch"
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 2
    test = json.loads((run_dir / "test_metrics.json").read_text())
    assert set(test["fields"]) == {"neutral_fraction", "los_velocity"}
    export = tmp_path / "prediction.h5"
    cli("predict", "--checkpoint", run_dir / "best.pt", "--cone-id", 0, "--out", export)
    _, model, dataset, _ = restore(run_dir / "best.pt", torch.device("cpu"))
    expected = model(dataset[0]["x"][None]).detach().numpy()[0]
    with h5py.File(export, "r") as f:
        for i, name in enumerate(dataset.mapping.targets):
            stats = dataset.normalization[name]
            assert np.allclose(f[f"prediction/{name}"][:], expected[i]*stats["scale"]+stats["offset"])
    dataset.close()
    # A second run must fail without modifying a completed run's metadata.
    before = (run_dir / "run_metadata.json").read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        cli("train", "--preparation", preparation, "--run-dir", run_dir,
            "--model-settings", json.dumps(small_config().to_dict()))
    assert (run_dir / "run_metadata.json").read_bytes() == before
    manifest = tmp_path / "plan.json"
    plan(Namespace(out=manifest, preparation=preparation, fields=",".join(FOUR_FIELDS),
                   stage="all", seeds=[42], epochs=2, batch_size=1, workers=0,
                   model_config=None, run_root=None))
    entries = json.loads(manifest.read_text())["entries"]
    assert len(entries) == 50
    assert all("--targets" in entry["argv"] for entry in entries)
    summarize(Namespace(plan=manifest, out=tmp_path / "summary"))
    status = json.loads((tmp_path / "summary/status.json").read_text())
    assert len(status["incomplete_job_indices"]) == 50


def test_summary_pairs_only_matched_single_target_baselines(tmp_path):
    from util.multifield_experiments import summarize
    import csv
    training = {"epochs": 20, "seed": 42, "batch_size": 1, "learning_rate": 0.001,
                "weight_decay": 0, "grad_clip": 1, "deterministic": False,
                "backbone_training": "from_scratch", "monitor": "mean"}
    metadata = {"status": "complete", "training": training, "preparation": {"id": "same"},
                "mapping": {"inputs": ["density"], "targets": ["neutral_fraction"],
                            "conditioning": "z_params"}, "model_config": {"kind": "localop"}}
    entries = []
    for index, (targets, error, seed) in enumerate([
        (["neutral_fraction"], 0.4, 42),
        (["neutral_fraction", "los_velocity"], 0.3, 42),
        (["neutral_fraction", "los_velocity"], 0.2, 43),
    ]):
        directory = tmp_path / f"run{index}"
        directory.mkdir()
        meta = deepcopy(metadata)
        meta["mapping"]["targets"] = targets
        meta["training"]["seed"] = seed
        (directory / "run_metadata.json").write_text(json.dumps(meta))
        (directory / "test_metrics.json").write_text(json.dumps({"fields": {
            "neutral_fraction": {"normalized_mse": error, "rmse": error**0.5}}}))
        entries.append({"index": index, "run_dir": str(directory)})
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"entries": entries}))
    output = tmp_path / "summary"
    summarize(Namespace(plan=plan, out=output))
    with (output / "per_seed.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert float(rows[1]["auxiliary_mse_gain"]) == pytest.approx(0.25)
    assert rows[2]["auxiliary_mse_gain"] == ""
