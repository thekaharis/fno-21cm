"""Native sampling fidelity, context isolation, tiling and actual CLI training."""
from copy import deepcopy
import json
import sys
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dataset.fields import FieldMapping
from dataset.lightcone_params import PARAM_NAMES
from dataset.los_windows import (LOSWindowConfig, LOSWindowDataset, NativeLightconeDataset,
                                 predict_native_cone)
from multifield_model import MultiFieldModel, field_mse
from modeling import ModelConfig


def make_native(tmp_path, lengths=(19, 25, 17, 22, 29)):
    files = []
    for row, n in enumerate(lengths):
        path = tmp_path / f"cone{row:03}.h5"
        z = 5 + np.arange(n)*0.017 + row*0.01
        # Nontrivial native structure: adjacent cells must not be interpolated.
        density = np.broadcast_to(np.arange(n, dtype=np.float32) % 7, (8, 8, n)).copy()
        density += row*0.1
        target = density*2-1
        reverse = row % 2 == 1
        with h5py.File(path, "w") as f:
            g = f.create_group("lightcone")
            for name, values in (("density", density), ("brightness_temp", target),
                                 ("neutral_fraction", (density > 3).astype(np.float32))):
                g.create_dataset(name, data=values[..., ::-1] if reverse else values,
                                 chunks=(8, 8, min(8, n)))
            g.create_dataset("lightcone_redshifts", data=z[::-1] if reverse else z)
            chi = np.arange(n)*1.43
            g.create_dataset("lightcone_distances", data=chi[::-1] if reverse else chi)
            params = f.create_group("params")
            params.create_dataset("names", data=np.asarray(PARAM_NAMES, dtype="S"))
            params.create_dataset("values", data=np.arange(len(PARAM_NAMES))+row)
        files.append(path)
    dataset = NativeLightconeDataset(FieldMapping.create("density", "brightness_temp", "z"), files=files)
    preparation = dataset.prepare()
    rows = dataset.install_preparation(preparation)
    return dataset, preparation, rows


def tiny_config(kind="fno"):
    return ModelConfig(kind=kind, ndim=3, hidden_channels=4, n_layers=1, modes=(1, 1, 2),
        localfno_base_width=4, localfno_spectral_rank=2, localfno_window=(4, 4, 4),
        localfno_modes=(2, 2, 2), ufno_width=8, ufno_norm="groupnorm")


def test_native_fidelity_reversal_coordinates_and_training_statistics(tmp_path):
    dataset, prep, rows = make_native(tmp_path)
    config = LOSWindowConfig(size=8, halo=2, context_xy=2)
    for row in (0, 1):
        sample = dataset.window(row, 3, config)
        density = np.broadcast_to((np.arange(3, 11) % 7)+row*0.1, (8, 8, 8))
        np.testing.assert_allclose(sample["x"][0].numpy(), density/10, atol=1e-7)
        np.testing.assert_allclose(sample["x"][1, 0, 0].numpy(), 1/(1+dataset.redshifts[row][3:11]))
        stats = dataset.normalization["brightness_temp"]
        np.testing.assert_allclose(sample["y"][0].numpy()*stats["scale"]+stats["offset"],
                                   density*2-1, atol=1e-6)
        assert sample["loss_mask"].flatten().tolist() == [False, False, True, True, True, True, False, False]
        np.testing.assert_allclose(sample["x"][-2, 0, 0], (np.arange(8)-3.5)*1.43/1000)
    expected = np.concatenate([dataset.read_fields(row)["brightness_temp"].ravel() for row in rows["train"]])
    assert prep["normalization"]["brightness_temp"]["train_mean"] == pytest.approx(expected.mean(), abs=1e-6)
    assert prep["normalization"]["brightness_temp"]["count"] == expected.size
    assert sum(map(len, rows.values())) == len(dataset)
    assert not set(rows["train"]) & set(rows["test"])


@pytest.mark.parametrize("n", [3, 8, 19, 20, 21])
def test_tiling_covers_every_native_cell_once_with_correct_endpoints(tmp_path, n):
    dataset, _, _ = make_native(tmp_path, lengths=(n, n+1, n+2, n+3, n+4))
    config = LOSWindowConfig(size=8, halo=2, context_xy=2)
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]
    prediction = predict_native_cone(Identity(), dataset, 0, config, "cpu")
    np.testing.assert_array_equal(prediction[0].numpy(), dataset.read_fields(0)["density"]/10)
    for start in (-2, n-2):
        sample = dataset.window(0, start, config)
        actual = np.arange(start, start+8)
        assert not sample["loss_mask"].flatten()[torch.from_numpy((actual < 0) | (actual >= n))].any()


def test_halo_and_padding_never_contribute_to_loss():
    pred = torch.zeros(2, 2, 3, 3, 8, requires_grad=True)
    target = torch.zeros_like(pred)
    target[..., :2] = 1000
    mask = torch.zeros(2, 1, 1, 1, 8, dtype=torch.bool)
    mask[..., 2:6] = True
    assert torch.equal(field_mse(pred, target, mask), torch.zeros(2))
    field_mse(pred, target+1, mask).sum().backward()
    assert torch.count_nonzero(pred.grad[..., :2]) == 0
    assert torch.count_nonzero(pred.grad[..., 2:6]) > 0


def test_random_draws_reproduce_across_workers_and_change_each_epoch(tmp_path):
    dataset, _, rows = make_native(tmp_path)
    config = LOSWindowConfig(size=8, halo=2, windows_per_cone=3, context_xy=2)
    windows = LOSWindowDataset(dataset, rows["train"], config, seed=42)
    a = [b["x"] for b in DataLoader(windows, batch_size=2, num_workers=0)]
    b = [b["x"] for b in DataLoader(windows, batch_size=2, num_workers=2)]
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    windows.set_epoch(1)
    c = [b["x"] for b in DataLoader(windows, batch_size=2, num_workers=0)]
    assert any(not torch.equal(x, y) for x, y in zip(a, c))
    windows.set_epoch(0)
    assert torch.equal(windows[0]["x"], a[0][0])


def test_coarse_context_is_box_averaged_and_target_free(tmp_path):
    dataset, _, _ = make_native(tmp_path)
    config = LOSWindowConfig(mode="coarse_context", size=8, halo=2, context_factor=2, context_xy=2)
    sample = dataset.window(0, 6, config)
    # The surrounding region is [2,18), pooled by two in all three axes.
    native = torch.from_numpy(dataset.read_fields(0, 2, 18, ("density",))["density"]/10)
    expected = torch.nn.functional.avg_pool3d(native[None, None], 2)[0, 0]
    assert torch.equal(sample["context"][0], expected)
    assert sample["context"].shape == (dataset.in_channels, 4, 4, 8)
    assert sample["context"][-1].min() == 1
    with h5py.File(dataset.file_paths[0], "r+") as f:
        f["lightcone/brightness_temp"][:] = 999
    changed = dataset.window(0, 6, config)
    assert torch.equal(changed["context"], sample["context"])
    assert torch.equal(changed["x"], sample["x"])
    assert not torch.equal(changed["y"], sample["y"])


@pytest.mark.parametrize("kind", ["fno", "localop", "ufno", "sirenfno"])
def test_surrounding_input_affects_predictions_and_receives_gradients(tmp_path, kind):
    dataset, _, _ = make_native(tmp_path)
    config = LOSWindowConfig(mode="coarse_context", size=8, halo=2, context_factor=2,
                             context_xy=2, context_features=2)
    sample = dataset.window(0, 6, config)
    torch.manual_seed(7)
    model = MultiFieldModel(tiny_config(kind), dataset.in_channels, dataset.mapping, window_config=config)
    context = sample["context"][None].clone().requires_grad_()
    out = model(sample["x"][None], context)
    out.square().mean().backward()
    assert context.grad[..., 0].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.context_encoder.parameters())
    changed = context.detach().clone()
    changed[:, 0, ..., 0] += 5
    assert not torch.allclose(out, model(sample["x"][None], changed))
    with pytest.raises(ValueError, match="requires"):
        model(sample["x"][None])


@pytest.mark.parametrize("mode", ["contiguous", "coarse_context"])
def test_native_cli_train_restore_evaluate_export_and_sweep(tmp_path, mode):
    from fno_multifield import main, restore
    from util.multifield_experiments import main as sweep_main
    dataset, _, _ = make_native(tmp_path)
    preparation = tmp_path / "prep.json"
    def cli(*args):
        with patch.object(sys, "argv", ["fno_multifield.py", *map(str, args)]):
            main()
    cli("prepare", "--data", tmp_path, "--glob", "cone*.h5", "--native-los",
        "--fields", "density,brightness_temp,neutral_fraction", "--conditioning", "z", "--out", preparation)
    run = tmp_path / "run"
    cli("train", "--preparation", preparation, "--targets", "brightness_temp,neutral_fraction",
        "--sampling", mode, "--window-size", 8, "--window-halo", 2, "--windows-per-cone", 2,
        "--context-factor", 2, "--context-xy", 2, "--context-features", 2,
        "--epochs", 1, "--batch-size", 2, "--run-dir", run, "--device", "cpu", "--spectral-bins", 2,
        "--model-settings", json.dumps(tiny_config().to_dict()))
    meta = json.loads((run / "run_metadata.json").read_text())
    assert meta["status"] == "complete"
    assert meta["sampling"]["mode"] == mode
    checkpoint, model, restored, rows = restore(run / "best.pt", "cpu")
    config = LOSWindowConfig(**checkpoint["metadata"]["sampling"])
    expected = predict_native_cone(model, restored, 1, config, "cpu")
    out = tmp_path / "export.h5"
    cli("predict", "--checkpoint", run / "best.pt", "--cone-id", 1, "--out", out)
    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["target_z"][:], dataset.redshifts[1])
        np.testing.assert_array_equal(f["lightcone_distances"][:], dataset.distances[1])
        for i, name in enumerate(restored.mapping.targets):
            stats = restored.normalization[name]
            np.testing.assert_allclose(f[f"prediction/{name}"][:], expected[i]*stats["scale"]+stats["offset"])
            np.testing.assert_allclose(f[f"target/{name}"][:], restored.read_fields(1)[name], atol=1e-6)
    report = tmp_path / "evaluate.json"
    cli("evaluate", "--checkpoint", run / "best.pt", "--out", report, "--spectral-bins", 2)
    assert json.loads(report.read_text())["fields"] == json.loads((run/"test_metrics.json").read_text())["fields"]
    sampling_file = tmp_path / "sampling.json"
    sampling_file.write_text(json.dumps(meta["sampling"]))
    plan = tmp_path / "plan.json"
    with patch.object(sys, "argv", ["sweep", "plan", "--preparation", str(preparation),
        "--fields", "density,brightness_temp", "--sampling-config", str(sampling_file), "--out", str(plan)]):
        sweep_main()
    entry = json.loads(plan.read_text())["entries"][0]
    assert entry["sampling"] == meta["sampling"]
    assert entry["argv"][entry["argv"].index("--sampling")+1] == mode


def test_invalid_geometry_sampling_and_incompatible_preparation_fail(tmp_path):
    from argparse import Namespace
    from fno_multifield import window_configuration
    dataset, _, _ = make_native(tmp_path)
    with pytest.raises(ValueError, match="requires"):
        window_configuration(Namespace(sampling="full"), dataset)
    with pytest.raises(ValueError, match="twice"):
        LOSWindowConfig(size=8, halo=4)
    with pytest.raises(ValueError, match="divide"):
        LOSWindowDataset(dataset, [0], LOSWindowConfig(mode="coarse_context", context_xy=3))
    with h5py.File(dataset.file_paths[0], "r+") as f:
        f["lightcone/lightcone_distances"][3] += 0.1
    with pytest.raises(ValueError, match="uniform"):
        NativeLightconeDataset(dataset.mapping, files=dataset.file_paths)


def test_native_multiple_inputs_and_parameter_channels_share_normalization(tmp_path):
    initial, _, _ = make_native(tmp_path)
    dataset = NativeLightconeDataset(FieldMapping.create("brightness_temp,density", "neutral_fraction", "z_params"),
                                     files=initial.file_paths)
    preparation = dataset.prepare()
    dataset.install_preparation(preparation)
    config = LOSWindowConfig(mode="coarse_context", size=8, halo=2, context_xy=2, context_factor=2)
    sample = dataset.window(0, 6, config)
    params = dataset.parameter_normalization.normalize(dataset.params[0])
    assert dataset.channel_names[:3] == ("brightness_temp", "density", "1/(1+z)")
    assert dataset.in_channels == 16
    for i, value in enumerate(params):
        assert torch.all(sample["x"][3+i] == value)
        assert torch.all(sample["context"][3+i] == value)
    expected = initial.read_fields(0, 6, 14)["density"]/10
    np.testing.assert_array_equal(sample["x"][1], expected)
    # The preparation can be reused with field roles reversed, without refitting.
    reverse = NativeLightconeDataset(FieldMapping.create("neutral_fraction", "brightness_temp,density", "z_params"),
                                     files=initial.file_paths)
    reverse.install_preparation(preparation)
    other = reverse.window(0, 6, config)
    assert torch.equal(other["y"], sample["x"][:2])


def test_comparison_never_pairs_different_sampling_recipes():
    from util.multifield_experiments import comparison_key
    base = {"training": {k: 1 for k in ("epochs", "seed", "batch_size", "learning_rate", "weight_decay",
             "grad_clip", "deterministic", "backbone_training", "monitor")},
             "mapping": {"inputs": ["density"], "conditioning": "z"}, "model_config": {}, "preparation": {}}
    fine = deepcopy(base)
    fine["sampling"] = LOSWindowConfig().to_dict()
    coarse = deepcopy(fine)
    coarse["sampling"]["mode"] = "coarse_context"
    assert len({comparison_key(v, "brightness_temp") for v in (base, fine, coarse)}) == 3
