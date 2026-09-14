#!/usr/bin/env python3
"""Prepare, train, evaluate and export configurable 3-D field mappings.

The existing x_HI entry points keep their historical defaults. This entry point
uses a common per-field normalized MSE recipe and retrains the shared backbone
for each mapping. One process/GPU per run; independent mappings can be scheduled
as a job array. See notes/multifield.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.fields import FOUR_FIELDS, FieldMapping, FieldRegistry
from dataset.lightcone_params import PARAM_NAMES
from dataset.multifield import MultiFieldDataset
from modeling import ModelConfig
from multifield_model import MultiFieldModel, weighted_objective
from util.multifield_metrics import FieldMetrics


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_dataset(args, mapping, registry):
    if args.cache:
        return MultiFieldDataset(mapping, cache=args.cache, registry=registry)
    files = sorted(Path(args.data).glob(args.glob))
    grid = (np.load(args.target_z) if args.target_z else
            np.linspace(args.z_min, args.z_max, args.n_z))
    return MultiFieldDataset(mapping, files=files, target_z=grid, registry=registry)


def prepared_dataset(preparation, mapping):
    source = preparation["source"]
    registry = FieldRegistry.from_dict(preparation["registry"])
    paths = [v["path"] for v in source["files"]]
    kwargs = ({"cache": paths[0]} if source["kind"] == "cache" else
              {"files": paths, "target_z": source["target_z"]})
    dataset = MultiFieldDataset(mapping, registry=registry, **kwargs)
    rows = dataset.install_preparation(preparation)
    return dataset, rows, registry


def prepare(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    registry = FieldRegistry.from_file(args.registry) if args.registry else FieldRegistry()
    fields = args.fields.split(",")
    mapping = FieldMapping.create(fields[:1], fields[1:], args.conditioning, registry)
    dataset = source_dataset(args, mapping, registry)
    try:
        print(f"Preparing {len(dataset)} cones, fields {mapping.fields}", flush=True)
        artifact = dataset.prepare(args.split_seed, args.val_fraction, args.test_fraction,
            progress=lambda done, total: print(f"Training statistics: {done}/{total} cones", flush=True))
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, artifact)
        print(f"Saved shared splits and training statistics to {out}")
    finally:
        dataset.close()


def cache_fields(args):
    """Create an additional cache; never rewrite a previous x_HI cache."""
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    registry = FieldRegistry.from_file(args.registry) if args.registry else FieldRegistry()
    fields = args.fields.split(",")
    mapping = FieldMapping.create(fields[:1], fields[1:], args.conditioning, registry)
    dataset = source_dataset(args, mapping, registry)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".{os.getpid()}.partial")
    try:
        with h5py.File(temporary, "x") as f:
            shape = (len(dataset), *dataset.spatial_shape)
            fields = {name: f.create_dataset(name, shape=shape, dtype="float32",
                      chunks=(1, *dataset.spatial_shape), compression="gzip", compression_opts=4)
                      for name in mapping.fields}
            for row in range(len(dataset)):
                for name, values in dataset.read_fields(row).items():
                    fields[name][row] = values
                if row % 25 == 0:
                    print(f"Cached {row+1}/{len(dataset)} cones", flush=True)
            f.create_dataset("cone_id", data=dataset.cone_ids)
            f.create_dataset("target_z", data=dataset.target_z)
            if dataset.params is not None:
                f.create_dataset("params", data=dataset.params)
                f.attrs["param_names"] = np.asarray(PARAM_NAMES, dtype="S")
            f.attrs["field_names"] = np.asarray(mapping.fields, dtype="S")
            f.attrs["source_description"] = json.dumps(dataset.source_description())
        temporary.replace(out)
        print(f"Saved {out}")
    finally:
        dataset.close()
        if temporary.exists():
            temporary.unlink()


def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def loader(dataset, rows, batch_size, workers, *, seed=0, shuffle=False):
    return DataLoader(Subset(dataset, rows), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate_model(model, batches, dataset, device, spectral_bins=0):
    model.eval()
    metrics = FieldMetrics(dataset.mapping.targets, dataset.normalization, spectral_bins)
    for batch in batches:
        x, y = batch["x"].to(device), batch["y"].to(device)
        prediction = model(x)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("nonfinite model predictions")
        metrics.update(prediction, y)
    return metrics.result()


def save_checkpoint(path, model, metadata, epoch):
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"schema_version": 1, "model": model.state_dict(),
                "metadata": metadata, "epoch": epoch}, temporary)
    temporary.replace(path)


def train(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("multi-field runs currently use one process/GPU; schedule mappings independently")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        raise ValueError("invalid training budget or learning rate")
    if args.weight_decay < 0 or args.grad_clip < 0 or args.spectral_bins < 0:
        raise ValueError("weight decay, gradient clipping and spectral bins must be nonnegative")
    preparation = read_json(args.preparation)
    registry = FieldRegistry.from_dict(preparation["registry"])
    mapping = FieldMapping.create(args.inputs, args.targets, preparation["conditioning"], registry)
    dataset, rows, registry = prepared_dataset(preparation, mapping)
    config = ModelConfig.from_dict(json.loads(args.model_settings))
    if config.ndim != 3:
        raise ValueError("multi-field lightcone training requires a 3-D model")
    weights_by_field = json.loads(args.loss_weights)
    if set(weights_by_field) - set(mapping.targets):
        raise ValueError("loss weight specified for a field outside the targets")
    weights = [float(weights_by_field.get(name, 1.0)) for name in mapping.targets]
    if not np.isfinite(weights).all() or min(weights) <= 0:
        raise ValueError("every selected target must have a finite, positive loss weight")
    if args.monitor != "mean" and args.monitor not in mapping.targets:
        raise ValueError("monitor must be mean or one of the target field names")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(args.deterministic)
    device = choose_device(args.device)
    model = MultiFieldModel(config, dataset.in_channels, mapping, registry).to(device)
    weights_tensor = torch.tensor(weights, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive lock also prevents accidental overwrites by duplicate array jobs.
    lock = out / "run.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    metadata = {
        "schema_version": 1, "mapping": mapping.to_dict(), "model_config": config.to_dict(),
        "preparation": preparation, "input_channels": list(dataset.channel_names),
        "training": {"epochs": args.epochs, "seed": args.seed, "batch_size": args.batch_size,
                     "learning_rate": args.lr, "weight_decay": args.weight_decay,
                     "grad_clip": args.grad_clip, "deterministic": args.deterministic,
                     "loss": "weighted mean of per-field normalized MSE",
                     "loss_weights": dict(zip(mapping.targets, weights)), "monitor": args.monitor,
                     "backbone_training": "from_scratch", "device": str(device)},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__), "status": "running"}
    started = False
    try:
        if any(p.name != "run.lock" for p in out.iterdir()):
            raise ValueError(f"run directory is not empty: {out}")
        write_json(out / "run_metadata.json", metadata)
        started = True
        print(f"Mapping: {mapping.slug}; device={device}; parameters={metadata['parameter_count']:,}; "
              f"train/val/test={len(rows['train'])}/{len(rows['val'])}/{len(rows['test'])}", flush=True)
        train_loader = loader(dataset, rows["train"], args.batch_size, args.workers,
                              seed=args.seed, shuffle=True)
        val_loader = loader(dataset, rows["val"], args.batch_size, args.workers)
        best = float("inf")
        start = time.monotonic()
        for epoch in range(args.epochs):
            model.train()
            total, samples = 0.0, 0
            for batch_index, batch in enumerate(train_loader):
                x, y = batch["x"].to(device), batch["y"].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                loss = weighted_objective(prediction, y, weights_tensor)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                    args.grad_clip if args.grad_clip else float("inf"), error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(x)
                samples += len(x)
                if (batch_index + 1) % 25 == 0:
                    print(f"Epoch {epoch+1}, batch {batch_index+1}/{len(train_loader)}: "
                          f"loss={total/samples:.6g}", flush=True)
            validation = evaluate_model(model, val_loader, dataset, device)
            score = (sum(validation[n]["normalized_mse"] * w for n, w in zip(mapping.targets, weights))
                     / sum(weights) if args.monitor == "mean"
                     else validation[args.monitor]["normalized_mse"])
            if score < best:
                best = score
                save_checkpoint(out / "best.pt", model, metadata, epoch)
            record = {"epoch": epoch, "train_loss": total/samples, "val_score": score,
                      "validation": validation, "learning_rate": optimizer.param_groups[0]["lr"],
                      "elapsed_seconds": time.monotonic()-start}
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            scheduler.step()
            print(f"[{mapping.slug}] epoch {epoch+1}/{args.epochs}: "
                  f"train={total/samples:.6g}, val={score:.6g}", flush=True)
        save_checkpoint(out / "final.pt", model, metadata, args.epochs-1)
        checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test = evaluate_model(model, loader(dataset, rows["test"], args.batch_size, args.workers),
                              dataset, device, args.spectral_bins)
        write_json(out / "test_metrics.json", {"checkpoint": "best.pt", "epoch": checkpoint["epoch"],
                    "split": "test", "fields": test})
        metadata.update(status="complete", best_epoch=checkpoint["epoch"],
                        elapsed_seconds=time.monotonic()-start)
        write_json(out / "run_metadata.json", metadata)
    except Exception as error:
        # Do not alter a previous run if the nonempty-directory check failed.
        if started:
            metadata.update(status="failed", error=str(error))
            write_json(out / "run_metadata.json", metadata)
        raise
    finally:
        dataset.close()
        lock.unlink()


def restore(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("expected a multi-field checkpoint")
    metadata = checkpoint["metadata"]
    preparation = metadata["preparation"]
    registry = FieldRegistry.from_dict(preparation["registry"])
    mapping = FieldMapping.create(**metadata["mapping"], registry=registry)
    dataset, rows, _ = prepared_dataset(preparation, mapping)
    model = MultiFieldModel(ModelConfig.from_dict(metadata["model_config"]),
                            dataset.in_channels, mapping, registry).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return checkpoint, model, dataset, rows


def evaluate(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    device = choose_device(args.device)
    checkpoint, model, dataset, rows = restore(args.checkpoint, device)
    try:
        values = evaluate_model(model, loader(dataset, rows[args.split], args.batch_size, args.workers),
                                dataset, device, args.spectral_bins)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, {"checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"],
                         "split": args.split, "fields": values})
    finally:
        dataset.close()


@torch.no_grad()
def predict(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    device = choose_device(args.device)
    checkpoint, model, dataset, rows = restore(args.checkpoint, device)
    try:
        matches = np.flatnonzero(dataset.cone_ids == args.cone_id)
        if not len(matches):
            raise ValueError("cone ID is absent from the prepared data")
        sample = dataset[int(matches[0])]
        prediction = model(sample["x"][None].to(device))[0].cpu().numpy()
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "x") as f:
            f.create_dataset("target_z", data=dataset.target_z)
            f.attrs["cone_id"] = args.cone_id
            f.attrs["mapping"] = json.dumps(dataset.mapping.to_dict())
            f.attrs["checkpoint"] = str(Path(args.checkpoint).resolve())
            for i, name in enumerate(dataset.mapping.targets):
                stats = dataset.normalization[name]
                for group, value in (("prediction", prediction[i]), ("target", sample["y"][i].numpy())):
                    d = f.create_dataset(f"{group}/{name}",
                        data=value*stats["scale"]+stats["offset"], compression="gzip")
                    d.attrs["units"] = dataset.registry[name].units
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("cache", cache_fields)):
        p = commands.add_parser(name)
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument("--cache")
        source.add_argument("--data")
        p.add_argument("--glob", default="21cmfast_11d_sample*.h5")
        p.add_argument("--target-z", help="optional increasing redshift grid (.npy)")
        p.add_argument("--n-z", type=int, default=256)
        p.add_argument("--z-min", type=float, default=5.0)
        p.add_argument("--z-max", type=float, default=25.0)
        p.add_argument("--fields", default=",".join(FOUR_FIELDS))
        p.add_argument("--conditioning", choices=("none", "z", "params", "z_params"), default="z_params")
        p.add_argument("--registry", help="optional complete field-registry JSON")
        p.add_argument("--out", required=True)
        if name == "prepare":
            p.add_argument("--split-seed", type=int, default=42)
            p.add_argument("--val-fraction", type=float, default=0.1)
            p.add_argument("--test-fraction", type=float, default=0.1)
        p.set_defaults(function=function)
    p = commands.add_parser("train")
    p.add_argument("--preparation", required=True)
    p.add_argument("--inputs", default="density")
    p.add_argument("--targets", default="neutral_fraction")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model-settings", default='{"kind":"localop","ndim":3}',
                   help="JSON ModelConfig; independent of architecture environment variables")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss-weights", default="{}", help='canonical field-name to positive weight JSON')
    p.add_argument("--monitor", default="mean")
    p.add_argument("--deterministic", action="store_true")
    p.set_defaults(function=train)
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--out", required=True)
    p.set_defaults(function=evaluate)
    for name in ("train", "evaluate"):
        p = commands.choices[name]
        p.add_argument("--batch-size", type=int, default=1)
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--device", default="auto")
        p.add_argument("--spectral-bins", type=int, default=12)
    p = commands.add_parser("predict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cone-id", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.set_defaults(function=predict)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
