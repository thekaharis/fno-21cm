"""Aligned scalar lightcone fields with reusable, train-only normalization."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.dataset_3d import ParameterNormalization, make_file_split
from dataset.fields import FieldMapping, FieldRegistry
from dataset.lightcone_params import PARAM_NAMES, read_sampled_params


class MultiFieldDataset(Dataset):
    """Read a cube cache or raw lightcones, always returning physical field data.

    Normalization is installed from a preparation artifact before tensor samples
    are requested. All selected fields must be finite scalar cubes on one grid.
    Missing fields and out-of-range interpolation fail instead of silently
    changing the sample population or inserting physical zeros.
    """

    def __init__(self, mapping=None, *, cache=None, files=None, target_z=None, registry=None):
        self.registry = registry or FieldRegistry()
        mapping = mapping or FieldMapping()
        self.mapping = FieldMapping.create(mapping.inputs, mapping.targets,
                                           mapping.conditioning, self.registry)
        if (cache is None) == (files is None):
            raise ValueError("provide exactly one of cache or files")
        self.cache_path = Path(cache).resolve() if cache is not None else None
        self.file_paths = sorted(Path(p).resolve() for p in files) if files is not None else []
        self._h5 = None
        self._pid = None
        self.normalization = None
        self.parameter_normalization = None
        self.keys = []
        self.params = None
        if self.cache_path:
            with h5py.File(self.cache_path, "r") as f:
                self.target_z = np.asarray(f["target_z"], dtype=np.float64)
                self.cone_ids = np.asarray(f["cone_id"], dtype=np.int64)
                self.keys = [self._resolve_fields(f)]
                first = f[self.keys[0][self.mapping.fields[0]]]
                if first.ndim != 4 or first.shape[0] != len(self.cone_ids):
                    raise ValueError("cached fields must have shape (cones, X, Y, Z)")
                self.spatial_shape = tuple(first.shape[1:])
                for key in self.keys[0].values():
                    if f[key].shape != first.shape:
                        raise ValueError(f"field {key} has a different grid/sample count")
                if self.mapping.use_params:
                    self.params = np.asarray(f["params"], dtype=np.float32)
                    stored = f.attrs.get("param_names", PARAM_NAMES)
                    stored = [v.decode() if isinstance(v, bytes) else str(v) for v in stored]
                    if len(stored) != self.params.shape[1] or len(set(stored)) != len(stored):
                        raise ValueError("invalid cached parameter schema")
                    self.params = self.params[:, [stored.index(n) for n in PARAM_NAMES]]
        else:
            if not self.file_paths or len(set(self.file_paths)) != len(self.file_paths):
                raise ValueError("raw source needs a nonempty list of distinct lightcones")
            self.target_z = np.asarray(target_z, dtype=np.float64)
            self.cone_ids = np.arange(len(self.file_paths), dtype=np.int64)
            rows = []
            for path in self.file_paths:
                with h5py.File(path, "r") as f:
                    group = f["lightcone"]
                    keys = self._resolve_fields(group)
                    self.keys.append(keys)
                    source_z = np.asarray(group["lightcone_redshifts"], dtype=np.float64)
                    self._validate_grid(source_z, increasing=False)
                    if self.target_z.ndim != 1 or self.target_z.size < 2:
                        raise ValueError("target_z must be a one-dimensional grid")
                    if (self.target_z.min() < source_z.min() - 1e-6
                            or self.target_z.max() > source_z.max() + 1e-6):
                        raise ValueError(f"requested redshift range is outside {path.name}")
                    shapes = [group[k].shape for k in keys.values()]
                    if any(len(s) != 3 or s != shapes[0] or s[-1] != len(source_z) for s in shapes):
                        raise ValueError(f"fields are not aligned scalar cubes in {path.name}")
                    shape = (*shapes[0][:2], len(self.target_z))
                    if hasattr(self, "spatial_shape") and self.spatial_shape != shape:
                        raise ValueError("lightcones have different transverse grids")
                    self.spatial_shape = shape
                    if self.mapping.use_params:
                        rows.append(read_sampled_params(f))
            if rows:
                self.params = np.stack(rows)
        self._validate_grid(self.target_z)
        if self.spatial_shape[-1] != len(self.target_z):
            raise ValueError("field LOS dimension does not match target_z")
        if not len(self.cone_ids) or len(set(self.cone_ids)) != len(self.cone_ids):
            raise ValueError("cone IDs must be nonempty and unique")
        if self.mapping.use_params:
            if (self.params is None or self.params.shape != (len(self), len(PARAM_NAMES))
                    or not np.isfinite(self.params).all()):
                raise ValueError("missing, nonfinite or incorrectly shaped simulation parameters")
        self.channel_names = (*self.mapping.inputs,
                              *(("1/(1+z)",) if self.mapping.use_redshift else ()),
                              *(PARAM_NAMES if self.mapping.use_params else ()))
        self.in_channels = len(self.channel_names)
        self.out_channels = len(self.mapping.targets)

    @staticmethod
    def _validate_grid(grid, increasing=True):
        if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
                or np.any(grid <= -1)):
            raise ValueError("redshift grid must be finite, one-dimensional and greater than -1")
        delta = np.diff(grid)
        if not (np.all(delta > 0) or (not increasing and np.all(delta < 0))):
            raise ValueError("redshift grid must be strictly monotonic")

    def _resolve_fields(self, group):
        return {name: self.registry.resolve_key(group.keys(), name) for name in self.mapping.fields}

    def __len__(self):
        return len(self.cone_ids)

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_pid"] = None
        return state

    def read_fields(self, idx):
        if self.cache_path:
            if self._h5 is None or self._pid != os.getpid():
                self.close()
                self._h5 = h5py.File(self.cache_path, "r")
                self._pid = os.getpid()
            result = {n: np.asarray(self._h5[k][idx], dtype=np.float32)
                      for n, k in self.keys[0].items()}
        else:
            from scipy.interpolate import interp1d
            with h5py.File(self.file_paths[idx], "r") as f:
                group = f["lightcone"]
                z = np.asarray(group["lightcone_redshifts"], dtype=np.float64)
                result = {}
                for name, key in self.keys[idx].items():
                    value = np.asarray(group[key], dtype=np.float32)
                    self._validate_values(name, value, idx)
                    if z[0] > z[-1]:
                        source_z, value = z[::-1], value[..., ::-1]
                    else:
                        source_z = z
                    if np.array_equal(self.target_z, source_z):
                        result[name] = value.copy()
                        continue
                    grid = np.clip(self.target_z, source_z[0], source_z[-1])
                    result[name] = interp1d(source_z, value, axis=-1, bounds_error=True,
                                            assume_sorted=True)(grid).astype(np.float32)
        for name, value in result.items():
            self._validate_values(name, value, idx)
        return result

    def _validate_values(self, name, value, idx):
        spec = self.registry[name]
        if (not np.isfinite(value).all()
                or (spec.minimum is not None and np.any(value < spec.minimum - 1e-5))
                or (spec.maximum is not None and np.any(value > spec.maximum + 1e-5))):
            raise ValueError(f"invalid/undefined values in {name}, cone {self.cone_ids[idx]}; "
                             "sparse or sentinel-valued fields require an explicit mask policy")

    def source_description(self):
        paths = [self.cache_path] if self.cache_path else self.file_paths
        return {
            "kind": "cache" if self.cache_path else "raw",
            "files": [{"path": str(p), "size": p.stat().st_size,
                       "mtime_ns": p.stat().st_mtime_ns} for p in paths],
            "cone_ids": self.cone_ids.tolist(), "target_z": self.target_z.tolist(),
            "spatial_shape": self.spatial_shape,
        }

    def fingerprint(self):
        payload = json.dumps(self.source_description(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def prepare(self, split_seed=42, val_fraction=0.1, test_fraction=0.1, progress=None):
        if len(self) < 3 or not (0 < val_fraction < 1 and 0 < test_fraction < 1
                                and val_fraction + test_fraction < 1):
            raise ValueError("need at least three cones and valid validation/test fractions")
        # Shuffle sorted physical IDs, not cache rows; works with missing IDs.
        ids = sorted(int(i) for i in self.cone_ids)
        split = make_file_split(len(ids), split_seed, val_fraction, test_fraction)
        if any(not part for part in split):
            raise ValueError("split leaves an empty training, validation or test set")
        split_ids = {name: [ids[i] for i in part]
                     for name, part in zip(("train", "val", "test"), split)}
        row_by_id = {int(cid): row for row, cid in enumerate(self.cone_ids)}
        train_rows = [row_by_id[cid] for cid in split_ids["train"]]
        # Streaming parallel-variance merge avoids catastrophic cancellation.
        moments = {f: [0, 0.0, 0.0] for f in self.mapping.fields}
        for index, row in enumerate(train_rows):
            for name, values in self.read_fields(row).items():
                values = values.astype(np.float64)
                count, mean, m2 = moments[name]
                n = values.size
                batch_mean = float(values.mean())
                delta = batch_mean - mean
                moments[name] = [count + n, mean + delta * n / (count + n),
                                 m2 + float(values.var()) * n + delta**2 * count * n / (count + n)]
            if progress is not None and (index % 25 == 0 or index + 1 == len(train_rows)):
                progress(index + 1, len(train_rows))
        stats = {}
        for name, (count, mean, m2) in moments.items():
            mode = self.registry[name].normalization
            std = float(np.sqrt(m2 / count))
            offset, scale = (mean, std if std > 1e-8 else 1.0)
            if mode != "standard":
                offset, scale = 0.0, 10.0 if mode == "density" else 1.0
            stats[name] = {"offset": offset, "scale": scale, "train_mean": mean,
                           "train_std": std, "count": count}
        parameters = (ParameterNormalization.fit(self.params, train_rows).to_dict()
                      if self.mapping.use_params else None)
        return {"schema_version": 1, "source": self.source_description(),
                "source_fingerprint": self.fingerprint(),
                "registry": self.registry.to_dict(), "conditioning": self.mapping.conditioning,
                "split_seed": split_seed, "split": split_ids,
                "normalization": stats, "parameter_normalization": parameters}

    def install_preparation(self, preparation):
        if preparation.get("schema_version") != 1:
            raise ValueError("unsupported preparation schema")
        if preparation["source_fingerprint"] != self.fingerprint():
            raise ValueError("source changed or differs from the preparation artifact")
        if preparation["conditioning"] != self.mapping.conditioning:
            raise ValueError("conditioning differs from preparation")
        saved_registry = FieldRegistry.from_dict(preparation["registry"])
        for name in self.mapping.fields:
            if saved_registry[name] != self.registry[name]:
                raise ValueError(f"field definition changed for {name}")
            stats = preparation["normalization"].get(name)
            if (stats is None or not np.isfinite([stats["offset"], stats["scale"]]).all()
                    or stats["scale"] <= 0):
                raise ValueError(f"invalid or missing normalization for {name}")
        parts = preparation["split"]
        if set(parts) != {"train", "val", "test"} or any(not p for p in parts.values()):
            raise ValueError("preparation needs three nonempty splits")
        all_ids = [cid for part in parts.values() for cid in part]
        if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(self.cone_ids):
            raise ValueError("preparation split overlaps or does not match source cones")
        self.normalization = preparation["normalization"]
        if self.mapping.use_params:
            self.parameter_normalization = ParameterNormalization.from_dict(
                preparation["parameter_normalization"])
        row_by_id = {int(cid): row for row, cid in enumerate(self.cone_ids)}
        return {name: [row_by_id[cid] for cid in part] for name, part in parts.items()}

    def __getitem__(self, idx):
        if self.normalization is None:
            raise RuntimeError("install train-only normalization before requesting samples")
        physical = self.read_fields(idx)
        values = {name: (v - self.normalization[name]["offset"])
                  / self.normalization[name]["scale"] for name, v in physical.items()}
        channels = [values[name] for name in self.mapping.inputs]
        if self.mapping.use_redshift:
            channels.append(np.broadcast_to(1 / (1 + self.target_z), self.spatial_shape))
        if self.mapping.use_params:
            channels.extend(np.broadcast_to(v, self.spatial_shape)
                            for v in self.parameter_normalization.normalize(self.params[idx]))
        return {"x": torch.from_numpy(np.ascontiguousarray(np.stack(channels), dtype=np.float32)),
                "y": torch.from_numpy(np.ascontiguousarray(
                    np.stack([values[name] for name in self.mapping.targets]), dtype=np.float32))}
