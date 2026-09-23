"""Native lightcone slabs, random training windows and deterministic tiling.

Only raw native arrays are accepted: a redshift-resampled cache cannot recover
the missing cells. Indices are exposed in increasing-redshift order; physical
distances and all fields follow the same reversal when needed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.fields import FieldMapping, FieldRegistry
from dataset.lightcone_params import PARAM_NAMES, read_sampled_params
from dataset.multifield import MultiFieldDataset


@dataclass(frozen=True)
class LOSWindowConfig:
    mode: str = "contiguous"
    size: int = 256
    halo: int = 32
    windows_per_cone: int = 8
    context_factor: int = 4
    context_xy: int = 4
    context_features: int = 8

    def __post_init__(self):
        if self.mode not in {"contiguous", "coarse_context"}:
            raise ValueError("window mode must be contiguous or coarse_context")
        for name in ("size", "halo", "windows_per_cone", "context_factor",
                     "context_xy", "context_features"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.size < 2 or self.halo < 0 or 2*self.halo >= self.size:
            raise ValueError("window size must exceed twice the nonnegative halo")
        if min(self.windows_per_cone, self.context_xy, self.context_features) < 1:
            raise ValueError("window count, context pooling and features must be positive")
        if self.mode == "coarse_context" and (self.context_factor < 2 or self.size*(self.context_factor-1) % 2):
            raise ValueError("context factor must be >=2 and allow a centered integer crop")

    @property
    def core(self):
        return self.size - 2*self.halo

    def to_dict(self):
        return asdict(self)


class NativeLightconeDataset(MultiFieldDataset):
    """One source row per simulation, with variable native LOS lengths.

    Splits and normalization reuse the multi-field preparation contract. The
    native grids are validated but never interpolated. Statistics stream slabs
    so preparation does not allocate a full lightcone in float64.
    """
    def __init__(self, mapping=None, *, files, registry=None):
        self.registry = registry or FieldRegistry()
        mapping = mapping or FieldMapping()
        self.mapping = FieldMapping.create(mapping.inputs, mapping.targets,
                                           mapping.conditioning, self.registry)
        if not self.mapping.use_redshift:
            raise ValueError("native windows require absolute redshift conditioning (z or z_params)")
        self.file_paths = sorted(Path(p).resolve() for p in files)
        if not self.file_paths or len(set(self.file_paths)) != len(self.file_paths):
            raise ValueError("raw source needs a nonempty list of distinct lightcones")
        self.cache_path = None
        self._h5 = self._pid = None
        self.normalization = self.parameter_normalization = None
        self.cone_ids = np.arange(len(self.file_paths), dtype=np.int64)
        self.keys, self.redshifts, self.distances, self.reversed = [], [], [], []
        rows = []
        for path in self.file_paths:
            with h5py.File(path, "r") as f:
                group = f["lightcone"]
                keys = self._resolve_fields(group)
                z = np.asarray(group["lightcone_redshifts"], dtype=np.float64)
                chi = np.asarray(group["lightcone_distances"], dtype=np.float64)
                self._validate_grid(z, increasing=False)
                if (chi.shape != z.shape or not np.isfinite(chi).all()
                        or not (np.all(np.diff(chi) > 0) or np.all(np.diff(chi) < 0))):
                    raise ValueError(f"invalid native comoving distances in {path.name}")
                step = np.abs(np.diff(chi))
                if not np.allclose(step, step.mean(), rtol=1e-4, atol=1e-6):
                    raise ValueError(f"native LOS must be uniform in comoving distance: {path.name}")
                shapes = [group[k].shape for k in keys.values()]
                if any(len(s) != 3 or s != shapes[0] or s[-1] != len(z) for s in shapes):
                    raise ValueError(f"fields are not aligned scalar cubes in {path.name}")
                xy = shapes[0][:2]
                if hasattr(self, "transverse_shape") and self.transverse_shape != xy:
                    raise ValueError("lightcones have different transverse grids")
                self.transverse_shape = xy
                reverse = bool(z[0] > z[-1])
                self.keys.append(keys)
                self.reversed.append(reverse)
                self.redshifts.append(z[::-1].copy() if reverse else z)
                self.distances.append(chi[::-1].copy() if reverse else chi)
                if self.mapping.use_params:
                    rows.append(read_sampled_params(f))
        self.params = np.stack(rows) if rows else None
        if self.mapping.use_params and (self.params.shape != (len(self), len(PARAM_NAMES))
                                         or not np.isfinite(self.params).all()):
            raise ValueError("missing or nonfinite simulation parameters")
        self.channel_names = (*self.mapping.inputs, "1/(1+z)",
                              *(PARAM_NAMES if self.mapping.use_params else ()),
                              "relative_los_Mpc/1000", "native_valid")
        self.in_channels = len(self.channel_names)
        self.out_channels = len(self.mapping.targets)

    def source_description(self):
        return {"kind": "native", "files": [
            {"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in self.file_paths], "cone_ids": self.cone_ids.tolist(),
            "transverse_shape": self.transverse_shape,
            "native_lengths": [len(z) for z in self.redshifts],
            "axis_order": "increasing_redshift", "distance_units": "comoving Mpc"}

    def read_fields(self, idx, start=0, stop=None, names=None):
        """Read a contiguous HDF5 slab; only the requested fields are loaded."""
        n = len(self.redshifts[idx])
        stop = n if stop is None else stop
        if not 0 <= start < stop <= n:
            raise ValueError("native slab lies outside the cone")
        a, b = (n-stop, n-start) if self.reversed[idx] else (start, stop)
        result = {}
        with h5py.File(self.file_paths[idx], "r") as f:
            for name in (self.mapping.fields if names is None else names):
                value = np.asarray(f["lightcone"][self.keys[idx][name]][..., a:b], dtype=np.float32)
                if self.reversed[idx]:
                    value = value[..., ::-1].copy()
                self._validate_values(name, value, idx)
                result[name] = value
        return result

    def iter_field_blocks(self, idx):
        n = len(self.redshifts[idx])
        for start in range(0, n, 256):
            yield self.read_fields(idx, start, min(start+256, n))

    def _region(self, idx, start, size, names, pool=(1, 1, 1)):
        """Normalize, edge-pad, then box-filter and decimate before stacking.

        Conditioning is pooled on the identical grid. Padding is explicitly
        marked and never contributes to supervised loss or evaluation.
        """
        if self.normalization is None:
            raise RuntimeError("install train-only normalization before requesting samples")
        n = len(self.redshifts[idx])
        a, b = max(0, start), min(n, start+size)
        if a >= b:
            raise ValueError("window has no native cells")
        left, right = max(0, -start), max(0, start+size-n)
        shape = (*self.transverse_shape, size)
        pooled_shape = tuple(s//k for s, k in zip(shape, pool))
        if any(s % k for s, k in zip(shape, pool)):
            raise ValueError("context pooling must divide each spatial dimension")
        fields = self.read_fields(idx, a, b, names)
        values = []
        for name in names:
            stats = self.normalization[name]
            v = (fields.pop(name) - stats["offset"]) / stats["scale"]
            v = np.pad(v, ((0, 0), (0, 0), (left, right)), mode="edge")
            v = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
            if pool != (1, 1, 1):
                v = F.avg_pool3d(v[None, None], pool, stride=pool)[0, 0]
            values.append(v)
        indices = np.arange(start, start+size)
        z = self.redshifts[idx][indices.clip(0, n-1)]
        valid = ((indices >= 0) & (indices < n)).astype(np.float32)
        # Relative physical positions keep fixed units, rather than normalizing
        # every physical extent independently to [0, 1].
        step = self.distances[idx][1] - self.distances[idx][0]
        relative = (np.arange(size) - (size-1)/2) * step / 1000
        def los_channel(v):
            v = np.asarray(v, dtype=np.float32).reshape(-1, pool[-1]).mean(axis=1)
            return torch.from_numpy(v).view(1, 1, -1).expand(pooled_shape)
        conditioning = [los_channel(1/(1+z))]
        if self.mapping.use_params:
            conditioning.extend(torch.full(pooled_shape, float(v))
                for v in self.parameter_normalization.normalize(self.params[idx]))
        conditioning.extend((los_channel(relative), los_channel(valid)))
        return values, conditioning

    def window(self, idx, start, config):
        fields, conditioning = self._region(idx, start, config.size, self.mapping.fields)
        by_name = dict(zip(self.mapping.fields, fields))
        sample = {"x": torch.stack([by_name[n] for n in self.mapping.inputs] + conditioning),
                  "y": torch.stack([by_name[n] for n in self.mapping.targets])}
        indices = np.arange(start, start+config.size)
        valid = (indices >= 0) & (indices < len(self.redshifts[idx]))
        valid[:config.halo] = False
        if config.halo:
            valid[-config.halo:] = False
        sample["loss_mask"] = torch.from_numpy(valid).view(1, 1, 1, -1)
        if config.mode == "coarse_context":
            extra = config.size*(config.context_factor-1)//2
            values, conditioning = self._region(idx, start-extra,
                config.size*config.context_factor, self.mapping.inputs,
                (config.context_xy, config.context_xy, config.context_factor))
            sample["context"] = torch.stack(values + conditioning)
        return sample

    def __getitem__(self, idx):
        raise RuntimeError("native cones have variable lengths; use LOSWindowDataset or tiled inference")


AUGMENTATIONS = ("none", "transverse")
# Fields that are vector components across the sky would change sign or swap
# under transverse flips/rotations; none are in the registry today.
TRANSVERSE_VECTOR_FIELDS = frozenset({"velocity_x", "velocity_y"})


def transverse_augment(sample, rng, context_xy=1):
    """Random element of the transverse symmetry group, applied consistently.

    The transverse planes of a 21cmFAST lightcone are the periodic box faces,
    and every field used here is a scalar or the LOS velocity component, so
    periodic shifts, 90-degree rotations and reflections in (x, y) are exact
    symmetries of the data distribution. The LOS axis is left untouched: it
    carries redshift evolution and the loss mask.

    The coarse context (pooled by ``context_xy`` transversally) receives the
    same rotation/reflection and a shift of ``shift/context_xy`` cells; fine
    shifts are restricted to multiples of ``context_xy`` so both grids stay
    aligned block for block.
    """
    nx, ny = sample["x"].shape[1:3]
    step = context_xy if "context" in sample else 1
    quarter_turns = int(rng.integers(4)) if nx == ny else 2*int(rng.integers(2))
    reflect = bool(rng.integers(2))
    shift = (int(rng.integers(nx//step))*step, int(rng.integers(ny//step))*step)

    def apply(t, scale):
        t = torch.rot90(t, quarter_turns, dims=(1, 2))
        if reflect:
            t = torch.flip(t, dims=(1,))
        return torch.roll(t, (shift[0]//scale, shift[1]//scale), dims=(1, 2)).contiguous()

    out = dict(sample)
    out["x"], out["y"] = apply(sample["x"], 1), apply(sample["y"], 1)
    if "context" in sample:
        out["context"] = apply(sample["context"], step)
    return out


class LOSWindowDataset(Dataset):
    """Equal draws per cone, reproducible across worker counts and epochs."""
    def __init__(self, source, rows, config, seed=0, augment="none"):
        self.source, self.rows, self.config = source, tuple(rows), config
        self.seed, self.epoch = int(seed), 0
        self.augment = augment
        if self.seed < 0 or not self.rows:
            raise ValueError("need a nonnegative sampling seed and a nonempty split")
        if config.mode == "coarse_context" and any(n % config.context_xy for n in source.transverse_shape):
            raise ValueError("context_xy must divide the native transverse dimensions")
        if augment not in AUGMENTATIONS:
            raise ValueError(f"augment must be one of {AUGMENTATIONS}")
        if augment == "transverse" and TRANSVERSE_VECTOR_FIELDS & set(source.mapping.fields):
            raise ValueError("transverse augmentation would corrupt transverse vector fields")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)*self.config.windows_per_cone

    def __getitem__(self, index):
        row = self.rows[index // self.config.windows_per_cone]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        n = len(self.source.redshifts[row])
        # Uniformly sample core centers, including cone boundaries. This also
        # trains the explicit edge-padding behavior used during reconstruction.
        center = int(rng.integers(n))
        start = center - self.config.core//2 - self.config.halo
        sample = self.source.window(row, start, self.config)
        # Drawn after the center, so window positions match unaugmented runs.
        if self.augment == "transverse":
            sample = transverse_augment(sample, rng, self.config.context_xy)
        return sample


@torch.no_grad()
def predict_native_cone(model, dataset, row, config, device):
    """Overlap input halos; retain disjoint valid cores, including both ends.

    Only a window lives on the accelerator. The native prediction is assembled
    on CPU; each real voxel is written once and padding is never exported.
    """
    model.eval()
    n = len(dataset.redshifts[row])
    result = torch.empty((dataset.out_channels, *dataset.transverse_shape, n))
    for core_start in range(0, n, config.core):
        sample = dataset.window(row, core_start-config.halo, config)
        kwargs = ({"context": sample["context"][None].to(device)} if "context" in sample else {})
        prediction = model(sample["x"][None].to(device), **kwargs)[0]
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("nonfinite model predictions")
        count = min(config.core, n-core_start)
        result[..., core_start:core_start+count] = prediction[..., config.halo:config.halo+count].cpu()
    return result
