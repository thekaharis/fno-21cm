"""Load a trained 2-D x_HI run and gather predictions on a set of slices.

Every contrast/sharpness probe needs the same three steps -- rebuild the model
from its recorded config, open the slice cache it was trained against, and run
inference on a chosen subset -- so they live here once.

Typical use::

    from legacy.xhi2d.slice_eval import gather, cone_split
    s = gather("checkpoints/2d_xhi/fno_whno/xhi2d_whno_glob_lr3e4", split="test", max_slices=512)
    fit, held = cone_split(s.cone, seed=1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dataset import paths
from dataset.slices import SliceCache
from dataset.dataset_3d import ParameterNormalization
from modeling import load_checkpoint
from viz.compare_xhi2d_models import build_model

BATCH = 32


@dataclass
class RunSlices:
    """Model output and truth for a set of slices, all on one device."""

    pred: torch.Tensor          # (N, H, W)
    truth: torch.Tensor         # (N, H, W)
    xhi: torch.Tensor           # (N,) mean neutral fraction of the truth
    z: torch.Tensor             # (N,) redshift
    cone: np.ndarray            # (N,) cone id, for grouped splits/bootstraps
    index: np.ndarray           # (N,) row in the slice cache

    def __len__(self) -> int:
        return len(self.pred)

    def subset(self, mask) -> "RunSlices":
        m = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
        t = torch.as_tensor(m, device=self.pred.device)
        return RunSlices(self.pred[t], self.truth[t], self.xhi[t], self.z[t],
                         self.cone[m], self.index[m])


def open_run(run_dir, checkpoint: str = "final_model_state_dict.pt",
             device: str = "cuda", cache_file=None):
    """Rebuild a run's model and its slice cache from recorded metadata."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    model = build_model(meta["model_config"])
    load_checkpoint(model, str(run_dir / checkpoint))
    model = model.to(device).eval()
    norm = meta.get("parameter_normalization")
    # Runs from before the datasets moved record an absolute path in the old
    # project root; fall back to the same filename under data/compressed.
    recorded = Path(meta["dataset"]["cache_file"])
    if cache_file is None and not recorded.exists():
        moved = paths.compressed(recorded.name)
        if not moved.exists():
            raise FileNotFoundError(
                f"cache {recorded} not found, and no {moved}")
        recorded = moved
    cache = SliceCache(
        cache_file or recorded,
        input_features=meta["input_features"]["name"],
        parameter_normalization=(ParameterNormalization.from_dict(norm)
                                 if norm else None),
    )
    return model, meta, cache


def split_indices(meta, cache, split: str = "test") -> np.ndarray:
    """Cache rows belonging to one split of the run's cone partition."""
    key = {"train": "train_cone_ids", "val": "val_cone_ids",
           "test": "test_cone_ids"}[split]
    cones = np.asarray(meta["split"][key], dtype=np.int64)
    return np.flatnonzero(np.isin(cache.cone_id, cones))


def gather(run_dir, split: str = "test", max_slices: int | None = None,
           indices=None, seed: int = 0, device: str = "cuda",
           checkpoint: str = "final_model_state_dict.pt",
           cache_file=None, xhi_range: tuple[float, float] | None = None,
           model_meta_cache=None) -> RunSlices:
    """Run the model over selected slices and return predictions with truth.

    ``xhi_range`` prescans the targets (cheap -- no model) and keeps only slices
    whose mean neutral fraction falls in the range, so the model runs on the
    subset alone.  Applied before ``max_slices``.
    """
    model, meta, cache = (model_meta_cache or
                          open_run(run_dir, checkpoint, device, cache_file))
    if indices is None:
        indices = split_indices(meta, cache, split)
    indices = np.asarray(indices)

    if xhi_range is not None:
        lo, hi = xhi_range
        means = np.array([float(cache[int(i)]["y"].mean()) for i in indices],
                         dtype=np.float32)
        indices = indices[(means >= lo) & (means <= hi)]
    if max_slices is not None and len(indices) > max_slices:
        indices = np.random.default_rng(seed).choice(
            indices, size=max_slices, replace=False)
    indices = np.sort(indices)

    preds, truths, zs = [], [], []
    with torch.inference_mode():
        for s in range(0, len(indices), BATCH):
            chunk = [cache[int(i)] for i in indices[s:s + BATCH]]
            xb = torch.stack([c["x"] for c in chunk]).to(device)
            preds.append(model(xb)[:, 0].float())
            truths.append(torch.stack([c["y"] for c in chunk]).to(device)[:, 0].float())
            # channel 1 is 1/(1+z) and is constant across a slice
            zs.append(1.0 / xb[:, 1].mean(dim=(-2, -1)).float() - 1.0)
    pred = torch.cat(preds)
    truth = torch.cat(truths)
    return RunSlices(pred, truth, truth.mean(dim=(-2, -1)), torch.cat(zs),
                     np.asarray(cache.cone_id[indices]), indices)


def cone_split(cone: np.ndarray, seed: int = 1, frac: float = 0.5):
    """Boolean (fit, held-out) masks that never split a cone across both.

    Slices from one lightcone are correlated; splitting by slice would leak.
    """
    uc = np.unique(cone)
    perm = np.random.default_rng(seed).permutation(len(uc))
    keep = set(uc[perm[: int(round(frac * len(uc)))]].tolist())
    fit = np.array([int(c) in keep for c in cone])
    return fit, ~fit
