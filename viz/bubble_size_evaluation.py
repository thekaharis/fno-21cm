#!/usr/bin/env python3
"""Transverse mean-free-path bubble-size evaluation for x_HI lightcones.

This diagnostic measures ionized-bubble sizes by launching isotropic rays from
uniformly sampled ionized points and recording the distance to the first
neutral cell. It is the standard mean-free-path (MFP) construction used by
21-cm analysis packages, implemented locally to keep the project independent
of tools21cm.

Lightcone-specific design
-------------------------
The LOS axis is uniform in redshift and mixes geometry with evolution, so a
single 3-D MFP distribution is not physically meaningful. This evaluator uses
periodic 2-D transverse slices (the true 200 Mpc spatial planes) and aggregates
them by the truth slice's mean x_HI. Truth determines the stage and selected
slices for both fields, ensuring paired model comparisons.

Rays that do not hit neutral gas within one transverse box length are retained
as a censored overflow probability. Distribution distances use exact periodic
cell-boundary traversal rather than finite ray-marching steps.
Slices where a prediction contains no ionized pixels are retained as a
zero-distance underflow category, preventing failed predictions from dropping
out of the aggregate score.

Each selected slice receives the same ray budget. Stage distributions are
therefore slice-balanced; within each slice, uniform ionized starting cells
give the usual ionized-area-weighted MFP estimator.

Examples
--------
Cluster checkpoint comparison::

    python -m viz.bubble_size_evaluation --checkpoints \
      ufno=checkpoints/checkpoints_3d_ufno/best_model_state_dict.pt \
      localsirenfno=checkpoints/checkpoints_3d_localsirenfno/best_model_state_dict.pt \
      --split test --n-cones 200 --out figures/bubble_size_out

Offline saved-cube comparison::

    python -m viz.bubble_size_evaluation --manifest cubes/manifest.json \
      --out figures/bubble_size_out

Synthetic verification::

    python -m viz.bubble_size_evaluation --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import wasserstein_distance

from util.metrics_21cm import mean_free_path_samples_2d


BOX_MPC = 200.0


@dataclass
class BubbleSizeConfig:
    """Sampling and binning configuration for the MFP diagnostic."""

    box_mpc: float = BOX_MPC
    threshold: float = 0.5
    rays_per_slice: int = 256
    slices_per_stage: int = 6
    n_bins: int = 24
    max_distance_mpc: float | None = None
    stage_edges: tuple[float, ...] = (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)
    active_range: tuple[float, float] = (0.05, 0.95)
    seed: int = 0

    def __post_init__(self):
        if not np.isfinite(self.box_mpc) or self.box_mpc <= 0:
            raise ValueError("box_mpc must be positive")
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("threshold must lie between zero and one")
        if self.rays_per_slice <= 0 or self.slices_per_stage <= 0:
            raise ValueError("ray and slice counts must be positive")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least two")
        if any(b <= a for a, b in zip(self.stage_edges[:-1], self.stage_edges[1:])):
            raise ValueError("stage_edges must be strictly increasing")
        if self.max_distance_mpc is None:
            self.max_distance_mpc = float(self.box_mpc)
        elif (
            not np.isfinite(self.max_distance_mpc)
            or self.max_distance_mpc <= 0
        ):
            raise ValueError("max_distance_mpc must be positive")

    @property
    def n_stages(self) -> int:
        return len(self.stage_edges) - 1

    @property
    def n_rows(self) -> int:
        return self.n_stages + 1

    def stage_labels(self) -> list[str]:
        labels = [
            f"xbar_HI {lo:.2f}-{hi:.2f}"
            for lo, hi in zip(self.stage_edges[:-1], self.stage_edges[1:])
        ]
        labels.append(
            f"active {self.active_range[0]:.2f}-{self.active_range[1]:.2f}"
        )
        return labels

    def radius_edges(self, cell_mpc: float) -> np.ndarray:
        # Sub-cell paths are real because starts are uniformly distributed
        # within cells. The first log bin collects paths below cell/8.
        minimum = max(float(cell_mpc) / 8.0, 1e-4)
        return np.geomspace(minimum, float(self.max_distance_mpc), self.n_bins + 1)


def _stable_token(value) -> int:
    return zlib.crc32(str(value).encode("utf-8")) & 0xFFFFFFFF


def _selected_slices(selection: np.ndarray, limit: int) -> np.ndarray:
    indices = np.flatnonzero(selection)
    if indices.size <= limit:
        return indices
    positions = np.linspace(0, indices.size - 1, limit)
    return indices[np.unique(np.rint(positions).astype(int))]


def _stage_slices(truth: np.ndarray, cfg: BubbleSizeConfig) -> list[np.ndarray]:
    xbar = truth.mean(axis=(0, 1))
    stage = np.digitize(xbar, cfg.stage_edges) - 1
    rows = [
        _selected_slices(stage == index, cfg.slices_per_stage)
        for index in range(cfg.n_stages)
    ]
    active = (xbar >= cfg.active_range[0]) & (xbar <= cfg.active_range[1])
    rows.append(_selected_slices(active, cfg.slices_per_stage))
    return rows


def _sample_stage(
    field: np.ndarray,
    slices: np.ndarray,
    cfg: BubbleSizeConfig,
    cone_id,
    stage_index: int,
    edges: np.ndarray,
) -> dict:
    distances: list[np.ndarray] = []
    n_requested = 0
    n_started = 0
    n_censored = 0
    n_underflow = 0
    ionized_fractions = []
    cell_size = (cfg.box_mpc / field.shape[0], cfg.box_mpc / field.shape[1])
    cone_token = _stable_token(cone_id)
    for slice_index in slices:
        mask = field[:, :, slice_index] < cfg.threshold
        seed = np.random.SeedSequence(
            [cfg.seed, cone_token, int(stage_index), int(slice_index)]
        )
        sample = mean_free_path_samples_2d(
            mask,
            cfg.rays_per_slice,
            cell_size,
            float(cfg.max_distance_mpc),
            seed=seed,
        )
        distances.append(sample["distances_mpc"])
        n_requested += sample["n_requested"]
        n_started += sample["n_started"]
        n_censored += sample["n_censored"]
        n_underflow += sample["n_requested"] - sample["n_started"]
        ionized_fractions.append(sample["ionized_fraction"])

    hits = (
        np.concatenate(distances)
        if distances
        else np.empty(0, dtype=np.float64)
    )
    if n_requested:
        clipped = np.clip(hits, edges[0], np.nextafter(edges[-1], -np.inf))
        counts = np.histogram(clipped, bins=edges)[0]
        mass = counts.astype(np.float64) / n_requested
        censored_fraction = n_censored / n_requested
        underflow_fraction = n_underflow / n_requested
        restricted = float(
            (hits.sum() + n_censored * float(cfg.max_distance_mpc)) / n_requested
        )
    else:
        mass = np.full(len(edges) - 1, np.nan)
        censored_fraction = np.nan
        underflow_fraction = np.nan
        restricted = np.nan
    return {
        "mass": mass,
        "distances": hits,
        "n_requested": int(n_requested),
        "n_started": int(n_started),
        "n_censored": int(n_censored),
        "n_underflow": int(n_underflow),
        "censored_fraction": float(censored_fraction),
        "underflow_fraction": float(underflow_fraction),
        "mean_hit_mpc": float(hits.mean()) if hits.size else np.nan,
        "median_hit_mpc": float(np.median(hits)) if hits.size else np.nan,
        "restricted_mean_mpc": restricted,
        "ionized_fraction": (
            float(np.mean(ionized_fractions)) if ionized_fractions else np.nan
        ),
    }


def _restricted_samples(stats: dict, cap: float) -> np.ndarray:
    if stats["n_requested"] == 0:
        return np.empty(0, dtype=np.float64)
    return np.concatenate([
        np.zeros(stats["n_underflow"], dtype=np.float64),
        stats["distances"],
        np.full(stats["n_censored"], cap, dtype=np.float64),
    ])


def _jensen_shannon(truth: dict, pred: dict) -> float:
    if truth["n_requested"] == 0 or pred["n_requested"] == 0:
        return np.nan
    p = np.concatenate([
        [truth["underflow_fraction"]], truth["mass"],
        [truth["censored_fraction"]],
    ])
    q = np.concatenate([
        [pred["underflow_fraction"]], pred["mass"],
        [pred["censored_fraction"]],
    ])
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(a, b):
        valid = a > 0
        return float(np.sum(a[valid] * np.log(a[valid] / b[valid])))

    return 0.5 * (kl(p, midpoint) + kl(q, midpoint))


def _nan_percentile(array: np.ndarray, quantiles, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(array, quantiles, axis=axis)


@dataclass
class BubbleSizeAccumulator:
    """Collect per-cone, stage-resolved truth/pred MFP distributions."""

    cfg: BubbleSizeConfig
    shape_xy: tuple[int, int]
    per_cone: list[dict] = field(default_factory=list)

    def __post_init__(self):
        cell = self.cfg.box_mpc / self.shape_xy[0]
        self.edges = self.cfg.radius_edges(cell)
        self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])

    def add_cone(self, cone_id, pred: np.ndarray, truth: np.ndarray):
        pred = np.asarray(pred, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        if pred.shape != truth.shape or truth.ndim != 3:
            raise ValueError("prediction and truth must be matching 3-D cubes")
        if truth.shape[:2] != self.shape_xy:
            raise ValueError("transverse cube shape changed between cones")

        stage_slices = _stage_slices(truth, self.cfg)
        records = []
        for stage_index, slices in enumerate(stage_slices):
            truth_stats = _sample_stage(
                truth, slices, self.cfg, cone_id, stage_index, self.edges
            )
            pred_stats = _sample_stage(
                pred, slices, self.cfg, cone_id, stage_index, self.edges
            )
            t_restricted = _restricted_samples(
                truth_stats, float(self.cfg.max_distance_mpc)
            )
            p_restricted = _restricted_samples(
                pred_stats, float(self.cfg.max_distance_mpc)
            )
            w1 = (
                float(wasserstein_distance(t_restricted, p_restricted))
                if t_restricted.size and p_restricted.size
                else np.nan
            )
            records.append({
                "n_slices": int(len(slices)),
                "truth": truth_stats,
                "pred": pred_stats,
                "restricted_wasserstein_mpc": w1,
                "js_divergence": _jensen_shannon(truth_stats, pred_stats),
            })
        self.per_cone.append({"cone_id": cone_id, "stages": records})

    def reduce(self) -> dict:
        if not self.per_cone:
            raise ValueError("no cones accumulated")
        n_cones = len(self.per_cone)
        n_rows = self.cfg.n_rows
        n_bins = self.cfg.n_bins

        def values(path: tuple[str, ...], shape=()):
            output = np.full((n_cones, n_rows) + shape, np.nan)
            for cone_index, cone in enumerate(self.per_cone):
                for stage_index, stage in enumerate(cone["stages"]):
                    value = stage
                    for key in path:
                        value = value[key]
                    output[(cone_index, stage_index)] = value
            return output

        truth_mass = values(("truth", "mass"), (n_bins,))
        pred_mass = values(("pred", "mass"), (n_bins,))
        truth_band = _nan_percentile(truth_mass, [16, 50, 84], axis=0)
        pred_band = _nan_percentile(pred_mass, [16, 50, 84], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            truth_mass_mean = np.nanmean(truth_mass, axis=0)
            pred_mass_mean = np.nanmean(pred_mass, axis=0)

        result = {
            "radius_edges_mpc": self.edges,
            "radius_centers_mpc": self.centers,
            "stage_labels": self.cfg.stage_labels(),
            "truth_mass_p16": truth_band[0],
            "truth_mass_med": truth_band[1],
            "truth_mass_p84": truth_band[2],
            "truth_mass_mean": truth_mass_mean,
            "pred_mass_p16": pred_band[0],
            "pred_mass_med": pred_band[1],
            "pred_mass_p84": pred_band[2],
            "pred_mass_mean": pred_mass_mean,
            "n_stage_slices_med": np.nanmedian(values(("n_slices",)), axis=0),
            "n_valid_cones": np.sum(
                np.isfinite(values(("restricted_wasserstein_mpc",))), axis=0
            ),
        }
        for output_name, path in (
            ("truth_restricted_mean_mpc", ("truth", "restricted_mean_mpc")),
            ("pred_restricted_mean_mpc", ("pred", "restricted_mean_mpc")),
            ("truth_median_hit_mpc", ("truth", "median_hit_mpc")),
            ("pred_median_hit_mpc", ("pred", "median_hit_mpc")),
            ("truth_censored_fraction", ("truth", "censored_fraction")),
            ("pred_censored_fraction", ("pred", "censored_fraction")),
            ("truth_underflow_fraction", ("truth", "underflow_fraction")),
            ("pred_underflow_fraction", ("pred", "underflow_fraction")),
            ("truth_ionized_fraction", ("truth", "ionized_fraction")),
            ("pred_ionized_fraction", ("pred", "ionized_fraction")),
            ("restricted_wasserstein_mpc", ("restricted_wasserstein_mpc",)),
            ("js_divergence", ("js_divergence",)),
        ):
            raw = values(path)
            band = _nan_percentile(raw, [16, 50, 84], axis=0)
            result[f"{output_name}_p16"] = band[0]
            result[f"{output_name}_med"] = band[1]
            result[f"{output_name}_p84"] = band[2]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result[f"{output_name}_mean"] = np.nanmean(raw, axis=0)

        truth_mean = values(("truth", "restricted_mean_mpc"))
        pred_mean = values(("pred", "restricted_mean_mpc"))
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_bias = np.where(truth_mean > 0, pred_mean / truth_mean - 1.0, np.nan)
        band = _nan_percentile(relative_bias, [16, 50, 84], axis=0)
        result["relative_mean_bias_p16"] = band[0]
        result["relative_mean_bias_med"] = band[1]
        result["relative_mean_bias_p84"] = band[2]
        return result


def plot_distributions(results: dict[str, dict], out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = next(iter(results.values()))
    labels = first["stage_labels"]
    n_cols = 3
    n_rows = int(np.ceil(len(labels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), squeeze=False)
    axes = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 2)))
    centers = first["radius_centers_mpc"]
    edges = first["radius_edges_mpc"]

    truth_drawn = False
    for model_index, (name, result) in enumerate(results.items()):
        for stage_index, label in enumerate(labels):
            ax = axes[stage_index]
            if not truth_drawn:
                ax.plot(centers, result["truth_mass_mean"][stage_index], color="black",
                        linewidth=2, label="truth")
                ax.fill_between(
                    centers,
                    result["truth_mass_p16"][stage_index],
                    result["truth_mass_p84"][stage_index],
                    color="black", alpha=0.12,
                )
                ax.scatter(
                    edges[0], result["truth_underflow_fraction_mean"][stage_index],
                    color="black", marker="v", s=26,
                )
                ax.scatter(
                    edges[-1], result["truth_censored_fraction_mean"][stage_index],
                    color="black", marker="^", s=26,
                )
            color = colors[model_index]
            ax.plot(centers, result["pred_mass_mean"][stage_index], color=color,
                    linewidth=1.8, label=name)
            ax.fill_between(
                centers,
                result["pred_mass_p16"][stage_index],
                result["pred_mass_p84"][stage_index],
                color=color, alpha=0.10,
            )
            ax.scatter(
                edges[0], result["pred_underflow_fraction_mean"][stage_index],
                color=color, marker="v", s=24,
            )
            ax.scatter(
                edges[-1], result["pred_censored_fraction_mean"][stage_index],
                color=color, marker="^", s=24,
            )
            ax.set_xscale("log")
            ax.set_xlim(edges[0], edges[-1])
            ax.set(title=label, xlabel="MFP distance R [cMpc]",
                   ylabel="probability per log bin")
            ax.grid(alpha=0.2)
        truth_drawn = True

    for ax in axes[len(labels):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=9)
    axes[0].text(
        0.02, 0.98, "v: no ionized pixels   ^: censored at box length",
        transform=axes[0].transAxes, va="top", fontsize=8,
    )
    fig.suptitle("Transverse mean-free-path ionized-bubble distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(results: dict[str, dict], out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = next(iter(results.values()))
    labels = first["stage_labels"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(results), 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for index, (name, result) in enumerate(results.items()):
        offset = (index - (len(results) - 1) / 2) * width
        axes[0].bar(
            x + offset, result["restricted_wasserstein_mpc_med"], width,
            label=name,
        )
        axes[1].bar(x + offset, 100 * result["relative_mean_bias_med"], width,
                    label=name)
    axes[0].set(ylabel="restricted Wasserstein distance [cMpc]",
                title="BSD distribution error (0 / box-length caps)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(ylabel="restricted mean MFP bias [%]",
                title="Bubble-size bias")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(results: dict[str, dict], out_path: Path):
    columns = [
        "model", "stage", "n_valid_cones", "n_slices_median",
        "truth_restricted_mean_mpc", "pred_restricted_mean_mpc",
        "relative_mean_bias", "restricted_wasserstein_mpc", "js_divergence",
        "truth_underflow_fraction", "pred_underflow_fraction",
        "truth_censored_fraction", "pred_censored_fraction",
        "truth_ionized_fraction", "pred_ionized_fraction",
    ]
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, result in results.items():
            for stage_index, label in enumerate(result["stage_labels"]):
                writer.writerow({
                    "model": name,
                    "stage": label,
                    "n_valid_cones": int(result["n_valid_cones"][stage_index]),
                    "n_slices_median": result["n_stage_slices_med"][stage_index],
                    "truth_restricted_mean_mpc": result["truth_restricted_mean_mpc_med"][stage_index],
                    "pred_restricted_mean_mpc": result["pred_restricted_mean_mpc_med"][stage_index],
                    "relative_mean_bias": result["relative_mean_bias_med"][stage_index],
                    "restricted_wasserstein_mpc": result["restricted_wasserstein_mpc_med"][stage_index],
                    "js_divergence": result["js_divergence_med"][stage_index],
                    "truth_underflow_fraction": result["truth_underflow_fraction_med"][stage_index],
                    "pred_underflow_fraction": result["pred_underflow_fraction_med"][stage_index],
                    "truth_censored_fraction": result["truth_censored_fraction_med"][stage_index],
                    "pred_censored_fraction": result["pred_censored_fraction_med"][stage_index],
                    "truth_ionized_fraction": result["truth_ionized_fraction_med"][stage_index],
                    "pred_ionized_fraction": result["pred_ionized_fraction_med"][stage_index],
                })


def write_npz(results: dict[str, dict], out_path: Path):
    payload = {}
    for name, result in results.items():
        for key, value in result.items():
            if key == "stage_labels":
                value = np.asarray(value, dtype=str)
            payload[f"{name}/{key}"] = value
    np.savez_compressed(out_path, **payload)


def run_from_cubes(
    cube_source: dict[str, Callable[[], "iter"]],
    cfg: BubbleSizeConfig,
    out_dir: Path,
) -> dict:
    results = {}
    for name, source in cube_source.items():
        accumulator = None
        count = 0
        for cone_id, pred, truth in source():
            if accumulator is None:
                accumulator = BubbleSizeAccumulator(cfg, truth.shape[:2])
            accumulator.add_cone(cone_id, pred, truth)
            count += 1
        if accumulator is None:
            raise ValueError(f"model {name!r} produced no cones")
        print(f"[{name}] accumulated {count} cones")
        results[name] = accumulator.reduce()

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_distributions(results, out_dir / "bubble_size_distribution.png")
    plot_summary(results, out_dir / "bubble_size_summary.png")
    write_csv(results, out_dir / "bubble_size_metrics.csv")
    write_npz(results, out_dir / "bubble_size_results.npz")
    config = {
        "method": "transverse_periodic_mean_free_path",
        **cfg.__dict__,
    }
    (out_dir / "bubble_size_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    for filename in (
        "bubble_size_distribution.png", "bubble_size_summary.png",
        "bubble_size_metrics.csv", "bubble_size_results.npz",
        "bubble_size_config.json",
    ):
        print(f"Wrote: {out_dir / filename}")
    return results


def _manifest_source(model_cones: list[dict]) -> Callable:
    def generate():
        for entry in model_cones:
            with np.load(entry["npz"]) as data:
                pred = np.asarray(data["pred"])
                truth = np.asarray(data["truth"])
            yield entry.get("cone_id", entry["npz"]), pred, truth

    return generate


def _validate_manifest_pairing(models: dict[str, list[dict]]) -> None:
    """Require identical ordered cones, geometry, and truth across models."""
    if not models:
        raise ValueError("manifest contains no models")
    reference_name, reference_entries = next(iter(models.items()))
    reference_ids = [entry.get("cone_id", entry["npz"]) for entry in reference_entries]
    for name, entries in models.items():
        ids = [entry.get("cone_id", entry["npz"]) for entry in entries]
        if ids != reference_ids:
            raise ValueError(
                f"manifest model {name!r} does not use the same ordered cone "
                f"IDs as {reference_name!r}"
            )
        if name == reference_name:
            continue
        for reference, candidate in zip(reference_entries, entries):
            with np.load(reference["npz"]) as reference_data, np.load(
                candidate["npz"]
            ) as candidate_data:
                reference_truth = reference_data["truth"]
                candidate_truth = candidate_data["truth"]
                if (
                    reference_truth.shape != candidate_truth.shape
                    or not np.array_equal(reference_truth, candidate_truth)
                ):
                    cone_id = reference.get("cone_id", reference["npz"])
                    raise ValueError(
                        f"manifest truth mismatch for model {name!r}, cone "
                        f"{cone_id}"
                    )


def run_from_manifest(manifest_path: Path, cfg: BubbleSizeConfig, out_dir: Path):
    spec = json.loads(manifest_path.read_text())
    _validate_manifest_pairing(spec["models"])
    sources = {
        name: _manifest_source(cones)
        for name, cones in spec["models"].items()
    }
    return run_from_cubes(sources, cfg, out_dir)


def run_from_checkpoints(
    checkpoints: dict[str, str],
    cfg: BubbleSizeConfig,
    out_dir: Path,
    n_cones: int,
    split: str,
    save_cubes: Path | None,
):
    import torch  # noqa: F401
    from dataset.dataset_3d import (
        InputFeatures,
        LightconeCubeCache,
        ParameterNormalization,
        resolve_split,
    )
    from modeling import ModelConfig
    from util.run_metadata import load_run_metadata
    from viz.visualize_3d import load_model, predict_cube

    cache = Path(os.environ.get("CUBES_CACHE", "cubes_3d.h5"))
    metadata_by_name = {
        name: load_run_metadata(Path(path).parent)
        for name, path in checkpoints.items()
    }
    missing = [name for name, metadata in metadata_by_name.items() if metadata is None]
    if missing:
        raise ValueError(f"run metadata is required for checkpoint models: {missing}")
    first_name = next(iter(checkpoints))
    first_metadata = metadata_by_name[first_name]
    for name, metadata in metadata_by_name.items():
        for contract in ("input_features", "parameter_normalization", "split"):
            if metadata.get(contract) != first_metadata.get(contract):
                raise ValueError(
                    f"checkpoint model {name!r} has different {contract}; "
                    "paired evaluation requires identical preprocessing and split"
                )

    input_features = InputFeatures(first_metadata["input_features"]["name"])
    dataset = LightconeCubeCache(cache, input_features=input_features)
    if first_metadata and first_metadata.get("parameter_normalization"):
        dataset.set_parameter_normalization(
            ParameterNormalization.from_dict(
                first_metadata["parameter_normalization"]
            )
        )
    train_idx, val_idx, test_idx, _ = resolve_split(dataset, first_metadata)
    rows = {"train": train_idx, "val": val_idx, "test": test_idx}[split][:n_cones]
    cone_ids = [int(dataset.cone_ids[row]) for row in rows]
    print(f"[checkpoints] {split} split: using {len(rows)} cones")

    def make_source(checkpoint_path: str, name: str):
        def generate():
            checkpoint = Path(checkpoint_path)
            metadata = metadata_by_name[name]
            config = ModelConfig.from_dict(metadata["model_config"])
            model = load_model(
                in_channels=dataset.in_channels,
                checkpoint=checkpoint,
                model_config=config,
            )
            for row, cone_id in zip(rows, cone_ids):
                _density, truth, pred = predict_cube(model, dataset[row])
                if save_cubes is not None:
                    save_cubes.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        save_cubes / f"{name}_cone{cone_id}.npz",
                        truth=truth.astype(np.float32),
                        pred=pred.astype(np.float32),
                    )
                yield cone_id, pred, truth

        return generate

    sources = {
        name: make_source(path, name)
        for name, path in checkpoints.items()
    }
    results = run_from_cubes(sources, cfg, out_dir)
    if save_cubes is not None:
        manifest = {
            "models": {
                name: [
                    {
                        "cone_id": cone_id,
                        "npz": str(save_cubes / f"{name}_cone{cone_id}.npz"),
                    }
                    for cone_id in cone_ids
                ]
                for name in checkpoints
            }
        }
        manifest_path = save_cubes / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote: {manifest_path}")
    return results


def _parse_kv(items: list[str]) -> dict[str, str]:
    parsed = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected name=path, got {item!r}")
        name, path = item.split("=", 1)
        parsed[name] = path
    return parsed


def _selftest() -> int:
    size = 64
    yy, xx = np.ogrid[:size, :size]
    small = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= 8 ** 2
    large = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= 16 ** 2

    first = mean_free_path_samples_2d(
        small, 4000, (1.0, 1.0), 64.0, seed=7
    )
    repeat = mean_free_path_samples_2d(
        small, 4000, (1.0, 1.0), 64.0, seed=7
    )
    bigger = mean_free_path_samples_2d(
        large, 4000, (1.0, 1.0), 64.0, seed=7
    )
    full = mean_free_path_samples_2d(
        np.ones((size, size), dtype=bool), 100, (1.0, 1.0), 64.0, seed=7
    )
    truth_cube = np.ones((size, size, 4), dtype=np.float64)
    truth_cube[small, :] = 0.0
    no_bubbles = np.ones_like(truth_cube)
    config = BubbleSizeConfig(
        box_mpc=64.0, rays_per_slice=64, slices_per_stage=2,
        n_bins=8, max_distance_mpc=64.0,
    )
    accumulator = BubbleSizeAccumulator(config, truth_cube.shape[:2])
    accumulator.add_cone(1, no_bubbles, truth_cube)
    reduced = accumulator.reduce()
    valid = reduced["n_valid_cones"] > 0

    checks = {
        "reproducible": np.array_equal(first["distances_mpc"], repeat["distances_mpc"]),
        "larger bubble has larger median": (
            np.median(bigger["distances_mpc"]) > np.median(first["distances_mpc"])
        ),
        "bounded bubble has no censoring": first["n_censored"] == 0,
        "fully ionized slice is censored": full["n_censored"] == 100,
        "missing predicted bubbles are penalized": bool(
            np.all(reduced["pred_underflow_fraction_med"][valid] == 1.0)
            and np.all(reduced["restricted_wasserstein_mpc_med"][valid] > 0)
        ),
    }
    for label, passed in checks.items():
        print(f"[{label}] {'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    print("SELFTEST:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--selftest", action="store_true")
    source.add_argument("--manifest", type=Path)
    source.add_argument("--checkpoints", nargs="+", metavar="name=path")
    parser.add_argument("--out", type=Path, default=Path("figures/bubble_size_out"))
    parser.add_argument("--n-cones", type=int, default=200)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--save-cubes", type=Path, default=None)
    parser.add_argument("--box-mpc", type=float, default=BOX_MPC)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--rays-per-slice", type=int, default=256)
    parser.add_argument("--slices-per-stage", type=int, default=6)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--max-distance-mpc", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.selftest:
        raise SystemExit(_selftest())

    cfg = BubbleSizeConfig(
        box_mpc=args.box_mpc,
        threshold=args.threshold,
        rays_per_slice=args.rays_per_slice,
        slices_per_stage=args.slices_per_stage,
        n_bins=args.n_bins,
        max_distance_mpc=args.max_distance_mpc,
        seed=args.seed,
    )
    if args.manifest:
        run_from_manifest(args.manifest, cfg, args.out)
    else:
        run_from_checkpoints(
            _parse_kv(args.checkpoints), cfg, args.out,
            n_cones=args.n_cones, split=args.split, save_cubes=args.save_cubes,
        )


if __name__ == "__main__":
    main()
