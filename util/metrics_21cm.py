"""Physics-oriented diagnostics for predicted neutral-fraction lightcones."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def trace_ray_to_neutral_2d(
    ionized_mask: np.ndarray,
    start_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    cell_size_mpc: tuple[float, float],
    max_distance_mpc: float,
) -> float:
    """Trace one periodic transverse ray to the first neutral cell.

    The ray starts at continuous cell coordinates ``start_xy`` inside an
    ionized cell. ``direction_xy`` is interpreted in physical coordinates and
    normalized internally. A 2-D digital differential analyzer advances
    exactly from cell boundary to cell boundary, so the result does not depend
    on an arbitrary ray-marching step size.

    Returns ``NaN`` when no neutral cell is encountered before
    ``max_distance_mpc``. This is a right-censored ray, as occurs for a
    percolating or fully ionized slice.
    """
    mask = np.asarray(ionized_mask, dtype=bool)
    if mask.ndim != 2 or not all(size > 0 for size in mask.shape):
        raise ValueError("ionized_mask must be a non-empty 2-D array")
    sx, sy = (float(value) for value in start_xy)
    ux, uy = (float(value) for value in direction_xy)
    dx, dy = (float(value) for value in cell_size_mpc)
    max_distance = float(max_distance_mpc)
    if not np.all(np.isfinite([sx, sy, ux, uy, dx, dy, max_distance])):
        raise ValueError("ray inputs and distances must be finite")
    if dx <= 0 or dy <= 0:
        raise ValueError("cell sizes must be positive")
    if max_distance <= 0:
        raise ValueError("max_distance_mpc must be positive")
    norm = float(np.hypot(ux, uy))
    if norm == 0:
        raise ValueError("direction_xy must be non-zero")
    ux, uy = ux / norm, uy / norm

    nx, ny = mask.shape
    sx %= nx
    sy %= ny
    ix, iy = int(np.floor(sx)), int(np.floor(sy))
    if not mask[ix, iy]:
        raise ValueError("ray must start inside an ionized cell")

    # Grid-coordinate velocities per physical Mpc; traversal time is therefore
    # already the desired comoving distance.
    vx, vy = ux / dx, uy / dy

    def axis_state(position: float, index: int, velocity: float):
        if velocity > 0:
            return 1, (index + 1.0 - position) / velocity, 1.0 / velocity
        if velocity < 0:
            speed = -velocity
            return -1, (position - index) / speed, 1.0 / speed
        return 0, np.inf, np.inf

    step_x, next_x, delta_x = axis_state(sx, ix, vx)
    step_y, next_y, delta_y = axis_state(sy, iy, vy)

    while True:
        distance = min(next_x, next_y)
        if distance > max_distance:
            return float("nan")

        # Ties are corner crossings. Advancing both axes chooses the diagonal
        # cell, which is the limiting cell for almost every direction; exact
        # ties have zero probability under isotropic random angles.
        tolerance = 1e-12 * max(1.0, distance)
        if next_x <= distance + tolerance:
            ix = (ix + step_x) % nx
            next_x += delta_x
        if next_y <= distance + tolerance:
            iy = (iy + step_y) % ny
            next_y += delta_y
        if not mask[ix, iy]:
            return float(distance)


def mean_free_path_samples_2d(
    ionized_mask: np.ndarray,
    n_rays: int,
    cell_size_mpc: tuple[float, float],
    max_distance_mpc: float,
    seed: int | np.random.SeedSequence | None = None,
) -> dict:
    """Sample a transverse mean-free-path bubble-size distribution.

    Starting cells are sampled uniformly over ionized area, sub-cell starting
    positions are uniform, and ray directions are isotropic. This makes the
    resulting distance distribution ionized-area weighted, matching the usual
    MFP bubble-size estimator. Transverse boundaries are periodic.

    The returned ``distances_mpc`` contains only boundary hits. Censored rays
    are reported separately and should be retained as an overflow category
    when comparing distributions.
    """
    mask = np.asarray(ionized_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("ionized_mask must be 2-D")
    n_rays = int(n_rays)
    if n_rays < 0:
        raise ValueError("n_rays must be non-negative")
    cells = np.argwhere(mask)
    if n_rays == 0 or cells.size == 0:
        return {
            "distances_mpc": np.empty(0, dtype=np.float64),
            "n_requested": n_rays,
            "n_started": 0,
            "n_hit": 0,
            "n_censored": 0,
            "ionized_fraction": float(mask.mean()),
        }

    rng = np.random.default_rng(seed)
    chosen = cells[rng.integers(0, len(cells), size=n_rays)]
    offsets = rng.random((n_rays, 2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_rays)
    distances = np.empty(n_rays, dtype=np.float64)
    for index in range(n_rays):
        start = chosen[index].astype(np.float64) + offsets[index]
        direction = (np.cos(angles[index]), np.sin(angles[index]))
        distances[index] = trace_ray_to_neutral_2d(
            mask,
            (float(start[0]), float(start[1])),
            direction,
            cell_size_mpc,
            max_distance_mpc,
        )
    hit = np.isfinite(distances)
    return {
        "distances_mpc": distances[hit],
        "n_requested": n_rays,
        "n_started": n_rays,
        "n_hit": int(hit.sum()),
        "n_censored": int((~hit).sum()),
        "ionized_fraction": float(mask.mean()),
    }


def find_low_z_cutoff_index(
    history: np.ndarray,
    target_z: np.ndarray,
    min_change: float = 0.01,
    baseline_points: int = 8,
    smoothing_points: int = 5,
    min_consecutive: int = 3,
    buffer_points: int = 1,
) -> int:
    """Find where x_HI first departs from its settled low-redshift state.

    The input history is smoothed before comparison with the median of its
    lowest-redshift samples. A change must persist for ``min_consecutive``
    samples, which prevents a single noisy slice from setting the crop.
    """
    history = np.asarray(history, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    if history.ndim != 1 or target_z.ndim != 1:
        raise ValueError("history and target_z must be one-dimensional")
    if len(history) != len(target_z) or len(history) == 0:
        raise ValueError("history and target_z must have the same non-zero length")
    if np.any(np.diff(target_z) <= 0):
        raise ValueError("target_z must be strictly increasing")

    n = len(history)
    baseline_points = max(1, min(int(baseline_points), n))
    smoothing_points = max(1, min(int(smoothing_points), n))
    min_consecutive = max(1, min(int(min_consecutive), n))
    buffer_points = max(0, int(buffer_points))

    if smoothing_points > 1:
        left = smoothing_points // 2
        right = smoothing_points - 1 - left
        padded = np.pad(history, (left, right), mode="edge")
        kernel = np.full(smoothing_points, 1.0 / smoothing_points)
        smoothed = np.convolve(padded, kernel, mode="valid")
    else:
        smoothed = history

    baseline = float(np.median(smoothed[:baseline_points]))
    changed = np.abs(smoothed - baseline) >= float(min_change)
    sustained = np.convolve(
        changed.astype(np.int8),
        np.ones(min_consecutive, dtype=np.int8),
        mode="valid",
    )
    starts = np.flatnonzero(sustained == min_consecutive)
    if starts.size == 0:
        return 0
    return max(0, int(starts[0]) - buffer_points)


def _radial_fourier_statistics(
    truth: np.ndarray,
    pred: np.ndarray,
    n_bins: int = 20,
) -> dict:
    shape = truth.shape
    truth_fft = np.fft.rfftn(truth - truth.mean())
    pred_fft = np.fft.rfftn(pred - pred.mean())
    k_axes = [
        np.fft.fftfreq(shape[0]),
        np.fft.fftfreq(shape[1]),
        np.fft.rfftfreq(shape[2]),
    ]
    k = np.sqrt(sum(axis.reshape(
        tuple(len(axis) if i == j else 1 for i in range(3))
    ) ** 2 for j, axis in enumerate(k_axes)))

    edges = np.linspace(0.0, float(k.max()), n_bins + 1)
    shell = np.digitize(k.ravel(), edges) - 1
    valid = (shell >= 0) & (shell < n_bins) & (k.ravel() > 0)
    shell = shell[valid]
    norm = float(np.prod(shape) ** 2)
    pt = (np.abs(truth_fft).ravel()[valid] ** 2) / norm
    pp = (np.abs(pred_fft).ravel()[valid] ** 2) / norm
    cross = (
        np.real(truth_fft * np.conj(pred_fft)).ravel()[valid] / norm
    )

    counts = np.bincount(shell, minlength=n_bins)
    sum_pt = np.bincount(shell, weights=pt, minlength=n_bins)
    sum_pp = np.bincount(shell, weights=pp, minlength=n_bins)
    sum_cross = np.bincount(shell, weights=cross, minlength=n_bins)
    nonempty = counts > 0
    power_truth = np.zeros_like(sum_pt)
    power_pred = np.zeros_like(sum_pp)
    np.divide(sum_pt, counts, out=power_truth, where=nonempty)
    np.divide(sum_pp, counts, out=power_pred, where=nonempty)
    coherence = sum_cross / np.sqrt(np.maximum(sum_pt * sum_pp, 1e-30))

    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "k_grid": centers[nonempty].tolist(),
        "power_truth": power_truth[nonempty].tolist(),
        "power_pred": power_pred[nonempty].tolist(),
        "cross_correlation": coherence[nonempty].tolist(),
    }


def _bubble_summary(field: np.ndarray, threshold: float = 0.5) -> dict:
    labels, count = ndimage.label(field < threshold)
    volumes = np.bincount(labels.ravel())[1:].astype(np.float64)
    if volumes.size == 0:
        return {
            "count": 0,
            "mean_effective_radius_cells": 0.0,
            "volume_weighted_radius_cells": 0.0,
        }
    radii = (3.0 * volumes / (4.0 * np.pi)) ** (1.0 / 3.0)
    return {
        "count": int(count),
        "mean_effective_radius_cells": float(radii.mean()),
        "volume_weighted_radius_cells": float(np.average(radii, weights=volumes)),
    }


def compute_physical_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    target_z: np.ndarray,
    active_variance_fraction: float = 0.05,
) -> dict:
    """Return JSON-serializable reionization-history and morphology metrics."""
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 3:
        raise ValueError("truth and prediction must have the same 3-D shape")
    if truth.shape[-1] != len(target_z):
        raise ValueError("target_z length must match the lightcone Z dimension")

    history_truth = truth.mean(axis=(0, 1))
    history_pred = pred.mean(axis=(0, 1))
    transverse_std = truth.std(axis=(0, 1))
    max_std = float(transverse_std.max())
    active = (
        transverse_std > active_variance_fraction * max_std
        if max_std > 1e-8
        else np.ones_like(transverse_std, dtype=bool)
    )
    error = pred - truth
    edge_width = max(1, min(truth.shape[0], truth.shape[1]) // 10)
    edge_mask = np.zeros(truth.shape[:2], dtype=bool)
    edge_mask[:edge_width, :] = True
    edge_mask[-edge_width:, :] = True
    edge_mask[:, :edge_width] = True
    edge_mask[:, -edge_width:] = True

    return {
        "voxel_rmse": float(np.sqrt(np.mean(error ** 2))),
        "transverse_edge_rmse": float(
            np.sqrt(np.mean(error[edge_mask, :] ** 2))
        ),
        "transverse_interior_rmse": float(
            np.sqrt(np.mean(error[~edge_mask, :] ** 2))
        ),
        "active_window_rmse": float(
            np.sqrt(np.mean(error[..., active] ** 2))
        ),
        "active_z_min": float(target_z[active].min()),
        "active_z_max": float(target_z[active].max()),
        "global_history_rmse": float(
            np.sqrt(np.mean((history_pred - history_truth) ** 2))
        ),
        "target_z": target_z.tolist(),
        "global_xhi_truth": history_truth.tolist(),
        "global_xhi_pred": history_pred.tolist(),
        "fourier": _radial_fourier_statistics(truth, pred),
        "ionized_regions_truth": _bubble_summary(truth),
        "ionized_regions_pred": _bubble_summary(pred),
    }
