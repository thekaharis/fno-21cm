"""Small loss adapters used by the neuralop Trainer."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F


class AbsoluteLoss:
    """Call a neuralop loss through its absolute-norm implementation."""

    def __init__(self, loss):
        self.loss = loss

    def __call__(self, out, y, **_):
        return self.loss.abs(out, y)


class RelativeLoss:
    """Call a neuralop loss through its relative-norm implementation."""

    def __init__(self, loss):
        self.loss = loss

    def __call__(self, out, y, **_):
        return self.loss.rel(out, y)


def los_volume_weights(target_z, omega_m: float = 0.31) -> torch.Tensor:
    """Per-slice comoving-thickness quadrature weights for the LOS axis.

    On a non-uniform LOS grid (see dataset/build_cubes.py --target-z-file),
    voxel count IS loss weight: a plain norm lets densely sampled epochs
    dominate in proportion to their slice count rather than the comoving
    volume they represent. These weights undo that: w_k ~ Delta chi_k via
    dchi/dz ~ 1/E(z) for flat LCDM (the absolute scale cancels in the
    mean-1 normalization, so omega_m precision is uncritical).
    """
    z = torch.as_tensor(target_z, dtype=torch.float64)
    ez = torch.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))
    dz = torch.gradient(z)[0]
    w = dz / ez
    return (w / w.mean()).to(torch.float32)


class LOSVolumeWeightedLoss:
    """Volume-weight a norm-based loss along the LOS (last) axis.

    Multiplies prediction and target by sqrt(w_k) (w normalized to mean 1)
    before delegating, which turns the wrapped norm into a physical volume
    quadrature -- for relative losses the denominator is weighted
    consistently: ||sqrt(w)(out - y)|| / ||sqrt(w) y||.

    Wrap only norm-based terms (Lp, H1). Value-semantic terms (BCE,
    threshold-based band losses) must not be wrapped: scaling moves their
    inputs out of [0, 1].
    """

    def __init__(self, loss, weights: torch.Tensor):
        self.loss = loss
        self._sqrt_w = torch.sqrt(
            torch.as_tensor(weights, dtype=torch.float32))

    def __call__(self, out, y, **kwargs):
        w = self._sqrt_w.to(device=out.device, dtype=out.dtype)
        return self.loss(out * w, y * w, **kwargs)


class WeightedLoss:
    """Combine ``(weight, loss)`` terms while preserving Trainer kwargs.

    When ``term_names`` are given, every evaluated (non-zero-weight) term's
    raw, *unweighted* value is accumulated per call; ``pop_term_means``
    returns the per-term means since the previous pop and resets the
    accumulator, which is how the trainer surfaces per-term training losses
    once per epoch.
    """

    def __init__(
        self,
        *terms: tuple[float, Callable],
        term_names: tuple[str, ...] | None = None,
    ):
        self.terms = tuple((float(weight), loss) for weight, loss in terms)
        if term_names is None:
            term_names = tuple(f"term{i}" for i in range(len(self.terms)))
        if len(term_names) != len(self.terms):
            raise ValueError(
                f"{len(self.terms)} loss terms require {len(self.terms)} "
                f"names, got {len(term_names)}"
            )
        self.term_names = tuple(str(name) for name in term_names)
        self._term_sums: dict[str, float] = {}
        self._term_batches = 0

    @property
    def active_weights(self) -> tuple[float, ...]:
        return tuple(weight for weight, _ in self.terms)

    def pop_term_means(self) -> dict[str, float]:
        """Return per-term mean raw values since the last pop, then reset."""
        batches = max(self._term_batches, 1)
        means = {name: total / batches for name, total in self._term_sums.items()}
        self._term_sums = {}
        self._term_batches = 0
        return means

    def __call__(self, out, y, **kwargs):
        total = 0.0
        for name, active_weight, (_, loss) in zip(
            self.term_names, self.active_weights, self.terms, strict=True
        ):
            if active_weight == 0.0:
                continue
            value = loss(out, y, **kwargs)
            self._term_sums[name] = (
                self._term_sums.get(name, 0.0) + float(value.detach())
            )
            total = total + active_weight * value
        self._term_batches += 1
        return total


class ScheduledWeightedLoss(WeightedLoss):
    """Weighted loss whose selected terms ramp in over early epochs."""

    def __init__(
        self,
        *terms: tuple[float, Callable],
        warmup_terms: tuple[int, ...] = (),
        warmup_epochs: int = 0,
        term_names: tuple[str, ...] | None = None,
    ):
        super().__init__(*terms, term_names=term_names)
        self.warmup_terms = frozenset(int(index) for index in warmup_terms)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    @property
    def warmup_factor(self) -> float:
        if self.warmup_epochs == 0:
            return 1.0
        return min(1.0, self.epoch / self.warmup_epochs)

    @property
    def active_weights(self) -> tuple[float, ...]:
        factor = self.warmup_factor
        return tuple(
            weight * factor if index in self.warmup_terms else weight
            for index, (weight, _) in enumerate(self.terms)
        )


class BinaryCrossEntropyTerm:
    """Voxel-mean BCE for neutral-fraction targets in ``[0, 1]``."""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def __call__(self, out, y, **_):
        prediction = out.clamp(self.eps, 1.0 - self.eps)
        return torch.nn.functional.binary_cross_entropy(prediction, y)


class IonizedWallRMSE:
    """One-sided RMSE for excess neutral fraction near true bubble walls.

    A transverse dilation of the target's neutral mask identifies ionized
    voxels near a true boundary. Only positive residuals are penalized, so the
    term specifically suppresses excess predicted neutral fraction on the
    ionized side. X/Y dilation is periodic; Z is not part of the band geometry.
    """

    def __init__(
        self,
        band_kernel_size: int = 7,
        threshold: float = 0.5,
        eps: float = 1e-12,
    ):
        kernel = int(band_kernel_size)
        if kernel <= 1 or kernel % 2 == 0:
            raise ValueError("band_kernel_size must be an odd integer greater than 1")
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1")
        self.band_kernel_size = kernel
        self.threshold = float(threshold)
        self.eps = float(eps)

    def wall_mask(self, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 5:
            raise ValueError("IonizedWallRMSE expects (B, C, X, Y, Z) tensors")
        neutral = target >= self.threshold
        radius = self.band_kernel_size // 2
        padded = F.pad(
            neutral.to(dtype=target.dtype),
            (0, 0, radius, radius, radius, radius),
            mode="circular",
        )
        near_neutral = F.max_pool3d(
            padded,
            kernel_size=(self.band_kernel_size, self.band_kernel_size, 1),
            stride=1,
        ) > 0
        return (target < self.threshold) & near_neutral

    def __call__(self, out: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
        if out.shape != y.shape:
            raise ValueError(
                f"prediction and target shapes differ: {out.shape} != {y.shape}"
            )
        mask = self.wall_mask(y)
        excess_neutral = F.relu(out - y)
        mask_float = mask.to(dtype=out.dtype)
        count = mask_float.sum()
        mean_squared_error = (
            (excess_neutral.square() * mask_float).sum()
            / count.clamp_min(1.0)
        )
        stable_rmse = (
            torch.sqrt(mean_squared_error + self.eps)
            - math.sqrt(self.eps)
        )
        return torch.where(count > 0, stable_rmse, out.sum() * 0.0)


class H2Loss2d:
    """Sobolev H2 norm for 2-D maps, periodic in both transverse axes.

    Terms: value, first derivatives, and all multi-index second derivatives
    (the mixed derivative enters with its combinatorial factor of 2, folded
    in as sqrt(2)*fxy). First derivatives use centered stencils; the pure
    second derivatives use the standard three-point stencil; the mixed
    derivative is centered in both axes. Quadrature scaling matches the
    neuralop H1Loss convention, so absolute and relative values are directly
    comparable with the Lp/H1 terms they are weighted against.
    """

    def __init__(
        self,
        measure: tuple[float, float] = (1.0, 1.0),
        reduction: str = "sum",
    ):
        if len(measure) != 2:
            raise ValueError("H2Loss2d requires two domain measures")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.d = 2
        self.measure = tuple(float(value) for value in measure)
        self.reduction = reduction
        self.periodic_in_x = True
        self.periodic_in_y = True

    def uniform_quadrature(self, x: torch.Tensor) -> tuple[float, float]:
        return (
            self.measure[0] / x.size(-2),
            self.measure[1] / x.size(-1),
        )

    def _reduce(self, value: torch.Tensor) -> torch.Tensor:
        if self.reduction == "sum":
            return value.sum()
        return value.mean()

    def compute_terms(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        quadrature: tuple[float, float] | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Return flattened value/first/second-derivative terms."""
        if x.shape != y.shape:
            raise ValueError(
                f"prediction and target shapes differ: {x.shape} != {y.shape}"
            )
        if x.ndim < 2:
            raise ValueError("H2Loss2d requires at least two spatial dims")
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        hx, hy = (float(value) for value in quadrature)
        sqrt2 = math.sqrt(2.0)

        def terms(field: torch.Tensor) -> dict[int, torch.Tensor]:
            fx = (
                torch.roll(field, shifts=-1, dims=-2)
                - torch.roll(field, shifts=1, dims=-2)
            ) / (2.0 * hx)
            fy = (
                torch.roll(field, shifts=-1, dims=-1)
                - torch.roll(field, shifts=1, dims=-1)
            ) / (2.0 * hy)
            fxx = (
                torch.roll(field, shifts=-1, dims=-2)
                - 2.0 * field
                + torch.roll(field, shifts=1, dims=-2)
            ) / (hx * hx)
            fyy = (
                torch.roll(field, shifts=-1, dims=-1)
                - 2.0 * field
                + torch.roll(field, shifts=1, dims=-1)
            ) / (hy * hy)
            fxy = (
                torch.roll(
                    torch.roll(field, shifts=-1, dims=-2), shifts=-1, dims=-1
                )
                - torch.roll(
                    torch.roll(field, shifts=-1, dims=-2), shifts=1, dims=-1
                )
                - torch.roll(
                    torch.roll(field, shifts=1, dims=-2), shifts=-1, dims=-1
                )
                + torch.roll(
                    torch.roll(field, shifts=1, dims=-2), shifts=1, dims=-1
                )
            ) / (4.0 * hx * hy)
            return {
                0: torch.flatten(field, start_dim=-2),
                1: torch.flatten(fx, start_dim=-2),
                2: torch.flatten(fy, start_dim=-2),
                3: torch.flatten(fxx, start_dim=-2),
                4: torch.flatten(sqrt2 * fxy, start_dim=-2),
                5: torch.flatten(fyy, start_dim=-2),
            }

        return terms(x), terms(y)

    def _squared_error_and_norm(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        quadrature: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample squared H2 error and squared H2 norm of the target."""
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        terms_x, terms_y = self.compute_terms(x, y, quadrature)
        scale = math.prod(quadrature)
        error = sum(
            scale * torch.sum(
                (terms_x[index] - terms_y[index]).square(),
                dim=-1,
            )
            for index in range(6)
        )
        norm = sum(
            scale * torch.sum(terms_y[index].square(), dim=-1)
            for index in range(6)
        )
        return error, norm

    def abs(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        quadrature: tuple[float, float] | None = None,
        take_root: bool = True,
    ) -> torch.Tensor:
        error, _ = self._squared_error_and_norm(x, y, quadrature)
        if take_root:
            error = error.sqrt()
        return self._reduce(error).squeeze()

    def rel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        quadrature: tuple[float, float] | None = None,
        take_root: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Relative H2 error, ``||x - y||_H2 / ||y||_H2`` per sample.

        Dimensionless, matching neuralop's relative Lp/H1 conventions
        (``eps`` guards near-zero target norms, e.g. fully clamped z_re
        maps).
        """
        error, norm = self._squared_error_and_norm(x, y, quadrature)
        if take_root:
            ratio = error.sqrt() / norm.sqrt().clamp_min(eps)
        else:
            ratio = error / norm.clamp_min(eps)
        return self._reduce(ratio).squeeze()


class LightconeH1Loss:
    """Absolute H1 norm with periodic X/Y and interior-only LOS differences.

    The lightcone endpoints are physically unrelated, so Z must not wrap.
    One-sided endpoint stencils can nevertheless dominate the loss because
    derivatives are scaled by the inverse grid spacing. This implementation
    evaluates the Z derivative only where a centered stencil is available,
    while retaining the value term and periodic X/Y derivatives on the full
    cube.
    """

    def __init__(
        self,
        measure: tuple[float, float, float] = (1.0, 1.0, 1.0),
        reduction: str = "sum",
    ):
        if len(measure) != 3:
            raise ValueError("LightconeH1Loss requires three domain measures")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.d = 3
        self.measure = tuple(float(value) for value in measure)
        self.reduction = reduction
        self.periodic_in_x = True
        self.periodic_in_y = True
        self.periodic_in_z = False

    def uniform_quadrature(self, x: torch.Tensor) -> tuple[float, float, float]:
        return tuple(
            self.measure[axis] / x.size(axis - 3)
            for axis in range(3)
        )

    def _reduce(self, value: torch.Tensor) -> torch.Tensor:
        if self.reduction == "sum":
            return value.sum()
        return value.mean()

    def compute_terms(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        quadrature: tuple[float, float, float] | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Return flattened value/derivative terms used by :meth:`abs`.

        Terms 0, 1, and 2 cover the full cube. Term 3 contains only
        ``Z[1:-1]`` because centered LOS differences are undefined at the
        physical endpoints.
        """
        if x.shape != y.shape:
            raise ValueError(
                f"prediction and target shapes differ: {x.shape} != {y.shape}"
            )
        if x.ndim < 3 or x.size(-1) < 3:
            raise ValueError(
                "LightconeH1Loss requires at least three spatial dimensions "
                "and three LOS cells"
            )
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        hx, hy, hz = (float(value) for value in quadrature)

        def terms(field: torch.Tensor) -> dict[int, torch.Tensor]:
            dx = (
                torch.roll(field, shifts=-1, dims=-3)
                - torch.roll(field, shifts=1, dims=-3)
            ) / (2.0 * hx)
            dy = (
                torch.roll(field, shifts=-1, dims=-2)
                - torch.roll(field, shifts=1, dims=-2)
            ) / (2.0 * hy)
            dz = (
                field[..., 2:] - field[..., :-2]
            ) / (2.0 * hz)
            return {
                0: torch.flatten(field, start_dim=-3),
                1: torch.flatten(dx, start_dim=-3),
                2: torch.flatten(dy, start_dim=-3),
                3: torch.flatten(dz, start_dim=-3),
            }

        return terms(x), terms(y)

    def _squared_error_and_norm(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        quadrature: tuple[float, float, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample squared H1 error and squared H1 norm of the target."""
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        quadrature = tuple(float(value) for value in quadrature)
        terms_x, terms_y = self.compute_terms(x, y, quadrature)
        scale = math.prod(quadrature)
        error = sum(
            scale * torch.sum(
                (terms_x[index] - terms_y[index]).square(),
                dim=-1,
            )
            for index in range(4)
        )
        norm = sum(
            scale * torch.sum(terms_y[index].square(), dim=-1)
            for index in range(4)
        )
        return error, norm

    def abs(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        quadrature: tuple[float, float, float] | None = None,
        take_root: bool = True,
    ) -> torch.Tensor:
        error, _ = self._squared_error_and_norm(x, y, quadrature)
        if take_root:
            error = error.sqrt()
        return self._reduce(error).squeeze()

    def rel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        quadrature: tuple[float, float, float] | None = None,
        take_root: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Relative H1 error, ``||x - y||_H1 / ||y||_H1`` per sample.

        Dimensionless, so it is directly comparable with (and weightable
        against) a relative L2 term. Full lightcones always contain a
        neutral high-z region, so ``||y||_H1`` is never near zero for this
        dataset; ``eps`` only guards degenerate inputs.
        """
        error, norm = self._squared_error_and_norm(x, y, quadrature)
        if take_root:
            ratio = error.sqrt() / norm.sqrt().clamp_min(eps)
        else:
            ratio = error / norm.clamp_min(eps)
        return self._reduce(ratio).squeeze()


def _edge_measure(field: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """|grad field| on periodic transverse axes, flattened per sample."""
    dx = torch.roll(field, shifts=-1, dims=-2) - field
    dy = torch.roll(field, shifts=-1, dims=-1) - field
    magnitude = torch.sqrt(dx * dx + dy * dy + eps)
    return magnitude.flatten(start_dim=-2)


class SlicedWassersteinEdges:
    """Sliced 1-Wasserstein distance between predicted and true edge measures.

    The edge measure is ``|grad x_HI|`` normalized to unit mass per sample: a
    probability distribution over *where* the field's transitions sit. Scoring
    it with optimal transport rather than pointwise is the whole point. A
    pointwise gradient loss (H1) is minimized by ``E[grad]``, which turns a
    tall narrow ridge into a low wide bump whenever the edge position is
    uncertain -- it rewards exactly the blurring we are trying to remove. The
    Wasserstein barycenter of shifted sharp edges is still a sharp edge, so
    the cost grows with *displacement* while smearing mass away from the true
    boundary increases it.

    The exact 2-D transport is replaced by the standard sliced approximation
    (Bonneel et al.): project pixel coordinates onto random unit directions
    and compare 1-D CDFs, which is the exact 1-D Wasserstein along each
    direction. On a fixed grid the projected coordinates are constant, so the
    per-direction sort order is precomputed once per spatial shape; each call
    is then a gather, a cumsum and an L1 difference, with no sorting in the
    training loop.

    Projections are scaled to unit extent, so the returned value is a mean
    transport distance in units of the box width and is directly comparable
    across resolutions.
    """

    def __init__(
        self,
        n_directions: int = 48,
        seed: int = 0,
        eps: float = 1e-8,
    ):
        if int(n_directions) <= 0:
            raise ValueError("n_directions must be positive")
        self.n_directions = int(n_directions)
        self.seed = int(seed)
        self.eps = float(eps)
        # (shape, device, dtype) -> (order, dt); deterministic in `seed`, so
        # every rank and every restart slices along the same directions.
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _projection(self, field: torch.Tensor):
        height, width = int(field.shape[-2]), int(field.shape[-1])
        key = (height, width, field.device, field.dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        angles = torch.rand(
            self.n_directions, generator=generator, dtype=torch.float64
        ) * math.pi
        rows = torch.arange(height, dtype=torch.float64)
        cols = torch.arange(width, dtype=torch.float64)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        coordinates = torch.stack(
            (grid_y.reshape(-1), grid_x.reshape(-1)), dim=0
        )
        # (n_directions, n_pixels)
        projected = (
            torch.cos(angles)[:, None] * coordinates[0][None, :]
            + torch.sin(angles)[:, None] * coordinates[1][None, :]
        )
        extent = (projected.amax(dim=1) - projected.amin(dim=1)).clamp_min(1.0)
        projected = projected / extent[:, None]

        order = torch.argsort(projected, dim=1)
        sorted_projection = torch.gather(projected, 1, order)
        # Trapezoid widths between consecutive sorted positions; the 1-D
        # Wasserstein-1 distance is the |CDF difference| integrated over these.
        dt = (sorted_projection[:, 1:] - sorted_projection[:, :-1])

        order = order.to(field.device)
        dt = dt.to(device=field.device, dtype=field.dtype)
        self._cache[key] = (order, dt)
        return order, dt

    def __call__(self, out: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
        if out.shape != y.shape:
            raise ValueError(
                f"prediction and target shapes differ: {out.shape} != {y.shape}"
            )
        order, dt = self._projection(out)

        prediction = _edge_measure(out, self.eps)
        target = _edge_measure(y, self.eps)
        # Flatten every leading (batch, channel, ...) axis into one.
        prediction = prediction.reshape(-1, prediction.shape[-1])
        target = target.reshape(-1, target.shape[-1])

        # Unit mass per sample: this term scores *where* the edges are, and
        # leaves *how much* edge there is to the spectral term.
        prediction = prediction / prediction.sum(dim=-1, keepdim=True).clamp_min(
            self.eps
        )
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # (n_samples, n_directions, n_pixels)
        expanded = order[None, :, :].expand(prediction.shape[0], -1, -1)
        cdf_prediction = torch.gather(
            prediction[:, None, :].expand_as(expanded), 2, expanded
        ).cumsum(dim=-1)[..., :-1]
        cdf_target = torch.gather(
            target[:, None, :].expand_as(expanded), 2, expanded
        ).cumsum(dim=-1)[..., :-1]

        distance = ((cdf_prediction - cdf_target).abs() * dt[None]).sum(dim=-1)
        return distance.mean()


class HighKPowerRatio:
    """Squared log-ratio of radially binned power above ``k_min``.

    Targets the measured small-scale power deficit directly. Being computed
    from ``|FFT|**2`` it is translation invariant, so -- unlike any pointwise
    loss -- it cannot be reduced by hedging on edge position; it only asks
    that the prediction carry the right *amount* of structure at each scale.
    Pair it with a positional term (see :class:`SlicedWassersteinEdges`),
    which is blind to the amplitude this one constrains.

    ``k_min`` is in cycles per pixel (Nyquist = 0.5). The default 0.2
    corresponds to k ~ 0.9 Mpc^-1 on the production 200 Mpc / 140 px grid,
    where the cylindrical power-spectrum diagnostics show the deficit setting
    in.
    """

    def __init__(
        self,
        k_min: float = 0.2,
        n_bins: int = 12,
        eps: float = 1e-12,
    ):
        if not 0.0 <= float(k_min) < 0.5:
            raise ValueError("k_min must lie in [0, 0.5) cycles per pixel")
        if int(n_bins) <= 0:
            raise ValueError("n_bins must be positive")
        self.k_min = float(k_min)
        self.n_bins = int(n_bins)
        self.eps = float(eps)
        self._cache: dict[tuple, tuple[torch.Tensor, int]] = {}

    def _bins(self, field: torch.Tensor) -> tuple[torch.Tensor, int]:
        height, width = int(field.shape[-2]), int(field.shape[-1])
        key = (height, width, field.device)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        ky = torch.fft.fftfreq(height, dtype=torch.float64)
        kx = torch.fft.rfftfreq(width, dtype=torch.float64)
        radius = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        edges = torch.linspace(
            self.k_min, float(radius.max()), self.n_bins + 1, dtype=torch.float64
        )
        index = torch.bucketize(radius, edges) - 1
        # Everything below k_min, and the k=0 mode, is excluded.
        index = torch.where(
            (radius < self.k_min) | (index >= self.n_bins),
            torch.full_like(index, -1),
            index,
        )
        index = index.reshape(-1).to(field.device)
        self._cache[key] = (index, self.n_bins)
        return index, self.n_bins

    def _binned_power(
        self, field: torch.Tensor, index: torch.Tensor, n_bins: int
    ) -> torch.Tensor:
        centered = field - field.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.rfft2(centered, norm="ortho")
        power = (spectrum.real**2 + spectrum.imag**2).flatten(start_dim=-2)
        power = power.reshape(-1, power.shape[-1])

        keep = index >= 0
        selected = index[keep]
        weights = power[:, keep]
        totals = torch.zeros(
            power.shape[0], n_bins, device=field.device, dtype=power.dtype
        )
        totals.index_add_(1, selected, weights)
        counts = torch.zeros(n_bins, device=field.device, dtype=power.dtype)
        counts.index_add_(0, selected, torch.ones_like(selected, dtype=power.dtype))
        return totals / counts.clamp_min(1.0)[None, :]

    def __call__(self, out: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
        if out.shape != y.shape:
            raise ValueError(
                f"prediction and target shapes differ: {out.shape} != {y.shape}"
            )
        index, n_bins = self._bins(out)
        power_prediction = self._binned_power(out, index, n_bins)
        power_target = self._binned_power(y, index, n_bins)
        ratio = torch.log(
            (power_prediction + self.eps) / (power_target + self.eps)
        )
        return ratio.square().mean()


# --------------------------------------------------------------- wall placement
def _drop_channel(field: torch.Tensor) -> torch.Tensor:
    """(N,1,...) -> (N,...); leave an already channel-less field alone."""
    return field.squeeze(1) if field.dim() in (4, 5) and field.shape[1] == 1 else field


def _match_rank(weight: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Give ``weight`` the rank of ``like`` by restoring a channel axis.

    Without this the weight is built at (N,H,W) while the prediction may be
    (N,1,H,W) -- or vice versa in the refit path, where predictions carry no
    channel axis. The two then broadcast to (N,N,H,W) instead of raising:
    every sample's weight lands on every other sample's error, the gradients
    are wrong, and memory grows as N^2. The values stay plausible, so it does
    not announce itself.
    """
    return weight.unsqueeze(1) if like.dim() == weight.dim() + 1 else weight

def _chamfer_distance(mask: torch.Tensor, cap: int) -> torch.Tensor:
    """Distance in pixels from the nearest ``True`` in ``mask``, capped.

    Iterative 3x3 min-propagation on the GPU: ``cap`` passes, each spreading the
    front one pixel.  This is the chessboard (L-inf) distance rather than the
    Euclidean one -- for penalising misplacement the difference is immaterial,
    and it avoids a ~27% epoch overhead from a CPU scipy EDT round trip.

    The cap is not only a speed knob: it bounds the loss gradient, since the
    per-pixel weight is exactly this distance.
    """
    d = torch.where(mask, torch.zeros_like(mask, dtype=torch.float32),
                    torch.full_like(mask, float(cap), dtype=torch.float32))
    d = d.unsqueeze(1)
    # 2-D slices or 3-D cubes; same propagation, different pooling rank.
    pool = {4: F.max_pool2d, 5: F.max_pool3d}.get(d.dim())
    if pool is None:
        raise ValueError(f"expected (N,H,W) or (N,D,H,W) mask, got {mask.shape}")
    for _ in range(int(cap)):
        # min-pool via -maxpool(-x); +1 per pixel of travel
        d = torch.minimum(d, -pool(-d, 3, stride=1, padding=1) + 1.0)
    return d.squeeze(1).clamp_(0.0, float(cap))


def signed_distance(target: torch.Tensor, cap: int = 32,
                    threshold: float = 0.5) -> torch.Tensor:
    """Signed distance to the neutral-region wall: <0 inside, >0 outside.

    Works on 2-D slices and 3-D cubes.  Note the distance is in *voxels*: for
    a lightcone the transverse and line-of-sight axes are not on the same
    physical scale, so the penalty is mildly anisotropic in Mpc.
    """
    inside = target > threshold
    return (_chamfer_distance(inside, cap)          # outside: positive
            - _chamfer_distance(~inside, cap))      # inside: negative


def transverse_signed_distance(target: torch.Tensor, cap: int = 32,
                               threshold: float = 0.5
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-XY-slice signed distance for a cube, plus a has-wall slice mask.

    Why this exists.  On the production lightcone cache a transverse cell is
    ~1.43 Mpc while a LOS cell is ~9.7 Mpc (41.7 Mpc at the z=5 end), so a
    voxel-space distance transform treats one transverse step as equal to a
    step ~7x longer in Mpc.  Worse, the measured truth front is 3.6 Mpc: 2.5
    transverse cells, but 0.37 of a LOS cell -- along the LOS the correct
    answer is not representable on this grid at all, so the 3-D transform
    spends most of its weight on a direction where nothing can be learned.

    There is a second effect, independent of units.  In 3-D a wall in an
    adjacent LOS slice is always a voxel or two away, so the transform rarely
    approaches ``cap``: measured phi range [-6, +7] and weight spread 1.5x,
    against [-32, +32] and 7x per-slice.  The exponential weighting that makes
    this loss work is effectively switched off in 3-D.

    Returns ``(phi, has_wall)`` where ``has_wall`` is True for slices that
    contain a front.  Single-phase slices carry no edge information, and every
    voxel in them sits at ``cap`` -- so they would otherwise receive the
    *largest* weight in the batch.  The caller must mask them out.
    """
    if target.dim() != 4:
        raise ValueError(
            f"expected a channel-less cube (N,X,Y,Z), got {tuple(target.shape)}"
        )
    n, x, y, z = target.shape
    flat = target.permute(0, 3, 1, 2).reshape(n * z, x, y)
    phi = signed_distance(flat, cap, threshold)
    inside = flat > threshold
    # A slice has a wall only if it contains both phases.
    has_wall = inside.any(dim=(-2, -1)) & (~inside).any(dim=(-2, -1))
    phi = phi.reshape(n, z, x, y).permute(0, 2, 3, 1)
    has_wall = has_wall.reshape(n, 1, 1, z)        # broadcasts over X, Y
    return phi, has_wall


class WallPlacementLoss:
    """Penalise every pixel by its distance from the true bubble wall.

        L = mean( phi * (pred - target) ),   phi = signed distance of the truth

    Why this and not H1.  Every pointwise-difference loss saturates with
    displacement: once a predicted wall no longer overlaps the true one, moving
    it further costs nothing more, so no gradient pulls it home.  Measured on a
    displaced step edge, going from 4 px to 48 px of error changes L2 by 3.5x,
    H1 by 2.9x, and the H1 *seminorm* and H2 by exactly 1.00x -- gradient-only
    losses are completely blind to misplacement.  H1's whole sensitivity comes
    from the L2 term inside it.  This loss changes by 144x over the same range.

    It is linear in the prediction, so its gradient never vanishes, and it is
    minimised exactly at ``pred = target`` for a binary target -- unlike the
    transport/spectral edge terms, which fix neither level nor position and
    diverged without an L2 anchor.

    ``cap`` bounds the per-pixel weight, hence the gradient.

    Do not use this term alone.  Being linear in the prediction, its gradient
    is the constant ``phi`` and never diminishes as the optimum is approached,
    so through a sigmoid output it drives the logits straight into saturation --
    where the sigmoid derivative is ~0 and learning stops.  Run alone it
    collapsed to a uniform field of 1.0 (std 0, 100% saturated) by epoch 0 and
    never moved again.  Pair it with a region term whose gradient is largest
    exactly where this one dies: BCE, whose gradient ``sigma(z) - y`` is maximal
    for a confidently wrong pixel.  BCE is not an L2 term, so "no L2" survives.
    """

    def __init__(self, cap: int = 32, threshold: float = 0.5,
                 normalize: bool = True):
        self.cap = int(cap)
        self.threshold = float(threshold)
        self.normalize = bool(normalize)

    def __call__(self, out: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
        if out.shape != y.shape:
            raise ValueError(f"shape mismatch: {out.shape} != {y.shape}")
        with torch.no_grad():
            phi = signed_distance(_drop_channel(y.detach()), self.cap,
                                  self.threshold)
            if self.normalize:
                phi = phi / self.cap
        return (_match_rank(phi, out) * (out - y)).mean()


class H1Seminorm:
    """Gradient-only Sobolev loss -- H1 with the L2 term removed.

    Provided because ``neuralop.H1Loss`` is *not* L2-free: it sums
    ``||u-v||^2`` and ``||grad u - grad v||^2``.  Be aware of what this costs:
    the seminorm has constants in its null space (any uniform offset is free)
    and, as measured above, it is exactly insensitive to how far a wall is
    misplaced.  Use it to reproduce that result, not to fix placement.
    """

    def __init__(self, cap: float | None = None):
        self.cap = cap

    def __call__(self, out: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
        du = torch.roll(out, -1, dims=-2) - out
        dv = torch.roll(out, -1, dims=-1) - out
        ty = torch.roll(y, -1, dims=-2) - y
        tx = torch.roll(y, -1, dims=-1) - y
        sq = (du - ty) ** 2 + (dv - tx) ** 2
        if self.cap is not None:
            sq = sq.clamp(max=float(self.cap) ** 2)
        return sq.mean().sqrt()


class ExponentialWallDistance:
    """Absolute error weighted exponentially by distance from the true wall.

        L = mean( w(phi) * |pred - y| ) / mean(w),   w = exp(|phi| / scale)

    Two deliberate choices.

    *Exponential in distance* -- a pixel wrong far from any true wall is
    punished exponentially harder than one wrong at the boundary.  A hedged
    ramp is wrong over a wide band, so its tails land in the expensive region;
    a sharp wall misplaced by d is wrong over a band of width d, costing
    ~scale*(exp(d/scale) - 1).  Both blur and displacement are punished, and
    neither saturates.

    *Absolute error, not squared* -- this is what removes hedging rather than
    merely discouraging it.  The minimiser of a weighted squared error is a
    weighted conditional *mean*, which is exactly the blurred compromise L2
    and BCE both converge to.  The minimiser of a weighted absolute error is a
    weighted conditional *median*, and the median of a binary field is binary:
    under uncertainty this loss picks a side instead of averaging.

    It also fixes the collapse of :class:`WallPlacementLoss`, whose gradient
    ``phi`` has no sign change and so drove the logits to saturation (a uniform
    1.0 field, 100% saturated, frozen from epoch 0).  Here the gradient is
    ``w * sign(pred - y)``: same non-vanishing magnitude, but it reverses at
    ``pred = y``, so the optimum is a fixed point rather than something the
    optimiser sails through.

    ``cap`` bounds the distance, hence ``w``, hence the gradient.

    ``axes`` selects the geometry of the distance transform.  ``"3d"`` is the
    historical behaviour.  ``"transverse"`` runs the transform independently in
    each XY slice and drops slices with no front -- see
    :func:`transverse_signed_distance` for why the LOS axis is not merely a
    different scale but an unrepresentable one on this grid.
    """

    def __init__(self, scale: float = 8.0, cap: int = 32,
                 threshold: float = 0.5, power: float = 1.0,
                 axes: str = "3d"):
        self.scale = float(scale)
        self.cap = int(cap)
        self.threshold = float(threshold)
        self.power = float(power)
        if str(axes).lower() not in {"3d", "transverse"}:
            raise ValueError(
                f"axes must be '3d' or 'transverse', got {axes!r}"
            )
        self.axes = str(axes).lower()

    def __call__(self, out: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
        if out.shape != y.shape:
            raise ValueError(f"shape mismatch: {out.shape} != {y.shape}")
        field = _drop_channel(y.detach())
        keep = None
        with torch.no_grad():
            if self.axes == "transverse" and field.dim() == 4:
                phi, has_wall = transverse_signed_distance(
                    field, self.cap, self.threshold)
                keep = _match_rank(has_wall.to(phi.dtype), out)
            else:
                # A 2-D field is already transverse, so both modes agree there.
                phi = signed_distance(field, self.cap, self.threshold)
            w = _match_rank(torch.exp(phi.abs() / self.scale), out)
            if keep is not None:
                w = w * keep
            # Normalise over the voxels that actually contribute, so dropping
            # single-phase slices rescales the loss rather than shrinking it.
            # A cube can legitimately contain no wall at all -- cube 85 of the
            # production cache is single-phase in all 256 slices -- and then
            # `keep` is all zero. Without the guard that is 0/0, and the NaN
            # propagates into w and out through the whole eval mean.
            no_wall = False
            if keep is None:
                denom = w.mean()
            else:
                n_keep = keep.expand_as(w).sum()
                no_wall = bool(n_keep == 0)
                denom = w.sum() / n_keep.clamp_min(1.0)
            w = w / denom.clamp_min(1e-12)
        if no_wall:
            # Return through `out` so the result still carries a grad_fn --
            # a bare constant would break backward(). Zero weight, zero grad.
            return (out * 0.0).sum()
        err = (out - y).abs()
        if self.power != 1.0:
            err = err.clamp_min(1e-12) ** self.power
        if keep is None:
            return (w * err).mean()
        return (w * err).sum() / keep.expand_as(err).sum().clamp_min(1.0)


class GranulometrySpectrum:
    """Distance between predicted and true bubble-size spectra.

    A differentiable stand-in for the mean-free-path bubble-size distribution
    that ``viz/bubble_size_evaluation.py`` reports. The MFP estimator cannot be
    a training term -- it thresholds the field, takes a first-crossing along
    each ray, and histograms the result, none of which has a useful gradient.

    Morphological opening by a ball of radius ``r`` (erode, then dilate) keeps
    only structures that ball fits inside. The volume lost between consecutive
    radii is therefore the mass of structures at that scale: a size
    distribution, and the same physics the MFP estimator samples. Erosion and
    dilation are min/max filters, so both are max-pooling and differentiable
    almost everywhere. Measured against the MFP estimator on discs of radius
    2-13, the two agree at Pearson r = 0.99.

    Two consequences worth stating plainly.

    *It needs an anchor.* A size spectrum is invariant to translation, to
    rotation, and to any rearrangement that preserves sizes. Measured: a field
    with every bubble displaced 37 px scores 21x *better* than one with the
    wrong sizes. That is the same hole that sank the L2-free edge run
    (``edgeonly``, 0.9098, never beating its epoch-0 value), and it is worse
    here. Use this only as an auxiliary term over L2 or expwall.

    *It works on the raw field.* Grayscale opening needs no threshold, so
    unlike the MFP estimator there is no cutoff to choose and no gradient
    killed by a hard mask.

    ``downsample`` and ``max_slices`` bound the cost, which scales with
    slices x plane area x radius. With the separable opening below, a whole
    256-slice cube at 4 radii and 2x downsampling measures ~164 ms/step on CPU
    and 32 slices ~32 ms, so the full LOS extent is affordable and
    ``max_slices`` is a knob rather than a necessity. Radii are in cells
    *after* downsampling.
    """

    #: Additive smoothing on the per-scale mass; see :meth:`spectrum`.
    EPS = 1e-4

    def __init__(
        self,
        radii: Sequence[int] = (1, 2, 4, 8),
        downsample: int = 2,
        max_slices: int | None = 32,
        seed: int = 0,
    ):
        radii = tuple(int(r) for r in radii)
        if len(radii) < 2 or any(r <= 0 for r in radii):
            raise ValueError("radii must be at least two positive values")
        if list(radii) != sorted(radii) or len(set(radii)) != len(radii):
            raise ValueError(f"radii must be strictly increasing, got {radii}")
        if int(downsample) < 1:
            raise ValueError("downsample must be at least 1")
        if max_slices is not None and int(max_slices) < 1:
            raise ValueError("max_slices must be positive or None")
        self.radii = radii
        self.downsample = int(downsample)
        self.max_slices = None if max_slices is None else int(max_slices)
        self.seed = int(seed)

    def _transverse(self, field: torch.Tensor) -> torch.Tensor:
        """Field -> ``(batch, 1, X, Y)`` transverse planes.

        A lightcone's bubble sizes are measured transversely, per LOS slice --
        the same convention as the evaluator -- so a cube is folded into the
        batch rather than opened with a 3-D ball.
        """
        if field.dim() == 5:
            field = field[:, :1].permute(0, 4, 1, 2, 3)          # (N,W,1,X,Y)
            field = field.reshape(-1, 1, *field.shape[-2:])
        elif field.dim() == 4:
            field = field[:, :1]
        else:
            raise ValueError(f"expected a 4-D or 5-D field, got {field.dim()}-D")
        if self.max_slices is not None and len(field) > self.max_slices:
            # A fixed stride, not a random draw: the term must be comparable
            # between the two calls that make up one loss evaluation.
            step = len(field) // self.max_slices
            field = field[::step][: self.max_slices]
        if self.downsample > 1:
            field = F.avg_pool2d(field, self.downsample)
        return field

    @staticmethod
    def _open(planes: torch.Tensor, radius: int) -> torch.Tensor:
        """Morphological opening by a square of side ``2*radius + 1``.

        Min/max filters over a rectangle are separable, so the square window
        is two 1-D passes rather than one 2-D one -- bit-identical output for
        O(k) work per pixel instead of O(k^2). Measured 5.2x faster over
        radii (1, 2, 4, 8) on 32 planes of 140x140.
        """
        size = 2 * radius + 1
        rows, cols = (size, 1), (1, size)
        pad_r, pad_c = (radius, 0), (0, radius)
        x = -F.max_pool2d(-planes, rows, stride=1, padding=pad_r)
        x = -F.max_pool2d(-x, cols, stride=1, padding=pad_c)      # erosion
        x = F.max_pool2d(x, rows, stride=1, padding=pad_r)
        return F.max_pool2d(x, cols, stride=1, padding=pad_c)     # dilation

    def spectrum(self, field: torch.Tensor) -> torch.Tensor:
        """Normalised opening pattern spectrum, ``(batch, len(radii) - 1)``."""
        planes = self._transverse(field)
        volumes = []
        for radius in self.radii:
            volumes.append(self._open(planes, radius).flatten(1).mean(1))
        volume = torch.stack(volumes, dim=1)
        mass = volume[:, :-1] - volume[:, 1:]
        # Openings are monotone in radius, so the differences are non-negative
        # up to floating point; clamp rather than let a -1e-9 flip a CDF.
        mass = mass.clamp_min(0.0)
        # Additive smoothing, not a clamped denominator. A field with no
        # structure at all -- a saturated or collapsed prediction, which is
        # exactly what pred_sat_low/high watches for -- has zero mass at every
        # scale, and normalising that by a clamped 1e-8 gives a gradient of
        # order 1e8. Measured 3e6 on a constant field before this. Smoothing
        # makes the featureless case a uniform spectrum, which is the honest
        # answer, and bounds the gradient by 1/(n * EPS).
        mass = mass + self.EPS
        return mass / mass.sum(dim=1, keepdim=True)

    def __call__(self, out: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
        if out.shape != y.shape:
            raise ValueError(f"shape mismatch: {out.shape} != {y.shape}")
        predicted = self.spectrum(out)
        with torch.no_grad():
            target = self.spectrum(y)
        # Compare CDFs, not densities: this is the 1-D Wasserstein distance
        # over scale, so getting a bubble population's size slightly wrong
        # costs less than getting it wrong by an order of magnitude. A
        # bin-wise difference would treat those the same.
        return (predicted.cumsum(1) - target.cumsum(1)).abs().mean()
