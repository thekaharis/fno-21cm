"""Small loss adapters used by the neuralop Trainer."""

from __future__ import annotations

import math
from collections.abc import Callable

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
