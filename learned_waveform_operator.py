"""Real, learned waveform transforms with orthonormal sampled columns.

One random bin table per axis generates periodic, dilated, phase-shifted
candidates. Real low-pass resampling precedes reduced QR on the deployment
grid. Analysis uses U.T and synthesis U: identity mixing is an orthogonal
projection, not an inverse of discarded modes. QR mixes the candidates, so
effective columns are not necessarily literal dilations of the mother table.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def validate_waveform_shape(sizes, modes, *, context="waveform"):
    if len(sizes) != len(modes):
        raise ValueError(f"{context}: spatial shape and modes must have equal length")
    # Exclude the even-grid Nyquist bin, which has only one real phase.
    limits = tuple(1 + 2 * ((int(n) - 1) // 2) for n in sizes)
    if any(n < 1 for n in sizes) or any(
        m < 1 or m > limit for m, limit in zip(modes, limits)
    ):
        raise ValueError(
            f"{context}: modes={tuple(modes)} exceeds the real waveform limits "
            f"{limits} for spatial shape {tuple(sizes)} (Nyquist excluded)"
        )


class WaveformBank(nn.Module):
    """Learned bin amplitudes; only fixed resamplers are cached across steps."""

    def __init__(self, ndim: int, modes: Sequence[int], bins: int = 31,
                 condition_limit: float = 1e4):
        super().__init__()
        self.ndim = int(ndim)
        self.n_modes = tuple(int(m) for m in modes)
        self.bins = int(bins)
        self.condition_limit = float(condition_limit)
        if self.ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")
        if len(self.n_modes) != self.ndim or any(m < 1 for m in self.n_modes):
            raise ValueError(f"modes must contain {self.ndim} positive integers")
        if self.bins < 3 or self.bins % 2 != 1:
            raise ValueError("waveform bins must be odd and at least 3")
        if not math.isfinite(self.condition_limit) or self.condition_limit <= 1:
            raise ValueError("waveform condition_limit must be finite and greater than 1")

        # DC-only axes have no unused trainable table (important for DDP).
        self.tables = nn.ParameterDict()
        centers = (torch.arange(self.bins) + 0.5) / self.bins
        fundamental = torch.stack((
            torch.cos(2 * math.pi * centers), torch.sin(2 * math.pi * centers)
        )) * math.sqrt(2 / self.bins)
        for axis, count in enumerate(self.n_modes):
            if count == 1:
                continue
            # Random throughout; reject starts whose first harmonic is tiny.
            # A nonzero fundamental supplies independent real phase pairs.
            for _ in range(64):
                table = torch.randn(self.bins)
                table = table - table.mean()
                if (fundamental @ table).norm() > 0.15 * table.norm():
                    break
            else:
                raise RuntimeError("could not initialize a nondegenerate random waveform")
            self.tables[str(axis)] = nn.Parameter(table)
        self._resampler_cache: dict[tuple, torch.Tensor] = {}
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def _apply(self, fn, recurse=True):
        # Plain cache tensors must not retain allocations on a previous device.
        self._resampler_cache.clear()
        self.last_diagnostics.clear()
        return super()._apply(fn, recurse=recurse)

    def _resampler(self, size, modes, *, device, dtype):
        """Return (N, M-1, B), the exact truncated Fourier series of bin steps.

        Fourier functions are only a fixed anti-aliasing resampler. Learned
        parameters and task coefficients are real waveform/bin quantities.
        The first nonconstant period is N cells; mode k has period N/k.
        """
        key = (size, modes, device, dtype)
        if key not in self._resampler_cache:
            centers = (torch.arange(self.bins, device=device, dtype=dtype) + .5) / self.bins
            grid = torch.arange(size, device=device, dtype=dtype) / size
            columns = []
            for j in range(modes - 1):
                k, phase = 1 + j // 2, (j % 2) / 4
                harmonics = min((self.bins - 1) // 2, (size - 1) // (2 * k))
                h = torch.arange(1, harmonics + 1, device=device, dtype=dtype)
                factor = (2 / self.bins) * torch.sinc(h / self.bins)
                angle = 2 * math.pi * h[:, None] * centers[None, :]
                sample_angle = 2 * math.pi * (k * grid[:, None] + phase) * h[None, :]
                columns.append(
                    (torch.cos(sample_angle) * factor) @ torch.cos(angle)
                    + (torch.sin(sample_angle) * factor) @ torch.sin(angle)
                )
            # Bound resolution/device caches; they contain no learned tensors.
            if len(self._resampler_cache) >= 16:
                self._resampler_cache.clear()
            self._resampler_cache[key] = torch.stack(columns, dim=1)
        return self._resampler_cache[key]

    def sampled_candidates(self, table, size: int, modes: int):
        """Mean-zero, unit-norm nonconstant columns before orthogonalization."""
        table = table - table.mean()
        raw = self._resampler(size, modes, device=table.device, dtype=table.dtype) @ table
        raw = raw - raw.mean(dim=0, keepdim=True)
        norm = torch.linalg.vector_norm(raw, dim=0, keepdim=True)
        # Do not amplify cancellation noise from a filtered-out harmonic into
        # a seemingly independent column. This threshold is scale-relative.
        tolerance = 64 * torch.finfo(raw.dtype).eps * table.norm()
        valid = torch.isfinite(raw.detach()).all() & (norm.detach() > tolerance.detach()).all()
        if not bool(valid):
            raise RuntimeError("zero, nonfinite, or numerically unresolved waveform candidates")
        return raw / norm

    def materialize_transform(self, spatial_shape, *, device, dtype):
        sizes = tuple(int(n) for n in spatial_shape)
        validate_waveform_shape(sizes, self.n_modes)
        output_device = torch.device(device)
        # MPS does not provide the required QR/SVD kernels in supported local
        # runtimes. Tiny CPU factorizations preserve the autograd copy path.
        device = torch.device("cpu") if output_device.type == "mps" else output_device
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        bases, diagnostics = [], {}
        with torch.autocast(device_type=device.type, enabled=False):
            for axis, (size, modes) in enumerate(zip(sizes, self.n_modes)):
                dc = torch.full((size, 1), size ** -0.5, device=device, dtype=compute_dtype)
                if modes == 1:
                    bases.append(dc)
                    continue
                table = self.tables[str(axis)].to(device=device, dtype=compute_dtype)
                try:
                    raw = self.sampled_candidates(table, size, modes)
                except RuntimeError as exc:
                    raise RuntimeError(f"waveform axis {axis}, size={size}: {exc}") from exc
                # Detach diagnostics, never the transform. Check BEFORE QR:
                # rank-deficient QR can return spurious columns/NaN gradients.
                singular = torch.linalg.svdvals(raw.detach())
                condition = singular[0] / singular[-1].clamp_min(torch.finfo(compute_dtype).tiny)
                valid = (torch.isfinite(singular).all()
                         & (singular[-1] >= 1 / self.condition_limit)
                         & (condition <= self.condition_limit))
                if not bool(valid):
                    raise RuntimeError(
                        f"waveform axis {axis}, size={size}, modes={modes}: "
                        f"degenerate candidates (condition={condition.item():.3g}); "
                        "reduce waveform learning rate or inspect the bin table"
                    )
                # Include DC first so finite-precision centering is respected.
                u, r = torch.linalg.qr(torch.cat((dc, raw), dim=1), mode="reduced")
                signs = torch.where(torch.diagonal(r) < 0, -1., 1.)
                u = u * signs
                bases.append(u)
                diagnostics[f"axis{axis}_condition"] = condition.detach()
                diagnostics[f"axis{axis}_min_singular"] = singular[-1].detach()
                diagnostics[f"axis{axis}_orthogonality_error"] = (
                    u.detach().T @ u.detach()
                    - torch.eye(modes, device=device, dtype=compute_dtype)
                ).abs().max()
        self.last_diagnostics = diagnostics
        return tuple(u.to(output_device) for u in bases)


class LearnedWaveformOperator(nn.Module):
    """Separable orthonormal transforms and real per-mode channel mixing."""

    def __init__(self, channels: int, ndim: int, modes: Sequence[int],
                 bins: int = 31, condition_limit: float = 1e4):
        super().__init__()
        self.channels = int(channels)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        self.bank = WaveformBank(ndim, modes, bins, condition_limit)
        self.ndim = self.bank.ndim
        self.n_modes = self.bank.n_modes
        self.weight = nn.Parameter(
            torch.randn(self.channels, self.channels, *self.n_modes) / math.sqrt(self.channels)
        )

    def materialize_transform(self, spatial_shape, *, device, dtype):
        if self.bank is None:
            raise RuntimeError("shared waveform operator requires its branch transform")
        return self.bank.materialize_transform(spatial_shape, device=device, dtype=dtype)

    @staticmethod
    def contract(x, bases, *, analysis):
        for axis, basis in enumerate(bases):
            position = 2 + axis
            x = torch.tensordot(x, basis, dims=([position], [0 if analysis else 1]))
            x = x.movedim(-1, position)
        return x

    def forward(self, x, *, transform=None):
        if x.ndim != self.ndim + 2 or x.shape[1] != self.channels:
            raise ValueError(f"expected (B, {self.channels}, *{self.ndim} spatial axes), got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError("waveform operator requires real floating-point inputs")
        if transform is None:
            transform = self.materialize_transform(x.shape[2:], device=x.device, dtype=x.dtype)
        if len(transform) != self.ndim or any(
            tuple(u.shape) != (n, m) for u, n, m in zip(transform, x.shape[2:], self.n_modes)
        ):
            raise ValueError("waveform transform does not match input shape/modes")
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = transform[0].dtype
            coefficients = self.contract(x.to(dtype), transform, analysis=True)
            letters = "xyz"[:self.ndim]
            mixed = torch.einsum(
                f"bi{letters},io{letters}->bo{letters}", coefficients, self.weight.to(dtype)
            )
            result = self.contract(mixed, transform, analysis=False)
        return result.to(x.dtype)


class SharedWaveformBottleneck(nn.Sequential):
    """One bank owned by block zero, distinct mixing in every residual block.

    Preserve sequential state-dict paths and reuse a graph-local transform.
    The other operators keep bank=None rather than registering aliases.
    """

    def __init__(self, *blocks):
        super().__init__(*blocks)
        for block in blocks[1:]:
            block.spectral.bank = None

    def forward(self, x):
        transform = self[0].spectral.materialize_transform(
            x.shape[2:], device=x.device, dtype=x.dtype
        )
        for block in self:
            x = block(x, operator_transform=transform)
        return x


def waveform_parameter_groups(model, *, lr, weight_decay, waveform_lr_ratio=0.1):
    """Keep legacy single-group optimizers unchanged; deduplicate shared tables."""
    tables = [p for module in model.modules() if isinstance(module, WaveformBank)
              for p in module.parameters() if p.requires_grad]
    if not tables:
        return model.parameters()
    if not math.isfinite(waveform_lr_ratio) or waveform_lr_ratio <= 0:
        raise ValueError("waveform_lr_ratio must be finite and positive")
    ids = {id(p) for p in tables}
    return [
        {"params": [p for p in model.parameters() if id(p) not in ids],
         "lr": lr, "weight_decay": weight_decay},
        {"params": tables, "lr": lr * waveform_lr_ratio, "weight_decay": 0.0},
    ]


def waveform_diagnostics(model):
    """Detached last-forward diagnostics, suitable for epoch metrics JSON."""
    return {
        f"waveform_{name.replace('.', '_')}_{metric}": float(value)
        for name, module in model.named_modules() if isinstance(module, WaveformBank)
        for metric, value in module.last_diagnostics.items()
    }
