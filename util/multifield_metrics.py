"""Per-field metrics in physical units, with a training-mean baseline."""
from __future__ import annotations

import math
import numpy as np
import torch


class FieldMetrics:
    def __init__(self, targets, normalization, spectral_bins=0):
        self.targets = targets
        self.normalization = normalization
        self.sums = {name: np.zeros(10, dtype=np.float64) for name in targets}
        self.spectral_bins = spectral_bins
        self.spectra = {name: np.zeros((4, spectral_bins), dtype=np.float64) for name in targets}
        self.edges = np.linspace(0, np.sqrt(0.5) + 1e-8, spectral_bins + 1)

    @torch.no_grad()
    def update(self, prediction, target):
        for i, name in enumerate(self.targets):
            stats = self.normalization[name]
            # Accumulate dimensionless errors and correlations. Physical-unit
            # epsilon tests incorrectly classify small native velocities as
            # constant and make their spectra/skill undefined.
            p = prediction[:, i].double()
            y = target[:, i].double()
            training_mean = (stats["train_mean"] - stats["offset"]) / stats["scale"]
            error = p - y
            values = [y.numel(), error.square().sum(), error.abs().sum(), error.sum(),
                      p.sum(), y.sum(), p.square().sum(), y.square().sum(), (p*y).sum(),
                      (y - training_mean).square().sum()]
            self.sums[name] += np.array([float(v) for v in values])
            if self.spectral_bins:
                # Each lightcone slice is periodic only in X/Y. Remove its
                # spatial mean and never Fourier-transform the evolving LOS.
                p = p - p.mean(dim=(-3, -2), keepdim=True)
                y = y - y.mean(dim=(-3, -2), keepdim=True)
                fp = torch.fft.rfft2(p, dim=(-3, -2), norm="ortho")
                fy = torch.fft.rfft2(y, dim=(-3, -2), norm="ortho")
                kx = torch.fft.fftfreq(p.shape[-3], device=p.device)
                ky = torch.fft.rfftfreq(p.shape[-2], device=p.device)
                radius = torch.sqrt(kx[:, None]**2 + ky[None, :]**2)
                # Account for conjugate frequencies omitted by the real FFT.
                multiplicity = torch.full_like(ky, 2)
                multiplicity[0] = 1
                if p.shape[-2] % 2 == 0:
                    multiplicity[-1] = 1
                for b in range(self.spectral_bins):
                    mask = (radius >= self.edges[b]) & (radius < self.edges[b+1]) & (radius > 0)
                    w = (mask * multiplicity[None, :])[None, :, :, None]
                    self.spectra[name][:, b] += [float((fp.abs().square()*w).sum()),
                                               float((fy.abs().square()*w).sum()),
                                               float(((fp*fy.conj()).real*w).sum()),
                                               float(w.sum()) * p.shape[0] * p.shape[-1]]

    def result(self):
        result = {}
        for name, v in self.sums.items():
            n, squared, absolute, bias, ps, ys, pp, yy, py, baseline = v
            if not n:
                raise ValueError("cannot evaluate an empty split")
            scale = self.normalization[name]["scale"]
            denominator = math.sqrt(max(0, pp-ps*ps/n) * max(0, yy-ys*ys/n))
            item = {"mse": squared/n*scale**2, "rmse": math.sqrt(squared/n)*scale,
                    "mae": absolute/n*scale, "mean_bias": bias/n*scale,
                    "normalized_mse": squared/n,
                    "pearson_r": float(np.clip((py-ps*ys/n)/denominator, -1, 1))
                    if denominator > 1e-12 else None,
                    "train_mean_baseline_rmse": math.sqrt(baseline/n)*scale,
                    "mse_skill_vs_train_mean": 1-squared/baseline if baseline > 1e-12 else None}
            if self.spectral_bins:
                pp, yy, py, counts = self.spectra[name]
                def finite(values):
                    return [float(v) if np.isfinite(v) else None for v in values]
                with np.errstate(divide="ignore", invalid="ignore"):
                    item["transverse_spectrum"] = {
                        "k_units": "cycles per transverse pixel", "bin_edges": self.edges.tolist(),
                        "power_ratio": finite(np.where((counts > 0) & (yy > 1e-20), pp/yy, np.nan)),
                        "cross_correlation": finite(np.where((counts > 0) & (pp*yy > 1e-40),
                                                            np.clip(py/np.sqrt(pp*yy), -1, 1), np.nan)),
                        "mode_count": counts.tolist()}
            result[name] = item
        return result
