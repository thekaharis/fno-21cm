"""Learnable per-sample contrast map for [0, 1]-valued field predictions.

    g(x; theta, tau) = [tanh((x - tau)/theta) - tanh(-tau/theta)]
                       / [tanh((1 - tau)/theta) - tanh(-tau/theta)]

A two-parameter monotonic reshaping of the prediction:

* ``tau`` is the threshold -- the value the transition is centred on.
* ``theta`` is the smoothness. ``theta -> 0`` gives a hard step at ``tau``;
  ``theta -> inf`` recovers the identity.

The affine renormalisation is what lets ``tau`` move off 1/2: a bare
``tanh((x - tau)/theta)`` no longer maps the endpoints to themselves once the
centre shifts, so predictions would leave [0, 1] and the mean level would
drift. Subtracting the value at 0 and dividing by the span pins ``g(0) = 0``
and ``g(1) = 1`` for every ``(theta, tau)``.

Note this is a *pointwise* map: it can steepen a transition, not move one.
Applied to a converged model trained under a pointwise loss it cannot improve
that loss by construction (the model's own output nonlinearity could already
have absorbed any such reshaping) -- unless the optimal reshaping differs
between samples, which is exactly what the per-sample head here provides.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# theta above this is numerically indistinguishable from the identity, and the
# normalisation denominator ~ 1/theta underflows if it grows without bound.
THETA_MIN, THETA_MAX = 0.02, 5.0
TAU_MIN, TAU_MAX = 0.05, 0.95


def apply_contrast(
    x: torch.Tensor,
    theta: torch.Tensor | float,
    tau: torch.Tensor | float = 0.5,
) -> torch.Tensor:
    """Apply the contrast map; ``theta``/``tau`` broadcast against ``x``."""
    if not torch.is_tensor(theta):
        theta = torch.as_tensor(theta, dtype=x.dtype, device=x.device)
    if not torch.is_tensor(tau):
        tau = torch.as_tensor(tau, dtype=x.dtype, device=x.device)
    theta = theta.clamp(THETA_MIN, THETA_MAX)
    lo = torch.tanh(-tau / theta)
    hi = torch.tanh((1.0 - tau) / theta)
    return (torch.tanh((x - tau) / theta) - lo) / (hi - lo).clamp_min(1e-12)


def prediction_statistics(x: torch.Tensor) -> torch.Tensor:
    """Pooled per-sample descriptors of a raw prediction, as head inputs.

    Deliberately computed from the prediction alone (never the target), so the
    head is a post-hoc module usable on any trained model and at inference.
    """
    flat = x.flatten(start_dim=1)
    dx = torch.roll(x, -1, dims=-2) - x
    dy = torch.roll(x, -1, dims=-1) - x
    grad = torch.sqrt(dx * dx + dy * dy + 1e-8).flatten(start_dim=1)
    return torch.stack(
        (
            flat.mean(dim=1),
            flat.std(dim=1),
            grad.mean(dim=1),
            ((flat > 0.1) & (flat < 0.9)).to(flat.dtype).mean(dim=1),
            flat.quantile(0.1, dim=1),
            flat.quantile(0.9, dim=1),
        ),
        dim=1,
    )


class ContrastHead(nn.Module):
    """Predict per-sample ``(theta, tau)`` from pooled prediction statistics.

    Initialised to sit at ``theta = THETA_MAX`` (identity) and ``tau = 0.5``,
    so attaching an untrained head leaves a model's predictions unchanged and
    any improvement is attributable to what the head learns.
    """

    def __init__(self, n_features: int = 6, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        # sigmoid(3) ~ 0.953 -> theta near THETA_MAX; sigmoid(0) = 0.5 -> tau = 1/2
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([3.0, 0.0]))

    def forward(self, prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(prediction_statistics(prediction.detach()))
        gate = torch.sigmoid(raw)
        theta = THETA_MIN + (THETA_MAX - THETA_MIN) * gate[:, 0]
        tau = TAU_MIN + (TAU_MAX - TAU_MIN) * gate[:, 1]
        shape = (-1,) + (1,) * (prediction.dim() - 1)
        return theta.view(shape), tau.view(shape)

    def apply(self, prediction: torch.Tensor) -> torch.Tensor:
        theta, tau = self(prediction)
        return apply_contrast(prediction, theta, tau)


class ContrastOutput(nn.Module):
    """Learnable output-stage contrast map, as part of the network.

    Modes:
      ``off``    -- identity passthrough.
      ``global`` -- two learned scalars shared by every sample.
      ``head``   -- per-sample ``(theta, tau)`` from :class:`ContrastHead`.

    Trained end to end this is not merely post-processing. The spectral
    operators synthesise a band-limited field; a pointwise nonlinearity
    applied *after* synthesis creates high-frequency content the operator
    itself cannot represent, and the network can co-adapt its pre-map output
    to the map rather than having a finished prediction reshaped afterwards.

    Initialised at the identity, so a run with the map enabled starts from
    exactly the baseline model.
    """

    def __init__(self, mode: str = "global"):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"off", "global", "head"}:
            raise ValueError(f"contrast mode must be off|global|head, got {mode!r}")
        self.mode = mode
        if mode == "global":
            # sigmoid(3) -> theta near THETA_MAX, sigmoid(0) -> tau = 1/2
            self.raw = nn.Parameter(torch.tensor([3.0, 0.0]))
        elif mode == "head":
            self.head = ContrastHead()

    def parameters_value(self) -> tuple[float, float] | None:
        """Current (theta, tau) for logging; None in head/off mode."""
        if self.mode != "global":
            return None
        gate = torch.sigmoid(self.raw.detach())
        return (
            float(THETA_MIN + (THETA_MAX - THETA_MIN) * gate[0]),
            float(TAU_MIN + (TAU_MAX - TAU_MIN) * gate[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return x
        if self.mode == "global":
            gate = torch.sigmoid(self.raw)
            theta = THETA_MIN + (THETA_MAX - THETA_MIN) * gate[0]
            tau = TAU_MIN + (TAU_MAX - TAU_MIN) * gate[1]
            return apply_contrast(x, theta, tau)
        return self.head.apply(x)


class ContrastComposed(nn.Module):
    """``base`` network followed by a learnable :class:`ContrastOutput`."""

    def __init__(self, base: nn.Module, mode: str = "global"):
        super().__init__()
        self.base = base
        self.contrast = ContrastOutput(mode)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.contrast(self.base(x, **kwargs))
