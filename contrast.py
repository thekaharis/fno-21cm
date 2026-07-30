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

import numpy as np
import torch
import torch.nn as nn

# theta above this is numerically indistinguishable from the identity, and the
# normalisation denominator ~ 1/theta underflows if it grows without bound.
THETA_MIN, THETA_MAX = 0.02, 5.0
TAU_MIN, TAU_MAX = 0.05, 0.95


def _squash(raw: torch.Tensor) -> torch.Tensor:
    """Unconstrained parameter -> theta in [THETA_MIN, THETA_MAX]."""
    return THETA_MIN + (THETA_MAX - THETA_MIN) * torch.sigmoid(raw)


def _inv_squash(theta: float) -> float:
    """Inverse of :func:`_squash`, for initialising from a target theta."""
    p = (float(theta) - THETA_MIN) / (THETA_MAX - THETA_MIN)
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


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


class ThetaSchedule(nn.Module):
    """theta as a smooth function of a field's mean value (ionisation state).

        theta(m) = theta_lo + (theta_hi - theta_lo) * sigmoid((log10(m+eps) - c)/s)

    Keyed on the *mean prediction*, which the model always has, so this is
    deployable at inference -- unlike the per-slice ``tau`` of section 3.3.

    Rationale (NOTES-contrast-map.md 6.1): the optimal sharpening is not one
    number.  At x_HI ~ 0.01 the field is a few isolated neutral islands the
    model renders as low-contrast blobs -- correctly placed, under-committed --
    and sharpening buys a held-out 2-9%.  By x_HI > 0.1 the field is a dense
    bubble network whose errors are positional, and sharpening does exactly
    nothing.  log10 because that transition spans a decade in x_HI.

    Defaults are the identity everywhere, so an unfitted schedule is a no-op.
    """

    EPS = 1e-4
    # The transition width is bounded rather than clamped. With an unbounded
    # log-width the fit degenerates whenever there is no signal to find: the
    # width collapsed 0.19 -> 0.00 -> 0.04 over three refits and then went
    # non-finite, which poisoned the model it was installed into.
    S_MIN, S_MAX = 0.02, 2.0

    def __init__(self, theta_lo: float = THETA_MAX, theta_hi: float = THETA_MAX,
                 c: float = -1.5, s: float = 0.4):
        super().__init__()
        self.raw_lo = nn.Parameter(torch.tensor(_inv_squash(theta_lo)))
        self.raw_hi = nn.Parameter(torch.tensor(_inv_squash(theta_hi)))
        self.c = nn.Parameter(torch.tensor(float(c)))
        self.raw_s = nn.Parameter(torch.tensor(self._inv_width(s)))

    @classmethod
    def _inv_width(cls, s: float) -> float:
        p = (float(s) - cls.S_MIN) / (cls.S_MAX - cls.S_MIN)
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    def width(self) -> torch.Tensor:
        return self.S_MIN + (self.S_MAX - self.S_MIN) * torch.sigmoid(self.raw_s)

    def forward(self, mean_value: torch.Tensor) -> torch.Tensor:
        lo = _squash(self.raw_lo)
        hi = _squash(self.raw_hi)
        m = torch.log10(mean_value.clamp_min(0.0) + self.EPS)
        gate = torch.sigmoid((m - self.c) / self.width())
        return lo + (hi - lo) * gate

    # -- persistence / reporting -----------------------------------------
    def state_dict_floats(self) -> dict:
        return {"theta_lo": float(_squash(self.raw_lo)),
                "theta_hi": float(_squash(self.raw_hi)),
                "c": float(self.c), "s": float(self.width())}

    def load_floats(self, d: dict) -> "ThetaSchedule":
        need = ("theta_lo", "theta_hi", "c", "s")
        if not all(np.isfinite(float(d[k])) for k in need):
            raise ValueError(f"non-finite schedule: {[float(d[k]) for k in need]}")
        with torch.no_grad():
            self.raw_lo.copy_(torch.tensor(_inv_squash(d["theta_lo"])))
            self.raw_hi.copy_(torch.tensor(_inv_squash(d["theta_hi"])))
            self.c.copy_(torch.tensor(float(d["c"])))
            self.raw_s.copy_(torch.tensor(self._inv_width(d["s"])))
        return self

    def describe(self) -> str:
        d = self.state_dict_floats()
        return (f"theta {d['theta_lo']:.3f}->{d['theta_hi']:.3f} "
                f"@ log10(m)={d['c']:.2f} width {d['s']:.2f}")


class SteppedThetaSchedule(nn.Module):
    """One independently learned theta per x_HI bin -- a lookup table, not a curve.

    :class:`ThetaSchedule` is a 4-parameter sigmoid, so it can only express a
    single monotone step: sharpen below some ionisation level, or above it.  The
    shape the data actually wants is a *band* (NOTES-contrast-map.md 6.1): the
    identity below x_HI ~ 0.005 where slices are near-empty and sharpening only
    thresholds noise, strong sharpening across 0.005-0.05 where the field is a
    few isolated neutral islands, and the identity again above 0.1 where the
    errors are positional rather than blur.  A sigmoid cannot represent that
    notch at all; a per-bin table can, and can also be non-monotonic.

    Bin edges are log-spaced by default because the interesting range spans a
    decade near zero while everything above 0.1 behaves identically.

    Empty bins keep their previous value rather than drifting: a refit batch
    will not populate every bin, and a bin with no samples gets no gradient.
    """

    def __init__(self, n_bins: int = 14, lo: float = 1e-3, hi: float = 1.0,
                 init: float = THETA_MAX, edges=None):
        super().__init__()
        if edges is None:
            inner = np.geomspace(lo, hi, n_bins - 1)
            edges = np.concatenate([[0.0], inner])
        edges = np.asarray(edges, dtype=np.float64)
        self.register_buffer("edges", torch.tensor(edges[1:], dtype=torch.float32))
        self.n_bins = len(edges)
        self.raw = nn.Parameter(
            torch.full((self.n_bins,), _inv_squash(init), dtype=torch.float32))

    def thetas(self) -> torch.Tensor:
        return _squash(self.raw)

    def bin_of(self, key: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(key.detach(), self.edges)

    def forward(self, mean_value: torch.Tensor) -> torch.Tensor:
        return self.thetas()[self.bin_of(mean_value)]

    # -- persistence / reporting -----------------------------------------
    def state_dict_floats(self) -> dict:
        return {"kind": "stepped",
                "edges": [0.0] + [float(e) for e in self.edges],
                "thetas": [float(v) for v in self.thetas()]}

    def load_floats(self, d: dict) -> "SteppedThetaSchedule":
        th = [float(v) for v in d["thetas"]]
        if not all(np.isfinite(v) for v in th):
            raise ValueError(f"non-finite stepped schedule: {th}")
        with torch.no_grad():
            self.raw.copy_(torch.tensor([_inv_squash(v) for v in th]))
        return self

    def describe(self) -> str:
        th = self.thetas()
        e = [0.0] + [float(x) for x in self.edges]
        parts = [f"{e[i]:.3g}:{float(th[i]):.2f}" for i in range(self.n_bins)]
        return "theta/bin " + " ".join(parts)


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

    MODES = ("off", "global", "head", "xhi")

    def __init__(self, mode: str = "global", schedule: dict | None = None,
                 freeze: bool = False, schedule_kind: str = "sigmoid",
                 n_bins: int = 14):
        super().__init__()
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(
                f"contrast mode must be one of {'|'.join(self.MODES)}, got {mode!r}")
        self.mode = mode
        self.freeze = bool(freeze)
        # Runtime gate for the alternating refit scheme: epoch 0 trains with no
        # map because no prediction exists yet to fit a schedule against.
        self.enabled = True
        if mode == "global":
            # sigmoid(3) -> theta near THETA_MAX, sigmoid(0) -> tau = 1/2
            self.raw = nn.Parameter(torch.tensor([3.0, 0.0]))
        elif mode == "head":
            self.head = ContrastHead()
        elif mode == "xhi":
            # "stepped" is a per-bin lookup table; "sigmoid" is the 4-parameter
            # curve, which cannot express a band (see SteppedThetaSchedule).
            kind = str(schedule_kind).lower()
            if kind not in ("sigmoid", "stepped"):
                raise ValueError(f"schedule kind must be sigmoid|stepped, got {kind!r}")
            self.schedule_kind = kind
            self.schedule = (SteppedThetaSchedule(n_bins=n_bins)
                             if kind == "stepped" else ThetaSchedule())
            if schedule:
                self.schedule.load_floats(schedule)
            if self.freeze:
                # A *learnable* map is neutralised by the network: g is monotone
                # and invertible, so it can emit g^-1(what it wanted) and cancel
                # the map (measured -- a learned global theta went to 4.43 ~=
                # identity). Freezing removes that escape from the map itself;
                # note the output sigmoid can still partly absorb it.
                for p in self.schedule.parameters():
                    p.requires_grad_(False)

    def describe(self) -> str | None:
        """Current parameters for logging; None when there is nothing scalar."""
        if self.mode == "xhi":
            return self.schedule.describe() + (" [frozen]" if self.freeze else "")
        v = self.parameters_value()
        return None if v is None else f"theta={v[0]:.3f} tau={v[1]:.3f}"

    def parameters_value(self) -> tuple[float, float] | None:
        """Current (theta, tau) for logging; None in head/xhi/off mode."""
        if self.mode != "global":
            return None
        gate = torch.sigmoid(self.raw.detach())
        return (
            float(THETA_MIN + (THETA_MAX - THETA_MIN) * gate[0]),
            float(TAU_MIN + (TAU_MAX - TAU_MIN) * gate[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off" or not self.enabled:
            return x
        if self.mode == "global":
            gate = torch.sigmoid(self.raw)
            theta = THETA_MIN + (THETA_MAX - THETA_MIN) * gate[0]
            tau = TAU_MIN + (TAU_MAX - TAU_MIN) * gate[1]
            return apply_contrast(x, theta, tau)
        if self.mode == "xhi":
            # Mean of the field is the ionisation-state proxy. Detached: the
            # schedule should react to the prediction, not give the network a
            # gradient path for gaming its own mean to pick a softer theta.
            if x.dim() == 5:
                # A lightcone cube spans the whole reionisation history along
                # its trailing line-of-sight axis, so a single mean per cube
                # would average x_HI ~ 0 and x_HI ~ 1 together and describe
                # neither. theta is therefore per LOS slice, which is also
                # exactly the quantity the 2-D schedule was fitted on.
                m = x.detach().mean(dim=(-3, -2))          # (N, C, W)
                if m.dim() == 3:
                    m = m[:, 0]                            # (N, W)
                theta = self.schedule(m.reshape(-1)).view(m.shape)
                theta = theta[:, None, None, None, :]      # (N,1,1,1,W)
            else:
                mean_value = x.detach().flatten(1).mean(1)
                theta = self.schedule(mean_value).view(
                    (-1,) + (1,) * (x.dim() - 1))
            return apply_contrast(x, theta, 0.5)
        return self.head.apply(x)


class ContrastComposed(nn.Module):
    """``base`` network followed by a learnable :class:`ContrastOutput`."""

    def __init__(self, base: nn.Module, mode: str = "global",
                 schedule: dict | None = None, freeze: bool = False,
                 schedule_kind: str = "sigmoid", n_bins: int = 14):
        super().__init__()
        self.base = base
        self.contrast = ContrastOutput(mode, schedule=schedule, freeze=freeze,
                                       schedule_kind=schedule_kind, n_bins=n_bins)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.contrast(self.base(x, **kwargs))
