"""Frozen global reionization-history emulator: parameters -> mean x_HI(z).

Trained by tools_global_history_emulator.py on every raw simulation except the
validation and test cones of the x_HI and multi-field studies. Field models
use it as one extra LOS conditioning channel, x_HI_global_emulated(z), so the
reionization timing comes from a model trained on ~5300 dense histories rather
than being inferred by the field model from its 1600 cones (which regresses to
the typical history at the edges of the prior).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


class MonotoneHistory(nn.Module):
    """History rising with z: sigmoid(b + cumsum softplus(d) - 10) over increasing z."""
    def __init__(self, n_in, n_nodes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_nodes+1))

    def forward(self, theta):
        raw = self.net(theta)
        logit = raw[:, :1] + torch.cumsum(nn.functional.softplus(raw[:, 1:]), dim=1) - 10.0
        return torch.sigmoid(logit)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class HistoryEmulator:
    CHANNEL = "x_HI_global_emulated"

    def __init__(self, path, expected_sha256=None):
        self.path = Path(path)
        self.sha256 = file_sha256(self.path)
        if expected_sha256 is not None and expected_sha256 != self.sha256:
            raise ValueError(f"history emulator {self.path} changed since training "
                             f"({self.sha256} != {expected_sha256})")
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        self.names = [str(n) for n in ck["names"]]
        self.z = np.asarray(ck["z"], dtype=np.float64)                  # increasing
        self.mean, self.std = np.asarray(ck["theta_mean"]), np.asarray(ck["theta_std"])
        self.models = []
        for state in ck["states"]:
            m = MonotoneHistory(len(self.names), len(self.z), ck.get("hidden", 256))
            m.load_state_dict(state)
            m.eval()
            self.models.append(m)

    def describe(self):
        return {"path": str(self.path.resolve()), "sha256": self.sha256,
                "members": len(self.models), "channel": self.CHANNEL}

    def ensemble(self, theta, names):
        """(members, n, nodes) histories for parameters given in ``names`` order."""
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        order = [list(names).index(n) for n in self.names]
        t = torch.tensor((theta[:, order]-self.mean)/self.std, dtype=torch.float32)
        with torch.no_grad():
            return torch.stack([m(t) for m in self.models]).numpy()

    def histories(self, theta, names):
        return self.ensemble(theta, names).mean(0)
