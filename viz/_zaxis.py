"""Cell-edge helpers for plotting against a redshift axis.

The 256-point multi-field cache is uniform in redshift, but native-resolution
predictions (LOS-window models) are uniform in comoving distance, so their
redshift spacing varies ~9x between z = 5 and z = 25 and differs per cone.
Stretching such an array linearly between z_min and z_max with ``imshow``
misplaces structure along the line of sight, and stacked cones would not share
an axis. Drawing with ``pcolormesh`` on true cell edges is exact for both.
"""
from __future__ import annotations

import numpy as np


def edges(centres):
    """Cell edges from monotone cell centres (length n+1)."""
    c = np.asarray(centres, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def transverse_edges(n, box_mpc):
    return np.linspace(0.0, box_mpc, n + 1)
