"""Shared schema and HDF5 reader for sampled lightcone parameters."""

from __future__ import annotations

import sys

import h5py
import numpy as np


PARAM_NAMES = (
    "F_ESC10",
    "F_STAR10",
    "ALPHA_ESC",
    "ALPHA_STAR",
    "L_X",
    "NU_X_THRESH",
    "M_TURN",
    "t_STAR",
    "X_RAY_SPEC_INDEX",
    "OMm",
    "SIGMA_8",
)


def read_sampled_params(
    h5_file: h5py.File,
    names: tuple[str, ...] = PARAM_NAMES,
) -> np.ndarray:
    """Read sampled parameters, returning NaNs when the layout is unexpected."""
    try:
        group = h5_file["params"]
        stored_names = [
            value.decode()
            if isinstance(value, (bytes, bytearray))
            else str(value)
            for value in np.asarray(group["names"])
        ]
        values = np.asarray(group["values"], dtype=np.float32).ravel()
        by_name = dict(zip(stored_names, values))
        return np.array(
            [by_name.get(name, np.nan) for name in names],
            dtype=np.float32,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"  [warn] could not read params ({exc}); storing NaNs",
            file=sys.stderr,
        )
        return np.full(len(names), np.nan, dtype=np.float32)
