"""Read 21cmFAST lightcone HDF5 files with h5py only.

Lightcones written by ``py21cmfast.LightCone.save`` (raw_lightcone_v2.0 schema)::

    /lightcone/density                (HII_DIM, HII_DIM, n_los) float32
    /lightcone/neutral_fraction       (HII_DIM, HII_DIM, n_los) float32
    /lightcone/brightness_temp        (HII_DIM, HII_DIM, n_los) float32
    /lightcone/los_velocity           (HII_DIM, HII_DIM, n_los) float32
    /lightcone/lightcone_distances    (n_los,) float64    [comoving Mpc]
    /lightcone/lightcone_redshifts    (n_los,) float64
    /params/                          (attrs: F_STAR10, OMm, SIGMA_8, ...)
    /params/fixed_cosmo_params/       (attrs: hlittle, OMb, POWER_INDEX)
    /params/fixed_matter_options/     (attrs: HMF, PERTURB_ALGORITHM, ...)
    /lightcone/global_quantities/{}   (n_nodes,) float64

The class below opens the file lazily so chunks of the cube can be sliced
without loading the full lightcone into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


# Comoving-distance to redshift inversion uses a flat ΛCDM cosmology.
# We avoid astropy here so the pipeline has no heavy dependency; the
# accuracy is sufficient for ordering / labelling LOS slices.
_C_KM_S = 299792.458


@dataclass(frozen=True)
class LightconeMeta:
    """Immutable summary of a lightcone file's geometry and parameters.

    Attributes:
        path: Filesystem path to the HDF5 file.
        fields: Tuple of field names present under ``/lightcones``.
        transverse: Resolution of the square transverse plane (``HII_DIM``).
        n_los: Number of cells along the line-of-sight axis.
        box_len: Transverse comoving box length in Mpc.
        cell_size: Size of a single transverse cell in Mpc.
        z_min: Minimum redshift reached along the LOS.
        z_max: Maximum redshift reached along the LOS.
        cosmo: Dictionary of cosmological parameters (e.g. ``hlittle``, ``OMm``).
        astro: Dictionary of astrophysical parameters (e.g. ``HII_EFF_FACTOR``).
        sim_options: Dictionary of simulation-option parameters (e.g. ``BOX_LEN``).
    """

    path: Path
    fields: tuple[str, ...]
    transverse: int  # HII_DIM (assumed square in transverse)
    n_los: int
    box_len: float  # transverse comoving Mpc
    cell_size: float  # transverse cell size, Mpc
    z_min: float
    z_max: float
    cosmo: dict
    astro: dict
    sim_options: dict


class LightconeFile:
    """Lazy reader for a single lightcone HDF5.

    Open with the context manager (preferred) or via ``open()`` / ``close()``.
    Field arrays are returned as ``np.float32`` (cast on read for FNO use).
    """

    REQUIRED_FIELDS = ("density", "neutral_fraction")

    def __init__(self, path: str | Path):
        """Initialise the reader without opening the file on disk.

        Args:
            path: Path to the ``lightcone.h5`` file.
        """
        self.path = Path(path)
        self._h5: h5py.File | None = None
        self._meta: LightconeMeta | None = None

    # ------------------------------------------------------------------ open
    def __enter__(self) -> "LightconeFile":
        """Context-manager entry: open the HDF5 and return *self*."""
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context-manager exit: unconditionally close the HDF5."""
        self.close()

    def open(self) -> None:
        """Open the HDF5 file in read-only mode.

        Safe to call multiple times: subsequent calls are no-ops if the
        file is already open.
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")

    def close(self) -> None:
        """Close the HDF5 file and release the handle.

        Safe to call multiple times: subsequent calls are no-ops.
        """
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    # --------------------------------------------------------------- meta
    def _ensure_open(self) -> h5py.File:
        """Return the live ``h5py.File`` handle or raise ``RuntimeError``.

        Returns:
            The currently open HDF5 file object.

        Raises:
            RuntimeError: If the file has not been opened.
        """
        if self._h5 is None:
            raise RuntimeError(f"LightconeFile {self.path} is not open")
        return self._h5

    @staticmethod
    def _attrs_to_dict(group: h5py.Group) -> dict:
        """Convert an HDF5 group's attributes to a plain Python dictionary.

        NumPy scalars and arrays are coerced to Python built-in types
        so that the metadata can be serialized to JSON later.

        Args:
            group: An ``h5py.Group`` whose attributes will be read.

        Returns:
            A dictionary with native Python types.
        """
        out: dict = {}
        for k, v in group.attrs.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.generic,)):
                out[k] = v.item() 
            else:
                out[k] = v
        return out

    @property
    def meta(self) -> LightconeMeta:
        """Read and cache the file's metadata.

        On first access the method inspects the HDF5 shape, parses
        parameter groups, and inverts comoving distances to redshifts.
        The result is cached so that repeated accesses are free.

        Returns:
            A :class:`LightconeMeta` dataclass describing this file.

        Raises:
            ValueError: If the lightcone shape is not 3-D or the
                transverse axes are not square.
        """
        if self._meta is not None:
            return self._meta
        f = self._ensure_open()
        lc = f["lightcone"]
        _3d = tuple(k for k, v in lc.items()
                     if isinstance(v, h5py.Dataset) and v.ndim == 3)
        if not _3d:
            raise ValueError(f"no 3-D fields in {self.path}")
        shape = lc[_3d[0]].shape
        if shape[0] != shape[1]:
            raise ValueError(
                f"unexpected lightcone shape {shape} in {self.path}; "
                "expected (HII_DIM, HII_DIM, n_los)"
            )
        transverse = int(shape[0])
        n_los = int(shape[2])
        cosmo = self._attrs_to_dict(f["params/fixed_cosmo_params"])
        astro = self._attrs_to_dict(f["params"])
        sim_opts = self._attrs_to_dict(f["params/fixed_matter_options"])
        box_len = float(f.attrs.get("box_len_mpc", 0.0))
        cell_size = box_len / transverse if transverse else 0.0

        zs = np.asarray(f["lightcone/lightcone_redshifts"], dtype=np.float64)
        z_min = float(zs.min())
        z_max = float(zs.max())

        self._meta = LightconeMeta(
            path=self.path,
            fields=_3d,
            transverse=transverse,
            n_los=n_los,
            box_len=box_len,
            cell_size=cell_size,
            z_min=z_min,
            z_max=z_max,
            cosmo=cosmo,
            astro=astro,
            sim_options=sim_opts,
        )
        return self._meta

    # ---------------------------------------------------------- field read
    def _field(self, name: str) -> h5py.Dataset:
        """Return an ``h5py.Dataset`` handle for a named lightcone field.

        Args:
            name: Field name (e.g. ``"density"`` or ``"neutral_fraction"``).

        Returns:
            The corresponding HDF5 dataset object.

        Raises:
            KeyError: If the field does not exist in this file.
        """
        f = self._ensure_open()
        if name not in f["lightcone"]:
            raise KeyError(f"field {name!r} not in {self.path}")
        return f["lightcone"][name]

    def field_shape(self, name: str) -> tuple[int, int, int]:
        """Return the shape of a named field as a 3-tuple.

        Args:
            name: Field name present in the file.

        Returns:
            Tuple ``(transverse_x, transverse_y, n_los)``.
        """
        return tuple(self._field(name).shape)

    def read_chunk(self, name: str, los_start: int, los_end: int) -> np.ndarray:
        """Extract a LOS slice ``[:, :, los_start:los_end]`` as float32.

        This is the primary I/O method used during training: it avoids
        loading the full cube into RAM.

        Args:
            name: Field name to read.
            los_start: Inclusive start index along the LOS axis.
            los_end: Exclusive end index along the LOS axis.

        Returns:
            A NumPy ``float32`` array of shape ``(Nx, Ny, los_end-los_start)``.

        Raises:
            ValueError: If the slice bounds are out of range or invalid.
        """
        ds = self._field(name)
        if los_start < 0 or los_end > ds.shape[2] or los_end <= los_start:
            raise ValueError(
                f"bad LOS slice [{los_start}:{los_end}] for n_los={ds.shape[2]}"
            )
        arr = ds[:, :, los_start:los_end]
        return np.asarray(arr, dtype=np.float32)

    def read_full(self, name: str) -> np.ndarray:
        """Read an entire field into memory as float32.

        Args:
            name: Field name to read.

        Returns:
            A NumPy ``float32`` array of shape ``(Nx, Ny, N_los)``.
        """
        ds = self._field(name)
        return np.asarray(ds[...], dtype=np.float32)

    # --------------------------------------------------- los interpolation
    def read_interpolated(self, name: str,
                          target_z: np.ndarray) -> np.ndarray:
        """Read a 3-D field and interpolate it to a target redshift grid.

        Uses linear interpolation along the LOS axis (axis=2) so each
        transverse pixel ``(x, y)`` is warped independently from the
        native ``los_redshifts()`` grid onto *target_z*.

        Args:
            name: Field name (e.g. ``"density"``).
            target_z: 1-D array of target redshifts, monotonically
                increasing.

        Returns:
            Float32 array of shape ``(Nx, Ny, len(target_z))``.
        """
        from scipy.interpolate import interp1d

        data = self.read_full(name)
        src_z = self.los_redshifts()
        n_x, n_y, _ = data.shape
        flat = data.reshape(-1, data.shape[2])
        interp_fn = interp1d(src_z, flat, kind="linear", axis=1,
                             bounds_error=False, fill_value=0.0,
                             assume_sorted=True)
        interp_flat = interp_fn(target_z)
        return interp_flat.reshape(n_x, n_y, -1).astype(np.float32)

    # --------------------------------------------------------- redshift map
    def los_redshifts(self) -> np.ndarray:
        """Return the redshift of each LOS cell (from pre-computed stored array).

        Returns:
            A 1-D ``float32`` array of length ``n_los`` containing the
            redshift of each LOS slice.
        """
        f = self._ensure_open()
        return np.asarray(f["lightcone/lightcone_redshifts"], dtype=np.float32)


# --------------------------------------------------------------------- records / discovery


@dataclass(frozen=True)
class LightconeRecord:
    """Immutable summary of a lightcone suitable for indexing and splitting.

    This is a lightweight view of :class:`LightconeMeta` tied to a single
    simulation directory.  It is consumed by the splitters, dataset, and
    standardiser.
    """

    sim_id: str
    path: Path
    fields: tuple[str, ...]
    transverse: int
    n_los: int
    box_len: float
    cell_size: float
    z_min: float
    z_max: float

    def has_required(self, fields: Iterable[str]) -> bool:
        """Check whether every field in *fields* is present in this record.

        Args:
            fields: Iterable of field names to require.

        Returns:
            ``True`` if all requested fields are available.
        """
        return all(f in self.fields for f in fields)


# --------------------------------------------------------------------- math


def _comoving_distance(z: np.ndarray, h: float, Om: float) -> np.ndarray:
    """Flat ΛCDM comoving distance in Mpc.

    Integrates ``c / H(z)`` on a fine trapezoid grid; precision is well below
    the LOS cell size for our resolutions.

    Args:
        z: Redshift value(s) as a NumPy array.
        h: Dimensionless Hubble constant (H₀ / 100 km s⁻¹ Mpc⁻¹).
        Om: Matter density parameter today.

    Returns:
        Comoving distance(s) in Mpc, same shape as *z*.
    """
    Ode = 1.0 - Om
    H0 = 100.0 * h  # km/s/Mpc
    z = np.atleast_1d(z).astype(np.float64)
    z_max = float(z.max())
    n = max(2048, int(z_max * 1024))
    grid = np.linspace(0.0, max(z_max, 1e-3), n)
    Ez = np.sqrt(Om * (1.0 + grid) ** 3 + Ode)
    integrand = _C_KM_S / (H0 * Ez)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))))
    return np.interp(z, grid, cum)


def _redshifts_from_comoving(d: np.ndarray, h: float, Om: float) -> np.ndarray:
    """Invert the comoving distance-redshift relation for flat ΛCDM.

    Args:
        d: Comoving distance(s) in Mpc.
        h: Dimensionless Hubble constant.
        Om: Matter density parameter today.

    Returns:
        Redshift(s) corresponding to *d*, same shape as input.
    """
    z_grid = np.linspace(0.0, 60.0, 8192)
    d_grid = _comoving_distance(z_grid, h=h, Om=Om)
    return np.interp(np.asarray(d, dtype=np.float64), d_grid, z_grid)


# Public helper for callers that want redshifts without constructing the
# loader — used by the manifest builder.
def comoving_to_z(d: Iterable[float], h: float = 0.6766, Om: float = 0.3097) -> np.ndarray:
    """Convert comoving distances to redshifts using a flat ΛCDM cosmology.

    This is a thin public wrapper around :func:`_redshifts_from_comoving`
    intended for scripts that do not need a full :class:`LightconeFile`.

    Args:
        d: Iterable of comoving distances in Mpc.
        h: Dimensionless Hubble constant. Defaults to Planck 2015 value.
        Om: Matter density parameter. Defaults to Planck 2015 value.

    Returns:
        NumPy array of redshifts, one per input distance.
    """
    return _redshifts_from_comoving(np.asarray(list(d), dtype=np.float64), h=h, Om=Om)
