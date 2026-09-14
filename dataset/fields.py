"""Field identities and disjoint input/target contracts for multi-field runs.

This module uses only the standard library so experiment plans can be prepared
on a login node without importing PyTorch or opening the training data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path


@dataclass(frozen=True)
class FieldSpec:
    name: str
    units: str
    aliases: tuple[str, ...] = ()
    normalization: str = "standard"
    bounded: bool = False
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self):
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid field name: {self.name!r}")
        if self.normalization not in {"standard", "identity", "density"}:
            raise ValueError(f"unknown normalization for {self.name}")
        if self.bounded and self.normalization != "identity":
            raise ValueError("fraction outputs require identity normalization")


SPECS = (
    FieldSpec("density", "dimensionless overdensity", ("matter_density",),
              "density", minimum=-1.0),
    FieldSpec("neutral_fraction", "dimensionless", ("x_HI", "xhi", "xH_box"),
              "identity", True, 0.0, 1.0),
    FieldSpec("brightness_temp", "mK", ("brightness_temperature", "delta_Tb")),
    # Exports sometimes rescale native 21cmFAST velocities. Do not assign km/s
    # without verifying the data producer's convention.
    FieldSpec("los_velocity", "source velocity units", ("velocity_z",)),
    FieldSpec("spin_temperature", "K", ("Ts_box",), minimum=0.0),
    FieldSpec("kinetic_temp_neutral", "K", ("Tk_box",), minimum=0.0),
    FieldSpec("kinetic_temperature", "K", ("temp_kinetic_all_gas",), minimum=0.0),
    FieldSpec("ionisation_rate_G12", "1e-12 s^-1", ("Gamma12_box",), minimum=0.0),
    FieldSpec("cumulative_recombinations", "per baryon", ("dNrec_box",), minimum=0.0),
    FieldSpec("z_reion", "redshift", ("z_re_box",), minimum=0.0),
    FieldSpec("xray_ionised_fraction", "dimensionless", ("x_e_box",),
              "identity", True, 0.0, 1.0),
)
FOUR_FIELDS = ("density", "neutral_fraction", "brightness_temp", "los_velocity")


class FieldRegistry:
    def __init__(self, specs=SPECS):
        self.specs = {}
        self.names = {}
        for spec in specs:
            if spec.name in self.specs:
                raise ValueError(f"duplicate field: {spec.name}")
            self.specs[spec.name] = spec
            for name in (spec.name, *spec.aliases):
                key = name.lower()
                if key in self.names:
                    raise ValueError(f"duplicate field alias: {name}")
                self.names[key] = spec.name

    def canonical(self, name):
        try:
            return self.names[name.strip().lower()]
        except KeyError:
            raise ValueError(f"unknown field {name!r}; registered: {list(self.specs)}") from None

    def __getitem__(self, name):
        return self.specs[self.canonical(name)]

    def resolve_key(self, available, name):
        spec = self[name]
        matches = [key for key in available
                   if key.lower() in {s.lower() for s in (spec.name, *spec.aliases)}]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one stored field for {name}; found {matches}")
        return matches[0]

    def to_dict(self):
        return [asdict(spec) for spec in self.specs.values()]

    @classmethod
    def from_dict(cls, values):
        return cls(FieldSpec(**{**v, "aliases": tuple(v.get("aliases", ()))}) for v in values)

    @classmethod
    def from_file(cls, path):
        """Load a full registry, including any explicitly configured extra fields."""
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class FieldMapping:
    inputs: tuple[str, ...] = ("density",)
    targets: tuple[str, ...] = ("neutral_fraction",)
    conditioning: str = "z_params"

    def __post_init__(self):
        if not self.inputs or not self.targets:
            raise ValueError("both inputs and targets must contain at least one field")
        if len(set(self.inputs)) != len(self.inputs) or len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate input or target fields")
        if set(self.inputs) & set(self.targets):
            raise ValueError("input and target fields must be disjoint")
        if self.conditioning not in {"none", "z", "params", "z_params"}:
            raise ValueError("conditioning must be none, z, params, or z_params")

    @classmethod
    def create(cls, inputs, targets, conditioning="z_params", registry=None):
        registry = registry or FieldRegistry()
        def names(value):
            if isinstance(value, str):
                value = value.split(",")
            return tuple(registry.canonical(s) for s in value)
        return cls(names(inputs), names(targets), conditioning)

    @property
    def fields(self):
        return tuple(dict.fromkeys((*self.inputs, *self.targets)))

    @property
    def use_redshift(self):
        return self.conditioning in {"z", "z_params"}

    @property
    def use_params(self):
        return self.conditioning in {"params", "z_params"}

    @property
    def slug(self):
        return "+".join(self.inputs) + "__to__" + "+".join(self.targets)

    def to_dict(self):
        return asdict(self)


def enumerate_mappings(fields=FOUR_FIELDS, stage="all", conditioning="z_params", registry=None):
    """Stages are cumulative: pairwise (12), single-target (28), all (50)."""
    registry = registry or FieldRegistry()
    names = tuple(registry.canonical(f) for f in fields)
    if len(set(names)) != len(names) or len(names) < 2:
        raise ValueError("select at least two distinct fields")
    if stage not in {"pairwise", "single-target", "all"}:
        raise ValueError("unknown mapping stage")
    result = []
    for roles in product(range(3), repeat=len(names)):
        inputs = tuple(f for f, role in zip(names, roles) if role == 1)
        targets = tuple(f for f, role in zip(names, roles) if role == 2)
        if not inputs or not targets:
            continue
        if stage != "all" and len(targets) != 1:
            continue
        if stage == "pairwise" and len(inputs) != 1:
            continue
        result.append(FieldMapping(inputs, targets, conditioning))
    return sorted(result, key=lambda m: (len(m.targets), len(m.inputs), m.slug))
