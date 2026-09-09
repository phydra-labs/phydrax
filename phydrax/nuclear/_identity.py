#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stable nuclear identities separated from evaluated physical data."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import ReferenceArtifactManifest


def _integer(value: int, name: str, /, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return normalized


@dataclass(frozen=True, slots=True)
class NuclideKey:
    """Dataset-independent identity of one nuclear ground or isomeric state."""

    proton_number: int
    mass_number: int
    isomer_index: int = 0
    nuclide_id: str = field(init=False)

    def __post_init__(self) -> None:
        protons = _integer(self.proton_number, "proton_number")
        mass = _integer(self.mass_number, "mass_number", minimum=1)
        isomer = _integer(self.isomer_index, "isomer_index")
        if protons > mass:
            raise ValueError("proton_number cannot exceed mass_number.")
        object.__setattr__(self, "proton_number", protons)
        object.__setattr__(self, "mass_number", mass)
        object.__setattr__(self, "isomer_index", isomer)
        object.__setattr__(
            self,
            "nuclide_id",
            canonical_fingerprint(
                {
                    "kind": "nuclide-key",
                    "proton_number": protons,
                    "mass_number": mass,
                    "isomer_index": isomer,
                }
            ),
        )


class NuclearParticleKind(StrEnum):
    """Elementary or unbound particles needed by nuclear reaction ledgers."""

    NEUTRON = "neutron"
    PHOTON = "photon"
    ELECTRON = "electron"
    POSITRON = "positron"
    NEUTRINO = "neutrino"
    ANTINEUTRINO = "antineutrino"


@dataclass(frozen=True, slots=True)
class NuclearSpeciesKey:
    """Exactly one nuclide or elementary nuclear-reaction particle."""

    nuclide: NuclideKey | None = None
    particle: NuclearParticleKind | None = None
    species_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.nuclide is None) == (self.particle is None):
            raise ValueError("A nuclear species key requires exactly one identity kind.")
        if self.nuclide is not None and not isinstance(self.nuclide, NuclideKey):
            raise TypeError("nuclide must be NuclideKey or None.")
        if self.particle is not None and not isinstance(
            self.particle, NuclearParticleKind
        ):
            raise TypeError("particle must be NuclearParticleKind or None.")
        identity = (
            self.nuclide.nuclide_id if self.nuclide is not None else self.particle.value
        )
        object.__setattr__(
            self,
            "species_id",
            canonical_fingerprint({"kind": "nuclear-species-key", "identity": identity}),
        )

    @classmethod
    def from_nuclide(cls, value: NuclideKey, /) -> NuclearSpeciesKey:
        return cls(nuclide=value)

    @classmethod
    def from_particle(cls, value: NuclearParticleKind, /) -> NuclearSpeciesKey:
        return cls(particle=value)

    @property
    def charge_number(self) -> int:
        return _charge_number(self)

    @property
    def baryon_number(self) -> int:
        return _baryon_number(self)

    @property
    def lepton_number(self) -> int:
        return _lepton_number(self)


class PreparedNuclearSpeciesTable(StrictModule, NonTrainableState):
    """Fixed-shape nuclear masses and conserved quantum-number vectors."""

    rest_masses_kg: Array
    charge_numbers: Array
    baryon_numbers: Array
    lepton_numbers: Array
    species_ids: tuple[str, ...] = eqx.field(static=True)
    table_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class NuclearSpeciesTable:
    """Evaluated masses on one immutable ordered nuclear-species axis."""

    species: tuple[NuclearSpeciesKey, ...]
    rest_masses_kg: np.ndarray
    mass_reference: ReferenceArtifactManifest
    table_id: str = field(init=False)

    def __post_init__(self) -> None:
        species = tuple(self.species)
        if not species or any(
            not isinstance(value, NuclearSpeciesKey) for value in species
        ):
            raise TypeError("species must contain NuclearSpeciesKey values.")
        identifiers = tuple(value.species_id for value in species)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Nuclear species must be unique and ordered explicitly.")
        masses = np.array(self.rest_masses_kg, dtype=np.float64, copy=True)
        if masses.shape != (len(species),):
            raise ValueError("rest_masses_kg must match the nuclear species axis.")
        if np.any(~np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("Nuclear rest masses must be finite and nonnegative.")
        if not isinstance(self.mass_reference, ReferenceArtifactManifest):
            raise TypeError("mass_reference must be ReferenceArtifactManifest.")
        masses.setflags(write=False)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "rest_masses_kg", masses)
        object.__setattr__(
            self,
            "table_id",
            canonical_fingerprint(
                {
                    "kind": "nuclear-species-table",
                    "species": list(identifiers),
                    "rest_masses_kg": array_tree_fingerprint(masses),
                    "reference": self.mass_reference.manifest_id,
                }
            ),
        )

    @property
    def species_count(self) -> int:
        return len(self.species)

    @property
    def charge_numbers(self) -> np.ndarray:
        values = np.asarray(
            [_charge_number(value) for value in self.species], dtype=np.int64
        )
        values.setflags(write=False)
        return values

    @property
    def baryon_numbers(self) -> np.ndarray:
        values = np.asarray(
            [_baryon_number(value) for value in self.species], dtype=np.int64
        )
        values.setflags(write=False)
        return values

    @property
    def lepton_numbers(self) -> np.ndarray:
        values = np.asarray(
            [_lepton_number(value) for value in self.species], dtype=np.int64
        )
        values.setflags(write=False)
        return values

    def index(self, species: NuclearSpeciesKey, /) -> int:
        if not isinstance(species, NuclearSpeciesKey):
            raise TypeError("species must be NuclearSpeciesKey.")
        identifiers = tuple(value.species_id for value in self.species)
        if species.species_id not in identifiers:
            raise KeyError("Nuclear species is absent from this table.")
        return identifiers.index(species.species_id)

    def prepare(self) -> PreparedNuclearSpeciesTable:
        return PreparedNuclearSpeciesTable(
            jnp.asarray(self.rest_masses_kg),
            jnp.asarray(self.charge_numbers),
            jnp.asarray(self.baryon_numbers),
            jnp.asarray(self.lepton_numbers),
            tuple(value.species_id for value in self.species),
            self.table_id,
        )


def _charge_number(value: NuclearSpeciesKey, /) -> int:
    if value.nuclide is not None:
        return value.nuclide.proton_number
    return {
        NuclearParticleKind.NEUTRON: 0,
        NuclearParticleKind.PHOTON: 0,
        NuclearParticleKind.ELECTRON: -1,
        NuclearParticleKind.POSITRON: 1,
        NuclearParticleKind.NEUTRINO: 0,
        NuclearParticleKind.ANTINEUTRINO: 0,
    }[value.particle]


def _baryon_number(value: NuclearSpeciesKey, /) -> int:
    if value.nuclide is not None:
        return value.nuclide.mass_number
    return int(value.particle is NuclearParticleKind.NEUTRON)


def _lepton_number(value: NuclearSpeciesKey, /) -> int:
    return {
        NuclearParticleKind.ELECTRON: 1,
        NuclearParticleKind.POSITRON: -1,
        NuclearParticleKind.NEUTRINO: 1,
        NuclearParticleKind.ANTINEUTRINO: -1,
    }.get(value.particle, 0)


__all__ = [
    "NuclearParticleKind",
    "NuclearSpeciesKey",
    "NuclearSpeciesTable",
    "NuclideKey",
    "PreparedNuclearSpeciesTable",
]
