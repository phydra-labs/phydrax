#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nuclide compositions and thermodynamic material state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._identity import NuclearSpeciesKey, NuclearSpeciesTable, NuclideKey


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return normalized


class CompositionBasis(StrEnum):
    ATOM_FRACTION = "atom_fraction"
    MASS_FRACTION = "mass_fraction"


@dataclass(frozen=True, slots=True)
class NuclideComposition:
    """Normalized nuclide fractions on one explicit immutable basis."""

    nuclides: tuple[NuclideKey, ...]
    fractions: np.ndarray
    basis: CompositionBasis
    source_id: str
    composition_id: str = field(init=False)

    def __post_init__(self) -> None:
        nuclides = tuple(self.nuclides)
        if not nuclides or any(not isinstance(value, NuclideKey) for value in nuclides):
            raise TypeError("nuclides must contain NuclideKey values.")
        identifiers = tuple(value.nuclide_id for value in nuclides)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Nuclide composition entries must be unique.")
        values = np.array(self.fractions, dtype=np.float64, copy=True)
        if values.shape != (len(nuclides),):
            raise ValueError("fractions must match the nuclide axis.")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Composition fractions must be finite and nonnegative.")
        tolerance = 256.0 * np.finfo(values.dtype).eps * max(1, values.size)
        if abs(float(np.sum(values)) - 1.0) > tolerance:
            raise ValueError(
                "Composition fractions must sum to one; they are not normalized implicitly."
            )
        if not isinstance(self.basis, CompositionBasis):
            raise TypeError("basis must be CompositionBasis.")
        source = _text(self.source_id, "source_id")
        values.setflags(write=False)
        object.__setattr__(self, "nuclides", nuclides)
        object.__setattr__(self, "fractions", values)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(
            self,
            "composition_id",
            canonical_fingerprint(
                {
                    "kind": "nuclide-composition",
                    "nuclides": list(identifiers),
                    "fractions": array_tree_fingerprint(values),
                    "basis": self.basis.value,
                    "source": source,
                }
            ),
        )

    def convert_basis(
        self, basis: CompositionBasis, species: NuclearSpeciesTable, /
    ) -> CompositionConversionResult:
        if not isinstance(basis, CompositionBasis):
            raise TypeError("basis must be CompositionBasis.")
        if not isinstance(species, NuclearSpeciesTable):
            raise TypeError("species must be NuclearSpeciesTable.")
        expected = tuple(
            NuclearSpeciesKey.from_nuclide(value).species_id for value in self.nuclides
        )
        actual = tuple(value.species_id for value in species.species)
        if expected != actual:
            raise ValueError(
                "Nuclear mass table must exactly match the composition axis."
            )
        masses = species.rest_masses_kg
        if np.any(masses <= 0.0):
            raise ValueError(
                "Nuclide basis conversion requires positive evaluated masses."
            )
        if basis is self.basis:
            converted = self
            residual = 0.0
        else:
            unnormalized = (
                self.fractions * masses
                if self.basis is CompositionBasis.ATOM_FRACTION
                else self.fractions / masses
            )
            converted = NuclideComposition(
                self.nuclides,
                unnormalized / np.sum(unnormalized),
                basis,
                canonical_fingerprint(
                    {
                        "kind": "composition-basis-conversion",
                        "source": self.composition_id,
                        "mass_table": species.table_id,
                        "target_basis": basis.value,
                    }
                ),
            )
            back_unnormalized = (
                converted.fractions / masses
                if self.basis is CompositionBasis.ATOM_FRACTION
                else converted.fractions * masses
            )
            back = back_unnormalized / np.sum(back_unnormalized)
            residual = float(np.max(np.abs(back - self.fractions)))
        return CompositionConversionResult(
            converted,
            residual,
            species.table_id,
            canonical_fingerprint(
                {
                    "kind": "composition-conversion-result",
                    "source": self.composition_id,
                    "target": converted.composition_id,
                    "mass_table": species.table_id,
                    "closure_residual": residual,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CompositionConversionResult:
    composition: NuclideComposition
    closure_residual: float
    species_table_id: str
    result_id: str


@dataclass(frozen=True, slots=True)
class NuclearMaterialState:
    """One composition at an explicit density and absolute temperature."""

    material_id: str
    composition: NuclideComposition
    mass_density_kg_m3: float
    temperature_k: float
    homogenization_id: str = "none"
    state_id: str = field(init=False)

    def __post_init__(self) -> None:
        material = _text(self.material_id, "material_id")
        if not isinstance(self.composition, NuclideComposition):
            raise TypeError("composition must be NuclideComposition.")
        density = float(self.mass_density_kg_m3)
        temperature = float(self.temperature_k)
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("mass_density_kg_m3 must be finite and positive.")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature_k must be finite and positive.")
        homogenization = _text(self.homogenization_id, "homogenization_id")
        object.__setattr__(self, "material_id", material)
        object.__setattr__(self, "mass_density_kg_m3", density)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "homogenization_id", homogenization)
        object.__setattr__(
            self,
            "state_id",
            canonical_fingerprint(
                {
                    "kind": "nuclear-material-state",
                    "material": material,
                    "composition": self.composition.composition_id,
                    "mass_density_kg_m3": density,
                    "temperature_k": temperature,
                    "homogenization": homogenization,
                }
            ),
        )


__all__ = [
    "CompositionBasis",
    "CompositionConversionResult",
    "NuclearMaterialState",
    "NuclideComposition",
]
