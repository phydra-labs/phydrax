#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nuclear reaction channels with exact conserved-quantity ledgers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral

from .._fingerprint import canonical_fingerprint
from ._identity import NuclearSpeciesKey, NuclearSpeciesTable
from ._provenance import NuclearDataProvenance


SPEED_OF_LIGHT_M_S = 299_792_458.0


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return normalized


@dataclass(frozen=True, slots=True)
class NuclearReactionParticipant:
    species: NuclearSpeciesKey
    multiplicity: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.species, NuclearSpeciesKey):
            raise TypeError("species must be NuclearSpeciesKey.")
        if isinstance(self.multiplicity, bool) or not isinstance(
            self.multiplicity, Integral
        ):
            raise TypeError("multiplicity must be an integer.")
        multiplicity = int(self.multiplicity)
        if multiplicity < 1:
            raise ValueError("multiplicity must be positive.")
        object.__setattr__(self, "multiplicity", multiplicity)


@dataclass(frozen=True, slots=True)
class NuclearReactionConservation:
    baryon_defect: int
    charge_defect: int
    lepton_defect: int

    @property
    def successful(self) -> bool:
        return self.baryon_defect == self.charge_defect == self.lepton_defect == 0


@dataclass(frozen=True, slots=True)
class NuclearReactionChannel:
    """One fully enumerated reaction branch and independently sourced rate data."""

    name: str
    reactants: tuple[NuclearReactionParticipant, ...]
    products: tuple[NuclearReactionParticipant, ...]
    q_value_j: float
    mass_table_id: str
    data: NuclearDataProvenance
    conservation: NuclearReactionConservation
    channel_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = _text(self.name, "name")
        reactants = tuple(self.reactants)
        products = tuple(self.products)
        if (
            not reactants
            or not products
            or any(
                not isinstance(item, NuclearReactionParticipant)
                for item in (*reactants, *products)
            )
        ):
            raise TypeError(
                "Reaction reactants and products must be non-empty participant tuples."
            )
        q_value = float(self.q_value_j)
        if not math.isfinite(q_value):
            raise ValueError("q_value_j must be finite.")
        mass_table = _text(self.mass_table_id, "mass_table_id")
        if not isinstance(self.data, NuclearDataProvenance):
            raise TypeError("data must be NuclearDataProvenance.")
        if not isinstance(self.conservation, NuclearReactionConservation):
            raise TypeError("conservation must be NuclearReactionConservation.")
        if not self.conservation.successful:
            raise ValueError(
                "Nuclear reaction branch violates a declared conserved quantity."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "reactants", reactants)
        object.__setattr__(self, "products", products)
        object.__setattr__(self, "q_value_j", q_value)
        object.__setattr__(self, "mass_table_id", mass_table)
        object.__setattr__(
            self,
            "channel_id",
            canonical_fingerprint(
                {
                    "kind": "nuclear-reaction-channel",
                    "name": name,
                    "reactants": [
                        [item.species.species_id, item.multiplicity] for item in reactants
                    ],
                    "products": [
                        [item.species.species_id, item.multiplicity] for item in products
                    ],
                    "q_value_j": q_value,
                    "mass_table": mass_table,
                    "data": self.data.provenance_id,
                    "conservation": [
                        self.conservation.baryon_defect,
                        self.conservation.charge_defect,
                        self.conservation.lepton_defect,
                    ],
                }
            ),
        )

    @classmethod
    def from_species_table(
        cls,
        name: str,
        reactants: tuple[NuclearReactionParticipant, ...],
        products: tuple[NuclearReactionParticipant, ...],
        species: NuclearSpeciesTable,
        data: NuclearDataProvenance,
        /,
    ) -> NuclearReactionChannel:
        if not isinstance(species, NuclearSpeciesTable):
            raise TypeError("species must be NuclearSpeciesTable.")
        reactants_ = tuple(reactants)
        products_ = tuple(products)
        reactant_mass = sum(
            item.multiplicity * species.rest_masses_kg[species.index(item.species)]
            for item in reactants_
        )
        product_mass = sum(
            item.multiplicity * species.rest_masses_kg[species.index(item.species)]
            for item in products_
        )
        conservation = NuclearReactionConservation(
            _quantum_total(products_, "baryon_number")
            - _quantum_total(reactants_, "baryon_number"),
            _quantum_total(products_, "charge_number")
            - _quantum_total(reactants_, "charge_number"),
            _quantum_total(products_, "lepton_number")
            - _quantum_total(reactants_, "lepton_number"),
        )
        return cls(
            name,
            reactants_,
            products_,
            float((reactant_mass - product_mass) * SPEED_OF_LIGHT_M_S**2),
            species.table_id,
            data,
            conservation,
        )


def _quantum_total(
    participants: tuple[NuclearReactionParticipant, ...], quantum: str, /
) -> int:
    if quantum == "baryon_number":
        values = tuple(item.species.baryon_number for item in participants)
    elif quantum == "charge_number":
        values = tuple(item.species.charge_number for item in participants)
    elif quantum == "lepton_number":
        values = tuple(item.species.lepton_number for item in participants)
    else:
        raise ValueError("Unknown nuclear conserved quantity.")
    return sum(
        item.multiplicity * value
        for item, value in zip(participants, values, strict=True)
    )


__all__ = [
    "NuclearReactionChannel",
    "NuclearReactionConservation",
    "NuclearReactionParticipant",
    "SPEED_OF_LIGHT_M_S",
]
