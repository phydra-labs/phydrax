#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Semantic volume compartments and their oriented adjacency complex."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._fingerprint import canonical_fingerprint


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty identifier.")
    return value


def _ordered_pair(first: str, second: str, /) -> tuple[str, str]:
    return (first, second) if first < second else (second, first)


@dataclass(frozen=True, slots=True)
class CompartmentDefinition:
    compartment_id: str
    label_ids: tuple[str, ...]
    material_role: str
    containment_parent_id: str | None = None
    allowed_neighbor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "compartment_id", _identifier(self.compartment_id, "compartment_id")
        )
        labels = tuple(sorted(_identifier(value, "label_id") for value in self.label_ids))
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("label_ids must be unique and non-empty.")
        neighbors = tuple(
            sorted(
                _identifier(value, "allowed_neighbor_id")
                for value in self.allowed_neighbor_ids
            )
        )
        if len(set(neighbors)) != len(neighbors) or self.compartment_id in neighbors:
            raise ValueError("allowed_neighbor_ids must be unique and exclude self.")
        object.__setattr__(self, "label_ids", labels)
        object.__setattr__(
            self, "material_role", _identifier(self.material_role, "material_role")
        )
        object.__setattr__(self, "allowed_neighbor_ids", neighbors)
        if self.containment_parent_id is not None:
            object.__setattr__(
                self,
                "containment_parent_id",
                _identifier(self.containment_parent_id, "containment_parent_id"),
            )


@dataclass(frozen=True, slots=True)
class CompartmentInterfaceDefinition:
    interface_id: str
    first_compartment_id: str
    second_compartment_id: str
    interface_role: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "interface_id", _identifier(self.interface_id, "interface_id")
        )
        object.__setattr__(
            self,
            "first_compartment_id",
            _identifier(self.first_compartment_id, "first_compartment_id"),
        )
        object.__setattr__(
            self,
            "second_compartment_id",
            _identifier(self.second_compartment_id, "second_compartment_id"),
        )
        object.__setattr__(
            self, "interface_role", _identifier(self.interface_role, "interface_role")
        )
        if self.first_compartment_id == self.second_compartment_id:
            raise ValueError("An interface must connect distinct compartments.")
        if not isinstance(self.required, bool):
            raise TypeError("required must be boolean.")

    @property
    def ordered_pair(self) -> tuple[str, str]:
        return _ordered_pair(self.first_compartment_id, self.second_compartment_id)


@dataclass(frozen=True, slots=True)
class CompartmentAdjacencyReport:
    expected_pairs: tuple[tuple[str, str], ...]
    observed_pairs: tuple[tuple[str, str], ...]
    missing_pairs: tuple[tuple[str, str], ...]
    forbidden_pairs: tuple[tuple[str, str], ...]
    successful: bool
    report_id: str


@dataclass(frozen=True, slots=True)
class CompartmentComplex:
    source_revision: str
    compartments: tuple[CompartmentDefinition, ...]
    interfaces: tuple[CompartmentInterfaceDefinition, ...]
    compartment_measures: tuple[tuple[str, float], ...]
    observed_adjacencies: tuple[tuple[str, str], ...]
    adjacency: CompartmentAdjacencyReport = field(init=False)
    complex_id: str = field(init=False)

    def __post_init__(self) -> None:
        revision = _identifier(self.source_revision, "source_revision")
        if not self.compartments or any(
            not isinstance(value, CompartmentDefinition) for value in self.compartments
        ):
            raise ValueError("compartments must contain CompartmentDefinition values.")
        identifiers = tuple(value.compartment_id for value in self.compartments)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Compartment identifiers must be unique.")
        known = set(identifiers)
        for compartment in self.compartments:
            if (
                compartment.containment_parent_id is not None
                and compartment.containment_parent_id not in known
            ):
                raise ValueError(
                    "Containment parents must identify declared compartments."
                )
            if any(
                neighbor not in known for neighbor in compartment.allowed_neighbor_ids
            ):
                raise ValueError("Allowed neighbors must identify declared compartments.")
        if any(
            not isinstance(value, CompartmentInterfaceDefinition)
            for value in self.interfaces
        ):
            raise TypeError(
                "interfaces must contain CompartmentInterfaceDefinition values."
            )
        interface_ids = [value.interface_id for value in self.interfaces]
        expected = tuple(
            sorted(value.ordered_pair for value in self.interfaces if value.required)
        )
        if len(set(interface_ids)) != len(interface_ids) or len(set(expected)) != len(
            expected
        ):
            raise ValueError(
                "Interface identifiers and required compartment pairs must be unique."
            )
        if any(first not in known or second not in known for first, second in expected):
            raise ValueError("Interface endpoints must identify declared compartments.")
        measures = tuple(
            sorted((str(name), float(value)) for name, value in self.compartment_measures)
        )
        if {name for name, _ in measures} != known or any(
            not np.isfinite(value) or value <= 0.0 for _, value in measures
        ):
            raise ValueError(
                "compartment_measures must provide one positive measure per compartment."
            )
        observed = tuple(
            sorted(_ordered_pair(pair[0], pair[1]) for pair in self.observed_adjacencies)
        )
        if len(set(observed)) != len(observed) or any(
            len(pair) != 2
            or pair[0] == pair[1]
            or pair[0] not in known
            or pair[1] not in known
            for pair in observed
        ):
            raise ValueError(
                "observed_adjacencies must contain unique valid compartment pairs."
            )
        allowed = {
            _ordered_pair(compartment.compartment_id, neighbor)
            for compartment in self.compartments
            for neighbor in compartment.allowed_neighbor_ids
        }
        allowed.update(value.ordered_pair for value in self.interfaces)
        missing = tuple(sorted(set(expected) - set(observed)))
        forbidden = tuple(sorted(set(observed) - allowed))
        report_id = canonical_fingerprint(
            {
                "kind": "compartment-adjacency-report",
                "expected": [list(pair) for pair in expected],
                "observed": [list(pair) for pair in observed],
                "missing": [list(pair) for pair in missing],
                "forbidden": [list(pair) for pair in forbidden],
            }
        )
        adjacency = CompartmentAdjacencyReport(
            expected,
            observed,
            missing,
            forbidden,
            not missing and not forbidden,
            report_id,
        )
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "compartment_measures", measures)
        object.__setattr__(self, "observed_adjacencies", observed)
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(
            self,
            "complex_id",
            canonical_fingerprint(
                {
                    "kind": "compartment-complex",
                    "source_revision": revision,
                    "compartments": [
                        {
                            "id": value.compartment_id,
                            "labels": list(value.label_ids),
                            "role": value.material_role,
                            "parent": value.containment_parent_id,
                            "neighbors": list(value.allowed_neighbor_ids),
                        }
                        for value in self.compartments
                    ],
                    "interfaces": [
                        {
                            "id": value.interface_id,
                            "pair": list(value.ordered_pair),
                            "role": value.interface_role,
                            "required": value.required,
                        }
                        for value in self.interfaces
                    ],
                    "measures": [list(value) for value in measures],
                    "adjacency": report_id,
                }
            ),
        )

    def require_valid_adjacency(self) -> None:
        if not self.adjacency.successful:
            raise ValueError(
                f"Compartment adjacency is invalid: missing={self.adjacency.missing_pairs}, "
                f"forbidden={self.adjacency.forbidden_pairs}."
            )


__all__ = [
    "CompartmentAdjacencyReport",
    "CompartmentComplex",
    "CompartmentDefinition",
    "CompartmentInterfaceDefinition",
]
