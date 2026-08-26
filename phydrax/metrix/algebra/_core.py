#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from fractions import Fraction
from operator import index
from typing import Any

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._frozendict import frozendict
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._properties import AlgebraPropertyEvidence, audit_algebra_properties
from ._resources import AlgebraResourceBudget, AlgebraResourceEvidence
from ._structure import (
    AlgebraRationalMap,
    AlgebraRationalVector,
    AlgebraStructureTable,
    AlgebraTerm,
)


class AbstractFiniteRealAlgebraSpec(StrictModule, NonTrainableState):
    """Finite-dimensional algebra over real coordinates with exact structure data."""

    family: str = eqx.field(static=True)
    coordinate_dimension: int = eqx.field(static=True)
    basis_ids: tuple[str, ...] = eqx.field(static=True)
    scalar_basis_index: int = eqx.field(static=True)
    structure: AlgebraStructureTable
    unit: AlgebraRationalVector
    conjugation: AlgebraRationalMap
    properties: AlgebraPropertyEvidence
    budget: AlgebraResourceBudget
    convention: frozendict[str, Any]
    resources: AlgebraResourceEvidence
    algebra_id: str = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def _family_marker(self) -> str:
        raise NotImplementedError

    def __init__(
        self,
        family: str,
        basis_ids: Sequence[str],
        terms: Sequence[AlgebraTerm],
        unit: Sequence[int | tuple[int, int] | Fraction],
        conjugation: Sequence[Sequence[int | tuple[int, int] | Fraction]],
        /,
        *,
        scalar_basis_index: int = 0,
        convention: Mapping[str, Any] | None = None,
        family_claims=None,
        budget: AlgebraResourceBudget | None = None,
    ):
        family_ = str(family)
        labels = tuple(str(value) for value in basis_ids)
        if not family_ or not labels or any(not value for value in labels):
            raise ValueError("Algebra family and basis labels must be non-empty.")
        if len(set(labels)) != len(labels):
            raise ValueError("Algebra basis labels must be unique.")
        if isinstance(scalar_basis_index, bool):
            raise TypeError("scalar_basis_index must be an integer.")
        scalar = index(scalar_basis_index)
        if scalar < 0 or scalar >= len(labels):
            raise ValueError("scalar_basis_index lies outside the coordinate basis.")
        budget_ = AlgebraResourceBudget() if budget is None else budget
        if not isinstance(budget_, AlgebraResourceBudget):
            raise TypeError("budget must be AlgebraResourceBudget or None.")
        budget_.admit_coordinates(len(labels))
        structure = AlgebraStructureTable(len(labels), terms, budget=budget_)
        unit_ = AlgebraRationalVector(unit)
        conjugation_ = AlgebraRationalMap(conjugation)
        if len(unit_.entries) != len(labels) or conjugation_.dimension != len(labels):
            raise ValueError(
                "Algebra unit/conjugation dimensions do not match its basis."
            )
        properties = audit_algebra_properties(
            structure,
            labels,
            unit_,
            conjugation_,
            budget_,
            family_claims=family_claims,
        )
        convention_ = {} if convention is None else dict(convention)
        algebra_id = canonical_fingerprint(
            {
                "kind": "finite-real-algebra-v1",
                "family": family_,
                "basis": list(labels),
                "scalar_basis_index": scalar,
                "structure": structure.table_id,
                "unit": unit_.vector_id,
                "conjugation": conjugation_.map_id,
                "convention": convention_,
            }
        )
        audit_terms = len(labels) ** 3 * 8 + len(labels) ** 2 * 8 + len(labels) * 4
        plan_bytes = structure.term_count * 5 * 8
        resources = AlgebraResourceEvidence(
            coordinate_count=len(labels),
            product_pairs=len(labels) ** 2,
            product_terms=structure.term_count,
            audit_terms=audit_terms,
            plan_bytes=plan_bytes,
            dense_kernel_bytes=0,
            budget=budget_,
        )
        self.family = family_
        self.coordinate_dimension = len(labels)
        self.basis_ids = labels
        self.scalar_basis_index = scalar
        self.structure = structure
        self.unit = unit_
        self.conjugation = conjugation_
        self.properties = properties
        self.convention = frozendict(convention_)
        self.budget = budget_
        self.resources = resources
        self.algebra_id = algebra_id
        self.spec_id = canonical_fingerprint(
            {
                "kind": "finite-real-algebra-spec-v1",
                "algebra": algebra_id,
                "budget": budget_.budget_id,
                "properties": properties.evidence_id,
                "resources": resources.evidence_id,
            }
        )

    def require_compatible(self, other: "AbstractFiniteRealAlgebraSpec", /) -> None:
        if not isinstance(other, AbstractFiniteRealAlgebraSpec):
            raise TypeError("Expected an AbstractFiniteRealAlgebraSpec.")
        if self.algebra_id != other.algebra_id:
            raise ValueError("Finite real algebra specifications do not match.")

    def basis_index(self, basis_id: str, /) -> int:
        identifier = str(basis_id)
        if identifier not in self.basis_ids:
            raise KeyError(f"Unknown algebra basis label {identifier!r}.")
        return self.basis_ids.index(identifier)

    def scalar_coordinates(self, value: int | float = 1, /) -> tuple[Fraction, ...]:
        coordinates = [Fraction(0) for _ in range(self.coordinate_dimension)]
        coordinates[self.scalar_basis_index] = Fraction(value)
        return tuple(coordinates)

    def conjugate_exact(self, value: Sequence[Fraction], /) -> tuple[Fraction, ...]:
        return self.conjugation.apply(value)

    def product_exact(
        self,
        left: Sequence[Fraction],
        right: Sequence[Fraction],
        /,
    ) -> tuple[Fraction, ...]:
        return self.structure.multiply(left, right)

    def prepare_product(self, **kwargs):
        from ._product import AlgebraProductPlan

        return AlgebraProductPlan(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "basis_ids": list(self.basis_ids),
            "scalar_basis_index": self.scalar_basis_index,
            "terms": [list(term) for term in self.structure.terms],
            "unit": [list(entry) for entry in self.unit.entries],
            "conjugation": [
                [list(entry) for entry in row] for row in self.conjugation.rows
            ],
            "convention": dict(self.convention),
            "claims": {
                claim.property_name: {
                    "status": claim.status,
                    "source": claim.source,
                    "witness": list(claim.witness),
                    "work": claim.work,
                }
                for claim in self.properties.claims
            },
            "budget": {
                "maximum_coordinates": self.budget.maximum_coordinates,
                "maximum_product_pairs": self.budget.maximum_product_pairs,
                "maximum_product_terms": self.budget.maximum_product_terms,
                "maximum_audit_terms": self.budget.maximum_audit_terms,
                "maximum_plan_bytes": self.budget.maximum_plan_bytes,
                "maximum_dense_kernel_bytes": self.budget.maximum_dense_kernel_bytes,
            },
        }


class FiniteRealAlgebraSpec(AbstractFiniteRealAlgebraSpec):
    """Public exact sparse-table finite real algebra specification."""

    def _family_marker(self) -> str:
        return self.family

    @staticmethod
    def from_dict(value: Mapping[str, Any], /) -> "FiniteRealAlgebraSpec":
        budget_value = value["budget"]
        if not isinstance(budget_value, Mapping):
            raise TypeError("Serialized algebra budget must be a mapping.")
        budget = AlgebraResourceBudget(
            maximum_coordinates=int(budget_value["maximum_coordinates"]),
            maximum_product_pairs=int(budget_value["maximum_product_pairs"]),
            maximum_product_terms=int(budget_value["maximum_product_terms"]),
            maximum_audit_terms=int(budget_value["maximum_audit_terms"]),
            maximum_plan_bytes=int(budget_value["maximum_plan_bytes"]),
            maximum_dense_kernel_bytes=int(budget_value["maximum_dense_kernel_bytes"]),
        )
        claims_value = value["claims"]
        if not isinstance(claims_value, Mapping):
            raise TypeError("Serialized algebra claims must be a mapping.")
        claims = {
            str(name): (
                claim["status"],
                claim["source"],
                tuple(str(item) for item in claim["witness"]),
                int(claim["work"]),
            )
            for name, claim in claims_value.items()
        }
        return FiniteRealAlgebraSpec(
            str(value["family"]),
            tuple(str(item) for item in value["basis_ids"]),
            tuple(tuple(int(item) for item in term) for term in value["terms"]),
            tuple(tuple(int(item) for item in entry) for entry in value["unit"]),
            tuple(
                tuple(tuple(int(item) for item in entry) for entry in row)
                for row in value["conjugation"]
            ),
            scalar_basis_index=int(value["scalar_basis_index"]),
            convention=value["convention"],
            family_claims=claims,
            budget=budget,
        )


__all__ = ["AbstractFiniteRealAlgebraSpec", "FiniteRealAlgebraSpec"]
