#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class AlgebraResourceBudget(StrictModule, NonTrainableState):
    maximum_coordinates: int = eqx.field(static=True)
    maximum_product_pairs: int = eqx.field(static=True)
    maximum_product_terms: int = eqx.field(static=True)
    maximum_audit_terms: int = eqx.field(static=True)
    maximum_plan_bytes: int = eqx.field(static=True)
    maximum_dense_kernel_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_coordinates: int = 32,
        maximum_product_pairs: int = 4096,
        maximum_product_terms: int = 131_072,
        maximum_audit_terms: int = 2_000_000,
        maximum_plan_bytes: int = 64 * 1024**2,
        maximum_dense_kernel_bytes: int = 16 * 1024**2,
    ):
        names = (
            "maximum_coordinates",
            "maximum_product_pairs",
            "maximum_product_terms",
            "maximum_audit_terms",
            "maximum_plan_bytes",
            "maximum_dense_kernel_bytes",
        )
        raw = (
            maximum_coordinates,
            maximum_product_pairs,
            maximum_product_terms,
            maximum_audit_terms,
            maximum_plan_bytes,
            maximum_dense_kernel_bytes,
        )
        if any(isinstance(value, bool) for value in raw):
            raise TypeError("Algebra resource limits must be integers.")
        values = tuple(index(value) for value in raw)
        if any(value <= 0 for value in values):
            raise ValueError("Algebra resource limits must be positive.")
        for name, value in zip(names, values, strict=True):
            setattr(self, name, value)
        self.budget_id = canonical_fingerprint(
            {
                "kind": "algebra-resource-budget-v1",
                **dict(zip(names, values, strict=True)),
            }
        )

    def admit_coordinates(self, count: int, /) -> None:
        value = index(count)
        if value <= 0 or value > self.maximum_coordinates:
            raise ValueError(
                f"Algebra requires {value} coordinates; maximum is "
                f"{self.maximum_coordinates}."
            )

    def admit_product_pairs(self, count: int, /) -> None:
        value = index(count)
        if value < 0 or value > self.maximum_product_pairs:
            raise ValueError(
                f"Algebra product requires {value} basis pairs; maximum is "
                f"{self.maximum_product_pairs}."
            )

    def admit_product(self, terms: int, plan_bytes: int, /) -> None:
        term_count = index(terms)
        byte_count = index(plan_bytes)
        if term_count < 0 or term_count > self.maximum_product_terms:
            raise ValueError("Algebra product term budget exceeded.")
        if byte_count < 0 or byte_count > self.maximum_plan_bytes:
            raise ValueError("Algebra product plan-byte budget exceeded.")

    def admit_audit(self, work: int, /) -> None:
        value = index(work)
        if value < 0 or value > self.maximum_audit_terms:
            raise ValueError(
                f"Algebra law audit requires {value} terms; maximum is "
                f"{self.maximum_audit_terms}."
            )


class AlgebraResourceEvidence(StrictModule, NonTrainableState):
    coordinate_count: int = eqx.field(static=True)
    product_pairs: int = eqx.field(static=True)
    product_terms: int = eqx.field(static=True)
    audit_terms: int = eqx.field(static=True)
    plan_bytes: int = eqx.field(static=True)
    dense_kernel_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coordinate_count: int,
        product_pairs: int,
        product_terms: int,
        audit_terms: int,
        plan_bytes: int,
        dense_kernel_bytes: int,
        budget: AlgebraResourceBudget,
    ):
        if not isinstance(budget, AlgebraResourceBudget):
            raise TypeError("budget must be AlgebraResourceBudget.")
        values = tuple(
            index(value)
            for value in (
                coordinate_count,
                product_pairs,
                product_terms,
                audit_terms,
                plan_bytes,
                dense_kernel_bytes,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("Algebra resource evidence must be nonnegative.")
        budget.admit_coordinates(values[0])
        budget.admit_product_pairs(values[1])
        budget.admit_product(values[2], values[4])
        budget.admit_audit(values[3])
        if values[5] > budget.maximum_dense_kernel_bytes:
            raise ValueError("Algebra dense-kernel budget exceeded.")
        (
            self.coordinate_count,
            self.product_pairs,
            self.product_terms,
            self.audit_terms,
            self.plan_bytes,
            self.dense_kernel_bytes,
        ) = values
        self.budget_id = budget.budget_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "algebra-resource-evidence-v1",
                "coordinates": values[0],
                "pairs": values[1],
                "terms": values[2],
                "audit_terms": values[3],
                "plan_bytes": values[4],
                "dense_kernel_bytes": values[5],
                "budget": budget.budget_id,
            }
        )


__all__ = ["AlgebraResourceBudget", "AlgebraResourceEvidence"]
