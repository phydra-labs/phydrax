#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class CliffordResourceBudget(StrictModule, NonTrainableState):
    """Hard pre-allocation limits for one Clifford algebra plan."""

    maximum_blades: int = eqx.field(static=True)
    maximum_product_terms: int = eqx.field(static=True)
    maximum_plan_bytes: int = eqx.field(static=True)
    maximum_dense_kernel_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_blades: int = 256,
        maximum_product_terms: int = 262_144,
        maximum_plan_bytes: int = 64 * 1024**2,
        maximum_dense_kernel_bytes: int = 8 * 1024**2,
    ):
        values = (
            int(maximum_blades),
            int(maximum_product_terms),
            int(maximum_plan_bytes),
            int(maximum_dense_kernel_bytes),
        )
        if any(value <= 0 for value in values):
            raise ValueError("Clifford resource limits must be positive.")
        (
            self.maximum_blades,
            self.maximum_product_terms,
            self.maximum_plan_bytes,
            self.maximum_dense_kernel_bytes,
        ) = values
        self.budget_id = canonical_fingerprint(
            {
                "kind": "clifford-resource-budget-v1",
                "maximum_blades": values[0],
                "maximum_product_terms": values[1],
                "maximum_plan_bytes": values[2],
                "maximum_dense_kernel_bytes": values[3],
            }
        )

    def admit_blades(self, blade_count: int, /) -> None:
        count = int(blade_count)
        if count < 0:
            raise ValueError("Clifford blade count must be nonnegative.")
        if count > self.maximum_blades:
            raise ValueError(
                f"Clifford layout requests {count} blades; budget allows "
                f"{self.maximum_blades}."
            )

    def admit_product_pairs(self, pair_count: int, /) -> None:
        count = int(pair_count)
        if count < 0:
            raise ValueError("Clifford product pair count must be nonnegative.")
        if count > self.maximum_product_terms:
            raise ValueError(
                f"Clifford product enumeration requests {count} blade pairs; budget "
                f"allows {self.maximum_product_terms}."
            )

    def admit_product(
        self,
        term_count: int,
        plan_bytes: int,
        /,
    ) -> None:
        terms = int(term_count)
        metadata = int(plan_bytes)
        if terms < 0 or metadata < 0:
            raise ValueError("Clifford plan resource counts must be nonnegative.")
        if terms > self.maximum_product_terms:
            raise ValueError(
                f"Clifford product requests {terms} terms; budget allows "
                f"{self.maximum_product_terms}."
            )
        if metadata > self.maximum_plan_bytes:
            raise ValueError(
                f"Clifford product requests {metadata} metadata bytes; budget allows "
                f"{self.maximum_plan_bytes}."
            )


class CliffordResourceEvidence(StrictModule, NonTrainableState):
    """Realized storage dimensions for one prepared Clifford object."""

    blade_count: int = eqx.field(static=True)
    product_terms: int = eqx.field(static=True)
    plan_bytes: int = eqx.field(static=True)
    dense_kernel_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        blade_count: int,
        product_terms: int,
        plan_bytes: int,
        dense_kernel_bytes: int,
        budget: CliffordResourceBudget,
    ):
        if not isinstance(budget, CliffordResourceBudget):
            raise TypeError("budget must be a CliffordResourceBudget.")
        blades = int(blade_count)
        terms = int(product_terms)
        metadata = int(plan_bytes)
        dense = int(dense_kernel_bytes)
        budget.admit_blades(blades)
        budget.admit_product(terms, metadata)
        if dense < 0:
            raise ValueError("Dense Clifford kernel bytes must be nonnegative.")
        self.blade_count = blades
        self.product_terms = terms
        self.plan_bytes = metadata
        self.dense_kernel_bytes = dense
        self.budget_id = budget.budget_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "clifford-resource-evidence-v1",
                "blade_count": blades,
                "product_terms": terms,
                "plan_bytes": metadata,
                "dense_kernel_bytes": dense,
                "budget": budget.budget_id,
            }
        )


__all__ = ["CliffordResourceBudget", "CliffordResourceEvidence"]
