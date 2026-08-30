#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._method import (
    AbstractLinearCombinatorialMethod,
    CombinatorialPlan,
    make_combinatorial_plan,
)
from ._problem import AbstractCombinatorialSpace, LinearCombinatorialProblem
from ._selection import stable_masked_order
from ._types import (
    CombinatorialCertificate,
    CombinatorialCertification,
    CombinatorialFeasibility,
    CombinatorialMethodCapabilities,
    CombinatorialProvenance,
    CombinatorialResult,
    CombinatorialStatus,
)


class CardinalityDecision(StrictModule):
    """Canonical ascending indices selected by a fixed-cardinality decision."""

    indices: Array


class CardinalitySpace(AbstractCombinatorialSpace):
    """Binary decisions selecting exactly `count` valid items."""

    valid: Array
    size: int = eqx.field(static=True)
    count: int = eqx.field(static=True)
    valid_count: int = eqx.field(static=True)
    _structure_id: str = eqx.field(static=True)

    def __init__(self, size: int, count: int, /, *, valid: Any | None = None):
        if isinstance(size, bool) or not isinstance(size, Integral):
            raise TypeError("size must be a positive integer.")
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("count must be a non-negative integer.")
        size_ = int(size)
        count_ = int(count)
        if size_ <= 0:
            raise ValueError("size must be positive.")
        if count_ < 0 or count_ > size_:
            raise ValueError("count must lie in [0, size].")
        if valid is None:
            validity = jnp.ones((size_,), dtype=bool)
        else:
            validity = jnp.asarray(valid, dtype=bool)
            if validity.shape != (size_,):
                raise ValueError(
                    f"valid must have shape {(size_,)}; got {validity.shape}."
                )
        self.valid = validity
        self.size = size_
        self.count = count_
        self.valid_count = int(jnp.sum(validity))
        self._structure_id = canonical_fingerprint(
            {
                "kind": "cardinality-space",
                "size": size_,
                "count": count_,
                "valid": array_tree_fingerprint(validity),
            }
        )

    @property
    def structure_id(self) -> str:
        return self._structure_id

    def decision_spec(self, /) -> CardinalityDecision:
        return CardinalityDecision(jax.ShapeDtypeStruct((self.count,), jnp.int32))

    def feature_spec(self, /) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.size,), jnp.float32)

    def canonicalize(self, decision: CardinalityDecision, /) -> CardinalityDecision:
        if not isinstance(decision, CardinalityDecision):
            raise TypeError("cardinality decisions must be CardinalityDecision values.")
        indices = jnp.asarray(decision.indices, dtype=jnp.int32)
        if indices.shape[-1:] != (self.count,):
            raise ValueError(
                f"cardinality indices must end with shape {(self.count,)}; "
                f"got {indices.shape}."
            )
        in_range = (indices >= 0) & (indices < self.size)
        safe = jnp.clip(indices, 0, self.size - 1)
        admissible = in_range & self.valid[safe]
        keys = jnp.where(admissible, indices, self.size)
        ordered = jnp.sort(keys, axis=-1)
        return CardinalityDecision(jnp.where(ordered < self.size, ordered, -1))

    def encode(self, decision: CardinalityDecision, /) -> Array:
        canonical = self.canonicalize(decision)
        indices = canonical.indices
        batch_shape = indices.shape[:-1]
        features = jnp.zeros(batch_shape + (self.size,), dtype=float)
        if self.count == 0:
            return features
        safe = jnp.clip(indices, 0, self.size - 1)
        valid = indices >= 0
        batch_indices = jnp.indices(batch_shape, sparse=False)
        scatter_index = tuple(
            batch_indices[position][..., None] for position in range(len(batch_shape))
        )
        return features.at[scatter_index + (safe,)].set(valid.astype(features.dtype))

    def audit(self, decision: CardinalityDecision, /) -> CombinatorialFeasibility:
        canonical = self.canonicalize(decision)
        indices = canonical.indices
        in_range = (indices >= 0) & (indices < self.size)
        safe = jnp.clip(indices, 0, self.size - 1)
        admissible = in_range & self.valid[safe]
        if self.count > 1:
            distinct = indices[..., 1:] != indices[..., :-1]
            duplicate_count = jnp.sum(~distinct, axis=-1)
        else:
            duplicate_count = jnp.zeros(indices.shape[:-1], dtype=jnp.int32)
        invalid_count = jnp.sum(~admissible, axis=-1)
        residual = invalid_count + duplicate_count
        return CombinatorialFeasibility(
            residual == 0,
            residual.astype(float),
        )


class StableCardinalityOracle(AbstractLinearCombinatorialMethod):
    """Exact stable sorting oracle for fixed-cardinality decisions."""

    maximum_items: int = eqx.field(static=True)

    def __init__(self, *, maximum_items: int = 10_000_000):
        if isinstance(maximum_items, bool) or not isinstance(maximum_items, Integral):
            raise TypeError("maximum_items must be a positive integer.")
        if int(maximum_items) <= 0:
            raise ValueError("maximum_items must be positive.")
        self.maximum_items = int(maximum_items)

    @property
    def method_id(self) -> str:
        return "native-stable-cardinality"

    @property
    def capabilities(self) -> CombinatorialMethodCapabilities:
        return CombinatorialMethodCapabilities(
            exact=True,
            jax_native=True,
            jit=True,
            batched=True,
            signed_costs=True,
            deterministic_ties=True,
            optimality_certificate=True,
            surrogate_pullback=True,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (("maximum_items", str(self.maximum_items)),)

    def plan(
        self,
        problem: LinearCombinatorialProblem,
        certification: CombinatorialCertification,
        /,
    ) -> CombinatorialPlan:
        if not isinstance(problem.space, CardinalitySpace):
            raise TypeError("StableCardinalityOracle requires CardinalitySpace.")
        if len(jax.tree_util.tree_leaves(problem.costs)) != 1:
            raise ValueError("CardinalitySpace requires one array-valued cost vector.")
        size = problem.space.size
        if size > self.maximum_items:
            raise ValueError(
                f"cardinality size {size} exceeds maximum_items {self.maximum_items}."
            )
        return make_combinatorial_plan(
            problem,
            self,
            certification,
            work_estimate=problem.batch_size * size,
            workspace_elements=problem.batch_size * size,
            certificate_kind="cardinality-boundary",
        )

    def solve(
        self,
        problem: LinearCombinatorialProblem,
        plan: CombinatorialPlan,
        /,
    ) -> CombinatorialResult:
        space = problem.space
        if not isinstance(space, CardinalitySpace):
            raise TypeError("StableCardinalityOracle requires CardinalitySpace.")
        costs = jax.tree_util.tree_leaves(problem.costs)[0]
        batch_shape = problem.batch_shape
        finite = jnp.all(jnp.isfinite(costs), axis=-1)
        validity = jnp.broadcast_to(space.valid, costs.shape)
        order = stable_masked_order(costs, validity)
        chosen = order[..., : space.count]
        chosen = jnp.sort(chosen, axis=-1)
        structurally_feasible = space.valid_count >= space.count
        decision = CardinalityDecision(
            jnp.where(structurally_feasible, chosen, -1).astype(jnp.int32)
        )
        features = space.encode(decision).astype(costs.dtype)
        objective = problem.objective(features)
        feasibility = space.audit(decision)

        if space.count > 0:
            selected_costs = jnp.take_along_axis(
                costs, order[..., : space.count], axis=-1
            )
            selected_max = selected_costs[..., -1]
        else:
            selected_max = jnp.full(batch_shape, -jnp.inf, dtype=costs.dtype)
        if 0 < space.count < space.valid_count:
            unselected_min = jnp.take_along_axis(
                costs,
                order[..., space.count : space.count + 1],
                axis=-1,
            )[..., 0]
            tie_available = jnp.full(batch_shape, structurally_feasible, dtype=bool)
            tie_margin = unselected_min - selected_max
        else:
            unselected_min = jnp.full(batch_shape, jnp.inf, dtype=costs.dtype)
            tie_available = jnp.zeros(batch_shape, dtype=bool)
            tie_margin = jnp.full(batch_shape, jnp.nan, dtype=costs.dtype)
        tolerance = plan.certification.threshold(objective, selected_max, unselected_min)
        boundary_valid = (selected_max <= unselected_min + tolerance) | ~tie_available
        recomputed = jnp.sum(costs * features, axis=-1)
        objective_residual = jnp.abs(objective - recomputed)
        objective_consistent = objective_residual <= plan.certification.threshold(
            objective, recomputed
        )
        optimality = (
            finite
            & feasibility.feasible
            & objective_consistent
            & boundary_valid
            & structurally_feasible
        )
        status = jnp.where(
            ~finite,
            int(CombinatorialStatus.NONFINITE_INPUT),
            jnp.where(
                not structurally_feasible,
                int(CombinatorialStatus.INFEASIBLE),
                jnp.where(
                    optimality,
                    int(CombinatorialStatus.OPTIMAL),
                    int(CombinatorialStatus.CERTIFICATION_FAILED),
                ),
            ),
        ).astype(jnp.int32)
        valid = status == int(CombinatorialStatus.OPTIMAL)
        zero = jnp.zeros(batch_shape, dtype=costs.dtype)
        certificate = CombinatorialCertificate(
            finite=finite,
            feasible=feasibility.feasible,
            objective_consistent=objective_consistent,
            optimality_proven=optimality,
            primal_residual=feasibility.residual.astype(costs.dtype),
            dual_residual=jnp.maximum(selected_max - unselected_min, 0.0),
            absolute_gap=zero,
            relative_gap=zero,
            tie_margin=jnp.where(tie_available, tie_margin, jnp.nan),
            dual_available=tie_available,
            gap_available=jnp.full(batch_shape, structurally_feasible, dtype=bool),
            tie_available=tie_available,
        )
        provenance = CombinatorialProvenance(
            problem_id=problem.problem_id,
            structure_id=problem.structure_id,
            method_id=self.method_id,
            plan_id=plan.plan_id,
            implementation="phydrax-native-jax",
            tie_policy="lowest-item-index",
            certificate_kind=plan.certificate_kind,
            exact=True,
            signed_costs=True,
            configuration=plan.configuration,
        )
        decision = CardinalityDecision(jnp.where(valid[..., None], decision.indices, -1))
        features = jnp.where(valid[..., None], features, jnp.zeros_like(features))
        return CombinatorialResult(
            decision=decision,
            features=features,
            objective_value=jnp.where(valid, objective, jnp.nan),
            status=status,
            valid=valid,
            certificate=certificate,
            iterations=jnp.ones(batch_shape, dtype=jnp.int32),
            work=jnp.full(batch_shape, space.size, dtype=jnp.int64),
            provenance=provenance,
            batch_shape=batch_shape,
        )


__all__ = [
    "CardinalityDecision",
    "CardinalitySpace",
    "StableCardinalityOracle",
]
