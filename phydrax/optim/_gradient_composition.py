#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import (
    tree_allfinite,
    tree_inner,
    tree_norm,
    tree_scale,
    validate_inexact_tree,
)
from ..linalg._dense_pseudoinverse import (
    apply_pseudoinverse,
    factor_pseudoinverse,
)
from ..linalg._policies import RankPolicy


ConflictFreeFailureMode = Literal["status", "error"]


class ConflictFreeGradientStatus(IntEnum):
    SUCCESS = 0
    STATIONARY = 1
    INFEASIBLE = 2
    NONFINITE = 3


class ConflictFreeGradientPolicy(StrictModule, NonTrainableState):
    """Numerical policy for inverse-gradient multi-objective composition."""

    rank_policy: RankPolicy
    minimum_norm: float = eqx.field(static=True)
    projection_tolerance: float = eqx.field(static=True)
    failure: ConflictFreeFailureMode = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        rank_policy: RankPolicy | None = None,
        minimum_norm: float = 1e-12,
        projection_tolerance: float = 1e-10,
        failure: ConflictFreeFailureMode = "status",
    ):
        rank = RankPolicy() if rank_policy is None else rank_policy
        if not isinstance(rank, RankPolicy):
            raise TypeError("rank_policy must be a RankPolicy.")
        norm = float(minimum_norm)
        tolerance = float(projection_tolerance)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("minimum_norm must be finite and strictly positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("projection_tolerance must be finite and nonnegative.")
        if failure not in ("status", "error"):
            raise ValueError("failure must be 'status' or 'error'.")
        self.rank_policy = rank
        self.minimum_norm = norm
        self.projection_tolerance = tolerance
        self.failure = failure
        self.policy_id = canonical_fingerprint(
            {
                "kind": "conflict-free-gradient-policy",
                "rank": {
                    "relative_cutoff": rank.relative_cutoff,
                    "absolute_cutoff": rank.absolute_cutoff,
                    "require_full_rank": rank.require_full_rank,
                },
                "minimum_norm": norm,
                "projection_tolerance": tolerance,
                "failure": failure,
            }
        )


class ConflictFreeGradientResult(StrictModule):
    """Composed direction and complete local multi-objective evidence."""

    direction: PyTree[Array]
    norms: Array
    cosine_matrix: Array
    projections: Array
    active: Array
    stationary: Array
    conflicts: Array
    rank: Array
    rank_cutoff: Array
    condition_estimate: Array
    direction_norm: Array
    successful: Array
    status: Array
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        direction: PyTree[Array],
        /,
        *,
        norms: Array,
        cosine_matrix: Array,
        projections: Array,
        active: Array,
        stationary: Array,
        conflicts: Array,
        rank: Array,
        rank_cutoff: Array,
        condition_estimate: Array,
        direction_norm: Array,
        successful: Array,
        status: Array,
        policy_id: str,
    ):
        self.direction = direction
        self.norms = jnp.asarray(norms)
        self.cosine_matrix = jnp.asarray(cosine_matrix)
        self.projections = jnp.asarray(projections)
        self.active = jnp.asarray(active, dtype=bool)
        self.stationary = jnp.asarray(stationary, dtype=bool)
        self.conflicts = jnp.asarray(conflicts, dtype=bool)
        self.rank = jnp.asarray(rank)
        self.rank_cutoff = jnp.asarray(rank_cutoff)
        self.condition_estimate = jnp.asarray(condition_estimate)
        self.direction_norm = jnp.asarray(direction_norm)
        self.successful = jnp.asarray(successful, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.policy_id = str(policy_id)


def _tree_add(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + y, left, right)


def conflict_free_gradient(
    gradients: Sequence[PyTree[Any]],
    /,
    *,
    active: Any | None = None,
    policy: ConflictFreeGradientPolicy | None = None,
) -> ConflictFreeGradientResult:
    """Compose objective gradients without materializing parameter coordinates."""

    values = tuple(
        validate_inexact_tree(value, name=f"objective gradient {index}")
        for index, value in enumerate(gradients)
    )
    if not values:
        raise ValueError("At least one objective gradient is required.")
    structure = jax.tree.structure(values[0])
    if any(jax.tree.structure(value) != structure for value in values[1:]):
        raise ValueError("Objective gradients must have congruent PyTree structures.")
    shapes = tuple(
        tuple(leaf.shape for leaf in jax.tree.leaves(value)) for value in values
    )
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError("Objective gradient leaves must have congruent shapes.")
    resolved = ConflictFreeGradientPolicy() if policy is None else policy
    if not isinstance(resolved, ConflictFreeGradientPolicy):
        raise TypeError("policy must be a ConflictFreeGradientPolicy.")
    count = len(values)
    active_mask = (
        jnp.ones((count,), dtype=bool)
        if active is None
        else jnp.asarray(active, dtype=bool)
    )
    if active_mask.shape != (count,):
        raise ValueError("active must contain one Boolean per objective gradient.")

    norms = jnp.stack(tuple(tree_norm(value) for value in values))
    finite = jnp.stack(tuple(tree_allfinite(value) for value in values))
    stationary = active_mask & finite & (norms <= resolved.minimum_norm)
    effective = active_mask & finite & ~stationary
    safe_norms = jnp.where(effective, norms, jnp.ones_like(norms))
    normalized = tuple(
        tree_scale(effective[index] / safe_norms[index], value)
        for index, value in enumerate(values)
    )
    gram = jnp.stack(
        tuple(
            jnp.stack(tuple(tree_inner(left, right) for right in normalized))
            for left in normalized
        )
    )
    factors = factor_pseudoinverse(
        gram,
        resolved.rank_policy,
        hermitian=True,
    )
    coefficients = apply_pseudoinverse(
        factors,
        effective.astype(gram.dtype),
    )
    candidate = tree_scale(0.0, values[0])
    for coefficient, direction in zip(coefficients, normalized, strict=True):
        candidate = _tree_add(candidate, tree_scale(coefficient, direction))
    candidate_norm = tree_norm(candidate)
    candidate_finite = tree_allfinite(candidate)
    usable_candidate = (
        candidate_finite & factors.finite & (candidate_norm > resolved.minimum_norm)
    )
    unit_candidate = tree_scale(
        jnp.where(usable_candidate, 1.0 / candidate_norm, 0.0),
        candidate,
    )
    unit_projections = jnp.stack(
        tuple(tree_inner(value, unit_candidate) for value in values)
    )
    scale = jnp.sum(jnp.where(effective, unit_projections, 0.0))
    positive_scale = jnp.maximum(scale, 0.0)
    direction = tree_scale(positive_scale, unit_candidate)
    projections = jnp.stack(tuple(tree_inner(value, direction) for value in values))
    conflicts = effective & (projections < -resolved.projection_tolerance)
    all_input_finite = jnp.all(~active_mask | finite)
    active_count = jnp.sum(effective.astype(jnp.int32))
    stationary_only = active_count == 0
    feasible = usable_candidate & ~jnp.any(conflicts) & (positive_scale > 0.0)
    successful = all_input_finite & (stationary_only | feasible)
    status = jnp.where(
        ~all_input_finite,
        int(ConflictFreeGradientStatus.NONFINITE),
        jnp.where(
            stationary_only,
            int(ConflictFreeGradientStatus.STATIONARY),
            jnp.where(
                feasible,
                int(ConflictFreeGradientStatus.SUCCESS),
                int(ConflictFreeGradientStatus.INFEASIBLE),
            ),
        ),
    )
    if resolved.failure == "error":
        direction = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                ~successful,
                "Objective gradients do not admit a finite conflict-free direction.",
            ),
            direction,
        )
    return ConflictFreeGradientResult(
        direction,
        norms=norms,
        cosine_matrix=gram,
        projections=projections,
        active=active_mask,
        stationary=stationary,
        conflicts=conflicts,
        rank=factors.rank,
        rank_cutoff=factors.rank_cutoff,
        condition_estimate=factors.condition_estimate,
        direction_norm=tree_norm(direction),
        successful=successful,
        status=status,
        policy_id=resolved.policy_id,
    )


__all__ = [
    "ConflictFreeFailureMode",
    "ConflictFreeGradientPolicy",
    "ConflictFreeGradientResult",
    "ConflictFreeGradientStatus",
    "conflict_free_gradient",
]
