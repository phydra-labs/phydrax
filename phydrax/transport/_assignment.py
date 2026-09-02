#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.optimize import linear_sum_assignment

from .._strict import StrictModule
from ._costs import AbstractGroundCost, PrecomputedCost


class AssignmentResult(StrictModule):
    """Exact represented finite bipartite assignment with frozen indices."""

    source_indices: Array
    target_indices: Array
    edge_costs: Array
    total_cost: Array
    unmatched_source: Array
    unmatched_target: Array
    valid: Array
    status: Array
    cardinality: int = eqx.field(static=True)
    selection_semantics: str = eqx.field(static=True)


def solve_multidimensional_assignment(
    source_points: ArrayLike,
    target_points: ArrayLike,
    /,
    *,
    cost: AbstractGroundCost | PrecomputedCost | ArrayLike,
    source_mask: ArrayLike | None = None,
    target_mask: ArrayLike | None = None,
    forbidden: ArrayLike | None = None,
    cardinality: int | None = None,
    maximum_atoms: int = 4096,
    tie_break: Literal["lexicographic"] = "lexicographic",
) -> AssignmentResult:
    """Solve one exact finite rectangular assignment on the host.

    Selection is deterministic and nondifferentiable.  This routine does not replace
    weighted optimal transport and never changes the supplied cardinality.
    """
    source = jnp.asarray(source_points)
    target = jnp.asarray(target_points)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("point arrays must be rank two with one common feature size.")
    if tie_break != "lexicographic":
        raise ValueError("Only deterministic lexicographic tie breaking is supported.")
    maximum = int(maximum_atoms)
    if maximum <= 0 or source.shape[0] > maximum or target.shape[0] > maximum:
        raise ValueError("assignment atom count exceeds maximum_atoms.")
    source_active = (
        np.ones((source.shape[0],), dtype=bool)
        if source_mask is None
        else np.asarray(source_mask, dtype=bool)
    )
    target_active = (
        np.ones((target.shape[0],), dtype=bool)
        if target_mask is None
        else np.asarray(target_mask, dtype=bool)
    )
    if source_active.shape != (source.shape[0],) or target_active.shape != (
        target.shape[0],
    ):
        raise ValueError("assignment masks must align their point arrays.")
    source_ids = np.flatnonzero(source_active)
    target_ids = np.flatnonzero(target_active)
    if isinstance(cost, (AbstractGroundCost, PrecomputedCost)):
        matrix = (
            cost.values
            if isinstance(cost, PrecomputedCost)
            else cost.matrix(source, target)
        )
    else:
        matrix = jnp.asarray(cost)
    matrix_host = np.asarray(matrix, dtype=float)
    if matrix_host.shape != (source.shape[0], target.shape[0]):
        raise ValueError("cost matrix must align all source and target points.")
    if np.any(~np.isfinite(matrix_host)) or np.any(matrix_host < 0.0):
        raise ValueError("assignment costs must be finite and nonnegative.")
    restricted = matrix_host[np.ix_(source_ids, target_ids)]
    prohibited = (
        np.zeros_like(matrix_host, dtype=bool)
        if forbidden is None
        else np.asarray(forbidden, dtype=bool)
    )
    if prohibited.shape != matrix_host.shape:
        raise ValueError("forbidden must align the complete cost matrix.")
    restricted_forbidden = prohibited[np.ix_(source_ids, target_ids)]
    n, m = restricted.shape
    requested = min(n, m) if cardinality is None else int(cardinality)
    if requested < 0 or requested > min(n, m):
        raise ValueError("cardinality must lie between zero and the active-side minimum.")
    if requested == 0:
        return AssignmentResult(
            source_indices=jnp.zeros((0,), dtype=jnp.int32),
            target_indices=jnp.zeros((0,), dtype=jnp.int32),
            edge_costs=jnp.zeros((0,), dtype=source.dtype),
            total_cost=jnp.asarray(0.0, dtype=source.dtype),
            unmatched_source=jnp.asarray(source_active),
            unmatched_target=jnp.asarray(target_active),
            valid=jnp.asarray(True),
            status=jnp.asarray(0, dtype=jnp.int32),
            cardinality=0,
            selection_semantics="exact-finite-frozen-lexicographic",
        )
    finite_max = max(float(np.max(restricted)), 1.0)
    barrier = finite_max * float(n + m + 2)
    size = n + m - requested
    augmented = np.full((size, size), barrier, dtype=float)
    augmented[:n, :m] = np.where(restricted_forbidden, barrier, restricted)
    augmented[:n, m:] = 0.0
    augmented[n:, :m] = 0.0
    # Dummy-to-dummy assignments would increase the number of real edges above k.
    augmented[n:, m:] = barrier
    scale = np.finfo(float).eps * max(1.0, float(np.max(np.abs(augmented))))
    lexicographic = np.arange(size * size, dtype=float).reshape((size, size))
    selected_rows, selected_columns = linear_sum_assignment(
        augmented + scale * lexicographic
    )
    real = (selected_rows < n) & (selected_columns < m)
    local_rows = selected_rows[real]
    local_columns = selected_columns[real]
    feasible = (
        local_rows.size == requested
        and not np.any(restricted_forbidden[local_rows, local_columns])
        and np.all(restricted[local_rows, local_columns] < barrier)
    )
    if not feasible:
        raise ValueError("The requested finite assignment is infeasible.")
    chosen_source = source_ids[local_rows]
    chosen_target = target_ids[local_columns]
    order = np.lexsort((chosen_target, chosen_source))
    chosen_source = chosen_source[order]
    chosen_target = chosen_target[order]
    edge_costs = matrix_host[chosen_source, chosen_target]
    unmatched_source = source_active.copy()
    unmatched_target = target_active.copy()
    unmatched_source[chosen_source] = False
    unmatched_target[chosen_target] = False
    return AssignmentResult(
        source_indices=jnp.asarray(chosen_source, dtype=jnp.int32),
        target_indices=jnp.asarray(chosen_target, dtype=jnp.int32),
        edge_costs=jnp.asarray(edge_costs, dtype=source.dtype),
        total_cost=jnp.asarray(np.sum(edge_costs), dtype=source.dtype),
        unmatched_source=jnp.asarray(unmatched_source),
        unmatched_target=jnp.asarray(unmatched_target),
        valid=jnp.asarray(True),
        status=jnp.asarray(0, dtype=jnp.int32),
        cardinality=requested,
        selection_semantics="exact-finite-frozen-lexicographic",
    )


__all__ = ["AssignmentResult", "solve_multidimensional_assignment"]
