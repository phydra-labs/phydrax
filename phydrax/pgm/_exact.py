#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ._model import (
    DiscreteFactorGraph,
    factor_graph_log_score,
    factor_group_cardinality_signature,
    VariableStateValues,
)
from ._types import (
    ExactFactorGraphResult,
    ExactFactorGraphStatus,
    FactorGraphProvenance,
)


def enumerate_assignments(cardinalities: Array, /) -> Array:
    """Enumerate mixed-radix assignments in deterministic lexicographic order."""
    cards = tuple(int(value) for value in np.asarray(cardinalities).tolist())
    total = prod(cards)
    if not cards:
        return jnp.zeros((1, 0), dtype=jnp.int32)
    indices = jnp.arange(total, dtype=jnp.int64)
    divisor = total
    columns: list[Array] = []
    for cardinality in cards:
        divisor //= cardinality
        columns.append((indices // divisor) % cardinality)
    return jnp.stack(columns, axis=-1).astype(jnp.int32)


def _variable_probabilities(
    graph: DiscreteFactorGraph,
    assignments: Array,
    probabilities: Array,
    /,
) -> Array:
    parts: list[Array] = []
    for variable, cardinality in enumerate(np.asarray(graph.cardinalities).tolist()):
        states = assignments[:, variable]
        parts.append(
            jnp.stack(
                [
                    jnp.sum(jnp.where(states == state, probabilities, 0.0))
                    for state in range(int(cardinality))
                ]
            )
        )
    return jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=probabilities.dtype)


def _configuration_indices(states: Array, signature: tuple[int, ...], /) -> Array:
    strides: list[int] = []
    stride = prod(signature)
    for cardinality in signature:
        stride //= cardinality
        strides.append(stride)
    return jnp.sum(states * jnp.asarray(strides, dtype=jnp.int32), axis=-1)


def _factor_probabilities(
    graph: DiscreteFactorGraph,
    assignments: Array,
    probabilities: Array,
    /,
) -> tuple[Array, ...]:
    outputs: list[Array] = []
    for group_index, scope in enumerate(graph.factor_scopes):
        signature = factor_group_cardinality_signature(graph, group_index)
        config_count = prod(signature)
        scope_states = assignments[:, scope]
        factors: list[Array] = []
        for factor in range(int(scope.shape[0])):
            config_indices = _configuration_indices(scope_states[:, factor, :], signature)
            marginal = jax.ops.segment_sum(
                probabilities,
                config_indices,
                num_segments=config_count,
            )
            factors.append(marginal.reshape(signature))
        if factors:
            outputs.append(jnp.stack(factors))
        else:
            outputs.append(jnp.zeros((0,) + signature, dtype=probabilities.dtype))
    return tuple(outputs)


def enumerate_factor_graph(
    graph: DiscreteFactorGraph,
    /,
    *,
    max_configurations: int = 65_536,
) -> ExactFactorGraphResult:
    """Run explicitly capped exact inference by complete finite-state enumeration."""
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be a DiscreteFactorGraph.")
    limit = int(max_configurations)
    if limit < 1:
        raise ValueError("max_configurations must be positive.")
    total = prod(int(value) for value in np.asarray(graph.cardinalities).tolist())
    if total > limit:
        raise ValueError(
            f"Exact factor-graph enumeration requires {total} configurations, "
            f"exceeding max_configurations={limit}."
        )

    assignments = enumerate_assignments(graph.cardinalities)
    scores = factor_graph_log_score(graph, assignments)
    finite = jnp.isfinite(scores)
    feasible_count = jnp.sum(finite.astype(jnp.int32))
    feasible = feasible_count > 0
    log_normalizer = jsp.special.logsumexp(scores)
    probabilities = jnp.where(feasible, jnp.exp(scores - log_normalizer), 0.0)
    variable_probabilities = _variable_probabilities(graph, assignments, probabilities)
    factor_probabilities = _factor_probabilities(graph, assignments, probabilities)
    map_index = jnp.argmax(scores)
    map_assignment = jnp.where(
        feasible,
        assignments[map_index],
        jnp.zeros((graph.num_variables,), dtype=jnp.int32),
    )
    map_log_score = jnp.where(feasible, scores[map_index], -jnp.inf)
    status = jnp.where(
        feasible,
        int(ExactFactorGraphStatus.SUCCESS),
        int(ExactFactorGraphStatus.INFEASIBLE),
    ).astype(jnp.int32)
    plan_id = canonical_fingerprint(
        {
            "kind": "exact-factor-graph-enumeration",
            "structure_id": graph.structure_id,
            "total_configurations": total,
            "max_configurations": limit,
        }
    )
    return ExactFactorGraphResult(
        log_normalizer=log_normalizer,
        variable_probabilities=VariableStateValues(
            variable_probabilities,
            structure_id=graph.structure_id,
        ),
        factor_probabilities=factor_probabilities,
        map_assignment=map_assignment,
        map_log_score=map_log_score,
        feasible_configurations=feasible_count,
        total_configurations=total,
        status=status,
        valid=feasible,
        provenance=FactorGraphProvenance(
            structure_id=graph.structure_id,
            plan_id=plan_id,
            method_id="exact-enumeration",
            implementation="mixed-radix-jax",
            exact=True,
            configuration=(("max_configurations", str(limit)),),
        ),
    )


__all__ = ["enumerate_assignments", "enumerate_factor_graph"]
