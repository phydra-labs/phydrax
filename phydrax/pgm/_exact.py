#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._model import (
    DiscreteFactorGraph,
    factor_group_cardinality_signature,
    factor_group_dense_tables,
    VariableStateValues,
)
from ._types import ExactFactorGraphResult, ExactFactorGraphStatus, FactorGraphProvenance


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


class PreparedExactFactorGraph(StrictModule):
    """Host-capped enumeration routes with runtime-replaceable dense log tables."""

    assignments: Array
    factor_indices: tuple[Array, ...]
    factor_tables: tuple[Array, ...]
    cardinalities: tuple[int, ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    provenance: FactorGraphProvenance


def prepare_exact_factor_graph(
    graph: DiscreteFactorGraph, /, *, max_configurations: int = 65_536
) -> PreparedExactFactorGraph:
    """Validate and allocate fixed enumeration support outside JIT."""
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be a DiscreteFactorGraph.")
    limit = int(max_configurations)
    if limit < 1:
        raise ValueError("max_configurations must be positive.")
    cards = tuple(int(value) for value in np.asarray(graph.cardinalities).tolist())
    total = prod(cards)
    if total > limit:
        raise ValueError(
            f"Exact factor-graph enumeration requires {total} configurations, "
            f"exceeding max_configurations={limit}."
        )
    assignments = enumerate_assignments(graph.cardinalities)
    routes = []
    tables = []
    for index, scope in enumerate(graph.factor_scopes):
        signature = factor_group_cardinality_signature(graph, index)
        stride = prod(signature)
        strides = []
        for cardinality in signature:
            stride //= cardinality
            strides.append(stride)
        routes.append(
            jnp.sum(
                assignments[:, scope] * jnp.asarray(strides, dtype=jnp.int32), axis=-1
            )
        )
        tables.append(factor_group_dense_tables(graph, index))
    plan_id = canonical_fingerprint(
        {
            "kind": "exact-factor-graph-enumeration",
            "structure_id": graph.structure_id,
            "total_configurations": total,
            "max_configurations": limit,
        }
    )
    return PreparedExactFactorGraph(
        assignments,
        tuple(routes),
        tuple(tables),
        cards,
        graph.structure_id,
        FactorGraphProvenance(
            structure_id=graph.structure_id,
            plan_id=plan_id,
            method_id="exact-enumeration",
            implementation="mixed-radix-jax",
            exact=True,
            configuration=(("max_configurations", str(limit)),),
        ),
    )


def run_exact_factor_graph(
    prepared: PreparedExactFactorGraph,
    factor_tables: tuple[Array, ...] | None = None,
    /,
) -> ExactFactorGraphResult:
    """Execute exact inference with numeric tables on immutable prepared support.

    Shapes are checked statically. NaN/+inf return NONFINITE_INPUT; -inf denotes
    impossible configurations. Parameters, log Z and marginals support JIT/grad.
    """
    tables = prepared.factor_tables if factor_tables is None else factor_tables
    if len(tables) != len(prepared.factor_tables):
        raise ValueError("Numeric factor count differs from prepared support.")
    total = int(prepared.assignments.shape[0])
    dtype = jnp.result_type(*tables) if tables else jnp.dtype(float)
    scores = jnp.zeros((total,), dtype=dtype)
    numeric_valid = jnp.asarray(True)
    arrays = []
    for value, original, indices in zip(
        tables, prepared.factor_tables, prepared.factor_indices
    ):
        table = jnp.asarray(value)
        if table.shape != original.shape or jnp.iscomplexobj(table):
            raise ValueError("Numeric factor tables must retain real prepared shapes.")
        arrays.append(table)
        numeric_valid = numeric_valid & ~jnp.any(jnp.isnan(table) | jnp.isposinf(table))
        flat = table.reshape((table.shape[0], prod(table.shape[1:])))
        scores = scores + jnp.sum(
            flat[jnp.arange(table.shape[0])[None, :], indices], axis=-1
        )
    finite = jnp.isfinite(scores)
    feasible_count = jnp.sum(finite.astype(jnp.int32))
    feasible = (feasible_count > 0) & numeric_valid
    log_normalizer = jsp.special.logsumexp(scores)
    probabilities = jnp.where(feasible, jnp.exp(scores - log_normalizer), 0.0)
    variable_parts = tuple(
        jax.ops.segment_sum(
            probabilities, prepared.assignments[:, index], num_segments=cardinality
        )
        for index, cardinality in enumerate(prepared.cardinalities)
    )
    variable_probabilities = (
        jnp.concatenate(variable_parts)
        if variable_parts
        else jnp.zeros((0,), dtype=dtype)
    )
    factor_probabilities = []
    for table, indices in zip(arrays, prepared.factor_indices):
        count = prod(table.shape[1:])
        factor_probabilities.append(
            jax.vmap(
                lambda route: jax.ops.segment_sum(
                    probabilities, route, num_segments=count
                ),
                in_axes=1,
                out_axes=0,
            )(indices).reshape(table.shape)
        )
    map_index = jnp.argmax(scores)
    return ExactFactorGraphResult(
        log_normalizer=log_normalizer,
        variable_probabilities=VariableStateValues(
            variable_probabilities, structure_id=prepared.structure_id
        ),
        factor_probabilities=tuple(factor_probabilities),
        map_assignment=jnp.where(
            feasible,
            prepared.assignments[map_index],
            jnp.zeros((len(prepared.cardinalities),), dtype=jnp.int32),
        ),
        map_log_score=jnp.where(feasible, scores[map_index], -jnp.inf),
        feasible_configurations=feasible_count,
        total_configurations=total,
        status=jnp.where(
            ~numeric_valid,
            int(ExactFactorGraphStatus.NONFINITE_INPUT),
            jnp.where(
                feasible,
                int(ExactFactorGraphStatus.SUCCESS),
                int(ExactFactorGraphStatus.INFEASIBLE),
            ),
        ).astype(jnp.int32),
        valid=feasible,
        provenance=prepared.provenance,
    )


def enumerate_factor_graph(
    graph: DiscreteFactorGraph, /, *, max_configurations: int = 65_536
) -> ExactFactorGraphResult:
    """Prepare and run explicitly capped complete finite-state enumeration.

    Use prepare_exact_factor_graph/run_exact_factor_graph when tables vary inside
    compiled execution; this convenience entry point performs host preparation.
    """
    return run_exact_factor_graph(
        prepare_exact_factor_graph(graph, max_configurations=max_configurations)
    )


__all__ = [
    "PreparedExactFactorGraph",
    "enumerate_assignments",
    "enumerate_factor_graph",
    "prepare_exact_factor_graph",
    "run_exact_factor_graph",
]
