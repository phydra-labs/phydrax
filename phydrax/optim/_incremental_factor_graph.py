#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ..linalg import PyTreeSpace
from ._residual_graph import (
    prepare_residual_graph,
    PreparedResidualGraph,
    refresh_residual_graph,
    ResidualBlock,
    ResidualGraphProblem,
)


class IncrementalFactorUpdate(StrictModule):
    affected_parameters: Array
    changed_factors: Array
    relinearized_parameters: Array
    topology_changed: bool
    update_count: Array


class IncrementalResidualGraph(StrictModule):
    """Persistent residual graph with deterministic affected-subgraph evidence."""

    prepared: PreparedResidualGraph
    parameters: PyTree[Array]
    linearization_parameters: PyTree[Array]
    factor_versions: Array
    parameter_versions: Array
    update_count: Array
    graph_version: Array
    relinearization_threshold: float

    def __init__(
        self,
        prepared: PreparedResidualGraph,
        parameters: PyTree[Any],
        linearization_parameters: PyTree[Any],
        factor_versions: Any,
        parameter_versions: Any,
        update_count: Any,
        graph_version: Any,
        /,
        *,
        relinearization_threshold: float,
    ):
        if not isinstance(prepared, PreparedResidualGraph):
            raise TypeError("prepared must be PreparedResidualGraph.")
        threshold = float(relinearization_threshold)
        if not isfinite(threshold) or threshold < 0.0:
            raise ValueError("relinearization_threshold must be finite and non-negative.")
        self.prepared = prepared
        self.parameters = parameters
        self.linearization_parameters = linearization_parameters
        self.factor_versions = jnp.asarray(factor_versions, dtype=jnp.int32)
        self.parameter_versions = jnp.asarray(parameter_versions, dtype=jnp.int32)
        self.update_count = jnp.asarray(update_count, dtype=jnp.int32)
        self.graph_version = jnp.asarray(graph_version, dtype=jnp.int32)
        self.relinearization_threshold = threshold


def prepare_incremental_factor_graph(
    graph: ResidualGraphProblem,
    parameters: PyTree[Any],
    /,
    *,
    args: Any = None,
    relinearization_threshold: float = 1e-3,
) -> IncrementalResidualGraph:
    prepared = prepare_residual_graph(graph, parameters, args=args)
    return IncrementalResidualGraph(
        prepared,
        parameters,
        parameters,
        jnp.zeros((len(graph.residual_blocks),), dtype=jnp.int32),
        jnp.zeros((len(graph.parameter_blocks),), dtype=jnp.int32),
        0,
        0,
        relinearization_threshold=relinearization_threshold,
    )


def update_incremental_factor_graph(
    incremental: IncrementalResidualGraph,
    graph: ResidualGraphProblem,
    parameters: PyTree[Any],
    /,
    *,
    changed_factors: tuple[str, ...] = (),
    args: Any = None,
) -> tuple[IncrementalResidualGraph, IncrementalFactorUpdate]:
    if not isinstance(incremental, IncrementalResidualGraph):
        raise TypeError("incremental must be IncrementalResidualGraph.")
    prepared = refresh_residual_graph(
        incremental.prepared,
        graph,
        parameters,
        args=args,
    )
    factor_index = {
        block.block_id: index for index, block in enumerate(graph.residual_blocks)
    }
    changed = tuple(str(value) for value in changed_factors)
    if any(value not in factor_index for value in changed):
        raise ValueError("changed_factors contains an unknown residual block ID.")
    changed_mask = jnp.zeros((len(graph.residual_blocks),), dtype=jnp.bool_)
    for identifier in changed:
        changed_mask = changed_mask.at[factor_index[identifier]].set(True)
    affected = jnp.any(
        prepared.adjacency & changed_mask[:, None],
        axis=0,
    )
    relinearized = jnp.zeros_like(affected)
    linearization_parameters = incremental.linearization_parameters
    parameter_versions = incremental.parameter_versions
    for index, block in enumerate(graph.parameter_blocks):
        current = block.extract(parameters)
        previous = block.extract(incremental.linearization_parameters)
        displacement = jnp.linalg.norm(
            PyTreeSpace(current).flatten(current)
            - PyTreeSpace(previous).flatten(previous)
        )
        relinearize = affected[index] | (
            displacement >= incremental.relinearization_threshold
        )
        relinearized = relinearized.at[index].set(relinearize)
        replacement = jax.tree.map(
            lambda new, old: jnp.where(relinearize, new, old),
            current,
            previous,
        )
        linearization_parameters = block.replace(
            linearization_parameters,
            replacement,
        )
        parameter_versions = parameter_versions.at[index].add(
            relinearize.astype(jnp.int32)
        )
    factor_versions = incremental.factor_versions + changed_mask.astype(jnp.int32)
    updated = IncrementalResidualGraph(
        prepared,
        parameters,
        linearization_parameters,
        factor_versions,
        parameter_versions,
        incremental.update_count + 1,
        incremental.graph_version,
        relinearization_threshold=incremental.relinearization_threshold,
    )
    evidence = IncrementalFactorUpdate(
        affected,
        changed_mask,
        relinearized,
        False,
        updated.update_count,
    )
    return updated, evidence


def add_incremental_factor(
    incremental: IncrementalResidualGraph,
    factor: ResidualBlock,
    parameters: PyTree[Any],
    /,
    *,
    args: Any = None,
) -> tuple[IncrementalResidualGraph, IncrementalFactorUpdate]:
    if not isinstance(factor, ResidualBlock):
        raise TypeError("factor must be ResidualBlock.")
    graph = incremental.prepared.graph
    if any(value.block_id == factor.block_id for value in graph.residual_blocks):
        raise ValueError("Added factor ID already exists.")
    expanded = ResidualGraphProblem(
        graph.parameter_blocks,
        graph.residual_blocks + (factor,),
        problem_id=graph.problem_id,
    )
    updated = prepare_incremental_factor_graph(
        expanded,
        parameters,
        args=args,
        relinearization_threshold=incremental.relinearization_threshold,
    )
    updated = IncrementalResidualGraph(
        updated.prepared,
        updated.parameters,
        updated.linearization_parameters,
        updated.factor_versions,
        updated.parameter_versions,
        incremental.update_count + 1,
        incremental.graph_version + 1,
        relinearization_threshold=incremental.relinearization_threshold,
    )
    changed = (
        jnp.zeros((len(expanded.residual_blocks),), dtype=jnp.bool_).at[-1].set(True)
    )
    affected = expanded.residual_blocks[-1].parameter_ids
    affected_mask = jnp.asarray(
        [block.block_id in affected for block in expanded.parameter_blocks]
    )
    return updated, IncrementalFactorUpdate(
        affected_mask,
        changed,
        affected_mask,
        True,
        updated.update_count,
    )


def remove_incremental_factor(
    incremental: IncrementalResidualGraph,
    factor_id: str,
    parameters: PyTree[Any],
    /,
    *,
    args: Any = None,
) -> tuple[IncrementalResidualGraph, IncrementalFactorUpdate]:
    graph = incremental.prepared.graph
    identifier = str(factor_id)
    retained = tuple(
        factor for factor in graph.residual_blocks if factor.block_id != identifier
    )
    if len(retained) == len(graph.residual_blocks):
        raise ValueError("Removed factor ID does not exist.")
    if not retained:
        raise ValueError("An incremental graph must retain at least one factor.")
    reduced = ResidualGraphProblem(
        graph.parameter_blocks,
        retained,
        problem_id=graph.problem_id,
    )
    updated = prepare_incremental_factor_graph(
        reduced,
        parameters,
        args=args,
        relinearization_threshold=incremental.relinearization_threshold,
    )
    updated = IncrementalResidualGraph(
        updated.prepared,
        updated.parameters,
        updated.linearization_parameters,
        updated.factor_versions,
        updated.parameter_versions,
        incremental.update_count + 1,
        incremental.graph_version + 1,
        relinearization_threshold=incremental.relinearization_threshold,
    )
    affected = jnp.ones((len(reduced.parameter_blocks),), dtype=jnp.bool_)
    return updated, IncrementalFactorUpdate(
        affected,
        jnp.ones((len(reduced.residual_blocks),), dtype=jnp.bool_),
        affected,
        True,
        updated.update_count,
    )


__all__ = [
    "IncrementalFactorUpdate",
    "IncrementalResidualGraph",
    "add_incremental_factor",
    "prepare_incremental_factor_graph",
    "remove_incremental_factor",
    "update_incremental_factor_graph",
]
