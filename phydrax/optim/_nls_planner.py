#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    DenseQR,
    DenseSVD,
    GeneralizedLSMR,
    LinearSolvePolicy,
    LinearSystem,
    prepare as prepare_linear,
    RankPolicy,
    solve as solve_linear,
    solve_many as solve_linear_many,
    TolerancePolicy,
)
from ._residual_graph import PreparedResidualGraph


LeastSquaresRoute: TypeAlias = Literal[
    "dense-qr",
    "dense-svd",
    "sparse-qr",
    "lsmr",
    "schur",
]


class LeastSquaresRoutePolicy(StrictModule):
    dense_dimension: int = eqx.field(static=True)
    sparse_density: float = eqx.field(static=True)
    rank_cutoff: float = eqx.field(static=True)
    iterative_tolerance: float = eqx.field(static=True)
    iterative_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        dense_dimension: int = 256,
        sparse_density: float = 0.2,
        rank_cutoff: float = 1e-10,
        iterative_tolerance: float = 1e-8,
        iterative_steps: int = 1000,
    ):
        dimension = int(dense_dimension)
        steps = int(iterative_steps)
        values = tuple(
            float(value) for value in (sparse_density, rank_cutoff, iterative_tolerance)
        )
        if dimension < 1 or steps < 1:
            raise ValueError("Route dimension and iterative steps must be positive.")
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Route tolerances must be finite and positive.")
        if values[0] > 1.0:
            raise ValueError("sparse_density must not exceed one.")
        self.dense_dimension = dimension
        self.sparse_density = values[0]
        self.rank_cutoff = values[1]
        self.iterative_tolerance = values[2]
        self.iterative_steps = steps


class LeastSquaresRoutePlan(StrictModule):
    linear_policy: LinearSolvePolicy
    route: LeastSquaresRoute = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    parameter_dimension: int = eqx.field(static=True)
    residual_dimension: int = eqx.field(static=True)
    adjacency_density: float = eqx.field(static=True)
    estimated_factor_bytes: int = eqx.field(static=True)


class SchurComplementPlan(StrictModule):
    eliminated: Array
    retained: Array
    plan_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)


def plan_least_squares_route(
    graph: PreparedResidualGraph,
    /,
    *,
    policy: LeastSquaresRoutePolicy | None = None,
) -> LeastSquaresRoutePlan:
    if not isinstance(graph, PreparedResidualGraph):
        raise TypeError("graph must be PreparedResidualGraph.")
    policy_ = LeastSquaresRoutePolicy() if policy is None else policy
    if not isinstance(policy_, LeastSquaresRoutePolicy):
        raise TypeError("policy must be LeastSquaresRoutePolicy or None.")
    active_indices = [
        index
        for index, block in enumerate(graph.graph.parameter_blocks)
        if not block.constant
    ]
    if not active_indices:
        raise ValueError("Least-squares routing requires one variable parameter block.")
    parameter_dimension = sum(
        int(graph.parameter_sizes[index]) for index in active_indices
    )
    residual_dimension = int(jnp.sum(graph.residual_sizes))
    active_adjacency = graph.adjacency[:, jnp.asarray(active_indices)]
    density = float(jnp.mean(active_adjacency))
    groups = {
        graph.graph.parameter_blocks[index].elimination_group for index in active_indices
    }
    has_schur = len(groups) > 1 and parameter_dimension > policy_.dense_dimension
    if has_schur:
        route: LeastSquaresRoute = "schur"
    elif parameter_dimension <= policy_.dense_dimension:
        route = "dense-svd" if residual_dimension < parameter_dimension else "dense-qr"
    elif density <= policy_.sparse_density:
        route = "lsmr"
    else:
        route = "lsmr"
    if route == "dense-qr":
        linear_policy = LinearSolvePolicy(
            DenseQR(),
            rank=RankPolicy(
                relative_cutoff=policy_.rank_cutoff,
                require_full_rank=True,
            ),
        )
    elif route == "dense-svd":
        linear_policy = LinearSolvePolicy(
            DenseSVD(),
            rank=RankPolicy(
                relative_cutoff=policy_.rank_cutoff,
            ),
        )
    elif route in ("lsmr", "sparse-qr"):
        linear_policy = LinearSolvePolicy(
            GeneralizedLSMR(),
            tolerance=TolerancePolicy(
                relative=policy_.iterative_tolerance,
                absolute=policy_.iterative_tolerance,
                max_steps=policy_.iterative_steps,
            ),
        )
    else:
        linear_policy = LinearSolvePolicy(
            rank=RankPolicy(
                relative_cutoff=policy_.rank_cutoff,
            )
        )
    factor_entries = (
        parameter_dimension * parameter_dimension
        if route in ("dense-qr", "dense-svd")
        else max(
            parameter_dimension,
            int(density * parameter_dimension * residual_dimension),
        )
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "least-squares-route",
            "graph": graph.graph_id,
            "route": route,
            "parameter_dimension": parameter_dimension,
            "residual_dimension": residual_dimension,
            "density": density,
        }
    )
    return LeastSquaresRoutePlan(
        linear_policy,
        route=route,
        plan_id=plan_id,
        parameter_dimension=parameter_dimension,
        residual_dimension=residual_dimension,
        adjacency_density=density,
        estimated_factor_bytes=8 * factor_entries,
    )


def prepare_schur_plan(graph: PreparedResidualGraph, /) -> SchurComplementPlan:
    if not isinstance(graph, PreparedResidualGraph):
        raise TypeError("graph must be PreparedResidualGraph.")
    offsets = []
    offset = 0
    for index, size in enumerate(graph.parameter_sizes.tolist()):
        if graph.graph.parameter_blocks[index].constant:
            offsets.append(None)
            continue
        offsets.append(jnp.arange(offset, offset + int(size), dtype=jnp.int32))
        offset += int(size)
    groups = tuple(block.elimination_group for block in graph.graph.parameter_blocks)
    minimum_group = min(
        group
        for index, group in enumerate(groups)
        if not graph.graph.parameter_blocks[index].constant
    )
    eliminated_values = [
        offsets[index]
        for index, group in enumerate(groups)
        if group == minimum_group and offsets[index] is not None
    ]
    retained_values = [
        offsets[index]
        for index, group in enumerate(groups)
        if group != minimum_group and offsets[index] is not None
    ]
    if not eliminated_values or not retained_values:
        raise ValueError(
            "Schur planning requires nonempty eliminated and retained groups."
        )
    eliminated = jnp.concatenate(eliminated_values)
    retained = jnp.concatenate(retained_values)
    plan_id = canonical_fingerprint(
        {
            "kind": "schur-complement",
            "graph": graph.graph_id,
            "eliminated": eliminated.tolist(),
            "retained": retained.tolist(),
        }
    )
    return SchurComplementPlan(
        eliminated,
        retained,
        plan_id=plan_id,
        dimension=offset,
    )


def solve_schur_system(
    normal_matrix: Any,
    gradient: Any,
    plan: SchurComplementPlan,
    /,
    *,
    linear: LinearSolvePolicy | None = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> Array:
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    linear_ = LinearSolvePolicy(DenseLU()) if linear is None else linear
    if not isinstance(linear_, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    matrix = precision_.accumulation(normal_matrix)
    vector = precision_.accumulation(gradient)
    if matrix.shape != (plan.dimension, plan.dimension):
        raise ValueError("normal_matrix shape does not match Schur plan.")
    if vector.shape != (plan.dimension,):
        raise ValueError("gradient shape does not match Schur plan.")
    eliminated = plan.eliminated
    retained = plan.retained
    a = matrix[jnp.ix_(eliminated, eliminated)]
    b = matrix[jnp.ix_(eliminated, retained)]
    c = matrix[jnp.ix_(retained, retained)]
    ge = vector[eliminated]
    gr = vector[retained]
    prepared_a = prepare_linear(
        LinearSystem(DenseLinearOperator(a)),
        precision_.bind_linear(linear_),
    )
    a_inverse_b = solve_linear_many(prepared_a, b).value
    a_inverse_g = solve_linear(prepared_a, ge).value
    schur = c - jnp.conj(b.T) @ a_inverse_b
    reduced_gradient = gr - jnp.conj(b.T) @ a_inverse_g
    prepared_schur = prepare_linear(
        LinearSystem(DenseLinearOperator(schur)),
        precision_.bind_linear(linear_),
    )
    retained_step = solve_linear(
        prepared_schur,
        -reduced_gradient,
    ).value
    eliminated_step = solve_linear(
        prepared_a,
        -(ge + b @ retained_step),
    ).value
    result = jnp.zeros((plan.dimension,), dtype=retained_step.dtype)
    result = result.at[eliminated].set(eliminated_step)
    return precision_.direction(result.at[retained].set(retained_step))


__all__ = [
    "LeastSquaresRoute",
    "LeastSquaresRoutePlan",
    "LeastSquaresRoutePolicy",
    "SchurComplementPlan",
    "plan_least_squares_route",
    "prepare_schur_plan",
    "solve_schur_system",
]
