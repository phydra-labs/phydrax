#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseQR,
    DenseSVD,
    FunctionLinearOperator,
    GeneralizedLSMR,
    GMRES,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSystem,
    MaterializationPolicy,
    prepare as prepare_linear,
    RankPolicy,
    solve as solve_linear,
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

    dense_max_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        dense_dimension: int = 256,
        sparse_density: float = 0.2,
        rank_cutoff: float = 1e-10,
        iterative_tolerance: float = 1e-8,
        iterative_steps: int = 1000,
        dense_max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        dimension = int(dense_dimension)
        steps = int(iterative_steps)
        values = tuple(
            float(value) for value in (sparse_density, rank_cutoff, iterative_tolerance)
        )
        if dimension < 1 or steps < 1:
            raise ValueError("Route dimension and iterative steps must be positive.")
        if dense_max_bytes < 1:
            raise ValueError("dense_max_bytes must be positive.")
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Route tolerances must be finite and positive.")
        if values[0] > 1.0:
            raise ValueError("sparse_density must not exceed one.")
        self.dense_dimension = dimension
        self.sparse_density = values[0]
        self.rank_cutoff = values[1]
        self.iterative_tolerance = values[2]
        self.iterative_steps = steps
        self.dense_max_bytes = dense_max_bytes


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
            materialization=MaterializationPolicy(
                max_entries=policy_.dense_dimension**2,
                max_bytes=policy_.dense_max_bytes,
            ),
            rank=RankPolicy(
                relative_cutoff=policy_.rank_cutoff,
                require_full_rank=True,
            ),
        )
    elif route == "dense-svd":
        linear_policy = LinearSolvePolicy(
            DenseSVD(),
            materialization=MaterializationPolicy(
                max_entries=policy_.dense_dimension**2,
                max_bytes=policy_.dense_max_bytes,
            ),
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
    eliminated_values: list[Array] = []
    retained_values: list[Array] = []
    for block_offsets, group in zip(offsets, groups, strict=True):
        if block_offsets is None:
            continue
        if group == minimum_group:
            eliminated_values.append(block_offsets)
        else:
            retained_values.append(block_offsets)
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
    normal_matrix: AbstractLinearOperator,
    gradient: Array,
    plan: SchurComplementPlan,
    /,
    *,
    linear: LinearSolvePolicy | None = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> Array:
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    linear_ = (
        LinearSolvePolicy(
            GeneralizedLSMR(),
            tolerance=TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=1000),
        )
        if linear is None
        else linear
    )
    if not isinstance(linear_, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    if not isinstance(normal_matrix, AbstractLinearOperator):
        raise TypeError("normal_matrix must be an AbstractLinearOperator.")
    if (
        normal_matrix.source.size != plan.dimension
        or normal_matrix.target.size != plan.dimension
    ):
        raise ValueError("normal_matrix shape does not match Schur plan.")
    vector = precision_.accumulation(gradient)
    if vector.shape != (plan.dimension,):
        raise ValueError("gradient shape does not match Schur plan.")
    eliminated, retained = plan.eliminated, plan.retained
    dtype = vector.dtype

    def restriction(rows: Array, columns: Array) -> FunctionLinearOperator:
        def action(value: Array) -> Array:
            full = (
                jnp.zeros((plan.dimension,), dtype=dtype)
                .at[columns]
                .set(value, unique_indices=True)
            )
            output = normal_matrix.mv(normal_matrix.source.unflatten(full))
            return normal_matrix.target.flatten(output)[rows].astype(dtype)

        def transpose(value: Array) -> Array:
            full = (
                jnp.zeros((plan.dimension,), dtype=dtype)
                .at[rows]
                .set(value, unique_indices=True)
            )
            output = normal_matrix.transpose_mv(normal_matrix.target.unflatten(full))
            return normal_matrix.source.flatten(output)[columns].astype(dtype)

        return FunctionLinearOperator(
            action,
            source=ArraySpace((columns.size,), dtype=dtype),
            target=ArraySpace((rows.size,), dtype=dtype),
            transpose_action=transpose,
        )

    a = restriction(eliminated, eliminated)
    b = restriction(eliminated, retained)
    lower = restriction(retained, eliminated)
    c = restriction(retained, retained)
    pivot_problem = (
        LeastSquaresProblem(a)
        if isinstance(linear_.method, GeneralizedLSMR)
        else LinearSystem(a)
    )
    prepared_a = prepare_linear(pivot_problem, precision_.bind_linear(linear_))
    a_inverse_g = solve_linear(prepared_a, vector[eliminated])

    def reduced_action(value: Array) -> Array:
        return c.mv(value) - lower.mv(solve_linear(prepared_a, b.mv(value)).value)

    # Curvature is Hermitian; its exact Schur complement has the same symmetry.
    def reduced_transpose(value: Array) -> Array:
        return jnp.conj(reduced_action(jnp.conj(value)))

    reduced = FunctionLinearOperator(
        reduced_action,
        source=c.source,
        target=c.target,
        transpose_action=reduced_transpose,
    )
    reduced_gradient = vector[retained] - lower.mv(a_inverse_g.value)
    reduced_policy = LinearSolvePolicy(
        GMRES(restart=min(20, retained.size)),
        tolerance=linear_.tolerance,
    )
    retained_solve = solve_linear(
        LinearSystem(reduced),
        -reduced_gradient,
        policy=precision_.bind_linear(reduced_policy),
    )
    eliminated_solve = solve_linear(
        prepared_a,
        -(vector[eliminated] + b.mv(retained_solve.value)),
    )
    valid = (
        a_inverse_g.successful
        & a_inverse_g.diagnostics.converged
        & retained_solve.successful
        & retained_solve.diagnostics.converged
        & eliminated_solve.successful
        & eliminated_solve.diagnostics.converged
    )
    result = jnp.zeros((plan.dimension,), dtype=dtype)
    result = result.at[eliminated].set(eliminated_solve.value)
    result = result.at[retained].set(retained_solve.value)
    full_residual = (
        precision_.accumulation(
            normal_matrix.target.flatten(
                normal_matrix.mv(normal_matrix.source.unflatten(result))
            )
        )
        + vector
    )
    threshold = linear_.tolerance.absolute + linear_.tolerance.relative * jnp.linalg.norm(
        vector
    )
    valid = valid & (jnp.linalg.norm(full_residual) <= threshold)
    return precision_.direction(jnp.where(valid, result, jnp.full_like(result, jnp.nan)))


__all__ = [
    "LeastSquaresRoute",
    "LeastSquaresRoutePlan",
    "LeastSquaresRoutePolicy",
    "SchurComplementPlan",
    "plan_least_squares_route",
    "prepare_schur_plan",
    "solve_schur_system",
]
