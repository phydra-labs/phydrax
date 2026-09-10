#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    LeastSquaresProblem as LinearLeastSquaresProblem,
    LinearSolveStatus,
    PyTreeSpace,
    solve as solve_linear,
)
from ._certificates import reconcile_optimization_status
from ._iterative import (
    LeastSquaresResult,
    OptimizationCertificate,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._nls_planner import (
    LeastSquaresRoutePlan,
    LeastSquaresRoutePolicy,
    plan_least_squares_route,
    prepare_schur_plan,
    SchurComplementPlan,
    solve_schur_system,
)
from ._residual_graph import (
    factor_graph_certificate,
    prepare_residual_graph,
    PreparedResidualGraph,
    ResidualGraphProblem,
)
from ._robust_losses import robustify_residual, squared_tree_norm


class ResidualGraphLinearization(StrictModule):
    """Block-assembled robust objective, gradient, and positive curvature."""

    residuals: tuple[PyTree[Array], ...]
    objective: Array
    gradient: Array
    curvature: Array
    robust_blocks: Array
    clipped_curvature_blocks: Array
    structural_jacobian_nonzeros: Array
    finite: Array


class ResidualGraphSolveEvidence(StrictModule):
    """Executed graph route and robust-curvature evidence."""

    route: str = eqx.field(static=True)
    route_plan_id: str = eqx.field(static=True)
    schur_plan_id: str = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)
    curvature_model: str = eqx.field(static=True)
    robust_blocks: Array
    clipped_curvature_blocks: Array
    structural_jacobian_nonzeros: Array
    linear_solves: Array
    linear_iterations: Array


class _ResidualGraphIterationState(StrictModule):
    parameters: PyTree[Any]
    model: ResidualGraphLinearization
    damping: Array
    status: Array
    iterations: Array
    accepted: Array
    rejected: Array
    linear_solves: Array
    linear_iterations: Array
    residual_evaluations: Array
    jacobian_evaluations: Array
    step_norm: Array
    ratio: Array


def _variable_layout(graph: ResidualGraphProblem, parameters: PyTree[Any], /):
    values = graph.parameter_values(parameters)
    spaces = {}
    slices = {}
    offset = 0
    for block in graph.parameter_blocks:
        if block.constant:
            continue
        space = PyTreeSpace(values[block.block_id])
        spaces[block.block_id] = space
        slices[block.block_id] = slice(offset, offset + space.size)
        offset += space.size
    if offset == 0:
        raise ValueError("Residual graph solve requires one variable parameter block.")
    return values, spaces, slices, offset


def _tangent_steps(
    graph: ResidualGraphProblem,
    parameters: PyTree[Any],
    step: Array,
    spaces,
    slices,
    /,
):
    values = graph.parameter_values(parameters)
    tangents = {}
    for block in graph.parameter_blocks:
        if block.constant:
            continue
        tangent = spaces[block.block_id].unflatten(step[slices[block.block_id]])
        if block.geometry is not None:
            tangent = block.geometry.project_tangent(
                values[block.block_id],
                tangent,
            )
        tangents[block.block_id] = tangent
    return tangents


def _block_linearization(
    graph: ResidualGraphProblem,
    parameters: PyTree[Any],
    block,
    values,
    spaces,
    slices,
    args: Any,
    /,
):
    local_ids = tuple(
        identifier for identifier in block.parameter_ids if identifier in spaces
    )
    local_slices = {}
    local_offset = 0
    global_indices = []
    for identifier in local_ids:
        size = spaces[identifier].size
        local_slices[identifier] = slice(local_offset, local_offset + size)
        local_offset += size
        global_slice = slices[identifier]
        global_indices.append(
            jnp.arange(global_slice.start, global_slice.stop, dtype=jnp.int32)
        )

    def flat_weighted(local_step):
        tangents = {
            identifier: spaces[identifier].unflatten(local_step[local_slices[identifier]])
            for identifier in local_ids
        }
        for parameter_block in graph.parameter_blocks:
            identifier = parameter_block.block_id
            if identifier in tangents and parameter_block.geometry is not None:
                tangents[identifier] = parameter_block.geometry.project_tangent(
                    values[identifier],
                    tangents[identifier],
                )
        candidate = graph.retract(parameters, tangents)
        candidate_values = graph.parameter_values(candidate)
        weighted = block.weighted_residual(
            tuple(candidate_values[identifier] for identifier in block.parameter_ids),
            args,
        )
        return PyTreeSpace(weighted).flatten(weighted), weighted

    zero = jnp.zeros((local_offset,), dtype=jax.tree.leaves(parameters)[0].dtype)
    if local_offset:
        jacobian, weighted = jax.jacfwd(
            flat_weighted,
            has_aux=True,
        )(zero)
        flat_residual = PyTreeSpace(weighted).flatten(weighted)
    else:
        flat_residual, weighted = flat_weighted(zero)
        jacobian = jnp.zeros(
            (flat_residual.size, 0),
            dtype=flat_residual.dtype,
        )
    squared_norm = squared_tree_norm(weighted)
    if block.loss is None:
        rho = squared_norm
        first = jnp.asarray(1.0, dtype=squared_norm.dtype)
        second = jnp.asarray(0.0, dtype=squared_norm.dtype)
        robust_residual = weighted
        robust = jnp.asarray(False)
    else:
        loss = block.loss.evaluate(squared_norm)
        rho = loss.rho
        first = jnp.maximum(loss.first, 0.0)
        second = loss.second
        robust_residual = robustify_residual(weighted, block.loss)[0]
        robust = jnp.asarray(True)
    raw_curvature = first + 2.0 * second * squared_norm
    radial_curvature = jnp.maximum(raw_curvature, 0.0)
    radial = (
        jnp.conj(jacobian.T) @ flat_residual / jnp.sqrt(jnp.maximum(squared_norm, 1e-30))
    )
    local_gradient = first * (jnp.conj(jacobian.T) @ flat_residual)
    local_curvature = first * (jnp.conj(jacobian.T) @ jacobian) + (
        radial_curvature - first
    ) * jnp.outer(radial, jnp.conj(radial))
    indices = (
        jnp.concatenate(global_indices)
        if global_indices
        else jnp.empty((0,), dtype=jnp.int32)
    )
    finite = (
        jnp.all(jnp.isfinite(flat_residual))
        & jnp.all(jnp.isfinite(jacobian))
        & jnp.isfinite(rho)
        & jnp.all(jnp.isfinite(local_gradient))
        & jnp.all(jnp.isfinite(local_curvature))
    )
    return (
        robust_residual,
        0.5 * rho,
        local_gradient,
        local_curvature,
        indices,
        robust,
        robust & (raw_curvature < 0.0),
        jnp.count_nonzero(jacobian),
        finite,
    )


def linearize_residual_graph(
    graph: ResidualGraphProblem,
    parameters: PyTree[Any],
    args: Any = None,
    /,
) -> ResidualGraphLinearization:
    """Assemble block-sparse robust normal equations in tangent coordinates."""
    if not isinstance(graph, ResidualGraphProblem):
        raise TypeError("graph must be ResidualGraphProblem.")
    values, spaces, slices, dimension = _variable_layout(graph, parameters)
    dtype = jax.tree.leaves(parameters)[0].dtype
    gradient = jnp.zeros((dimension,), dtype=dtype)
    curvature = jnp.zeros((dimension, dimension), dtype=dtype)
    objective = jnp.asarray(0.0, dtype=dtype)
    robust_blocks = jnp.asarray(0, dtype=jnp.int32)
    clipped_blocks = jnp.asarray(0, dtype=jnp.int32)
    nonzeros = jnp.asarray(0, dtype=jnp.int32)
    finite = jnp.asarray(True)
    residuals = []
    for block in graph.residual_blocks:
        (
            residual,
            block_objective,
            block_gradient,
            block_curvature,
            indices,
            robust,
            clipped,
            block_nonzeros,
            block_finite,
        ) = _block_linearization(
            graph,
            parameters,
            block,
            values,
            spaces,
            slices,
            args,
        )
        residuals.append(residual)
        objective = objective + block_objective
        if indices.size:
            gradient = gradient.at[indices].add(block_gradient)
            curvature = curvature.at[jnp.ix_(indices, indices)].add(block_curvature)
        robust_blocks = robust_blocks + robust.astype(jnp.int32)
        clipped_blocks = clipped_blocks + clipped.astype(jnp.int32)
        nonzeros = nonzeros + block_nonzeros.astype(jnp.int32)
        finite = finite & block_finite
    curvature = 0.5 * (curvature + jnp.conj(curvature.T))
    return ResidualGraphLinearization(
        tuple(residuals),
        objective,
        gradient,
        curvature,
        robust_blocks,
        clipped_blocks,
        nonzeros,
        finite,
    )


def _solve_route(
    matrix: Array,
    gradient: Array,
    route: LeastSquaresRoutePlan,
    schur: SchurComplementPlan | None,
    /,
):
    if route.route == "schur":
        if schur is None:
            raise ValueError("Schur route requires a SchurComplementPlan.")
        step = solve_schur_system(matrix, gradient, schur)
        return step, jnp.asarray(1, dtype=jnp.int32), jnp.all(jnp.isfinite(step))
    result = solve_linear(
        LinearLeastSquaresProblem(DenseLinearOperator(matrix)),
        -gradient,
        policy=route.linear_policy,
    )
    usable = (
        (result.status == int(LinearSolveStatus.SUCCESS))
        | (result.status == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED))
        | (result.status == int(LinearSolveStatus.STAGNATION))
        | (result.status == int(LinearSolveStatus.CONDITION_LIMIT_REACHED))
    )
    return (
        result.value,
        result.diagnostics.iterations,
        usable & jnp.all(jnp.isfinite(result.value)),
    )


def _graph_feasibility(graph: ResidualGraphProblem, parameters: PyTree[Any], /):
    violation = jnp.asarray(0.0)
    for block in graph.parameter_blocks:
        if block.bounds is not None:
            violation = jnp.maximum(
                violation,
                block.bounds.violation(block.extract(parameters)),
            )
    return violation


def _graph_parameter_norm(graph: ResidualGraphProblem, parameters: PyTree[Any], /):
    squared_norm = jnp.asarray(0.0)
    for block in graph.parameter_blocks:
        if block.constant:
            continue
        squared_norm = squared_norm + sum(
            jnp.real(jnp.vdot(leaf, leaf))
            for leaf in jax.tree.leaves(block.extract(parameters))
        )
    return jnp.sqrt(jnp.maximum(squared_norm, 0.0))


def solve_residual_graph(
    graph: ResidualGraphProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination | None = None,
    route_policy: LeastSquaresRoutePolicy | None = None,
    args: Any = None,
    initial_damping: float = 1e-3,
) -> LeastSquaresResult:
    """Solve a residual graph through its planned dense, Krylov, or Schur route."""
    if not isinstance(graph, ResidualGraphProblem):
        raise TypeError("graph must be ResidualGraphProblem.")
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination or None.")
    damping_ = float(initial_damping)
    if not isfinite(damping_) or damping_ <= 0.0:
        raise ValueError("initial_damping must be finite and positive.")
    prepared: PreparedResidualGraph = prepare_residual_graph(
        graph,
        initial_parameters,
        args=args,
    )
    route = plan_least_squares_route(prepared, policy=route_policy)
    schur = prepare_schur_plan(prepared) if route.route == "schur" else None
    _, spaces, slices, dimension = _variable_layout(graph, initial_parameters)
    model = linearize_residual_graph(graph, initial_parameters, args)
    initial_optimality = jnp.linalg.norm(model.gradient, ord=jnp.inf)
    dtype = model.objective.dtype
    state = _ResidualGraphIterationState(
        parameters=initial_parameters,
        model=model,
        damping=jnp.asarray(damping_, dtype=dtype),
        status=jnp.asarray(int(OptimizationStatus.ITERATING), dtype=jnp.int32),
        iterations=jnp.asarray(0, dtype=jnp.int32),
        accepted=jnp.asarray(0, dtype=jnp.int32),
        rejected=jnp.asarray(0, dtype=jnp.int32),
        linear_solves=jnp.asarray(0, dtype=jnp.int32),
        linear_iterations=jnp.asarray(0, dtype=jnp.int32),
        residual_evaluations=jnp.asarray(1, dtype=jnp.int32),
        jacobian_evaluations=jnp.asarray(1, dtype=jnp.int32),
        step_norm=jnp.asarray(0.0, dtype=dtype),
        ratio=jnp.asarray(jnp.nan, dtype=dtype),
    )
    evaluation_limit = (
        termination_.maximum_steps + 1
        if termination_.maximum_evaluations is None
        else termination_.maximum_evaluations
    )
    optimality_threshold = termination_.optimality_threshold(initial_optimality)

    def select_tree(predicate, candidate, current):
        return jax.tree_util.tree_map(
            lambda new, old: jnp.where(predicate, new, old) if eqx.is_array(new) else old,
            candidate,
            current,
        )

    def iteration(_, current):
        optimality = jnp.linalg.norm(current.model.gradient, ord=jnp.inf)
        entry_status = jnp.where(
            ~current.model.finite,
            int(OptimizationStatus.NONFINITE_EVALUATION),
            jnp.where(
                optimality <= optimality_threshold,
                int(OptimizationStatus.SUCCESS),
                jnp.where(
                    current.residual_evaluations >= evaluation_limit,
                    int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                    current.status,
                ),
            ),
        ).astype(jnp.int32)
        current = eqx.tree_at(lambda value: value.status, current, entry_status)
        active = entry_status == int(OptimizationStatus.ITERATING)

        def attempt(value):
            matrix = value.model.curvature + value.damping * jnp.eye(
                dimension,
                dtype=value.model.curvature.dtype,
            )
            step, route_iterations, usable = _solve_route(
                matrix,
                value.model.gradient,
                route,
                schur,
            )

            def failed_linear_solve(operand):
                return eqx.tree_at(
                    lambda item: (
                        item.status,
                        item.linear_solves,
                        item.linear_iterations,
                    ),
                    operand,
                    (
                        jnp.asarray(
                            int(OptimizationStatus.LINEAR_SOLVE_FAILED),
                            dtype=jnp.int32,
                        ),
                        operand.linear_solves + 1,
                        operand.linear_iterations + route_iterations,
                    ),
                )

            def evaluate_candidate(operand):
                tangent_steps = _tangent_steps(
                    graph,
                    operand.parameters,
                    step,
                    spaces,
                    slices,
                )
                candidate_parameters = graph.retract(
                    operand.parameters,
                    tangent_steps,
                )
                candidate_model = linearize_residual_graph(
                    graph,
                    candidate_parameters,
                    args,
                )
                predicted = -jnp.real(
                    jnp.vdot(operand.model.gradient, step)
                ) - 0.5 * jnp.real(jnp.vdot(step, operand.model.curvature @ step))
                actual = operand.model.objective - candidate_model.objective
                ratio = actual / jnp.maximum(predicted, 1e-30)
                accept = (
                    candidate_model.finite
                    & (predicted > 0.0)
                    & (actual > 0.0)
                    & (ratio >= 1e-4)
                )
                parameters = select_tree(
                    accept,
                    candidate_parameters,
                    operand.parameters,
                )
                selected_model = select_tree(
                    accept,
                    candidate_model,
                    operand.model,
                )
                damping = jnp.where(
                    accept,
                    jnp.maximum(1e-12, 0.25 * operand.damping),
                    jnp.minimum(1e12, 4.0 * operand.damping),
                )
                step_norm = jnp.linalg.norm(step)
                stagnated = accept & (
                    step_norm
                    <= termination_.step_threshold(
                        _graph_parameter_norm(graph, parameters)
                    )
                )
                trust_failed = ~accept & (damping >= 1e12)
                status = jnp.where(
                    stagnated,
                    int(OptimizationStatus.STAGNATION),
                    jnp.where(
                        trust_failed,
                        int(OptimizationStatus.TRUST_REGION_FAILED),
                        int(OptimizationStatus.ITERATING),
                    ),
                ).astype(jnp.int32)
                return _ResidualGraphIterationState(
                    parameters=parameters,
                    model=selected_model,
                    damping=damping,
                    status=status,
                    iterations=operand.iterations + 1,
                    accepted=operand.accepted + accept.astype(jnp.int32),
                    rejected=operand.rejected + (~accept).astype(jnp.int32),
                    linear_solves=operand.linear_solves + 1,
                    linear_iterations=(operand.linear_iterations + route_iterations),
                    residual_evaluations=operand.residual_evaluations + 1,
                    jacobian_evaluations=operand.jacobian_evaluations + 1,
                    step_norm=step_norm,
                    ratio=ratio,
                )

            return jax.lax.cond(
                usable,
                evaluate_candidate,
                failed_linear_solve,
                value,
            )

        return jax.lax.cond(active, attempt, lambda value: value, current)

    state = jax.lax.fori_loop(
        0,
        termination_.maximum_steps,
        iteration,
        state,
    )
    state = eqx.tree_at(
        lambda value: value.status,
        state,
        jnp.where(
            state.status == int(OptimizationStatus.ITERATING),
            int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
            state.status,
        ).astype(jnp.int32),
    )
    parameters = state.parameters
    model = state.model
    damping = state.damping
    status = state.status
    iterations = state.iterations
    accepted = state.accepted
    rejected = state.rejected
    linear_solves = state.linear_solves
    linear_iterations = state.linear_iterations
    residual_evaluations = state.residual_evaluations
    jacobian_evaluations = state.jacobian_evaluations
    step_norm = state.step_norm
    ratio = state.ratio
    physical = factor_graph_certificate(
        graph,
        parameters,
        args,
        tolerance=termination_.absolute_optimality,
    )
    feasibility = _graph_feasibility(graph, parameters)
    finite = physical.finite & jnp.isfinite(feasibility)
    certified = physical.certified & (feasibility <= termination_.absolute_optimality)
    certificate = OptimizationCertificate(
        kind="least-squares-normal",
        tolerance=termination_.absolute_optimality,
        optimality_norm=jnp.maximum(physical.gradient_norm, feasibility),
        primal_feasibility=feasibility,
        projected_stationarity=physical.gradient_norm,
        finite=finite,
        regular=True,
        certified=certified,
        evaluation_work=2,
        certificate_id=f"{graph.problem_id}/residual-graph-normal",
    )
    status_evidence = reconcile_optimization_status(
        status,
        certificate,
        allow_certificate_promotion=True,
    )
    evidence = ResidualGraphSolveEvidence(
        route.route,
        route.plan_id,
        "" if schur is None else schur.plan_id,
        prepared.graph_id,
        "block-robust-psd-clipped",
        model.robust_blocks,
        model.clipped_curvature_blocks,
        model.structural_jacobian_nonzeros,
        jnp.asarray(linear_solves, dtype=jnp.int32),
        jnp.asarray(linear_iterations, dtype=jnp.int32),
    )
    diagnostics = OptimizationDiagnostics(
        iterations=iterations,
        accepted_steps=accepted,
        rejected_steps=rejected,
        residual_evaluations=residual_evaluations + certificate.evaluation_work,
        jacobian_evaluations=jacobian_evaluations + 1,
        linear_solves=linear_solves,
        linear_iterations=linear_iterations,
        initial_optimality_norm=initial_optimality,
        final_optimality_norm=certificate.optimality_norm,
        final_step_norm=step_norm,
        accepted_step_size=jnp.where(accepted > 0, 1.0, 0.0),
        damping=damping,
        reduction_ratio=ratio,
        primal_feasibility=feasibility,
    )
    provenance = OptimizationProvenance(
        problem_id=graph.problem_id,
        method="residual-graph-gauss-newton",
        backend="phydrax-native",
        globalization="levenberg-marquardt-ratio",
        matrix_free=route.route == "lsmr",
        implicit_differentiation=False,
        notes=(
            f"route={route.route};plan={route.plan_id};"
            f"schur={'' if schur is None else schur.plan_id};"
            f"clipped-blocks={int(model.clipped_curvature_blocks)};"
            f"internal-status={int(status)}"
        ),
    )
    return LeastSquaresResult(
        parameters,
        model.residuals,
        model.objective,
        None,
        status_evidence.public_status,
        diagnostics,
        provenance,
        optimality_certificate=certificate,
        status_evidence=status_evidence,
        method_evidence=evidence,
    )


__all__ = [
    "linearize_residual_graph",
    "ResidualGraphLinearization",
    "ResidualGraphSolveEvidence",
    "solve_residual_graph",
]
