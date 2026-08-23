#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .._nonlinear_precision import NonlinearPrecisionPolicy
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    solve as solve_linear,
)
from ._constrained_model import PreparedConstrainedModel
from ._iterative import (
    ConstrainedOptimalityCertificate,
    NonlinearLeastSquaresProblem,
    OptimizationCertificate,
    OptimizationStatus,
    OptimizationStatusEvidence,
    OptimizationTermination,
)


def _least_squares_solve(
    matrix,
    right_hand_side,
    precision: NonlinearPrecisionPolicy,
    linear: LinearSolvePolicy | None,
    /,
):
    linear_ = LinearSolvePolicy(DenseSVD()) if linear is None else linear
    if not isinstance(linear_, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    return solve_linear(
        LeastSquaresProblem(DenseLinearOperator(precision.accumulation(matrix))),
        precision.accumulation(right_hand_side),
        policy=precision.bind_linear(linear_),
    )


def reconcile_optimization_status(
    internal_status: Any,
    certificate: OptimizationCertificate,
    /,
    *,
    allow_certificate_promotion: bool,
) -> OptimizationStatusEvidence:
    if not isinstance(certificate, OptimizationCertificate):
        raise TypeError("certificate must be OptimizationCertificate.")
    internal = int(internal_status)
    internal_success = internal == int(OptimizationStatus.SUCCESS)
    certified = bool(certificate.certified)
    if internal_success and certified:
        public = internal
        promoted = False
        demoted = False
        reason = "internal-and-certificate-success"
    elif internal_success:
        promoted = False
        demoted = True
        if not bool(certificate.finite):
            public = int(OptimizationStatus.NONFINITE_EVALUATION)
            reason = "nonfinite-certificate"
        elif float(certificate.primal_feasibility) > float(certificate.tolerance):
            public = int(OptimizationStatus.INFEASIBLE)
            reason = "feasibility-failed"
        elif not bool(certificate.regular):
            public = int(OptimizationStatus.CONSTRAINT_QUALIFICATION_FAILED)
            reason = "regularity-failed"
        elif float(certificate.complementarity) > float(certificate.tolerance):
            public = int(OptimizationStatus.CERTIFICATION_FAILED)
            reason = "complementarity-failed"
        else:
            public = int(OptimizationStatus.CERTIFICATION_FAILED)
            reason = "stationarity-failed"
    elif certified and allow_certificate_promotion:
        public = int(OptimizationStatus.SUCCESS)
        promoted = True
        demoted = False
        reason = "certificate-promoted"
    else:
        public = internal
        promoted = False
        demoted = False
        reason = (
            "certificate-unavailable"
            if not bool(certificate.finite)
            else "stationarity-failed"
        )
    return OptimizationStatusEvidence(
        internal_status=internal,
        public_status=public,
        certificate=certificate,
        promoted=promoted,
        demoted=demoted,
        decision_reason=reason,
    )


def _least_squares_objective(
    problem: NonlinearLeastSquaresProblem,
    parameters: Any,
    args: Any,
    /,
    precision: NonlinearPrecisionPolicy,
):
    residual, _ = problem.value(parameters, args)
    flat = precision.accumulation(ravel_pytree(residual)[0])
    return precision.decision(0.5 * jnp.real(jnp.sum(jnp.conj(flat) * flat)))


def certify_least_squares_physical(
    problem: NonlinearLeastSquaresProblem,
    parameters: Any,
    args: Any,
    termination: OptimizationTermination,
    /,
    *,
    certificate_step: float | None = None,
    kind: Literal[
        "least-squares-normal", "derivative-free-stationarity"
    ] = "derivative-free-stationarity",
    linear: LinearSolvePolicy | None = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> OptimizationCertificate:
    if not isinstance(problem, NonlinearLeastSquaresProblem):
        raise TypeError("problem must be NonlinearLeastSquaresProblem.")
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination.")
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    precision_.validate_tolerance(termination.absolute_optimality)
    model_parameters = precision_.state(parameters)
    physical_parameters = precision_.certificate(parameters)
    coordinates, unflatten = ravel_pytree(physical_parameters)
    physical_residual, _ = problem.value(physical_parameters, args)
    objective = _least_squares_objective(
        problem,
        physical_parameters,
        args,
        precision_,
    )
    epsilon_dtype = (
        coordinates.real.dtype
        if precision_.certificate_dtype is None
        else jnp.dtype(precision_.certificate_dtype)
    )
    epsilon = jnp.finfo(epsilon_dtype).eps
    lower = jnp.full_like(coordinates, -jnp.inf)
    upper = jnp.full_like(coordinates, jnp.inf)
    if problem.bounds is not None:
        lower_tree, upper_tree = problem.bounds.materialize(physical_parameters)
        lower = ravel_pytree(lower_tree)[0]
        upper = ravel_pytree(upper_tree)[0]
    gradients = []
    evaluations = 1
    for index in range(coordinates.size):
        scale_step = jnp.sqrt(epsilon) * (1.0 + jnp.abs(coordinates[index]))
        step = (
            scale_step
            if certificate_step is None
            else jnp.maximum(scale_step, certificate_step)
        )
        lower_room = jnp.maximum(coordinates[index] - lower[index], 0.0)
        upper_room = jnp.maximum(upper[index] - coordinates[index], 0.0)
        central_step = jnp.minimum(
            step,
            jnp.minimum(lower_room, upper_room),
        )
        forward_step = jnp.minimum(step, 0.5 * upper_room)
        backward_step = jnp.minimum(step, 0.5 * lower_room)
        fixed = (
            jnp.isfinite(lower[index])
            & jnp.isfinite(upper[index])
            & (lower[index] == upper[index])
        )
        use_central = (
            (central_step > 0.0)
            & (central_step >= forward_step)
            & (central_step >= backward_step)
        )
        use_forward = (forward_step > 0.0) & (forward_step >= backward_step)
        if bool(jax.device_get(fixed)):
            gradient_value = jnp.asarray(0.0, dtype=objective.dtype)
            used = 0
        elif bool(jax.device_get(use_central)):
            direction = jnp.zeros_like(coordinates).at[index].set(central_step)
            plus = _least_squares_objective(
                problem,
                unflatten(coordinates + direction),
                args,
                precision_,
            )
            minus = _least_squares_objective(
                problem,
                unflatten(coordinates - direction),
                args,
                precision_,
            )
            gradient_value = (plus - minus) / (2.0 * central_step)
            used = 2
        elif bool(jax.device_get(use_forward)):
            direction = jnp.zeros_like(coordinates).at[index].set(forward_step)
            plus = _least_squares_objective(
                problem,
                unflatten(coordinates + direction),
                args,
                precision_,
            )
            plus_two = _least_squares_objective(
                problem,
                unflatten(coordinates + 2.0 * direction),
                args,
                precision_,
            )
            gradient_value = (-3.0 * objective + 4.0 * plus - plus_two) / (
                2.0 * forward_step
            )
            used = 2
        elif bool(jax.device_get(backward_step > 0.0)):
            direction = jnp.zeros_like(coordinates).at[index].set(backward_step)
            minus = _least_squares_objective(
                problem,
                unflatten(coordinates - direction),
                args,
                precision_,
            )
            minus_two = _least_squares_objective(
                problem,
                unflatten(coordinates - 2.0 * direction),
                args,
                precision_,
            )
            gradient_value = (3.0 * objective - 4.0 * minus + minus_two) / (
                2.0 * backward_step
            )
            used = 2
        else:
            gradient_value = jnp.asarray(jnp.nan, dtype=objective.dtype)
            used = 0
        gradients.append(gradient_value)
        evaluations += used
    gradient_coordinates = jnp.stack(gradients)
    gradient = unflatten(gradient_coordinates)
    projected = (
        gradient
        if problem.bounds is None
        else problem.bounds.projected_gradient(physical_parameters, gradient)
    )
    projected_coordinates = ravel_pytree(projected)[0]
    stationarity = precision_.decision(
        jnp.linalg.norm(
            precision_.accumulation(projected_coordinates),
            ord=jnp.inf,
        )
    )
    feasibility = (
        jnp.asarray(0.0, dtype=objective.dtype)
        if problem.bounds is None
        else problem.bounds.violation(physical_parameters)
    )
    finite = (
        jnp.isfinite(objective)
        & jnp.all(jnp.isfinite(gradient_coordinates))
        & jnp.isfinite(feasibility)
    )
    tolerance = precision_.decision(termination.absolute_optimality)
    certified = finite & (stationarity <= tolerance) & (feasibility <= tolerance)
    return OptimizationCertificate(
        kind=kind,
        tolerance=tolerance,
        optimality_norm=stationarity,
        primal_feasibility=feasibility,
        projected_stationarity=stationarity,
        finite=finite,
        regular=True,
        certified=certified,
        evaluation_work=evaluations,
        certificate_id=f"{problem.problem_id}/{kind}",
        precision_evidence=precision_.evidence_for(
            model_parameters,
            precision_.residual(physical_residual),
        ),
    )


def _independent_active_multipliers(
    gradient,
    equality_jacobian,
    inequality_jacobian,
    active_mask,
    precision,
    linear,
):
    active_indices = [
        int(index) for index in jax.device_get(jnp.where(active_mask)[0]).tolist()
    ]
    selected = list(active_indices)
    equality_count = equality_jacobian.shape[0]
    while True:
        selected_indices = jnp.asarray(selected, dtype=jnp.int32)
        selected_jacobian = inequality_jacobian[selected_indices]
        multiplier_matrix = jnp.concatenate(
            [
                jnp.conj(equality_jacobian.T),
                -jnp.conj(selected_jacobian.T),
            ],
            axis=1,
        )
        if multiplier_matrix.shape[1]:
            multiplier_result = _least_squares_solve(
                multiplier_matrix,
                -gradient,
                precision,
                linear,
            )
            multipliers = precision.direction(multiplier_result.value)
        else:
            multipliers = jnp.empty((0,), dtype=gradient.dtype)
        selected_multipliers = multipliers[equality_count:]
        selected_values = jax.device_get(selected_multipliers).tolist()
        negative = [index for index, value in enumerate(selected_values) if value < 0.0]
        if not negative:
            break
        worst = min(negative, key=selected_values.__getitem__)
        selected.pop(worst)
    equality_multipliers = multipliers[:equality_count]
    inequality_multipliers = jnp.zeros(
        (inequality_jacobian.shape[0],),
        dtype=gradient.dtype,
    )
    if selected:
        inequality_multipliers = inequality_multipliers.at[
            jnp.asarray(selected, dtype=jnp.int32)
        ].set(selected_multipliers)
    return equality_multipliers, inequality_multipliers


def certify_constrained_physical(
    prepared: PreparedConstrainedModel,
    parameters: Any,
    canonical: ConstrainedOptimalityCertificate,
    tolerance: Any,
    /,
    *,
    kind: Literal["active-kkt", "barrier-kkt"] = "active-kkt",
    args: Any = None,
    linear: LinearSolvePolicy | None = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> OptimizationCertificate:
    if not isinstance(prepared, PreparedConstrainedModel):
        raise TypeError("prepared must be PreparedConstrainedModel.")
    if not isinstance(canonical, ConstrainedOptimalityCertificate):
        raise TypeError("canonical must be ConstrainedOptimalityCertificate.")
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    model_parameters = precision_.state(parameters)
    physical_parameters = precision_.certificate(parameters)
    evaluation = prepared.evaluate(physical_parameters, args)
    equality_jacobian = evaluation.constraint_jacobian[prepared.equality_indices]
    lower_jacobian = evaluation.constraint_jacobian[prepared.lower_indices]
    upper_jacobian = -evaluation.constraint_jacobian[prepared.upper_indices]
    inequality_jacobian = jnp.concatenate(
        [lower_jacobian, upper_jacobian],
        axis=0,
    )
    tolerance_ = precision_.decision(tolerance)
    active_mask = evaluation.inequality_slacks <= jnp.sqrt(tolerance_)
    if kind == "active-kkt":
        equality_multipliers, inequality_multipliers = _independent_active_multipliers(
            evaluation.gradient,
            equality_jacobian,
            inequality_jacobian,
            active_mask,
            precision_,
            linear,
        )
    else:
        equality_multipliers = canonical.equality_multipliers
        inequality_multipliers = canonical.inequality_multipliers
    gradient_ = precision_.accumulation(evaluation.gradient)
    equality_jacobian_ = precision_.accumulation(equality_jacobian)
    inequality_jacobian_ = precision_.accumulation(inequality_jacobian)
    equality_multipliers_ = precision_.accumulation(equality_multipliers)
    inequality_multipliers_ = precision_.accumulation(inequality_multipliers)
    stationarity_coordinates = (
        gradient_
        + jnp.conj(equality_jacobian_.T) @ equality_multipliers_
        - jnp.conj(inequality_jacobian_.T) @ inequality_multipliers_
    )
    stationarity = precision_.decision(
        jnp.linalg.norm(stationarity_coordinates, ord=jnp.inf)
    )
    physical_slacks = jnp.maximum(evaluation.inequality_slacks, 0.0)
    physical_complementarity = jnp.max(
        jnp.abs(physical_slacks * inequality_multipliers),
        initial=0.0,
    )
    multiplier_violation = jnp.max(
        jnp.maximum(-inequality_multipliers, 0.0),
        initial=0.0,
    )
    if kind == "active-kkt":
        primal = evaluation.primal_feasibility
        dual = jnp.maximum(stationarity, multiplier_violation)
        complementarity = physical_complementarity
    else:
        primal = jnp.maximum(
            evaluation.primal_feasibility,
            canonical.primal_feasibility,
        )
        dual = jnp.maximum(
            stationarity,
            jnp.maximum(canonical.dual_feasibility, multiplier_violation),
        )
        complementarity = jnp.maximum(
            canonical.complementarity,
            physical_complementarity,
        )
    active_rows = jnp.concatenate(
        [
            equality_jacobian,
            inequality_jacobian[active_mask],
        ],
        axis=0,
    )
    if active_rows.shape[0] == 0:
        regular = jnp.asarray(True)
    else:
        regularity_result = _least_squares_solve(
            jnp.conj(active_rows.T),
            jnp.zeros((active_rows.shape[1],), dtype=active_rows.dtype),
            precision_,
            linear,
        )
        regular = (
            regularity_result.diagnostics.rank >= active_rows.shape[0]
        ) & jnp.isfinite(regularity_result.diagnostics.condition_estimate)
    finite = (
        evaluation.finite
        & jnp.all(jnp.isfinite(equality_multipliers))
        & jnp.all(jnp.isfinite(inequality_multipliers))
        & jnp.all(jnp.isfinite(stationarity_coordinates))
        & jnp.isfinite(primal)
        & jnp.isfinite(dual)
        & jnp.isfinite(complementarity)
    )
    certified = (
        finite
        & regular
        & (stationarity <= tolerance_)
        & (primal <= tolerance_)
        & (dual <= tolerance_)
        & (complementarity <= tolerance_)
    )
    return OptimizationCertificate(
        kind=kind,
        tolerance=tolerance_,
        optimality_norm=jnp.maximum(
            stationarity,
            jnp.maximum(primal, jnp.maximum(dual, complementarity)),
        ),
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        projected_stationarity=stationarity,
        finite=finite,
        regular=regular,
        certified=certified,
        evaluation_work=1,
        certificate_id=f"{prepared.problem.problem_id}/{kind}",
        precision_evidence=precision_.evidence_for(
            model_parameters,
            precision_.residual(evaluation.gradient),
            children=(
                {}
                if canonical.precision_evidence is None
                else {"canonical-kkt": canonical.precision_evidence}
            ),
        ),
    )


__all__ = [
    "certify_constrained_physical",
    "certify_least_squares_physical",
    "reconcile_optimization_status",
]
