#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe
from jax import core as jax_core
from jaxtyping import Array

from .._numerics import weight_ess
from ..linalg import DenseLinearOperator
from ..optim import (
    AbstractScalarIterativeMethod,
    implicit_minimize,
    minimize,
    NewtonKrylov,
    OptimizationStatus,
    OptimizationTermination,
)
from ._geometry import (
    initial_coordinates,
    physical_dual,
    prepare_moment_geometry,
    weighted_covariance,
    weights_from_coordinates,
)
from ._problem import (
    ExactMoments,
    MomentCalibrationExecutionPolicy,
    MomentCalibrationPolicy,
    MomentCalibrationProblem,
    QuadraticMoments,
)
from ._results import (
    MomentCalibrationDiagnostics,
    MomentCalibrationProvenance,
    MomentCalibrationResult,
    MomentCalibrationStatus,
)


def calibrate_moments(
    problem: MomentCalibrationProblem,
    /,
    *,
    method: AbstractScalarIterativeMethod | None = None,
    termination: OptimizationTermination | None = None,
    policy: MomentCalibrationPolicy | None = None,
    execution: MomentCalibrationExecutionPolicy | None = None,
    initial_dual: Array | None = None,
) -> MomentCalibrationResult:
    """Return an audited relative-entropy calibration of one finite prior."""

    execution_ = MomentCalibrationExecutionPolicy() if execution is None else execution
    if execution_.route == "canonical-conic":
        from ._canonical import calibrate_moments_conic

        return calibrate_moments_conic(problem, execution_.solver)
    if execution_.route == "mixed-integer":
        from ._subset import calibrate_moments_subset

        return calibrate_moments_subset(problem, execution_.solver)
    _require_dual_compatible(problem, execution_)
    method_, termination_, policy_ = _resolve_configuration(
        method,
        termination,
        policy,
    )
    geometry = _require_geometry(prepare_moment_geometry(problem, policy_))
    initial = initial_coordinates(geometry, initial_dual)

    def objective(coordinates, _):
        return _dual_objective(problem, geometry, coordinates)

    optimization = minimize(
        objective,
        initial,
        method=method_,
        termination=termination_,
    )
    return _result(
        problem,
        geometry,
        optimization.parameters,
        optimization.status,
        optimization.diagnostics,
        optimization.provenance,
        termination_,
        policy_,
    )


def implicit_calibrate_moments(
    problem: MomentCalibrationProblem,
    /,
    *,
    method: AbstractScalarIterativeMethod | None = None,
    termination: OptimizationTermination | None = None,
    policy: MomentCalibrationPolicy | None = None,
    execution: MomentCalibrationExecutionPolicy | None = None,
    initial_dual: Array | None = None,
) -> Array:
    """Return calibrated weights with regular-stationarity implicit derivatives."""

    execution_ = MomentCalibrationExecutionPolicy() if execution is None else execution
    _require_dual_compatible(problem, execution_)
    method_, termination_, policy_ = _resolve_configuration(
        method,
        termination,
        policy,
    )
    geometry = _require_geometry(prepare_moment_geometry(problem, policy_))
    if isinstance(problem.target, ExactMoments):
        affine_tolerance = _affine_tolerance(problem, geometry, policy_)
        geometry = _error_if_geometry(
            geometry,
            geometry.affine_residual_norm > affine_tolerance,
            "Implicit exact calibration requires an affine-consistent target.",
        )
    initial = initial_coordinates(geometry, initial_dual)

    def objective(coordinates, _):
        return _dual_objective(problem, geometry, coordinates)

    coordinates = implicit_minimize(
        objective,
        initial,
        method=method_,
        termination=termination_,
    )
    _, weights = weights_from_coordinates(problem, geometry, coordinates)
    achieved, final_covariance = weighted_covariance(problem, weights)
    residual = achieved - problem.target.values
    final_hessian = _coordinate_hessian(
        problem,
        geometry,
        final_covariance,
    )
    eigenvalues = jnp.linalg.eigvalsh(final_hessian)
    minimum_eigenvalue = jnp.min(eigenvalues)
    maximum_eigenvalue = jnp.maximum(jnp.max(eigenvalues), 1.0)
    regularity_threshold = (
        jnp.maximum(
            policy_.regularity_relative_tolerance,
            jnp.sqrt(jnp.finfo(weights.dtype).eps),
        )
        * maximum_eigenvalue
    )
    scaled_residual = residual / geometry.moment_scales
    residual_threshold = termination_.optimality_threshold(
        jnp.linalg.norm(
            (geometry.prior_moments - problem.target.values) / geometry.moment_scales
        )
    )
    invalid = ~jnp.all(jnp.isfinite(weights))
    if isinstance(problem.target, ExactMoments):
        invalid = (
            invalid
            | (jnp.max(jnp.abs(scaled_residual)) > residual_threshold)
            | (minimum_eigenvalue <= regularity_threshold)
        )
    return _error_if_array(
        weights,
        invalid,
        "Implicit moment calibration requires a finite regular solution.",
    )


def _dual_objective(problem, geometry, coordinates):
    dual = physical_dual(geometry, coordinates)
    scores = problem.moment_map.transpose_mv(dual)
    centered_scores = scores - jnp.vdot(geometry.prior_moments, dual).real
    logits = jnp.where(
        geometry.active_support,
        geometry.prior_log_weights + centered_scores,
        -jnp.inf,
    )
    target_delta = problem.target.values - geometry.prior_moments
    value = jsp.special.logsumexp(logits) - jnp.vdot(target_delta, dual).real
    if isinstance(problem.target, ExactMoments):
        inactive = ~geometry.retained_directions
        return value + 0.5 * jnp.sum(jnp.where(inactive, coordinates**2, 0.0))
    assert isinstance(problem.target, QuadraticMoments)
    covariance_dual = problem.target.covariance.mv(dual)
    return value + 0.5 * oe.contract("i,i->", dual, covariance_dual)


def _result(
    problem,
    geometry,
    coordinates,
    optimizer_status,
    optimization_diagnostics,
    optimization_provenance,
    termination,
    policy,
):
    log_weights, weights = weights_from_coordinates(
        problem,
        geometry,
        coordinates,
    )
    dual = physical_dual(geometry, coordinates)
    achieved, final_covariance = weighted_covariance(problem, weights)
    residual = achieved - problem.target.values
    scaled_residual = residual / geometry.moment_scales
    gradient = _dual_gradient(problem, geometry, coordinates, residual)
    final_hessian = _coordinate_hessian(problem, geometry, final_covariance)
    final_eigenvalues = jnp.linalg.eigvalsh(final_hessian)
    minimum_final = jnp.min(final_eigenvalues)
    maximum_final = jnp.max(final_eigenvalues)
    condition = maximum_final / jnp.maximum(
        minimum_final,
        jnp.finfo(weights.dtype).tiny,
    )
    retained = geometry.retained_directions
    minimum_prior = jnp.min(jnp.where(retained, geometry.covariance_eigenvalues, jnp.inf))
    minimum_prior = jnp.where(
        jnp.any(retained),
        minimum_prior,
        0.0,
    )
    maximum_prior = jnp.max(jnp.where(retained, geometry.covariance_eigenvalues, 0.0))
    safe_log_weights = jnp.where(geometry.active_support, log_weights, 0.0)
    safe_log_prior = jnp.where(
        geometry.active_support,
        geometry.prior_log_weights,
        0.0,
    )
    relative_entropy = jnp.sum(weights * (safe_log_weights - safe_log_prior))
    minimum_weight = jnp.min(jnp.where(geometry.active_support, weights, jnp.inf))
    maximum_weight = jnp.max(jnp.where(geometry.active_support, weights, 0.0))
    maximum_log_ratio = jnp.max(
        jnp.where(
            geometry.active_support,
            safe_log_weights - safe_log_prior,
            -jnp.inf,
        )
    )
    normalization_residual = jnp.abs(jnp.sum(weights) - 1.0)
    initial_residual_norm = jnp.linalg.norm(
        (geometry.prior_moments - problem.target.values) / geometry.moment_scales
    )
    residual_threshold = termination.optimality_threshold(initial_residual_norm)
    affine_tolerance = _affine_tolerance(problem, geometry, policy)
    regularity_threshold = jnp.maximum(
        policy.regularity_relative_tolerance,
        jnp.sqrt(jnp.finfo(weights.dtype).eps),
    ) * jnp.maximum(maximum_final, 1.0)
    all_finite = (
        jnp.all(jnp.isfinite(weights))
        & jnp.all(jnp.isfinite(dual))
        & jnp.all(jnp.isfinite(achieved))
        & jnp.all(jnp.isfinite(gradient))
        & jnp.isfinite(relative_entropy)
        & jnp.isfinite(minimum_final)
        & jnp.isfinite(maximum_final)
    )
    optimizer_success = optimizer_status == int(OptimizationStatus.SUCCESS)
    status = jnp.asarray(int(MomentCalibrationStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~optimizer_success,
        int(MomentCalibrationStatus.OPTIMIZATION_FAILED),
        status,
    )
    if isinstance(problem.target, ExactMoments):
        status = jnp.where(
            minimum_final <= regularity_threshold,
            int(MomentCalibrationStatus.REGULARITY_NOT_CERTIFIED),
            status,
        )
        status = jnp.where(
            jnp.max(jnp.abs(scaled_residual)) > residual_threshold,
            int(MomentCalibrationStatus.TARGET_RESIDUAL_NOT_MET),
            status,
        )
        status = jnp.where(
            geometry.affine_residual_norm > affine_tolerance,
            int(MomentCalibrationStatus.AFFINE_TARGET_INCONSISTENT),
            status,
        )
    status = jnp.where(
        ~all_finite,
        int(MomentCalibrationStatus.NONFINITE_RESULT),
        status,
    ).astype(jnp.int32)
    diagnostics = MomentCalibrationDiagnostics(
        optimizer_status=jnp.asarray(optimizer_status, dtype=jnp.int32),
        optimization=optimization_diagnostics,
        prior_moments=geometry.prior_moments,
        target_residual=residual,
        scaled_target_residual=scaled_residual,
        maximum_absolute_residual=jnp.max(jnp.abs(residual)),
        maximum_scaled_residual=jnp.max(jnp.abs(scaled_residual)),
        affine_residual_norm=geometry.affine_residual_norm,
        numerical_affine_rank=geometry.numerical_affine_rank,
        rank_cutoff=geometry.rank_cutoff,
        minimum_prior_eigenvalue=minimum_prior,
        maximum_prior_eigenvalue=maximum_prior,
        minimum_final_eigenvalue=minimum_final,
        final_condition_estimate=condition,
        dual_gradient_norm=jnp.linalg.norm(gradient),
        dual_norm=jnp.linalg.norm(dual),
        relative_entropy=relative_entropy,
        effective_sample_size=weight_ess(weights),
        active_support=jnp.sum(geometry.active_support.astype(jnp.int32)),
        minimum_active_weight=minimum_weight,
        maximum_active_weight=maximum_weight,
        maximum_log_weight_ratio=maximum_log_ratio,
        normalization_residual=normalization_residual,
        geometry_finite=geometry.finite,
        spectrum=geometry.spectrum_diagnostics,
    )
    provenance = MomentCalibrationProvenance(
        problem_id=problem.problem_id,
        operator_id=problem.moment_map.operator_id,
        target_kind=(
            "exact" if isinstance(problem.target, ExactMoments) else "quadratic"
        ),
        source_points=problem.source_points,
        moment_count=problem.moment_count,
        execution=(
            "dense" if isinstance(problem.moment_map, DenseLinearOperator) else "operator"
        ),
        differentiation="explicit",
        optimizer=optimization_provenance,
    )
    return MomentCalibrationResult(
        problem=problem,
        log_weights=log_weights,
        dual_variables=dual,
        achieved_moments=achieved,
        status=status,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _dual_gradient(problem, geometry, coordinates, residual):
    gradient = jnp.swapaxes(geometry.transform, -1, -2) @ residual
    if isinstance(problem.target, ExactMoments):
        return gradient + jnp.where(
            geometry.retained_directions,
            0.0,
            coordinates,
        )
    assert isinstance(problem.target, QuadraticMoments)
    dual = physical_dual(geometry, coordinates)
    covariance_dual = problem.target.covariance.mv(dual)
    return gradient + oe.contract("ji,j->i", geometry.transform, covariance_dual)


def _coordinate_hessian(problem, geometry, covariance):
    physical_hessian = covariance
    if isinstance(problem.target, QuadraticMoments):
        physical_hessian = physical_hessian + problem.target.covariance._materialize()
    transformed = (
        jnp.swapaxes(geometry.transform, -1, -2) @ physical_hessian @ geometry.transform
    )
    if isinstance(problem.target, ExactMoments):
        transformed = transformed + jnp.diag(
            (~geometry.retained_directions).astype(transformed.dtype)
        )
    return 0.5 * (transformed + jnp.swapaxes(transformed, -1, -2))


def _affine_tolerance(problem, geometry, policy):
    scale = jnp.maximum(
        jnp.linalg.norm(
            (problem.target.values - geometry.prior_moments) / geometry.moment_scales
        ),
        1.0,
    )
    return policy.affine_absolute_tolerance + policy.affine_relative_tolerance * scale


def _require_dual_compatible(problem, execution):
    if not isinstance(execution, MomentCalibrationExecutionPolicy):
        raise TypeError("execution must be a MomentCalibrationExecutionPolicy or None.")
    if execution.route != "dual-relative-entropy":
        raise ValueError(
            "This entry point executes only the declared dual-relative-entropy "
            "route; canonical conic and mixed-integer routes require their "
            "corresponding prepared program."
        )
    if (
        problem.group_constraints is not None
        or problem.subset is not None
        or problem.boundary is not None
    ):
        raise ValueError(
            "Group, subset, and boundary structures are incompatible with the "
            "regular dual-relative-entropy route."
        )


def _resolve_configuration(method, termination, policy):
    method_ = NewtonKrylov() if method is None else method
    termination_ = OptimizationTermination() if termination is None else termination
    policy_ = MomentCalibrationPolicy() if policy is None else policy
    if not isinstance(method_, AbstractScalarIterativeMethod):
        raise TypeError("method must be an AbstractScalarIterativeMethod or None.")
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination or None.")
    if not isinstance(policy_, MomentCalibrationPolicy):
        raise TypeError("policy must be a MomentCalibrationPolicy or None.")
    return method_, termination_, policy_


def _require_geometry(geometry):
    return _error_if_geometry(
        geometry,
        ~geometry.finite,
        "Moment calibration inputs or affine geometry are invalid.",
    )


def _error_if_geometry(geometry, predicate, message):
    if not isinstance(predicate, jax_core.Tracer):
        if bool(predicate):
            raise ValueError(message)
        return geometry
    checked = eqx.error_if(geometry.transform, predicate, message)
    return eqx.tree_at(lambda item: item.transform, geometry, checked)


def _error_if_array(value, predicate, message):
    if not isinstance(predicate, jax_core.Tracer):
        if bool(predicate):
            raise eqx.EquinoxRuntimeError(message)
        return value
    return eqx.error_if(value, predicate, message)


__all__ = ["calibrate_moments", "implicit_calibrate_moments"]
