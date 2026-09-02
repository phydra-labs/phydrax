#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._numerics import log_normalize
from .._strict import StrictModule
from ..linalg import DenseLinearOperator, OperatorProperties
from ..linalg.eigen import (
    Eigenproblem,
    self_adjoint_spectrum,
    SelfAdjointSpectrumStatus,
)
from ._problem import (
    ExactMoments,
    IntervalMoments,
    MomentCalibrationPolicy,
    MomentCalibrationProblem,
    QuadraticMoments,
)


class MomentGeometry(StrictModule):
    """Fixed-capacity affine geometry and dual-coordinate preconditioner."""

    prior_log_weights: Array
    prior_weights: Array
    active_support: Array
    prior_moments: Array
    covariance: Array
    moment_scales: Array
    transform: Array
    retained_directions: Array
    covariance_eigenvalues: Array
    rank_cutoff: Array
    affine_residual: Array
    affine_residual_norm: Array
    finite: Array
    feature_values_finite: Array
    spectrum_diagnostics: Any

    @property
    def numerical_affine_rank(self) -> Array:
        return jnp.sum(self.retained_directions.astype(jnp.int32))


def prepare_moment_geometry(
    problem: MomentCalibrationProblem,
    policy: MomentCalibrationPolicy,
    /,
) -> MomentGeometry:
    """Prepare normalized prior geometry without materializing source Gram matrices."""

    if not isinstance(problem, MomentCalibrationProblem):
        raise TypeError("problem must be a MomentCalibrationProblem.")
    if not isinstance(policy, MomentCalibrationPolicy):
        raise TypeError("policy must be a MomentCalibrationPolicy.")
    if isinstance(problem.target, IntervalMoments):
        raise ValueError(
            "IntervalMoments requires the canonical-conic calibration execution route."
        )
    if problem.moment_count > policy.maximum_moments:
        raise ValueError(
            f"Moment count {problem.moment_count} exceeds the configured maximum "
            f"{policy.maximum_moments}."
        )

    prior_weights, log_mass, prior_valid = log_normalize(
        problem.prior_log_weights,
        axes=0,
        mask=problem.mask,
    )
    active = problem.mask & jnp.isfinite(problem.prior_log_weights)
    normalized_log = jnp.where(
        active,
        problem.prior_log_weights - log_mass,
        -jnp.inf,
    )
    prior_moments, covariance, feature_finite = _weighted_covariance(
        problem,
        prior_weights,
    )
    dtype = prior_weights.dtype
    epsilon = jnp.finfo(dtype).eps
    covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    covariance_diagonal = jnp.maximum(jnp.diag(covariance), 0.0)
    maximum_variance = jnp.maximum(jnp.max(covariance_diagonal), 1.0)
    variance_floor = 64.0 * epsilon * maximum_variance
    moment_scales = jnp.where(
        covariance_diagonal > variance_floor,
        jnp.sqrt(covariance_diagonal),
        1.0,
    )
    standardized_covariance = covariance / (
        moment_scales[:, None] * moment_scales[None, :]
    )
    covariance_spectrum = _spectrum(
        standardized_covariance,
        problem,
        policy,
        suffix="covariance",
    )
    covariance_eigenvalues = covariance_spectrum.eigenvalues
    covariance_vectors = covariance_spectrum.eigenvectors
    maximum_eigenvalue = jnp.maximum(
        jnp.max(jnp.abs(covariance_eigenvalues)),
        1.0,
    )
    relative_cutoff = (
        64.0 * epsilon * problem.moment_count
        if policy.rank.relative_cutoff is None
        else policy.rank.relative_cutoff
    )
    rank_cutoff = jnp.asarray(relative_cutoff, dtype=dtype) * maximum_eigenvalue
    retained = covariance_eigenvalues > rank_cutoff
    target_delta = problem.target.values - prior_moments
    standardized_delta = target_delta / moment_scales
    retained_projection = covariance_vectors @ (
        retained * (jnp.swapaxes(covariance_vectors, -1, -2) @ standardized_delta)
    )
    affine_residual = standardized_delta - retained_projection

    if isinstance(problem.target, ExactMoments):
        safe_eigenvalues = jnp.maximum(covariance_eigenvalues, rank_cutoff)
        factors = jnp.where(retained, jax.lax.rsqrt(safe_eigenvalues), 0.0)
        transform = (covariance_vectors * factors[None, :]) / moment_scales[:, None]
        preconditioner_diagnostics = covariance_spectrum.diagnostics
        preconditioner_finite = covariance_spectrum.diagnostics.finite
    else:
        assert isinstance(problem.target, QuadraticMoments)
        if not problem.target.covariance.capabilities.materialize:
            raise ValueError(
                "Dual calibration preconditioning requires materializable covariance."
            )
        target_covariance = problem.target.covariance._materialize()
        target_covariance = 0.5 * (
            target_covariance + jnp.swapaxes(target_covariance, -1, -2)
        )
        base_hessian = covariance + target_covariance
        base_diagonal = jnp.maximum(jnp.diag(base_hessian), 0.0)
        base_scales = jnp.where(
            base_diagonal > variance_floor,
            jnp.sqrt(base_diagonal),
            1.0,
        )
        standardized_base = base_hessian / (base_scales[:, None] * base_scales[None, :])
        base_spectrum = _spectrum(
            standardized_base,
            problem,
            policy,
            suffix="soft-hessian",
        )
        safe_eigenvalues = jnp.maximum(base_spectrum.eigenvalues, rank_cutoff)
        transform = (
            base_spectrum.eigenvectors * jax.lax.rsqrt(safe_eigenvalues)[None, :]
        ) / base_scales[:, None]
        preconditioner_diagnostics = base_spectrum.diagnostics
        preconditioner_finite = base_spectrum.diagnostics.finite

    covariance_success = covariance_spectrum.status == int(
        SelfAdjointSpectrumStatus.SUCCESS
    )
    materially_negative = jnp.min(covariance_eigenvalues) < -rank_cutoff
    target_finite = jnp.all(jnp.isfinite(problem.target.values))
    covariance_valid = (
        jnp.asarray(True)
        if isinstance(problem.target, ExactMoments)
        else jnp.all(jnp.isfinite(problem.target.covariance._materialize()))
    )
    finite = (
        prior_valid
        & feature_finite
        & target_finite
        & covariance_valid
        & jnp.all(jnp.isfinite(prior_moments))
        & jnp.all(jnp.isfinite(covariance))
        & covariance_success
        & preconditioner_finite
        & (~materially_negative)
    )
    if policy.rank.require_full_rank:
        finite = finite & jnp.all(retained)

    return MomentGeometry(
        prior_log_weights=normalized_log,
        prior_weights=prior_weights,
        active_support=jax.lax.stop_gradient(active),
        prior_moments=prior_moments,
        covariance=covariance,
        moment_scales=moment_scales,
        transform=jax.lax.stop_gradient(transform),
        retained_directions=jax.lax.stop_gradient(retained),
        covariance_eigenvalues=covariance_eigenvalues,
        rank_cutoff=rank_cutoff,
        affine_residual=affine_residual,
        affine_residual_norm=jnp.linalg.norm(affine_residual),
        finite=finite,
        feature_values_finite=feature_finite,
        spectrum_diagnostics=(
            covariance_spectrum.diagnostics,
            preconditioner_diagnostics,
        ),
    )


def initial_coordinates(
    geometry: MomentGeometry,
    initial_dual: Array | None,
    /,
) -> Array:
    """Map an optional physical dual warm start into prepared coordinates."""

    if initial_dual is None:
        return jnp.zeros_like(geometry.prior_moments)
    dual = jnp.asarray(initial_dual, dtype=geometry.prior_weights.dtype)
    if dual.shape != geometry.prior_moments.shape:
        raise ValueError(f"initial_dual must have shape {geometry.prior_moments.shape}.")
    coordinates = jnp.linalg.lstsq(geometry.transform, dual, rcond=None)[0]
    return jnp.where(geometry.retained_directions, coordinates, 0.0)


def physical_dual(geometry: MomentGeometry, coordinates: Array, /) -> Array:
    return geometry.transform @ coordinates


def log_weights_from_coordinates(
    problem: MomentCalibrationProblem,
    geometry: MomentGeometry,
    coordinates: Array,
    /,
) -> Array:
    dual = physical_dual(geometry, coordinates)
    scores = problem.moment_map.transpose_mv(dual)
    centered_scores = scores - jnp.vdot(geometry.prior_moments, dual).real
    logits = jnp.where(
        geometry.active_support,
        geometry.prior_log_weights + centered_scores,
        -jnp.inf,
    )
    _, log_normalizer, valid = log_normalize(logits, axes=0)
    return jnp.where(
        valid & geometry.active_support,
        logits - log_normalizer,
        -jnp.inf,
    )


def weights_from_coordinates(
    problem: MomentCalibrationProblem,
    geometry: MomentGeometry,
    coordinates: Array,
    /,
) -> tuple[Array, Array]:
    log_weights = log_weights_from_coordinates(problem, geometry, coordinates)
    weights = jnp.where(jnp.isfinite(log_weights), jnp.exp(log_weights), 0.0)
    return log_weights, weights


def weighted_covariance(
    problem: MomentCalibrationProblem,
    weights: Array,
    /,
) -> tuple[Array, Array]:
    moments, covariance, _ = _weighted_covariance(problem, weights)
    return moments, covariance


def _weighted_covariance(
    problem: MomentCalibrationProblem,
    weights: Array,
    /,
) -> tuple[Array, Array, Array]:
    operator = problem.moment_map
    if isinstance(operator, DenseLinearOperator):
        matrix = operator.matrix
        finite = jnp.all(jnp.isfinite(matrix))
        safe_matrix = jnp.where(jnp.isfinite(matrix), matrix, 0.0)
        moments = safe_matrix @ weights
        second = (safe_matrix * weights[None, :]) @ jnp.swapaxes(safe_matrix, -1, -2)
    else:
        moments = operator.mv(weights)
        identity = jnp.eye(problem.moment_count, dtype=weights.dtype)

        def column(finite, basis):
            row = operator.transpose_mv(basis)
            row_finite = jnp.all(jnp.isfinite(row))
            safe_row = jnp.where(jnp.isfinite(row), row, 0.0)
            return finite & row_finite, operator.mv(weights * safe_row)

        finite, columns = jax.lax.scan(column, jnp.asarray(True), identity)
        second = jnp.swapaxes(columns, -1, -2)
    covariance = second - moments[:, None] * moments[None, :]
    return moments, covariance, finite


def _spectrum(
    matrix: Array,
    problem: MomentCalibrationProblem,
    policy: MomentCalibrationPolicy,
    /,
    *,
    suffix: str,
):
    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
        operator_id=f"moment-{suffix}:{problem.problem_id}",
    )
    eigenproblem = Eigenproblem(
        operator,
        problem_id=f"moment-{suffix}:{problem.problem_id}",
    )
    return self_adjoint_spectrum(eigenproblem, policy=policy.spectrum)


__all__ = [
    "MomentGeometry",
    "initial_coordinates",
    "log_weights_from_coordinates",
    "physical_dual",
    "prepare_moment_geometry",
    "weighted_covariance",
    "weights_from_coordinates",
]
