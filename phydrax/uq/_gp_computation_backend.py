#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from ..kernels import AbstractPositiveDefiniteKernel
from ..linalg import (
    AbstractLinearOperator,
    DenseCholesky,
    DenseLinearOperator,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolveDiagnostics,
    LinearSolvePolicy,
    LinearSystem,
    MaterializationPolicy,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    solve,
    SolveResourcePolicy,
)
from ..linalg.backends._jax_dense import DenseCholeskyState


class _PositiveFactor(NamedTuple):
    prepared: PreparedLinearSolve
    cholesky: Array
    diagnostics: LinearSolveDiagnostics
    status: Array


class _ProjectedGaussianProcessState(NamedTuple):
    kernel_action: Array
    active_mask: Array
    prior_diagonal: Array
    effective_observation_variance: Array
    projected_noise: Array
    projected_covariance: Array
    noise_factor: _PositiveFactor
    covariance_factor: _PositiveFactor
    row_batch_size: int
    workspace_bytes: int
    kernel_entry_count: int


class _PredictiveGeometry(NamedTuple):
    query_action: Array
    solved_query_action: Array
    covariance: Array | None
    variance: Array
    row_batch_size: int
    workspace_bytes: int
    kernel_entry_count: int


def _build_projected_state(
    observation_points: Array,
    /,
    *,
    kernel: AbstractPositiveDefiniteKernel,
    noise_scale: Array,
    jitter: Array,
    actions: AbstractLinearOperator,
    active_mask: Array,
    max_workspace_bytes: int,
    max_factorization_bytes: int,
    checkpoint: bool,
) -> _ProjectedGaussianProcessState:
    observation_count = int(observation_points.shape[0])
    noise = jnp.broadcast_to(jnp.asarray(noise_scale), (observation_count,))
    effective_variance = noise * noise + jnp.asarray(jitter)
    effective_variance = eqx.error_if(
        effective_variance,
        jnp.any(~jnp.isfinite(effective_variance)) | jnp.any(effective_variance <= 0.0),
        "Effective GP observation variance must be finite and positive.",
    )
    kernel_action, batch_size, workspace_bytes, kernel_entries = _kernel_matrix_action(
        kernel,
        observation_points,
        observation_points,
        actions,
        max_workspace_bytes=max_workspace_bytes,
        checkpoint=checkpoint,
    )
    mask = jnp.asarray(active_mask, dtype=bool)
    if mask.shape != (actions.source.size,):
        raise ValueError("Action active_mask must align with action capacity.")
    active_float = mask.astype(effective_variance.dtype)
    projected_noise = _weighted_action_gram(actions, effective_variance)
    projected_kernel = _transpose_action_columns(actions, kernel_action)
    projected_noise = projected_noise * active_float[:, None] * active_float[
        None, :
    ] + jnp.diag(1.0 - active_float)
    projected_covariance = (
        projected_kernel * active_float[:, None] * active_float[None, :] + projected_noise
    )
    projected_noise = _symmetrize(projected_noise)
    projected_covariance = _symmetrize(projected_covariance)
    noise_factor = _factorize_positive(
        projected_noise,
        name="projected-noise",
        max_factorization_bytes=max_factorization_bytes,
    )
    covariance_factor = _factorize_positive(
        projected_covariance,
        name="projected-covariance",
        max_factorization_bytes=max_factorization_bytes,
    )
    return _ProjectedGaussianProcessState(
        active_mask=mask,
        kernel_action=kernel_action,
        prior_diagonal=kernel.diagonal(observation_points),
        effective_observation_variance=effective_variance,
        projected_noise=projected_noise,
        projected_covariance=projected_covariance,
        noise_factor=noise_factor,
        covariance_factor=covariance_factor,
        row_batch_size=batch_size,
        workspace_bytes=workspace_bytes,
        kernel_entry_count=kernel_entries,
    )


def _kernel_matrix_action(
    kernel: AbstractPositiveDefiniteKernel,
    left_points: Array,
    right_points: Array,
    actions: AbstractLinearOperator,
    /,
    *,
    max_workspace_bytes: int,
    checkpoint: bool,
) -> tuple[Array, int, int, int]:
    left_count = int(left_points.shape[0])
    right_count = int(right_points.shape[0])
    action_count = actions.source.size
    itemsize = int(jnp.dtype(left_points.dtype).itemsize)
    bytes_per_row = max(1, (right_count + action_count) * itemsize)
    batch_size = min(left_count, max_workspace_bytes // bytes_per_row)
    if batch_size < 1:
        raise ValueError(
            "Kernel-action workspace cannot hold one kernel row and action output."
        )

    def one(point: Array) -> Array:
        kernel_row = kernel.matrix(point[None, ...], right_points)[0]
        return actions.transpose_mv(kernel_row)

    body = jax.checkpoint(one) if checkpoint else one
    result = jax.lax.map(body, left_points, batch_size=batch_size)
    workspace_bytes = batch_size * bytes_per_row
    return result, batch_size, workspace_bytes, left_count * right_count


def _predictive_geometry(
    query_points: Array,
    observation_points: Array,
    /,
    *,
    kernel: AbstractPositiveDefiniteKernel,
    actions: AbstractLinearOperator,
    covariance_factor: _PositiveFactor,
    max_workspace_bytes: int,
    checkpoint: bool,
    full_covariance: bool,
) -> _PredictiveGeometry:
    query_action, batch_size, workspace_bytes, kernel_entries = _kernel_matrix_action(
        kernel,
        query_points,
        observation_points,
        actions,
        max_workspace_bytes=max_workspace_bytes,
        checkpoint=checkpoint,
    )
    solved, successful = _solve_columns(covariance_factor, query_action.T)
    solved = eqx.error_if(
        solved,
        ~successful,
        "Projected GP covariance solve failed during prediction.",
    )
    prior_diagonal = kernel.diagonal(query_points)
    downdate_diagonal = ein.contract("qm,mq->q", query_action, solved)
    variance = _validated_variance(
        prior_diagonal - downdate_diagonal,
        prior_diagonal,
        action_count=actions.source.size,
    )
    covariance = None
    if full_covariance:
        prior_covariance = kernel.matrix(query_points, query_points)
        covariance = _symmetrize(prior_covariance - query_action @ solved)
    return _PredictiveGeometry(
        query_action=query_action,
        solved_query_action=solved,
        covariance=covariance,
        variance=variance,
        row_batch_size=batch_size,
        workspace_bytes=workspace_bytes,
        kernel_entry_count=kernel_entries,
    )


def _posterior_mean(
    residual: Array,
    /,
    *,
    actions: AbstractLinearOperator,
    covariance_factor: _PositiveFactor,
    query_action: Array,
) -> Array:
    projected_residual = actions.transpose_mv(residual)
    solved = _solve_vector(covariance_factor, projected_residual)
    return ein.contract("qm,m->q", query_action, solved)


def _posterior_mean_from_solved_geometry(
    residual: Array,
    /,
    *,
    actions: AbstractLinearOperator,
    solved_query_action: Array,
) -> Array:
    projected_residual = actions.transpose_mv(residual)
    return ein.contract("mq,m->q", solved_query_action, projected_residual)


def _computation_aware_elbo(
    residual: Array,
    /,
    *,
    actions: AbstractLinearOperator,
    projected: _ProjectedGaussianProcessState,
) -> Array:
    projected_residual = actions.transpose_mv(residual)
    alpha = _solve_vector(projected.covariance_factor, projected_residual)
    mean_correction = ein.contract("nm,m->n", projected.kernel_action, alpha)
    solved_kernel_action, successful = _solve_columns(
        projected.covariance_factor,
        projected.kernel_action.T,
    )
    solved_kernel_action = eqx.error_if(
        solved_kernel_action,
        ~successful,
        "Projected GP covariance solve failed while computing the ELBO.",
    )
    downdate_diagonal = ein.contract(
        "nm,mn->n",
        projected.kernel_action,
        solved_kernel_action,
    )
    variance = _validated_variance(
        projected.prior_diagonal - downdate_diagonal,
        projected.prior_diagonal,
        action_count=actions.source.size,
    )
    effective_variance = projected.effective_observation_variance
    error = residual - mean_correction
    expected_log_likelihood = -0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi * effective_variance)
        + (error * error + variance) / effective_variance
    )

    solved_noise, successful = _solve_columns(
        projected.covariance_factor,
        projected.projected_noise,
    )
    solved_noise = eqx.error_if(
        solved_noise,
        ~successful,
        "Projected GP covariance solve failed while computing the KL divergence.",
    )
    trace_term = jnp.trace(solved_noise)
    quadratic = ein.contract("m,m->", projected_residual, alpha)
    noise_correction = ein.contract(
        "m,mn,n->",
        alpha,
        projected.projected_noise,
        alpha,
    )
    logdet_covariance = _positive_logdet(projected.covariance_factor)
    logdet_noise = _positive_logdet(projected.noise_factor)
    action_count = jnp.sum(projected.active_mask, dtype=projected_residual.dtype)
    kl = 0.5 * (
        trace_term
        + quadratic
        - action_count
        + logdet_covariance
        - logdet_noise
        - noise_correction
    )
    return expected_log_likelihood - kl


def _factorize_positive(
    matrix: Array,
    /,
    *,
    name: str,
    max_factorization_bytes: int,
) -> _PositiveFactor:
    itemsize = int(jnp.dtype(matrix.dtype).itemsize)
    required_bytes = int(matrix.size) * itemsize
    if required_bytes > max_factorization_bytes:
        raise ValueError(
            f"{name} requires {required_bytes} factorization bytes, exceeding "
            f"the policy limit {max_factorization_bytes}."
        )
    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "asserted",
                "positive_semidefinite": "transformed",
            },
        ),
        operator_id=f"computation-aware-gp:{name}:{matrix.shape[0]}",
    )
    prepared = prepare(
        LinearSystem(operator),
        LinearSolvePolicy(
            DenseCholesky(),
            materialization=MaterializationPolicy(
                max_entries=max(1, int(matrix.size)),
                max_bytes=max(1, required_bytes),
            ),
            differentiation=DifferentiationPolicy("mathematical"),
            failure=FailurePolicy("status"),
            resources=SolveResourcePolicy(
                factorization_bytes=max_factorization_bytes,
                workspace_bytes=max_factorization_bytes,
            ),
        ),
    )
    if not isinstance(prepared.state, DenseCholeskyState):
        raise TypeError("Projected GP factorization must resolve to dense Cholesky.")
    probe = solve(prepared, jnp.ones((matrix.shape[0],), dtype=matrix.dtype))
    return _PositiveFactor(
        prepared,
        prepared.state.factor,
        probe.diagnostics,
        probe.status,
    )


def _solve_vector(factor: _PositiveFactor, right: Array, /) -> Array:
    result = solve(factor.prepared, right)
    return eqx.error_if(
        result.value,
        ~result.successful,
        "Projected GP positive-definite solve failed.",
    )


def _solve_columns(
    factor: _PositiveFactor,
    right: Array,
    /,
) -> tuple[Array, Array]:
    def solve_one(column: Array) -> tuple[Array, Array]:
        result = solve(factor.prepared, column)
        return result.value, result.successful

    values, successful = jax.vmap(solve_one, in_axes=1, out_axes=(1, 0))(right)
    return values, jnp.all(successful)


def _positive_logdet(factor: _PositiveFactor, /) -> Array:
    return 2.0 * jnp.sum(jnp.log(jnp.diag(factor.cholesky)))


def _weighted_action_gram(
    actions: AbstractLinearOperator,
    diagonal: Array,
    /,
) -> Array:
    if isinstance(actions, DenseLinearOperator):
        return ein.contract(
            "ni,n,nj->ij",
            actions.matrix,
            diagonal,
            actions.matrix,
        )

    action_count = actions.source.size
    dtype = actions.source.structure().dtype

    def one(index: Array) -> Array:
        basis = jax.nn.one_hot(index, action_count, dtype=dtype)
        column = actions.mv(basis)
        return actions.transpose_mv(diagonal * column)

    return jax.lax.map(one, jnp.arange(action_count, dtype=jnp.int32)).T


def _transpose_action_columns(
    actions: AbstractLinearOperator,
    matrix: Array,
    /,
) -> Array:
    return jax.vmap(actions.transpose_mv, in_axes=1, out_axes=1)(matrix)


def _validated_variance(
    variance: Array,
    scale: Array,
    /,
    *,
    action_count: int,
) -> Array:
    magnitude = jnp.maximum(jnp.max(jnp.abs(scale)), jnp.asarray(1.0, scale.dtype))
    tolerance = (
        32.0
        * jnp.finfo(scale.dtype).eps
        * max(int(scale.shape[0]), int(action_count), 1)
        * magnitude
    )
    variance = eqx.error_if(
        variance,
        jnp.any(~jnp.isfinite(variance)) | jnp.any(variance < -tolerance),
        "Computation-aware GP variance is materially negative or nonfinite.",
    )
    return jnp.maximum(variance, 0.0)


def _symmetrize(matrix: Array, /) -> Array:
    return 0.5 * (matrix + matrix.T)


__all__: list[str] = []
