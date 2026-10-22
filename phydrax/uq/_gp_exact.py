#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native dense exact GP execution with bounded kernel/score tiles.

The dense covariance and native factor remain necessary O(n²) storage. Byte
limits here govern retained numerical payloads and explicit kernel/contraction
work arrays, not opaque backend scratch or compiler peak memory.
"""

from __future__ import annotations

from math import prod
from numbers import Integral
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from phydrax import ein
from phydrax.kernels import AbstractPositiveDefiniteKernel
from phydrax.linalg import (
    condition_target_covariance,
    ConditionTargetPolicy,
    ConditionTargetResult,
    DenseLinearOperator,
    DifferentiationPolicy,
    FactorizationPolicy,
    factorize,
    FailurePolicy,
    LinearDerivativeSolvePolicy,
    LinearSolveResult,
    MaterializationPolicy,
    OperatorProperties,
    PreparedFactorization,
    SolveResourcePolicy,
)

from .._strict import StrictModule


@final
class ExactGaussianProcessExecutionPolicy(StrictModule):
    """Finite explicit tile and retained-factor budgets for exact scalar GPs.

    ``max_workspace_bytes`` bounds package-owned covariance/derivative tiles,
    selected inverse columns, and the native RHS workspace estimate. Mandatory
    dense factorization preparation and backend/compiler scratch are excluded.
    ``max_factor_storage_bytes`` bounds the distinct array payloads retained by
    the native factor, numerical evidence, and opt-in regularization evidence;
    it excludes the observation design and the likelihood state's parameters.
    """

    max_workspace_bytes: int = eqx.field(static=True)
    max_factor_storage_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_workspace_bytes: int = 64 * 1024 * 1024,
        max_factor_storage_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        for value in (max_workspace_bytes, max_factor_storage_bytes):
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError("Exact GP byte limits must be integers.")
            if value < 1:
                raise ValueError("Exact GP byte limits must be positive.")
        self.max_workspace_bytes = int(max_workspace_bytes)
        self.max_factor_storage_bytes = int(max_factor_storage_bytes)

    def column_batch_size(self, points: Array, /) -> int:
        """Capacity of the shared covariance and exact-score column tile.

        The conservative explicit-array estimate reserves ``8*n+16`` scalar
        entries plus ``10*n+2*d+8`` per column. The latter includes selected
        identity columns and native dense RHS work, kernel values/tangents, and
        the score tile; ``d`` is the flattened coordinate input size. Callback
        internals and native factorization preparation are outside this scope.
        """
        count = points.shape[0]
        if count == 0:
            raise ValueError("Exact GP execution requires at least one observation.")
        fixed = 8 * count + 16
        per_column = 10 * count + 2 * prod(points.shape[1:]) + 8
        capacity = (
            self.max_workspace_bytes // points.dtype.itemsize - fixed
        ) // per_column
        if capacity < 1:
            required = (fixed + per_column) * points.dtype.itemsize
            raise ValueError(
                f"Exact GP workspace requires at least {required} bytes for one "
                f"kernel/score column, exceeding the limit {self.max_workspace_bytes}."
            )
        return min(count, capacity)


type _CovariancePrimals = tuple[Array, AbstractPositiveDefiniteKernel, Array, Array]
type _CovarianceTangents = tuple[
    Array | None, AbstractPositiveDefiniteKernel, Array | None, Array | None
]


def exact_gp_factorization(
    points: Array,
    /,
    *,
    kernel: AbstractPositiveDefiniteKernel,
    noise_scale: Array,
    jitter: Array,
    execution: ExactGaussianProcessExecutionPolicy,
    conditioning: ConditionTargetPolicy | None,
) -> tuple[PreparedFactorization, ConditionTargetResult | None]:
    """Prepare one native covariance factor; do not repair failed factors."""
    if points.ndim != kernel.input_ndim + 1:
        raise ValueError("Observation design must match the kernel input rank.")
    width = execution.column_batch_size(points)
    count = points.shape[0]
    matrix_bytes = count * count * points.dtype.itemsize
    if 2 * matrix_bytes > execution.max_factor_storage_bytes:
        raise ValueError(
            f"Exact GP covariance and factor require at least {2 * matrix_bytes} "
            f"retained bytes, exceeding the limit {execution.max_factor_storage_bytes}."
        )
    covariance = _assemble_covariance(points, kernel, noise_scale, jitter, width)
    regularization = (
        None
        if conditioning is None
        else condition_target_covariance(covariance, conditioning)
    )
    matrix = covariance if regularization is None else regularization.covariance
    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"self_adjoint": "construction", "positive_definite": "asserted"},
        ),
    )
    prepared = factorize(
        operator,
        FactorizationPolicy(
            "cholesky",
            materialization=MaterializationPolicy(
                max_entries=count * count,
                max_bytes=matrix_bytes,
            ),
            differentiation=DifferentiationPolicy("mathematical"),
            derivative_solve=LinearDerivativeSolvePolicy(route="primal-factors"),
            failure=FailurePolicy("status"),
            resources=SolveResourcePolicy(
                factorization_bytes=execution.max_factor_storage_bytes,
                # A dense factorization necessarily prepares a dense matrix.
                # The separate tile limit is enforced before every score solve.
                workspace_bytes=max(execution.max_workspace_bytes, 2 * matrix_bytes),
            ),
        ),
    )
    require_exact_factor_storage(prepared, None, regularization, execution)
    return prepared, regularization


def exact_factor_storage(
    factorization: PreparedFactorization,
    evidence: LinearSolveResult | None,
    regularization: ConditionTargetResult | None,
    /,
) -> tuple[int, int]:
    """Count distinct Python array payloads, including native matrix and factor.

    Shared references in the prepared problem/operator/state are counted once.
    Compilers may alias or repack this representation independently.
    """
    seen: set[int] = set()
    elements = 0
    storage_bytes = 0
    for leaf in jax.tree_util.tree_leaves((factorization, evidence, regularization)):
        if eqx.is_array(leaf) and id(leaf) not in seen:
            seen.add(id(leaf))
            elements += leaf.size
            storage_bytes += leaf.size * leaf.dtype.itemsize
    return elements, storage_bytes


def require_exact_factor_storage(
    factorization: PreparedFactorization,
    evidence: LinearSolveResult | None,
    regularization: ConditionTargetResult | None,
    execution: ExactGaussianProcessExecutionPolicy,
    /,
) -> None:
    _, retained_bytes = exact_factor_storage(factorization, evidence, regularization)
    if retained_bytes > execution.max_factor_storage_bytes:
        raise ValueError(
            f"Exact GP native factor and evidence retain {retained_bytes} bytes, "
            f"exceeding the limit {execution.max_factor_storage_bytes}."
        )


def exact_gp_log_probability_from_factor(
    residual: Array,
    factorization: PreparedFactorization,
    /,
) -> Array:
    """Evaluate a fixed-factor Gaussian density through native solve evidence."""
    solved = factorization.solve(residual)
    return _log_probability_from_solve(residual, factorization, solved)


def _log_probability_from_solve(
    residual: Array,
    factorization: PreparedFactorization,
    solved: LinearSolveResult,
    /,
) -> Array:
    alpha = eqx.error_if(
        solved.value,
        ~jnp.all(solved.successful),
        "Exact GP covariance solve failed; inspect native factor evidence.",
    )
    logdet = factorization.log_abs_determinant()
    return -0.5 * (
        residual.size * jnp.log(2.0 * jnp.pi)
        + logdet
        + ein.contract("n,n->", residual, alpha)
    )


@eqx.filter_custom_jvp
def exact_gp_log_marginal_likelihood(
    points: Array,
    residual: Array,
    kernel: AbstractPositiveDefiniteKernel,
    noise_scale: Array,
    jitter: Array,
    execution: ExactGaussianProcessExecutionPolicy,
    conditioning: ConditionTargetPolicy | None,
    /,
) -> Array:
    """Exact scalar likelihood with streamed, rematerialized score derivatives.

    The score contracts every covariance entry and every inverse column. There
    is no stochastic trace, low-rank approximation, or parameter Jacobian.
    """
    factorization, regularization = exact_gp_factorization(
        points,
        kernel=kernel,
        noise_scale=noise_scale,
        jitter=jitter,
        execution=execution,
        conditioning=conditioning,
    )
    evidence = factorization.solve(residual)
    require_exact_factor_storage(factorization, evidence, regularization, execution)
    return _log_probability_from_solve(residual, factorization, evidence)


@exact_gp_log_marginal_likelihood.def_jvp
def _exact_gp_log_marginal_likelihood_jvp(
    primals: tuple[
        Array,
        Array,
        AbstractPositiveDefiniteKernel,
        Array,
        Array,
        ExactGaussianProcessExecutionPolicy,
        ConditionTargetPolicy | None,
    ],
    tangents: tuple[
        Array | None,
        Array | None,
        AbstractPositiveDefiniteKernel,
        Array | None,
        Array | None,
        object,
        object,
    ],
) -> tuple[Array, Array]:
    points, residual, kernel, noise_scale, jitter, execution, conditioning = primals
    points_dot, residual_dot, kernel_dot, noise_dot, jitter_dot, _, _ = tangents
    factorization, regularization = exact_gp_factorization(
        points,
        kernel=kernel,
        noise_scale=noise_scale,
        jitter=jitter,
        execution=execution,
        conditioning=conditioning,
    )
    solved = factorization.solve(residual)
    require_exact_factor_storage(factorization, solved, regularization, execution)
    alpha = eqx.error_if(
        solved.value,
        ~jnp.all(solved.successful),
        "Exact GP likelihood derivative requires a successful native solve.",
    )
    value = -0.5 * (
        residual.size * jnp.log(2.0 * jnp.pi)
        + factorization.log_abs_determinant()
        + ein.contract("n,n->", residual, alpha)
    )
    covariance_primals = (points, kernel, noise_scale, jitter)
    covariance_tangents = (points_dot, kernel_dot, noise_dot, jitter_dot)
    if not any(
        eqx.is_array(leaf) for leaf in jax.tree_util.tree_leaves(covariance_tangents)
    ):
        tangent = jnp.zeros((), dtype=points.dtype)
        if residual_dot is not None:
            tangent = -ein.contract("n,n->", alpha, residual_dot)
        return value, tangent
    width = execution.column_batch_size(points)
    added_diagonal_dot: Array | None = None
    if regularization is not None:
        diagonal = _observation_diagonal(*covariance_primals)
        _, diagonal_dot = eqx.filter_jvp(
            _observation_diagonal, covariance_primals, covariance_tangents
        )
        if diagonal_dot is None:
            diagonal_dot = jnp.zeros_like(diagonal)
        eta_dot = _regularization_eta_tangent(
            covariance_primals,
            covariance_tangents,
            diagonal,
            diagonal_dot,
            regularization,
            width,
        )
        added_diagonal_dot = regularization.eta * diagonal_dot + eta_dot * diagonal

    def contract_columns(start: Array, size: int) -> Array:
        columns = start + jnp.arange(size, dtype=jnp.int32)
        inverse = factorization.inverse_columns(
            columns,
            max_workspace_bytes=execution.max_workspace_bytes,
        )
        inverse_values = eqx.error_if(
            inverse.value,
            ~jnp.all(inverse.successful),
            "Exact GP inverse-column solve failed during likelihood differentiation.",
        )
        _, covariance_dot = _covariance_block_jvp(
            covariance_primals, covariance_tangents, start, size
        )
        if added_diagonal_dot is not None:
            covariance_dot = (
                covariance_dot
                + (jnp.arange(points.shape[0])[:, None] == columns[None, :])
                * added_diagonal_dot[:, None]
            )
        score = alpha[:, None] * alpha[columns][None, :] - inverse_values
        return 0.5 * ein.contract("nc,nc->", score, covariance_dot)

    def accumulate(index: Array, total: Array) -> Array:
        return total + contract_columns(index * width, width)

    count = points.shape[0]
    tangent = jax.lax.fori_loop(
        0,
        count // width,
        jax.checkpoint(accumulate),
        jnp.zeros((), dtype=points.dtype),
    )
    remainder = count % width
    if remainder:
        tangent = tangent + jax.checkpoint(contract_columns, static_argnums=(1,))(
            jnp.asarray(count - remainder, dtype=jnp.int32), remainder
        )
    if residual_dot is not None:
        tangent = tangent - ein.contract("n,n->", alpha, residual_dot)
    return value, tangent


def _observation_diagonal(
    points: Array,
    kernel: AbstractPositiveDefiniteKernel,
    noise_scale: Array,
    jitter: Array,
    /,
) -> Array:
    noise = jnp.broadcast_to(noise_scale, (points.shape[0],))
    return kernel.diagonal(points) + noise * noise + jitter


def _covariance_block(
    points: Array,
    kernel: AbstractPositiveDefiniteKernel,
    noise_scale: Array,
    jitter: Array,
    start: Array,
    size: int,
    /,
) -> Array:
    query = jax.lax.dynamic_slice_in_dim(points, start, size, axis=0)
    noise = jnp.broadcast_to(noise_scale, (points.shape[0],))
    columns = start + jnp.arange(size, dtype=jnp.int32)
    return (
        kernel.matrix(points, query)
        + (jnp.arange(points.shape[0])[:, None] == columns[None, :])
        * (noise * noise + jitter)[:, None]
    )


def _covariance_block_jvp(
    primals: _CovariancePrimals,
    tangents: _CovarianceTangents,
    start: Array,
    size: int,
    /,
) -> tuple[Array, Array]:
    def block(
        points: Array,
        kernel: AbstractPositiveDefiniteKernel,
        noise_scale: Array,
        jitter: Array,
    ) -> Array:
        return _covariance_block(points, kernel, noise_scale, jitter, start, size)

    value, tangent = eqx.filter_jvp(block, primals, tangents)
    return value, jnp.zeros_like(value) if tangent is None else tangent


def _assemble_covariance(
    points: Array,
    kernel: AbstractPositiveDefiniteKernel,
    noise_scale: Array,
    jitter: Array,
    width: int,
    /,
) -> Array:
    count = points.shape[0]

    def assemble(index: Array, matrix: Array) -> Array:
        start = index * width
        block = _covariance_block(points, kernel, noise_scale, jitter, start, width)
        return jax.lax.dynamic_update_slice(matrix, block, (0, start))

    matrix = jax.lax.fori_loop(
        0,
        count // width,
        jax.checkpoint(assemble),
        jnp.zeros((count, count), dtype=points.dtype),
    )
    remainder = count % width
    if remainder:
        block = _covariance_block(
            points,
            kernel,
            noise_scale,
            jitter,
            jnp.asarray(count - remainder, dtype=jnp.int32),
            remainder,
        )
        matrix = jax.lax.dynamic_update_slice(matrix, block, (0, count - remainder))
    return matrix


def _regularization_eta_tangent(
    primals: _CovariancePrimals,
    tangents: _CovarianceTangents,
    diagonal: Array,
    diagonal_dot: Array,
    regularization: ConditionTargetResult,
    width: int,
    /,
) -> Array:
    count = diagonal.size
    if count == 1:
        return jnp.zeros((), dtype=diagonal.dtype)
    scale = regularization.normalization_scale
    _, scale_dot = jax.jvp(jnp.sqrt, (diagonal,), (diagonal_dot,))

    def normalized(matrix: Array, left: Array, right: Array) -> Array:
        return matrix / left[:, None] / right[None, :]

    match regularization.policy.bound:
        case "trace":

            def normalized_trace(current: Array) -> Array:
                root = jnp.sqrt(current)
                return jnp.sum(current / root / root)

            _, upper_dot = jax.jvp(normalized_trace, (diagonal,), (diagonal_dot,))
        case "gershgorin":
            row_sums = regularization.normalized_row_sums
            if row_sums is None:
                raise ValueError("Gershgorin evidence requires normalized row sums.")

            def rows(start: Array, size: int) -> Array:
                columns = start + jnp.arange(size, dtype=jnp.int32)
                matrix, matrix_dot = _covariance_block_jvp(primals, tangents, start, size)
                block, block_dot = jax.jvp(
                    normalized,
                    (matrix, scale, scale[columns]),
                    (matrix_dot, scale_dot, scale_dot[columns]),
                )
                _, absolute_dot = jax.jvp(jnp.abs, (block,), (block_dot,))
                return jnp.sum(absolute_dot, axis=1)

            def accumulate(index: Array, total: Array) -> Array:
                return total + rows(index * width, width)

            sums_dot = jax.lax.fori_loop(
                0,
                count // width,
                jax.checkpoint(accumulate),
                jnp.zeros_like(diagonal),
            )
            remainder = count % width
            if remainder:
                row_dot = jax.checkpoint(rows, static_argnums=(1,))(
                    jnp.asarray(count - remainder, dtype=jnp.int32), remainder
                )
                sums_dot = sums_dot + row_dot
            _, upper_dot = jax.jvp(jnp.max, (row_sums,), (sums_dot,))
        case invalid:
            raise ValueError(f"Unknown condition-target bound {invalid!r}.")
    return upper_dot / (regularization.policy.max_condition_number - 1.0)


__all__ = ["ExactGaussianProcessExecutionPolicy"]
