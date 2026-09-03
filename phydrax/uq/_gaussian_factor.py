#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule


GaussianFactorStatus = Literal[0, 1, 2, 3, 4]
GAUSSIAN_FACTOR_SUCCESS: GaussianFactorStatus = 0
GAUSSIAN_FACTOR_NONFINITE: GaussianFactorStatus = 1
GAUSSIAN_FACTOR_INVALID_REGULARIZATION: GaussianFactorStatus = 2
GAUSSIAN_FACTOR_NON_HERMITIAN: GaussianFactorStatus = 3
GAUSSIAN_FACTOR_NOT_POSITIVE_SEMIDEFINITE: GaussianFactorStatus = 4


class GaussianFactor(StrictModule):
    """A rectangular covariance root ``F`` representing ``F Fᴴ``.

    Leading dimensions are batch dimensions, the penultimate dimension is the
    Gaussian event, and the final dimension indexes factor directions.  A zero
    final dimension is the exact zero covariance.  Numerical diagnostics are
    arrays so construction remains valid under JAX transformations.
    """

    factor: Array
    regularization: Array
    rank_tolerance: Array
    numerical_rank: Array
    valid: Array
    status: Array
    factor_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        factor: ArrayLike,
        /,
        *,
        regularization: ArrayLike = 0.0,
        rank_tolerance: ArrayLike = 0.0,
        factor_id: str = "gaussian-factor",
        resolved_method: str = "provided-rectangular-factor",
    ):
        value = jnp.asarray(factor)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("Gaussian factors must have an inexact dtype.")
        if value.ndim < 2 or value.shape[-2] <= 0:
            raise ValueError(
                "Gaussian factors must have shape (..., event_size, rank) with "
                "a positive event_size."
            )
        if not isinstance(factor_id, str) or not factor_id:
            raise ValueError("factor_id must be a non-empty string.")
        if not isinstance(resolved_method, str) or not resolved_method:
            raise ValueError("resolved_method must be a non-empty string.")

        real_dtype = jnp.real(value).dtype
        regularization_value = jnp.asarray(regularization)
        tolerance_value = jnp.asarray(rank_tolerance)
        if regularization_value.ndim != 0 or not jnp.issubdtype(
            regularization_value.dtype, jnp.floating
        ):
            raise TypeError("regularization must be a real scalar.")
        if tolerance_value.ndim != 0 or not jnp.issubdtype(
            tolerance_value.dtype, jnp.floating
        ):
            raise TypeError("rank_tolerance must be a real scalar.")
        regularization_value = regularization_value.astype(real_dtype)
        tolerance_value = tolerance_value.astype(real_dtype)

        batch_shape = value.shape[:-2]
        if value.shape[-1] == 0:
            singular_values = jnp.empty((*batch_shape, 0), dtype=real_dtype)
        else:
            singular_values = jnp.linalg.svd(value, compute_uv=False)
        numerical_rank = jnp.sum(
            singular_values > tolerance_value, axis=-1, dtype=jnp.int32
        )
        finite = jnp.all(jnp.isfinite(value), axis=(-2, -1))
        regularization_valid = jnp.isfinite(regularization_value) & (
            regularization_value >= 0.0
        )
        tolerance_valid = jnp.isfinite(tolerance_value) & (tolerance_value >= 0.0)
        valid = finite & regularization_valid & tolerance_valid
        status = jnp.where(
            ~finite,
            GAUSSIAN_FACTOR_NONFINITE,
            jnp.where(
                ~(regularization_valid & tolerance_valid),
                GAUSSIAN_FACTOR_INVALID_REGULARIZATION,
                GAUSSIAN_FACTOR_SUCCESS,
            ),
        ).astype(jnp.int32)

        self.factor = value
        self.regularization = regularization_value
        self.rank_tolerance = tolerance_value
        self.numerical_rank = numerical_rank
        self.valid = valid
        self.status = status
        self.factor_id = factor_id
        self.resolved_method = resolved_method

    @property
    def covariance(self) -> Array:
        """Materialize the represented Hermitian covariance."""
        return self.factor @ _adjoint(self.factor)

    @property
    def rank(self) -> int:
        """Return the number of stored factor directions."""
        return self.factor.shape[-1]

    @property
    def event_size(self) -> int:
        """Return the flattened Gaussian event size."""
        return self.factor.shape[-2]


def gaussian_covariance(factor: GaussianFactor, /) -> Array:
    """Materialize ``F Fᴴ`` from a rectangular Gaussian factor."""
    if not isinstance(factor, GaussianFactor):
        raise TypeError("factor must be a GaussianFactor.")
    return factor.covariance


def gaussian_factor_from_covariance(
    covariance: ArrayLike,
    /,
    *,
    regularization: ArrayLike = 0.0,
    rank_tolerance: ArrayLike = 0.0,
    hermitian_tolerance: ArrayLike = 0.0,
    factor_id: str = "covariance-factor",
) -> GaussianFactor:
    """Construct an eigenvalue factor without implicit repair or jitter.

    ``regularization`` is the sole diagonal modification and is retained on the
    result.  ``rank_tolerance`` explicitly classifies eigenvalues in
    ``[-rank_tolerance, 0)`` as numerical null directions; more negative values
    remain observable as invalid nonfinite factor entries.
    """
    matrix = jnp.asarray(covariance)
    if not jnp.issubdtype(matrix.dtype, jnp.inexact):
        raise TypeError("covariance must have an inexact dtype.")
    if matrix.ndim < 2 or matrix.shape[-2] <= 0 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError("covariance must have shape (..., event_size, event_size).")

    real_dtype = jnp.real(matrix).dtype
    hermitian_tolerance_value = jnp.asarray(hermitian_tolerance)
    if hermitian_tolerance_value.ndim != 0 or not jnp.issubdtype(
        hermitian_tolerance_value.dtype, jnp.floating
    ):
        raise TypeError("hermitian_tolerance must be a real scalar.")
    hermitian_tolerance_value = hermitian_tolerance_value.astype(real_dtype)
    regularization_value = jnp.asarray(regularization)
    if regularization_value.ndim != 0 or not jnp.issubdtype(
        regularization_value.dtype, jnp.floating
    ):
        raise TypeError("regularization must be a real scalar.")
    regularization_value = regularization_value.astype(real_dtype)
    rank_tolerance_value = jnp.asarray(rank_tolerance)
    if rank_tolerance_value.ndim != 0 or not jnp.issubdtype(
        rank_tolerance_value.dtype, jnp.floating
    ):
        raise TypeError("rank_tolerance must be a real scalar.")
    rank_tolerance_value = rank_tolerance_value.astype(real_dtype)

    identity = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    regularized = matrix + regularization_value * identity
    eigenvalues, eigenvectors = jnp.linalg.eigh(regularized, symmetrize_input=False)
    tolerance_valid = jnp.isfinite(rank_tolerance_value) & (rank_tolerance_value >= 0.0)
    numerical_null = (eigenvalues < 0.0) & (eigenvalues >= -rank_tolerance_value)
    root_eigenvalues = jnp.where(numerical_null, 0.0, eigenvalues)
    root = eigenvectors * jnp.sqrt(root_eigenvalues)[..., None, :]
    result = GaussianFactor(
        root,
        regularization=regularization_value,
        rank_tolerance=rank_tolerance_value,
        factor_id=factor_id,
        resolved_method="hermitian-eigendecomposition",
    )

    finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    hermitian_defect = jnp.max(jnp.abs(matrix - _adjoint(matrix)), axis=(-2, -1))
    hermitian_valid = (
        jnp.isfinite(hermitian_tolerance_value)
        & (hermitian_tolerance_value >= 0.0)
        & (hermitian_defect <= hermitian_tolerance_value)
    )
    regularization_valid = jnp.isfinite(regularization_value) & (
        regularization_value >= 0.0
    )
    psd = jnp.all(eigenvalues >= -rank_tolerance_value, axis=-1)
    valid = (
        result.valid
        & finite
        & hermitian_valid
        & regularization_valid
        & tolerance_valid
        & psd
    )
    configuration_valid = regularization_valid & tolerance_valid
    status = jnp.where(
        ~finite,
        GAUSSIAN_FACTOR_NONFINITE,
        jnp.where(
            ~configuration_valid,
            GAUSSIAN_FACTOR_INVALID_REGULARIZATION,
            jnp.where(
                ~hermitian_valid,
                GAUSSIAN_FACTOR_NON_HERMITIAN,
                jnp.where(
                    ~psd,
                    GAUSSIAN_FACTOR_NOT_POSITIVE_SEMIDEFINITE,
                    result.status,
                ),
            ),
        ),
    ).astype(jnp.int32)
    result = eqx.tree_at(lambda node: node.valid, result, valid)
    return eqx.tree_at(lambda node: node.status, result, status)


def compress_gaussian_factor(
    factor: GaussianFactor,
    /,
    *,
    factor_id: str = "qr-compressed-factor",
) -> GaussianFactor:
    """QR-compress a wide factor while preserving its covariance exactly."""
    if not isinstance(factor, GaussianFactor):
        raise TypeError("factor must be a GaussianFactor.")
    if factor.rank <= factor.event_size:
        compressed = factor.factor
        method = "qr-compression-not-required"
    else:
        _, upper = jnp.linalg.qr(_adjoint(factor.factor), mode="reduced")
        compressed = _adjoint(upper)
        method = "qr-compression"
    return GaussianFactor(
        compressed,
        regularization=factor.regularization,
        rank_tolerance=factor.rank_tolerance,
        factor_id=factor_id,
        resolved_method=method,
    )


def add_independent_gaussian_factors(
    left: GaussianFactor,
    right: GaussianFactor,
    /,
    *,
    compress: bool = True,
    factor_id: str = "independent-factor-sum",
) -> GaussianFactor:
    """Add independent covariance roots by column concatenation and optional QR."""
    if not isinstance(left, GaussianFactor) or not isinstance(right, GaussianFactor):
        raise TypeError("left and right must be GaussianFactor instances.")
    if left.event_size != right.event_size:
        raise ValueError("Independent factors must have the same event_size.")
    batch_shape = jnp.broadcast_shapes(left.factor.shape[:-2], right.factor.shape[:-2])
    left_value = jnp.broadcast_to(left.factor, (*batch_shape, left.event_size, left.rank))
    right_value = jnp.broadcast_to(
        right.factor, (*batch_shape, right.event_size, right.rank)
    )
    left_regularization_valid = jnp.isfinite(left.regularization) & (
        left.regularization >= 0.0
    )
    right_regularization_valid = jnp.isfinite(right.regularization) & (
        right.regularization >= 0.0
    )
    regularization = jnp.where(
        ~left_regularization_valid,
        left.regularization,
        jnp.where(
            ~right_regularization_valid,
            right.regularization,
            left.regularization + right.regularization,
        ),
    )
    left_tolerance_valid = jnp.isfinite(left.rank_tolerance) & (
        left.rank_tolerance >= 0.0
    )
    right_tolerance_valid = jnp.isfinite(right.rank_tolerance) & (
        right.rank_tolerance >= 0.0
    )
    rank_tolerance = jnp.where(
        ~left_tolerance_valid,
        left.rank_tolerance,
        jnp.where(
            ~right_tolerance_valid,
            right.rank_tolerance,
            jnp.maximum(left.rank_tolerance, right.rank_tolerance),
        ),
    )
    combined = GaussianFactor(
        jnp.concatenate((left_value, right_value), axis=-1),
        regularization=regularization,
        rank_tolerance=rank_tolerance,
        factor_id=factor_id,
        resolved_method="independent-column-concatenation",
    )
    result = (
        compress_gaussian_factor(combined, factor_id=factor_id) if compress else combined
    )
    left_valid = jnp.broadcast_to(left.valid, batch_shape)
    right_valid = jnp.broadcast_to(right.valid, batch_shape)
    valid = result.valid & left_valid & right_valid
    status = jnp.where(
        ~left_valid,
        jnp.broadcast_to(left.status, batch_shape),
        jnp.where(
            ~right_valid,
            jnp.broadcast_to(right.status, batch_shape),
            result.status,
        ),
    ).astype(jnp.int32)
    result = eqx.tree_at(lambda node: node.valid, result, valid)
    return eqx.tree_at(lambda node: node.status, result, status)


def gaussian_cross_covariance(
    left: GaussianFactor,
    right: GaussianFactor,
    /,
) -> Array:
    """Return ``F Gᴴ`` for factors sharing the same latent directions."""
    if not isinstance(left, GaussianFactor) or not isinstance(right, GaussianFactor):
        raise TypeError("left and right must be GaussianFactor instances.")
    if left.rank != right.rank:
        raise ValueError("Cross-covariance factors must share their final rank axis.")
    return left.factor @ _adjoint(right.factor)


def solve_triangular_rank_aware(
    triangular: ArrayLike,
    right: ArrayLike,
    /,
    *,
    lower: bool = True,
    conjugate_transpose: bool = False,
    rank_tolerance: ArrayLike = 0.0,
) -> Array:
    """Solve a triangular system, using its SVD on explicitly detected rank loss."""
    matrix = jnp.asarray(triangular)
    rhs = jnp.asarray(right)
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError("triangular must have shape (..., size, size).")
    if rhs.ndim not in (matrix.ndim - 1, matrix.ndim):
        raise ValueError("right must be a compatible vector or matrix right-hand side.")
    if rhs.shape[-2 if rhs.ndim == matrix.ndim else -1] != matrix.shape[-1]:
        raise ValueError("right has an incompatible solve dimension.")
    tolerance = jnp.asarray(rank_tolerance)
    if tolerance.ndim != 0 or not jnp.issubdtype(tolerance.dtype, jnp.floating):
        raise TypeError("rank_tolerance must be a real scalar.")

    solve_matrix = _adjoint(matrix) if conjugate_transpose else matrix
    solve_lower = not lower if conjugate_transpose else lower
    diagonal = jnp.diagonal(solve_matrix, axis1=-2, axis2=-1)
    full_rank = (tolerance >= 0.0) & jnp.all(
        jnp.isfinite(diagonal) & (jnp.abs(diagonal) > tolerance), axis=-1
    )
    identity = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    safe_matrix = jnp.where(full_rank[..., None, None], solve_matrix, identity)
    direct = jsp.linalg.solve_triangular(
        safe_matrix, rhs, lower=solve_lower, unit_diagonal=False
    )
    rank_aware = _rank_aware_solve(solve_matrix, rhs, tolerance)
    trailing_axes = rhs.ndim - (matrix.ndim - 2)
    selector = full_rank[(...,) + (None,) * trailing_axes]
    return jnp.where(selector, direct, rank_aware)


def gaussian_factor_log_determinant(
    factor: GaussianFactor,
    /,
    *,
    rank_tolerance: ArrayLike | None = None,
) -> Array:
    """Return ``log(det(F Fᴴ))``, or ``-inf`` for a singular covariance."""
    if not isinstance(factor, GaussianFactor):
        raise TypeError("factor must be a GaussianFactor.")
    tolerance = (
        factor.rank_tolerance if rank_tolerance is None else jnp.asarray(rank_tolerance)
    )
    if factor.rank == 0:
        return jnp.full(
            factor.factor.shape[:-2], -jnp.inf, dtype=jnp.real(factor.factor).dtype
        )
    singular_values = jnp.linalg.svd(factor.factor, compute_uv=False)
    active = (tolerance >= 0.0) & (singular_values > tolerance)
    safe = jnp.where(active, singular_values, jnp.ones_like(singular_values))
    value = 2.0 * jnp.sum(jnp.log(safe), axis=-1)
    full_row_rank = (factor.rank >= factor.event_size) & (
        jnp.sum(active, axis=-1) == factor.event_size
    )
    return jnp.where(full_row_rank, value, -jnp.inf)


def gaussian_factor_quadratic_form(
    factor: GaussianFactor,
    residual: ArrayLike,
    /,
    *,
    rank_tolerance: ArrayLike | None = None,
    support_tolerance: ArrayLike = 0.0,
) -> Array:
    """Return the covariance-inverse quadratic form with singular support checks."""
    if not isinstance(factor, GaussianFactor):
        raise TypeError("factor must be a GaussianFactor.")
    vector = jnp.asarray(residual)
    if vector.ndim < 1 or vector.shape[-1] != factor.event_size:
        raise ValueError("residual must end in factor.event_size.")
    tolerance = (
        factor.rank_tolerance if rank_tolerance is None else jnp.asarray(rank_tolerance)
    )
    support = jnp.asarray(support_tolerance)
    if support.ndim != 0 or not jnp.issubdtype(support.dtype, jnp.floating):
        raise TypeError("support_tolerance must be a real scalar.")
    if factor.rank == 0:
        residual_norm = jnp.linalg.norm(vector, axis=-1)
        return jnp.where(residual_norm <= support, 0.0, jnp.inf)

    left_vectors, singular_values, _ = jnp.linalg.svd(factor.factor, full_matrices=False)
    active = (tolerance >= 0.0) & (singular_values > tolerance)
    coefficients = ein.contract("...ji,...j->...i", jnp.conj(left_vectors), vector)
    safe_values = jnp.where(active, singular_values, jnp.ones_like(singular_values))
    solved = jnp.where(active, coefficients / safe_values, 0.0)
    projected = ein.contract(
        "...ij,...j->...i", left_vectors, jnp.where(active, coefficients, 0.0)
    )
    support_error = jnp.linalg.norm(vector - projected, axis=-1)
    quadratic = jnp.real(jnp.sum(jnp.conj(solved) * solved, axis=-1))
    supported = (support >= 0.0) & (support_error <= support)
    full_row_rank = jnp.sum(active, axis=-1) == factor.event_size
    return jnp.where(supported | full_row_rank, quadratic, jnp.inf)


def _rank_aware_solve(matrix: Array, right: Array, tolerance: Array) -> Array:
    left_vectors, singular_values, adjoint_right_vectors = jnp.linalg.svd(
        matrix, full_matrices=False
    )
    active = (tolerance >= 0.0) & (singular_values > tolerance)
    safe_values = jnp.where(active, singular_values, jnp.ones_like(singular_values))
    if right.ndim == matrix.ndim - 1:
        projected = ein.contract("...ji,...j->...i", jnp.conj(left_vectors), right)
        scaled = jnp.where(active, projected / safe_values, 0.0)
        return ein.contract("...ij,...j->...i", _adjoint(adjoint_right_vectors), scaled)
    projected = _adjoint(left_vectors) @ right
    scaled = jnp.where(active[..., :, None], projected / safe_values[..., :, None], 0.0)
    return _adjoint(adjoint_right_vectors) @ scaled


def _adjoint(matrix: Array) -> Array:
    return jnp.conj(jnp.swapaxes(matrix, -1, -2))


__all__ = [
    "GAUSSIAN_FACTOR_INVALID_REGULARIZATION",
    "GAUSSIAN_FACTOR_NONFINITE",
    "GAUSSIAN_FACTOR_NON_HERMITIAN",
    "GAUSSIAN_FACTOR_NOT_POSITIVE_SEMIDEFINITE",
    "GAUSSIAN_FACTOR_SUCCESS",
    "GaussianFactor",
    "GaussianFactorStatus",
    "add_independent_gaussian_factors",
    "compress_gaussian_factor",
    "gaussian_covariance",
    "gaussian_cross_covariance",
    "gaussian_factor_from_covariance",
    "gaussian_factor_log_determinant",
    "gaussian_factor_quadratic_form",
    "solve_triangular_rank_aware",
]
