# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import math
from typing import assert_never, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.linalg as scipy_linalg
from jax import Array, core as jax_core

from .._execution_device import device_platform
from ._dense_svd import economy_svd
from ._materialization import materialize
from ._policies import RankPolicy
from ._spaces import (
    _coordinate_pairing_matrix,
    _coordinate_pairing_weights,
    _has_diagonal_pairing,
    AbstractVectorSpace,
)
from ._svd_contracts import (
    _default_svd_device,
    DenseSVD,
    DenseSVDAlgorithm,
    DenseSVDState,
    HostSVDRankEvidence,
    SVDProblem,
    SVDSolvePlan,
    SVDSolveStatus,
    SVDTolerancePolicy,
)
from ._svd_evidence import exact_spectrum_rank_evidence, primal_svd_status


class HostSVDPreparation(NamedTuple):
    """Read-only float64 right frame and rank evidence for immutable host tables."""

    right_coordinates: np.ndarray
    primal_status: np.ndarray
    rank_evidence: HostSVDRankEvidence


def _host_fused_product(left: np.ndarray, right: np.ndarray, /) -> np.ndarray:
    """Deterministic binary64 lift: ascending inner index, one FMA per term.

    Each dot starts at +0.0 and rounds once per multiply-add. BLAS can use other
    reduction trees; CPU XLA agrees for the certified coupling baselines, but
    that agreement is not an assertion about every native matmul shape. This
    bounded setup-only operation is never a runtime numerical fallback.
    """
    product = np.empty((left.shape[0], right.shape[1]), dtype=np.float64)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            value = 0.0
            for inner in range(left.shape[1]):
                value = math.fma(
                    float(left[row, inner]), float(right[inner, column]), value
                )
            product[row, column] = value
    return product


def prepare_host_svd(matrix: np.ndarray, rank: RankPolicy, /) -> HostSVDPreparation:
    """Certify one concrete float64 table with the native dense SVD policy.

    This is host preparation, not a solver dispatch route: it accepts no JAX
    values, selects no devices, attaches no derivatives, and returns no runtime
    solve state. The driver and 2:1 QR reduction match native CPU DenseSVD.
    """
    if not isinstance(matrix, np.ndarray) or matrix.dtype != np.dtype(np.float64):
        raise TypeError("Host SVD preparation requires a NumPy float64 matrix.")
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("Host SVD preparation requires a nonempty matrix.")
    if not isinstance(rank, RankPolicy):
        raise TypeError("rank must be a RankPolicy.")
    rows, columns = matrix.shape
    count = min(rows, columns)
    status = np.asarray(SVDSolveStatus.SUCCESS, dtype=np.int32)
    if not np.all(np.isfinite(matrix)):
        status = np.asarray(SVDSolveStatus.PREPARATION_FAILED, dtype=np.int32)
    left = np.full((rows, count), np.nan, dtype=np.float64)
    values = np.full((count,), np.nan, dtype=np.float64)
    right = np.full((columns, count), np.nan, dtype=np.float64)
    if status == SVDSolveStatus.SUCCESS:
        try:
            reduced = max(rows, columns) >= 2 * count
            if reduced:
                tall = matrix if rows >= columns else matrix.T
                # Native CPU QR is GEQRF followed by ORGQR. High-level SciPy
                # economic QR can choose another Q formation algorithm.
                geqrf, orgqr = scipy_linalg.lapack.get_lapack_funcs(
                    ("geqrf", "orgqr"), (tall,)
                )
                _, _, work, info = geqrf(tall, lwork=-1)
                if info != 0:
                    raise scipy_linalg.LinAlgError("GEQRF workspace query failed.")
                raw, tau, _, info = geqrf(tall, lwork=int(work[0]))
                if info != 0:
                    raise scipy_linalg.LinAlgError("GEQRF failed.")
                triangular = np.triu(raw[:count, :])
                _, work, info = orgqr(raw, tau, lwork=-1)
                if info != 0:
                    raise scipy_linalg.LinAlgError("ORGQR workspace query failed.")
                basis, _, info = orgqr(raw, tau, lwork=int(work[0]), overwrite_a=True)
                if info != 0:
                    raise scipy_linalg.LinAlgError("ORGQR failed.")
                value = triangular if rows >= columns else triangular.T
            else:
                value = matrix
            left, values, right_adjoint = scipy_linalg.svd(
                value, full_matrices=False, check_finite=False, lapack_driver="gesdd"
            )
            if reduced:
                if rows >= columns:
                    left = _host_fused_product(basis, left)
                else:
                    right_adjoint = _host_fused_product(right_adjoint, basis.T)
            right = right_adjoint.T
        except scipy_linalg.LinAlgError:
            # Native nonconverged LAPACK drivers return nonfinite factors.
            pass
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):

        def norms(value: np.ndarray) -> np.ndarray:
            scale = np.max(np.abs(value), axis=0)
            safe = np.where(scale > 0, scale, 1)
            return scale * np.sqrt(np.sum((value / safe[None, :]) ** 2, axis=0))

        bound = np.max(values)
        forward, backward = matrix @ right, matrix.T @ left
        left_norm, right_norm = norms(left), norms(right)
        tiny = np.finfo(values.dtype).tiny
        relative = np.maximum(
            norms(forward - left * values[None, :])
            / np.maximum(bound * right_norm + values * left_norm, tiny),
            norms(backward - right * values[None, :])
            / np.maximum(bound * left_norm + values * right_norm, tiny),
        )
        orthogonality = np.maximum(
            np.max(np.abs(left.T @ left - np.eye(count))),
            np.max(np.abs(right.T @ right - np.eye(count))),
        )
        evidence = exact_spectrum_rank_evidence(
            values,
            rows,
            columns,
            rank,
            np.asarray(np.all(np.isfinite(values)) & (status == SVDSolveStatus.SUCCESS)),
            host=True,
        )
        primal = primal_svd_status(
            values,
            relative,
            np.asarray(orthogonality),
            status,
            evidence,
            count,
            SVDTolerancePolicy(),
            rank,
            host=True,
        )
    for array in (
        right,
        evidence.lower_bound,
        evidence.upper_bound,
        evidence.threshold_lower,
        evidence.threshold_upper,
        evidence.available,
        evidence.deterministic_exact,
        evidence.failure_probability,
    ):
        array.setflags(write=False)
    primal.setflags(write=False)
    return HostSVDPreparation(right, primal, evidence)


def validate_dense_algorithm(
    algorithm: DenseSVDAlgorithm, matrix: Array | None = None, /
) -> None:
    # Tracing cannot expose concrete placement. Plan with the current default
    # device, then recheck materialized preparation when placement is available.
    devices = (
        matrix.devices()
        if matrix is not None and not isinstance(matrix, jax_core.Tracer)
        else (_default_svd_device(),)
    )
    if algorithm == "divide-and-conquer" and any(
        device_platform(device) == "cuda" for device in devices
    ):
        raise ValueError(
            "DenseSVD algorithm='divide-and-conquer' is unsupported on platform "
            "'cuda'; use algorithm='qr'."
        )


def diagonal_scales(space: AbstractVectorSpace, /) -> tuple[Array, Array]:
    weights = _coordinate_pairing_weights(space)
    valid = (
        jnp.all(jnp.isfinite(weights))
        & jnp.all(weights.real > 0)
        & jnp.all(weights.imag == 0)
    )
    return jnp.sqrt(weights.real).astype(weights.dtype), valid


def pairing_factor(space: AbstractVectorSpace, /) -> tuple[Array, Array]:
    pairing = _coordinate_pairing_matrix(space)
    scale = jnp.maximum(jnp.max(jnp.abs(pairing)), 1)
    tolerance = 64 * max(space.size, 1) * jnp.finfo(pairing.real.dtype).eps * scale
    valid = jnp.all(jnp.isfinite(pairing)) & (
        jnp.max(jnp.abs(pairing - pairing.conj().T)) <= tolerance
    )
    factor = jax.lax.cond(
        valid,
        lambda matrix: jnp.linalg.cholesky(matrix, symmetrize_input=False),
        lambda matrix: jnp.full_like(matrix, jnp.nan),
        pairing,
    )
    valid = valid & jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.diag(factor).real > 0)
    return factor, valid


def restore_coordinates(factor: Array, columns: Array, diagonal: bool, /) -> Array:
    if diagonal:
        return columns / factor[:, None]
    return jsp.linalg.solve_triangular(factor.conj().T, columns, lower=False)


def prepare_dense(problem: SVDProblem, plan: SVDSolvePlan, /) -> DenseSVDState:
    operator = problem.operator
    diagonal = _has_diagonal_pairing(operator.source) and _has_diagonal_pairing(
        operator.target
    )
    if diagonal:
        source, source_valid = diagonal_scales(operator.source)
        target, target_valid = diagonal_scales(operator.target)
    else:
        source, source_valid = pairing_factor(operator.source)
        target, target_valid = pairing_factor(operator.target)
    matrix = materialize(operator, plan.policy.materialization)
    method = plan.policy.method
    if not isinstance(method, DenseSVD):
        raise TypeError("Dense SVD preparation requires DenseSVD.")
    validate_dense_algorithm(method.algorithm, matrix)
    valid = source_valid & target_valid & jnp.all(jnp.isfinite(matrix))

    def transform(value: Array) -> Array:
        if diagonal:
            return target[:, None] * value / source[None, :]
        scaled = jsp.linalg.solve_triangular(source, value.conj().T, lower=True).conj().T
        return target.conj().T @ scaled

    reduced = jax.lax.cond(
        valid, transform, lambda value: jnp.full_like(value, jnp.nan), matrix
    )
    valid = valid & jnp.all(jnp.isfinite(reduced))
    status = jnp.where(
        valid, int(SVDSolveStatus.SUCCESS), int(SVDSolveStatus.PREPARATION_FAILED)
    ).astype(jnp.int32)
    return DenseSVDState(reduced, source, target, status, diagonal)


def thin_decomposition(
    matrix: Array, available: Array, algorithm: DenseSVDAlgorithm, /
) -> tuple[Array, Array, Array]:
    validate_dense_algorithm(algorithm, matrix)
    rows, columns = matrix.shape
    rank = min(rows, columns)
    match algorithm:
        case "divide-and-conquer":
            driver = jax.lax.linalg.SvdAlgorithm.DIVIDE_AND_CONQUER
        case "qr":
            driver = jax.lax.linalg.SvdAlgorithm.QR
        case unknown:
            assert_never(unknown)

    def decompose(value: Array) -> tuple[Array, Array, Array]:
        left, values, right_adjoint = economy_svd(value, algorithm=driver)
        return left, values, right_adjoint.conj().T

    def unavailable(value: Array) -> tuple[Array, Array, Array]:
        return (
            jnp.full((rows, rank), jnp.nan, value.dtype),
            jnp.full((rank,), jnp.nan, value.real.dtype),
            jnp.full((columns, rank), jnp.nan, value.dtype),
        )

    return jax.lax.cond(available, decompose, unavailable, matrix)
