#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import core as jax_core
from jaxtyping import Array

from ..._strict import StrictModule
from .._materialization import materialize
from .._operators import DenseLinearOperator
from .._pairings import DiagonalPairing, EuclideanPairing
from .._plans import LinearSolvePlan
from .._policies import DenseCholesky, DenseLU, DenseQR, DenseSVD
from .._problems import LeastSquaresProblem, MinimumNormProblem
from .._properties import LinearCapabilityError
from .._results import LinearSolveStatus
from .._space_extensions import CoordaxSpace, TensorProductSpace
from .._spaces import ArraySpace, BlockSpace, DualSpace, PyTreeSpace


class DenseBackendOutput(StrictModule):
    value: Array
    status: Array
    iterations: Array
    refinement_steps: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None


class DenseLUState(StrictModule):
    matrix: Array
    factor: Array
    pivots: Array
    singular: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)


class DenseMixedPrecisionLUState(StrictModule):
    """High-precision operator storage with reusable low-precision LU factors."""

    matrix: Array
    factor: Array
    pivots: Array
    singular: Array
    condition_estimate: Array
    condition_limit: float = eqx.field(static=True)
    maximum_refinement_steps: int = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)


class DenseCholeskyState(StrictModule):
    matrix: Array
    factor: Array
    square_root_metric: Array
    inverse_square_root_metric: Array
    invalid: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)


class DenseQRState(StrictModule):
    original_matrix: Array
    design: Array
    q: Array
    r: Array
    square_root_weights: Array | None
    rank: Array
    condition_estimate: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    target_size: int = eqx.field(static=True)


class DenseSVDState(StrictModule):
    original_matrix: Array
    design: Array
    u: Array
    singular_values: Array
    vh: Array
    reported_singular_values: Array
    square_root_weights: Array | None
    source_inverse_square_root: Array | None
    source_projection: Array | None
    retained: Array
    rank: Array
    condition_estimate: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    target_size: int = eqx.field(static=True)


def prepare_dense(problem, plan: LinearSolvePlan, /) -> Any:
    """Materialize and factor one dense execution plan."""
    matrix = (
        problem.operator.matrix
        if isinstance(problem.operator, DenseLinearOperator)
        else materialize(problem.operator, plan.policy.materialization)
    )
    supported_dtypes = (
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float64),
        jnp.dtype(jnp.complex64),
        jnp.dtype(jnp.complex128),
    )
    if matrix.dtype not in supported_dtypes:
        raise TypeError(
            "Dense solve backends require float32, float64, complex64, or complex128 "
            f"coordinates; got {matrix.dtype}."
        )
    method = plan.policy.method
    if method.name == "auto":
        method_name = plan.method
    else:
        method_name = method.name
    if method_name == DenseLU().name:
        precision = plan.policy.precision
        factorization_dtype = (
            matrix.dtype
            if precision is None or precision.factorization_dtype is None
            else jnp.dtype(precision.factorization_dtype)
        )
        if factorization_dtype == matrix.dtype:
            return _prepare_lu(matrix, problem.operator.batch_shape)
        return _prepare_mixed_precision_lu(
            matrix,
            problem.operator.batch_shape,
            plan,
        )
    if method_name == DenseCholesky().name:
        return _prepare_cholesky(
            matrix,
            problem.operator.source,
            problem.operator.batch_shape,
        )
    if method_name in (DenseQR().name, DenseSVD().name):
        design, square_root_weights, source_inverse_square_root, rank_design = (
            _least_squares_design(problem, matrix, plan)
        )
        if method_name == DenseQR().name:
            return _prepare_qr(
                matrix,
                design,
                square_root_weights,
                source_inverse_square_root,
                problem.operator.batch_shape,
                problem.operator.target.size,
                plan,
            )
        return _prepare_svd(
            matrix,
            design,
            square_root_weights,
            source_inverse_square_root,
            problem.operator.batch_shape,
            problem.operator.target.size,
            rank_design,
            plan,
        )
    raise ValueError(f"Unsupported dense method {method_name!r}.")


def solve_dense(state: Any, rhs: Array, plan: LinearSolvePlan, /) -> DenseBackendOutput:
    if isinstance(state, DenseMixedPrecisionLUState):
        return _solve_mixed_precision_lu(state, rhs, plan)
    if isinstance(state, DenseLUState):
        return _solve_lu(state, rhs)
    if isinstance(state, DenseCholeskyState):
        return _solve_cholesky(state, rhs)
    if isinstance(state, DenseQRState):
        return _solve_qr(state, rhs, plan)
    if isinstance(state, DenseSVDState):
        return _solve_svd(state, rhs, plan)
    raise TypeError(f"Unsupported dense prepared state {type(state).__name__}.")


def solve_dense_transformed(
    state: Any,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
    *,
    adjoint: bool,
) -> DenseBackendOutput:
    """Reuse square direct factors for algebraic-transpose or Hilbert-adjoint solves."""
    if isinstance(state, DenseMixedPrecisionLUState):
        return _solve_mixed_precision_lu(
            state,
            rhs,
            plan,
            trans=2 if adjoint else 1,
        )
    if isinstance(state, DenseLUState):
        factor = _flat_batch(state.factor, state.batch_shape)
        pivots = _flat_batch(state.pivots, state.batch_shape)
        rhs_flat = _flat_batch(rhs, state.batch_shape)
        trans = 2 if adjoint else 1
        value = jax.vmap(
            lambda lu, pivot, b: jsp.linalg.lu_solve((lu, pivot), b, trans=trans)
        )(factor, pivots, rhs_flat).reshape(rhs.shape)
        matrix = (
            jnp.conj(jnp.swapaxes(state.matrix, -1, -2))
            if adjoint
            else jnp.swapaxes(state.matrix, -1, -2)
        )
        status = _direct_status(matrix, rhs, value, state.singular)
        return DenseBackendOutput(
            value=value,
            status=status,
            iterations=jnp.zeros_like(status, dtype=jnp.int32),
            refinement_steps=jnp.zeros_like(status, dtype=jnp.int32),
            rank=jnp.where(
                state.singular,
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(state.matrix.shape[-1], dtype=jnp.int32),
            ),
            condition_estimate=jnp.full(state.batch_shape, jnp.nan),
            singular_values=None,
        )
    if isinstance(state, DenseCholeskyState):
        if adjoint:
            return _solve_cholesky(state, rhs)
        factor = _flat_batch(state.factor, state.batch_shape)
        transformed_rhs = state.inverse_square_root_metric[..., :, None] * rhs
        rhs_flat = _flat_batch(transformed_rhs, state.batch_shape)

        def solve_one(cholesky, b):
            intermediate = jsp.linalg.solve_triangular(jnp.conj(cholesky), b, lower=True)
            return jsp.linalg.solve_triangular(cholesky.T, intermediate, lower=False)

        transformed_value = jax.vmap(solve_one)(factor, rhs_flat).reshape(rhs.shape)
        value = state.square_root_metric[..., :, None] * transformed_value
        status = _direct_status(
            jnp.swapaxes(state.matrix, -1, -2), rhs, value, state.invalid
        )
        return DenseBackendOutput(
            value=value,
            status=status,
            iterations=jnp.zeros_like(status, dtype=jnp.int32),
            refinement_steps=jnp.zeros_like(status, dtype=jnp.int32),
            rank=jnp.where(
                state.invalid,
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(state.matrix.shape[-1], dtype=jnp.int32),
            ),
            condition_estimate=jnp.full(state.batch_shape, jnp.nan),
            singular_values=None,
        )
    raise TypeError("Transformed direct solves require prepared LU or Cholesky state.")


def _prepare_lu(matrix: Array, batch_shape: tuple[int, ...], /) -> DenseLUState:
    size = int(matrix.shape[-1])
    count = prod(batch_shape) if batch_shape else 1
    flattened = matrix.reshape((count, size, size))
    factor, pivots = jax.vmap(jsp.linalg.lu_factor)(flattened)
    diagonal = jnp.diagonal(factor, axis1=-2, axis2=-1)
    scale = jnp.maximum(jnp.max(jnp.abs(flattened), axis=(-2, -1)), 1.0)
    threshold = jnp.finfo(matrix.real.dtype).eps * float(size) * scale
    singular = jnp.any(jnp.abs(diagonal) <= threshold[:, None], axis=-1)
    singular = singular | jnp.any(~jnp.isfinite(flattened), axis=(-2, -1))
    return DenseLUState(
        matrix=matrix,
        factor=factor.reshape(batch_shape + (size, size)),
        pivots=pivots.reshape(batch_shape + (size,)),
        singular=singular.reshape(batch_shape),
        batch_shape=batch_shape,
    )


def _prepare_mixed_precision_lu(
    matrix: Array,
    batch_shape: tuple[int, ...],
    plan: LinearSolvePlan,
    /,
) -> DenseMixedPrecisionLUState:
    precision = plan.policy.precision
    if precision is None or precision.factorization_dtype is None:
        raise ValueError("Mixed-precision LU requires a factorization dtype.")
    factorization_dtype = jnp.dtype(precision.factorization_dtype)
    condition = jnp.linalg.cond(matrix)
    factorization_real_dtype = (
        jnp.float32
        if factorization_dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.complex64))
        else jnp.float64
    )
    automatic_limit = 0.1 / float(jnp.finfo(factorization_real_dtype).eps)
    condition_limit = (
        automatic_limit
        if precision.condition_limit is None
        else min(automatic_limit, precision.condition_limit)
    )
    unsafe = jnp.any(~jnp.isfinite(condition) | (condition > condition_limit))
    rejection = (
        "Mixed-precision LU capability rejected before low-precision "
        f"factorization: condition estimate exceeds safe limit {condition_limit:.6g}."
    )
    if isinstance(unsafe, jax_core.Tracer):
        matrix = eqx.error_if(matrix, unsafe, rejection)
    elif bool(unsafe):
        raise LinearCapabilityError(rejection)

    low_matrix = matrix.astype(factorization_dtype)
    size = int(low_matrix.shape[-1])
    count = prod(batch_shape) if batch_shape else 1
    flattened = low_matrix.reshape((count, size, size))
    factor, pivots = jax.vmap(jsp.linalg.lu_factor)(flattened)
    diagonal = jnp.diagonal(factor, axis1=-2, axis2=-1)
    scale = jnp.maximum(jnp.max(jnp.abs(flattened), axis=(-2, -1)), 1.0)
    threshold = jnp.finfo(low_matrix.real.dtype).eps * float(size) * scale
    singular = jnp.any(jnp.abs(diagonal) <= threshold[:, None], axis=-1)
    singular = singular | jnp.any(~jnp.isfinite(flattened), axis=(-2, -1))
    return DenseMixedPrecisionLUState(
        matrix=matrix,
        factor=factor.reshape(batch_shape + (size, size)),
        pivots=pivots.reshape(batch_shape + (size,)),
        singular=singular.reshape(batch_shape),
        condition_estimate=condition,
        condition_limit=condition_limit,
        maximum_refinement_steps=precision.maximum_refinement_steps,
        batch_shape=batch_shape,
    )


def _prepare_cholesky(
    matrix: Array,
    space,
    batch_shape: tuple[int, ...],
    /,
) -> DenseCholeskyState:
    metric = _metric_diagonal(space)
    square_root_metric = jnp.sqrt(metric)
    inverse_square_root_metric = jax.lax.rsqrt(metric)
    transformed = (
        square_root_metric[..., :, None]
        * matrix
        * inverse_square_root_metric[..., None, :]
    )
    factor = jnp.linalg.cholesky(transformed)
    diagonal = jnp.real(jnp.diagonal(factor, axis1=-2, axis2=-1))
    invalid = jnp.any(~jnp.isfinite(factor), axis=(-2, -1)) | jnp.any(
        diagonal <= 0.0, axis=-1
    )
    return DenseCholeskyState(
        matrix=matrix,
        factor=factor,
        square_root_metric=square_root_metric,
        inverse_square_root_metric=inverse_square_root_metric,
        invalid=invalid,
        batch_shape=batch_shape,
    )


def _metric_diagonal(space, /) -> Array:
    if isinstance(space, (CoordaxSpace, TensorProductSpace)):
        return _metric_diagonal(space.delegate)
    if isinstance(space, (ArraySpace, PyTreeSpace)):
        pairing = space.pairing
        if isinstance(pairing, (EuclideanPairing, DiagonalPairing)):
            dtype = jnp.result_type(
                *[spec.dtype for spec in jax.tree.leaves(space.structure())]
            )
            coordinates = jnp.ones((space.size,), dtype=dtype)
            vector = space.unflatten(coordinates)
            return jnp.real(space.flatten(space.riesz(vector)))
        raise TypeError(
            "Linear solve backends currently require Euclidean or diagonal pairings."
        )
    if isinstance(space, BlockSpace):
        return jnp.concatenate(tuple(_metric_diagonal(block) for block in space.spaces))
    if isinstance(space, DualSpace):
        return 1.0 / _metric_diagonal(space.primal)
    raise TypeError("Linear solve backends do not support this vector-space pairing.")


def _least_squares_design(
    problem,
    matrix: Array,
    plan: LinearSolvePlan,
    /,
) -> tuple[Array, Array | None, Array | None, Array | None]:
    method = plan.policy.method
    if (
        isinstance(problem, MinimumNormProblem)
        and isinstance(method, DenseSVD)
        and method.damping > 0.0
    ):
        raise ValueError("DenseSVD damping is defined only for least-squares problems.")
    batch_shape = problem.operator.batch_shape
    if isinstance(problem, MinimumNormProblem):
        source_metric = _metric_diagonal(problem.operator.source)
        source_inverse_square_root = jax.lax.rsqrt(source_metric)
        design = matrix * source_inverse_square_root
        return design, None, source_inverse_square_root, None
    if not isinstance(problem, LeastSquaresProblem):
        raise TypeError("Dense rectangular methods require a least-squares problem.")
    target_size = problem.operator.target.size
    metric = _metric_diagonal(problem.operator.target)
    metric = jnp.broadcast_to(metric, batch_shape + (target_size,))
    if problem.weights is not None:
        if not isinstance(problem.operator.target, ArraySpace):
            raise TypeError("Explicit weights currently require ArraySpace targets.")
        weights = jnp.asarray(problem.weights, dtype=matrix.real.dtype)
        event_shape = problem.operator.target.shape
        if weights.shape == event_shape:
            weights = jnp.broadcast_to(weights, batch_shape + event_shape)
        expected = batch_shape + event_shape
        if weights.shape != expected:
            raise ValueError(f"weights must have shape {event_shape} or {expected}.")
        metric = metric * weights.reshape(batch_shape + (target_size,))
    square_root_weights = jnp.sqrt(metric)
    rank_design = square_root_weights[..., :, None] * matrix
    design = rank_design
    if problem.regularizer is not None:
        regularizer = materialize(problem.regularizer, plan.policy.materialization)
        regularizer_metric = _metric_diagonal(problem.regularizer.target)
        regularizer = jnp.sqrt(regularizer_metric)[..., :, None] * regularizer
        design = jnp.concatenate((design, regularizer), axis=-2)
    return (
        design,
        square_root_weights,
        None,
        (
            rank_design
            if problem.regularizer is not None
            and plan.policy.rank.relative_cutoff is not None
            else None
        ),
    )


def _rank_data(
    singular_values: Array,
    rows: int,
    columns: int,
    plan: LinearSolvePlan,
    /,
) -> tuple[Array, Array, Array]:
    maximum = jnp.max(singular_values, axis=-1)
    if plan.policy.rank.relative_cutoff is None:
        cutoff = jnp.finfo(singular_values.dtype).eps * float(max(rows, columns))
    else:
        cutoff = plan.policy.rank.relative_cutoff
    retained = singular_values > cutoff * maximum[..., None]
    rank = jnp.sum(retained, axis=-1, dtype=jnp.int32)
    minimum = jnp.min(jnp.where(retained, singular_values, jnp.inf), axis=-1)
    condition = jnp.where(rank > 0, maximum / minimum, jnp.inf)
    return retained, rank, condition


def _prepare_qr(
    original_matrix: Array,
    design: Array,
    square_root_weights: Array | None,
    source_inverse_square_root: Array | None,
    batch_shape: tuple[int, ...],
    target_size: int,
    plan: LinearSolvePlan,
    /,
) -> DenseQRState:
    if source_inverse_square_root is not None:
        raise ValueError("Dense QR does not implement minimum-norm semantics.")
    rows, columns = (int(size) for size in design.shape[-2:])
    if rows < columns:
        raise ValueError("Dense QR requires at least as many rows as columns.")
    q, r = jnp.linalg.qr(design, mode="reduced")
    singular_values = jnp.linalg.svd(r, compute_uv=False)
    retained, rank, condition = _rank_data(singular_values, rows, columns, plan)
    del retained
    return DenseQRState(
        original_matrix=original_matrix,
        design=design,
        q=q,
        r=r,
        square_root_weights=square_root_weights,
        rank=rank,
        condition_estimate=condition,
        batch_shape=batch_shape,
        target_size=target_size,
    )


def _prepare_svd(
    original_matrix: Array,
    design: Array,
    square_root_weights: Array | None,
    source_inverse_square_root: Array | None,
    batch_shape: tuple[int, ...],
    target_size: int,
    rank_design: Array | None,
    plan: LinearSolvePlan,
    /,
) -> DenseSVDState:
    factor_design = design
    source_projection = None
    if rank_design is None:
        u, singular_values, vh = jnp.linalg.svd(factor_design, full_matrices=False)
        rows, columns = (int(size) for size in factor_design.shape[-2:])
        retained, rank, condition = _rank_data(singular_values, rows, columns, plan)
        reported_singular_values = singular_values
    else:
        _, reported_singular_values, rank_vh = jnp.linalg.svd(
            rank_design, full_matrices=False
        )
        rank_rows, rank_columns = (int(size) for size in rank_design.shape[-2:])
        rank_retained, rank, condition = _rank_data(
            reported_singular_values,
            rank_rows,
            rank_columns,
            plan,
        )
        source_projection = (
            jnp.conj(jnp.swapaxes(rank_vh, -1, -2)) * rank_retained[..., None, :]
        )
        factor_design = jnp.matmul(design, source_projection)
        u, singular_values, vh = jnp.linalg.svd(factor_design, full_matrices=False)
        factor_rows, factor_columns = (int(size) for size in factor_design.shape[-2:])
        retained, _, _ = _rank_data(
            singular_values,
            factor_rows,
            factor_columns,
            plan,
        )
    return DenseSVDState(
        original_matrix=original_matrix,
        design=design,
        u=u,
        singular_values=singular_values,
        vh=vh,
        reported_singular_values=reported_singular_values,
        square_root_weights=square_root_weights,
        source_inverse_square_root=source_inverse_square_root,
        source_projection=source_projection,
        retained=retained,
        rank=rank,
        condition_estimate=condition,
        batch_shape=batch_shape,
        target_size=target_size,
    )


def _flat_batch(value: Array, batch_shape: tuple[int, ...], /) -> Array:
    count = prod(batch_shape) if batch_shape else 1
    return value.reshape((count,) + value.shape[len(batch_shape) :])


def _solve_lu(state: DenseLUState, rhs: Array, /) -> DenseBackendOutput:
    factor = _flat_batch(state.factor, state.batch_shape)
    pivots = _flat_batch(state.pivots, state.batch_shape)
    rhs_flat = _flat_batch(rhs, state.batch_shape)
    value = jax.vmap(lambda lu, pivot, b: jsp.linalg.lu_solve((lu, pivot), b))(
        factor, pivots, rhs_flat
    )
    value = value.reshape(rhs.shape)
    status = _direct_status(state.matrix, rhs, value, state.singular)
    return DenseBackendOutput(
        value=value,
        status=status,
        iterations=jnp.zeros_like(status, dtype=jnp.int32),
        refinement_steps=jnp.zeros_like(status, dtype=jnp.int32),
        rank=jnp.where(
            state.singular,
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(state.matrix.shape[-1], dtype=jnp.int32),
        ),
        condition_estimate=jnp.full(state.batch_shape, jnp.nan),
        singular_values=None,
    )


def _solve_mixed_precision_lu(
    state: DenseMixedPrecisionLUState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
    *,
    trans: int = 0,
) -> DenseBackendOutput:
    factor = _flat_batch(state.factor, state.batch_shape)
    pivots = _flat_batch(state.pivots, state.batch_shape)

    def solve_low(value):
        flattened = _flat_batch(value.astype(state.factor.dtype), state.batch_shape)
        solved = jax.vmap(
            lambda lu, pivot, b: jsp.linalg.lu_solve(
                (lu, pivot),
                b,
                trans=trans,
            )
        )(factor, pivots, flattened)
        return solved.reshape(value.shape).astype(rhs.dtype)

    matrix = (
        jnp.conj(jnp.swapaxes(state.matrix, -1, -2))
        if trans == 2
        else jnp.swapaxes(state.matrix, -1, -2)
        if trans == 1
        else state.matrix
    )
    value = solve_low(rhs)
    residual = rhs - jnp.matmul(matrix, value)
    residual_norm = jnp.linalg.norm(residual, axis=-2)
    rhs_norm = jnp.linalg.norm(rhs, axis=-2)
    relative = max(
        plan.policy.tolerance.relative,
        10.0 * float(jnp.finfo(rhs.real.dtype).eps) * float(matrix.shape[-1]),
    )
    threshold = plan.policy.tolerance.absolute + relative * rhs_norm
    active = (
        (residual_norm > threshold)
        & jnp.isfinite(residual_norm)
        & ~state.singular[..., None]
    )
    refinement_steps = jnp.zeros_like(residual_norm, dtype=jnp.int32)

    def refine(_, carry):
        current, current_residual, current_norm, refining, counts = carry
        correction_rhs = jnp.where(
            refining[..., None, :],
            current_residual,
            0,
        )
        candidate = current + solve_low(correction_rhs)
        candidate_residual = rhs - jnp.matmul(matrix, candidate)
        candidate_norm = jnp.linalg.norm(candidate_residual, axis=-2)
        accepted = (
            refining & jnp.isfinite(candidate_norm) & (candidate_norm < current_norm)
        )
        accepted_columns = accepted[..., None, :]
        next_value = jnp.where(accepted_columns, candidate, current)
        next_residual = jnp.where(
            accepted_columns,
            candidate_residual,
            current_residual,
        )
        next_norm = jnp.where(accepted, candidate_norm, current_norm)
        next_counts = counts + refining.astype(jnp.int32)
        next_refining = accepted & (next_norm > threshold)
        return (
            next_value,
            next_residual,
            next_norm,
            next_refining,
            next_counts,
        )

    value, _, _, _, refinement_steps = jax.lax.fori_loop(
        0,
        state.maximum_refinement_steps,
        refine,
        (value, residual, residual_norm, active, refinement_steps),
    )
    status = _direct_status(matrix, rhs, value, state.singular)
    return DenseBackendOutput(
        value=value,
        status=status,
        iterations=jnp.zeros_like(status, dtype=jnp.int32),
        refinement_steps=refinement_steps,
        rank=jnp.where(
            state.singular,
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(state.matrix.shape[-1], dtype=jnp.int32),
        ),
        condition_estimate=state.condition_estimate,
        singular_values=None,
    )


def _solve_cholesky(state: DenseCholeskyState, rhs: Array, /) -> DenseBackendOutput:
    factor = _flat_batch(state.factor, state.batch_shape)
    transformed_rhs = state.square_root_metric[..., :, None] * rhs
    rhs_flat = _flat_batch(transformed_rhs, state.batch_shape)

    def solve_one(cholesky, b):
        intermediate = jsp.linalg.solve_triangular(cholesky, b, lower=True)
        return jsp.linalg.solve_triangular(
            jnp.conj(cholesky.T), intermediate, lower=False
        )

    transformed_value = jax.vmap(solve_one)(factor, rhs_flat).reshape(rhs.shape)
    value = state.inverse_square_root_metric[..., :, None] * transformed_value
    status = _direct_status(state.matrix, rhs, value, state.invalid)
    diagonal = jnp.real(jnp.diagonal(state.factor, axis1=-2, axis2=-1))
    condition = (jnp.max(diagonal, axis=-1) / jnp.min(diagonal, axis=-1)) ** 2
    return DenseBackendOutput(
        value=value,
        status=status,
        iterations=jnp.zeros_like(status, dtype=jnp.int32),
        refinement_steps=jnp.zeros_like(status, dtype=jnp.int32),
        rank=jnp.where(
            state.invalid,
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(state.matrix.shape[-1], dtype=jnp.int32),
        ),
        condition_estimate=condition,
        singular_values=None,
    )


def _transformed_rhs(
    state: DenseQRState | DenseSVDState,
    rhs: Array,
    /,
) -> Array:
    target = rhs[..., : state.target_size, :]
    if state.square_root_weights is not None:
        target = state.square_root_weights[..., :, None] * target
    extra_rows = int(state.design.shape[-2]) - state.target_size
    if extra_rows:
        zeros = jnp.zeros(rhs.shape[:-2] + (extra_rows, rhs.shape[-1]), dtype=rhs.dtype)
        target = jnp.concatenate((target, zeros), axis=-2)
    return target


def _solve_qr(
    state: DenseQRState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
) -> DenseBackendOutput:
    transformed = _transformed_rhs(state, rhs)
    projected = jnp.matmul(jnp.conj(jnp.swapaxes(state.q, -1, -2)), transformed)
    r = _flat_batch(state.r, state.batch_shape)
    projected_flat = _flat_batch(projected, state.batch_shape)
    value = jax.vmap(
        lambda factor, b: jsp.linalg.solve_triangular(factor, b, lower=False)
    )(r, projected_flat)
    output_shape = state.batch_shape + (state.r.shape[-1], rhs.shape[-1])
    value = value.reshape(output_shape)
    rank_deficient = state.rank < state.r.shape[-1]
    status = _rectangular_status(
        state.design,
        transformed,
        value,
        rank_deficient,
        True,
    )
    return DenseBackendOutput(
        value=value,
        status=status,
        iterations=jnp.zeros_like(status, dtype=jnp.int32),
        refinement_steps=jnp.zeros_like(status, dtype=jnp.int32),
        rank=state.rank,
        condition_estimate=state.condition_estimate,
        singular_values=None,
    )


def _solve_svd(
    state: DenseSVDState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
) -> DenseBackendOutput:
    transformed = _transformed_rhs(state, rhs)
    projected = jnp.matmul(jnp.conj(jnp.swapaxes(state.u, -1, -2)), transformed)
    safe_singular_values = jnp.where(
        state.retained,
        state.singular_values,
        jnp.ones_like(state.singular_values),
    )
    method = plan.policy.method
    damping = method.damping if isinstance(method, DenseSVD) else 0.0
    if damping == 0.0:
        filter_factors = jnp.where(
            state.retained,
            1.0 / safe_singular_values,
            0.0,
        )
    else:
        squared_damping = jnp.asarray(
            damping**2,
            dtype=state.singular_values.dtype,
        )
        filter_factors = jnp.where(
            state.retained,
            state.singular_values / (state.singular_values**2 + squared_damping),
            0.0,
        )
    scaled = filter_factors[..., :, None] * projected
    value = jnp.matmul(jnp.conj(jnp.swapaxes(state.vh, -1, -2)), scaled)
    if state.source_projection is not None:
        value = jnp.matmul(state.source_projection, value)
    if state.source_inverse_square_root is not None:
        value = state.source_inverse_square_root[..., :, None] * value
    required_rank = min(state.original_matrix.shape[-2], state.original_matrix.shape[-1])
    rank_deficient = state.rank < required_rank
    status = _rectangular_status(
        state.design,
        transformed,
        value,
        rank_deficient,
        plan.policy.rank.require_full_rank,
    )
    return DenseBackendOutput(
        value=value,
        status=status,
        iterations=jnp.zeros_like(status, dtype=jnp.int32),
        refinement_steps=jnp.zeros_like(status, dtype=jnp.int32),
        rank=state.rank,
        condition_estimate=state.condition_estimate,
        singular_values=state.reported_singular_values,
    )


def _direct_status(matrix: Array, rhs: Array, value: Array, singular: Array, /) -> Array:
    rhs_finite = jnp.all(jnp.isfinite(rhs), axis=-2)
    matrix_finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    output_finite = jnp.all(jnp.isfinite(value), axis=-2)
    status = jnp.zeros(rhs_finite.shape, dtype=jnp.int32)
    status = jnp.where(
        ~matrix_finite[..., None] | ~rhs_finite,
        int(LinearSolveStatus.NONFINITE_INPUT),
        status,
    )
    status = jnp.where(
        (status == 0) & singular[..., None],
        int(LinearSolveStatus.SINGULAR),
        status,
    )
    return jnp.where(
        (status == 0) & ~output_finite,
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    )


def _rectangular_status(
    design: Array,
    rhs: Array,
    value: Array,
    rank_deficient: Array,
    require_full_rank: bool,
    /,
) -> Array:
    rhs_finite = jnp.all(jnp.isfinite(rhs), axis=-2)
    design_finite = jnp.all(jnp.isfinite(design), axis=(-2, -1))
    output_finite = jnp.all(jnp.isfinite(value), axis=-2)
    status = jnp.zeros(rhs_finite.shape, dtype=jnp.int32)
    status = jnp.where(
        ~design_finite[..., None] | ~rhs_finite,
        int(LinearSolveStatus.NONFINITE_INPUT),
        status,
    )
    if require_full_rank:
        status = jnp.where(
            (status == 0) & rank_deficient[..., None],
            int(LinearSolveStatus.RANK_DEFICIENT),
            status,
        )
    return jnp.where(
        (status == 0) & ~output_finite,
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    )


__all__ = [
    "DenseBackendOutput",
    "DenseMixedPrecisionLUState",
    "prepare_dense",
    "solve_dense",
    "solve_dense_transformed",
]
