#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from types import ModuleType
from typing import Literal, overload, Protocol, Self, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._hermitian_spectral import (
    fraction_matrix_profile,
    fraction_storage_upper,
    reserve_fraction_work,
)


type _SmallArray = Array | np.ndarray


type _NumpyCofactorValue = np.ndarray | np.number | int | Fraction
type _CofactorValue = Array | _NumpyCofactorValue


class _CofactorArithmetic(Protocol):
    """Homogeneous scalar/array algebra used by the direct cofactor formula."""

    def __mul__(self, other: Self, /) -> Self: ...

    def __sub__(self, other: Self, /) -> Self: ...

    def __add__(self, other: Self, /) -> Self: ...


def _cofactor_2_terms[T: _CofactorArithmetic](a: T, b: T, c: T, d: T, /) -> T:
    return a * d - b * c


def _cofactor_3_terms[T: _CofactorArithmetic](component: Callable[[int, int], T], /) -> T:
    # Repeat scalar indexing at each authored occurrence so AD accumulation
    # retains the authored gather/scatter graph as well as the primal term order.
    return (
        component(0, 0)
        * (component(1, 1) * component(2, 2) - component(1, 2) * component(2, 1))
        - component(0, 1)
        * (component(1, 0) * component(2, 2) - component(1, 2) * component(2, 0))
        + component(0, 2)
        * (component(1, 0) * component(2, 1) - component(1, 1) * component(2, 0))
    )


@overload
def cofactor_determinant(matrix: Array, /) -> Array: ...


@overload
def cofactor_determinant(matrix: np.ndarray, /) -> _NumpyCofactorValue: ...


def cofactor_determinant(matrix: _SmallArray, /) -> _CofactorValue:
    """Evaluate the unscaled square-matrix cofactor expansion in fixed term order.

    NumPy inputs return arrays with batch axes and scalars without them; JAX
    inputs return JAX arrays. Integer/Fraction arithmetic is unrounded absent
    overflow; floating arithmetic rounds as the written formula. Scalar indexing
    retains the authored differentiation accumulation graph.
    """
    match matrix.shape[-1]:
        case 1:
            return matrix[..., 0, 0]
        case 2:
            return _cofactor_2_terms(
                matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 0], matrix[..., 1, 1]
            )
        case 3:

            def component(row: int, column: int) -> _SmallArray:
                return matrix[..., row, column]

            return _cofactor_3_terms(component)
        case _:
            raise ValueError(
                "cofactor_determinant.matrix.shape must have trailing dimension 1, 2, or 3; got "
                + str(matrix.shape[-1])
            )


@overload
def cofactor_determinant_2(u: Array, v: Array, /) -> Array: ...


@overload
def cofactor_determinant_2(u: np.ndarray, v: np.ndarray, /) -> _NumpyCofactorValue: ...


def cofactor_determinant_2(
    u: Array | np.ndarray, v: Array | np.ndarray, /
) -> _CofactorValue:
    """Evaluate the unscaled 2-D cofactor expansion in fixed term order.

    NumPy inputs return arrays with batch axes and scalars without them; JAX
    inputs return JAX arrays. Integer/Fraction arithmetic is unrounded absent
    overflow; floating arithmetic rounds as the written formula.
    """
    return _cofactor_2_terms(u[..., 0], u[..., 1], v[..., 0], v[..., 1])


@overload
def cofactor_determinant_3(u: Array, v: Array, w: Array, /) -> Array: ...


@overload
def cofactor_determinant_3(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, /
) -> _NumpyCofactorValue: ...


def cofactor_determinant_3(
    u: Array | np.ndarray, v: Array | np.ndarray, w: Array | np.ndarray, /
) -> _CofactorValue:
    """Evaluate the unscaled 3-D cofactor expansion in fixed term order.

    NumPy inputs return arrays with batch axes and scalars without them; JAX
    inputs return JAX arrays. Integer/Fraction arithmetic is unrounded absent
    overflow; floating arithmetic rounds as the written formula.
    """
    rows = (u, v, w)

    def component(row: int, column: int) -> _CofactorValue:
        return rows[row][..., column]

    return _cofactor_3_terms(component)


def exact_cofactor_determinant_2(
    u: tuple[Fraction, Fraction], v: tuple[Fraction, Fraction], /
) -> Fraction:
    """Return the rational 2-D cofactor determinant without conversion or charges."""
    return u[0] * v[1] - u[1] * v[0]


def exact_cofactor_action_2(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
    /,
) -> tuple[Fraction, Fraction] | None:
    """Solve two exact rows once; ``None`` proves singularity.

    Like the exact cofactor determinants, this scalar primitive leaves work
    and storage admission to its caller. No reusable factor is needed for
    a single right-hand side.
    """
    determinant = exact_cofactor_determinant_2(first, second)
    if determinant == 0:
        return None
    return (
        (right[0] * second[1] - first[1] * right[1]) / determinant,
        (first[0] * right[1] - right[0] * second[0]) / determinant,
    )


def exact_cofactor_determinant_3(
    u: tuple[Fraction, Fraction, Fraction],
    v: tuple[Fraction, Fraction, Fraction],
    w: tuple[Fraction, Fraction, Fraction],
    /,
) -> Fraction:
    """Return the rational scalar cofactor determinant without conversion or charges."""
    rows = (u, v, w)

    def component(row: int, column: int) -> Fraction:
        return rows[row][column]

    return _cofactor_3_terms(component)


def _small_product(xp: ModuleType, left: _SmallArray, right: _SmallArray) -> _SmallArray:
    if xp is np:
        # The contracted axis is bounded by the tiny-system plan. Explicit
        # accumulation makes the host recipe independent of BLAS thread policy.
        product = left[..., :, 0, None] * right[..., None, 0, :]
        for column in range(1, left.shape[-1]):
            product = product + left[..., :, column, None] * right[..., None, column, :]
        return product
    return contract("...ij,...jk->...ik", left, right)


def _small_array(xp: ModuleType, value: ArrayLike, /) -> _SmallArray:
    array = xp.asarray(value)
    return array.astype(jax.typeof(array).dtype, copy=False) if xp is np else array


if TYPE_CHECKING:
    from ..discretization._coordinate_budget import CoordinateEnclosureBudget


class SmallLinearSolvePlan(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    singular_tolerance: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    refinement_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        singular_tolerance: float = 1e-12,
        maximum_condition: float = 1e12,
        refinement_iterations: int = 1,
    ) -> None:
        dimension_ = int(dimension)
        if dimension_ not in (1, 2, 3, 4):
            raise ValueError(
                f"SmallLinearSolvePlan.dimension must be 1 through 4; got {dimension_}"
            )
        if singular_tolerance <= 0.0 or maximum_condition <= 1.0:
            raise ValueError(
                "SmallLinearSolvePlan.tolerances must have singular_tolerance > 0 "
                "and maximum_condition > 1; got "
                f"singular_tolerance={singular_tolerance!r}, "
                f"maximum_condition={maximum_condition!r}"
            )
        if refinement_iterations < 0:
            raise ValueError(
                "SmallLinearSolvePlan.refinement_iterations must be non-negative; "
                f"got {refinement_iterations!r}"
            )
        self.dimension = dimension_
        self.singular_tolerance = float(singular_tolerance)
        self.maximum_condition = float(maximum_condition)
        self.refinement_iterations = int(refinement_iterations)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "small-linear-solve-plan",
                "dimension": dimension_,
                "singular_tolerance": singular_tolerance,
                "maximum_condition": maximum_condition,
                "refinement_iterations": refinement_iterations,
            }
        )

    @classmethod
    def from_stored_fields(
        cls,
        value: SmallLinearSolvePlan,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> SmallLinearSolvePlan:
        """Authenticate static inverse controls, never a retained factorization."""
        if not isinstance(value, cls):
            raise TypeError(
                "SmallLinearSolvePlan.value must be a small linear solve plan; "
                f"got {type(value).__name__}"
            )
        if budget is not None:
            budget.reserve(5)
        if (
            type(value.dimension) is not int
            or type(value.refinement_iterations) is not int
            or type(value.singular_tolerance) is not float
            or type(value.maximum_condition) is not float
            or type(value.plan_id) is not str
        ):
            raise ValueError(
                "SmallLinearSolvePlan.controls must retain their canonical scalar "
                "types; got changed stored controls"
            )
        rebuilt = cls(
            value.dimension,
            singular_tolerance=value.singular_tolerance,
            maximum_condition=value.maximum_condition,
            refinement_iterations=value.refinement_iterations,
        )
        if rebuilt.plan_id != value.plan_id:
            raise ValueError(
                "SmallLinearSolvePlan.plan_id must bind all retained inverse "
                "controls; got a stale stored identity"
            )
        return value


class SmallLinearSolveResult(StrictModule):
    value: Array
    determinant: Array
    rank: Array
    condition_estimate: Array
    residual_norm: Array
    refinement_iterations: Array
    successful: Array
    status: Array


def _lu_factor_four(
    matrix: _SmallArray, xp: ModuleType, /
) -> tuple[_SmallArray, _SmallArray, _SmallArray]:
    """Factor scaled 4-by-4 batches with deterministic partial pivoting."""
    factors = matrix
    batch_shape = matrix.shape[:-2]
    identity = xp.broadcast_to(xp.eye(4, dtype=matrix.dtype), matrix.shape)
    permutation = identity
    parity = xp.ones(batch_shape, dtype=matrix.real.dtype)
    row_indices = xp.arange(4, dtype=xp.int32)
    for column in range(3):
        pivot = column + xp.argmax(xp.abs(factors[..., column:, column]), axis=-1)
        column_row = identity[..., column, :]
        pivot_row = (
            (row_indices == pivot[..., None]).astype(matrix.dtype)
            if xp is np
            else jax.nn.one_hot(pivot, 4, dtype=matrix.dtype)
        )
        swap = (
            identity
            - column_row[..., :, None] * column_row[..., None, :]
            - pivot_row[..., :, None] * pivot_row[..., None, :]
            + column_row[..., :, None] * pivot_row[..., None, :]
            + pivot_row[..., :, None] * column_row[..., None, :]
        )
        factors = _small_product(xp, swap, factors)
        permutation = _small_product(xp, swap, permutation)
        parity = parity * xp.where(pivot == column, 1.0, -1.0)
        pivot_value = factors[..., column, column]
        safe_pivot = xp.where(xp.abs(pivot_value) > 0.0, pivot_value, 1.0)
        multipliers = factors[..., :, column] / safe_pivot[..., None]
        active_rows = xp.broadcast_to(row_indices > column, batch_shape + (4,))
        multipliers = xp.where(active_rows, multipliers, 0.0)
        update = multipliers[..., :, None] * factors[..., None, column, :]
        active_trailing = active_rows[..., :, None] & active_rows[..., None, :]
        factors = factors - xp.where(active_trailing, update, 0.0)
        column_values = xp.where(active_rows, multipliers, factors[..., :, column])
        if xp is np:
            factors[..., :, column] = column_values
        else:
            factors = factors.at[..., :, column].set(column_values)
    determinant = parity.astype(matrix.dtype) * xp.prod(
        xp.diagonal(factors, axis1=-2, axis2=-1), axis=-1
    )
    return factors, permutation, determinant


def _lu_solve_four(
    factors: _SmallArray, permutation: _SmallArray, right: _SmallArray, xp: ModuleType, /
) -> _SmallArray:
    """Solve a factored 4-by-4 batch for one or more right-hand sides."""
    permuted = _small_product(xp, permutation, right)
    lower_solution = xp.zeros_like(permuted)
    for row in range(4):
        contribution = xp.sum(factors[..., row, :, None] * lower_solution, axis=-2)
        value = permuted[..., row, :] - contribution
        if xp is np:
            lower_solution[..., row, :] = value
        else:
            lower_solution = lower_solution.at[..., row, :].set(value)
    solution = xp.zeros_like(permuted)
    for row in range(3, -1, -1):
        contribution = xp.sum(factors[..., row, :, None] * solution, axis=-2)
        pivot = factors[..., row, row]
        safe_pivot = xp.where(xp.abs(pivot) > 0.0, pivot, 1.0)
        value = (lower_solution[..., row, :] - contribution) / safe_pivot[..., None]
        if xp is np:
            solution[..., row, :] = value
        else:
            solution = solution.at[..., row, :].set(value)
    return solution


def _inverse(
    matrix: _SmallArray, dimension: int, xp: ModuleType, /
) -> tuple[_SmallArray, _SmallArray]:
    if dimension == 1:
        determinant = matrix[..., 0, 0]
        inverse = (1.0 / xp.where(determinant != 0.0, determinant, 1.0))[..., None, None]
        return inverse, determinant
    if dimension == 2:
        a, b = matrix[..., 0, 0], matrix[..., 0, 1]
        c, d = matrix[..., 1, 0], matrix[..., 1, 1]
        determinant = a * d - b * c
        adjugate = xp.stack((d, -b, -c, a), axis=-1).reshape(matrix.shape)
        return adjugate / xp.where(determinant != 0.0, determinant, 1.0)[
            ..., None, None
        ], determinant
    if dimension == 3:
        first, second, third = matrix[..., 0, :], matrix[..., 1, :], matrix[..., 2, :]
        cofactor_rows = xp.stack(
            (xp.cross(second, third), xp.cross(third, first), xp.cross(first, second)),
            axis=-2,
        )
        determinant = xp.sum(first * cofactor_rows[..., 0, :], axis=-1)
        inverse = (
            xp.swapaxes(cofactor_rows, -1, -2)
            / xp.where(determinant != 0.0, determinant, 1.0)[..., None, None]
        )
        return inverse, determinant
    factors, permutation, determinant = _lu_factor_four(matrix, xp)
    identity = xp.broadcast_to(xp.eye(4, dtype=matrix.dtype), matrix.shape)
    return _lu_solve_four(factors, permutation, identity, xp), determinant


def _determinant_kernel(
    plan: SmallLinearSolvePlan, matrix: ArrayLike, xp: ModuleType, /
) -> _SmallArray:
    value = _small_array(xp, matrix)
    if not jax.dtypes.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(jax.dtypes.canonicalize_dtype(np.float64))
    dimension = plan.dimension
    if value.shape[-2:] != (dimension, dimension):
        raise ValueError(
            f"determinant_small_linear.matrix.shape must end in {(dimension, dimension)}; "
            f"got {value.shape}"
        )
    scale = xp.max(xp.abs(value), axis=(-2, -1))
    safe_scale = xp.where(scale > 0.0, scale, 1.0).astype(value.dtype)
    scaled = value / safe_scale[..., None, None]
    if dimension == 1:
        determinant = scaled[..., 0, 0]
    elif dimension == 2:
        determinant = (
            scaled[..., 0, 0] * scaled[..., 1, 1] - scaled[..., 0, 1] * scaled[..., 1, 0]
        )
    elif dimension == 3:
        determinant = xp.sum(
            scaled[..., 0, :] * xp.cross(scaled[..., 1, :], scaled[..., 2, :]), axis=-1
        )
    else:
        _, _, determinant = _lu_factor_four(scaled, xp)
    return determinant * safe_scale**dimension


def _solve_kernel(
    plan: SmallLinearSolvePlan,
    matrix: ArrayLike,
    right_hand_side: ArrayLike,
    xp: ModuleType,
    /,
) -> tuple[_SmallArray, ...]:
    matrix_ = _small_array(xp, matrix)
    if not jax.dtypes.issubdtype(matrix_.dtype, jnp.inexact):
        matrix_ = matrix_.astype(jax.dtypes.canonicalize_dtype(np.float64))
    right = _small_array(xp, right_hand_side)
    dtype = jax.dtypes.canonicalize_dtype(np.result_type(matrix_.dtype, right.dtype))
    matrix_, right = matrix_.astype(dtype, copy=False), right.astype(dtype, copy=False)
    dimension = plan.dimension
    if matrix_.shape[-2:] != (dimension, dimension):
        raise ValueError(
            f"solve_small_linear.matrix.shape must end in {(dimension, dimension)}; "
            f"got {matrix_.shape}"
        )
    vector_rhs = right.shape == matrix_.shape[:-1]
    if vector_rhs:
        right = right[..., :, None]
    if right.shape[:-2] != matrix_.shape[:-2] or right.shape[-2] != dimension:
        raise ValueError(
            "solve_small_linear.right_hand_side.shape must have batch shape "
            f"{matrix_.shape[:-2]} and row dimension {dimension}; got {right.shape}"
        )
    scale = xp.max(xp.abs(matrix_), axis=(-2, -1))
    safe_scale = xp.where(scale > 0.0, scale, 1.0)
    field_scale = safe_scale.astype(matrix_.dtype)
    scaled_matrix = matrix_ / field_scale[..., None, None]
    if dimension == 4:
        factors, permutation, scaled_determinant = _lu_factor_four(scaled_matrix, xp)
        identity = xp.broadcast_to(xp.eye(4, dtype=matrix_.dtype), matrix_.shape)
        scaled_inverse = _lu_solve_four(factors, permutation, identity, xp)
        value = _lu_solve_four(
            factors, permutation, right / field_scale[..., None, None], xp
        )
    else:
        scaled_inverse, scaled_determinant = _inverse(scaled_matrix, dimension, xp)
        value = _small_product(xp, scaled_inverse / field_scale[..., None, None], right)
    inverse = scaled_inverse / field_scale[..., None, None]
    determinant = scaled_determinant * field_scale**dimension
    dtype_tolerance = float(dimension) * jax.dtypes.finfo(matrix_.real.dtype).eps
    singular_tolerance = xp.maximum(
        xp.asarray(plan.singular_tolerance, dtype=matrix_.real.dtype),
        xp.asarray(dtype_tolerance, dtype=matrix_.real.dtype),
    )
    nonsingular = xp.abs(scaled_determinant) > singular_tolerance
    refinement_count = xp.zeros(determinant.shape, dtype=xp.int32)
    for _ in range(plan.refinement_iterations):
        residual = right - _small_product(xp, matrix_, value)
        correction = (
            _lu_solve_four(
                factors, permutation, residual / field_scale[..., None, None], xp
            )
            if dimension == 4
            else _small_product(xp, inverse, residual)
        )
        candidate = value + correction
        candidate_residual = right - _small_product(xp, matrix_, candidate)
        residual_size = xp.sum(xp.abs(residual) ** 2, axis=(-2, -1))
        candidate_size = xp.sum(xp.abs(candidate_residual) ** 2, axis=(-2, -1))
        apply = (
            nonsingular
            & xp.all(xp.isfinite(correction), axis=(-2, -1))
            & (candidate_size < residual_size)
        )
        value = xp.where(apply[..., None, None], candidate, value)
        refinement_count = refinement_count + apply.astype(xp.int32)
    residual = right - _small_product(xp, matrix_, value)
    residual_norm = xp.sqrt(xp.sum(xp.abs(residual) ** 2, axis=(-2, -1)))
    matrix_norm = xp.max(xp.sum(xp.abs(matrix_), axis=-1), axis=-1)
    inverse_norm = xp.max(xp.sum(xp.abs(inverse), axis=-1), axis=-1)
    condition = matrix_norm * inverse_norm
    finite = (
        xp.all(xp.isfinite(matrix_), axis=(-2, -1))
        & xp.all(xp.isfinite(right), axis=(-2, -1))
        & xp.all(xp.isfinite(value), axis=(-2, -1))
    )
    successful = nonsingular & finite & (condition <= plan.maximum_condition)
    rank = xp.where(successful, dimension, 0).astype(xp.int32)
    status = xp.where(successful, 0, xp.where(nonsingular, 2, 1)).astype(xp.int32)
    value = xp.where(successful[..., None, None], value, 0.0)
    if vector_rhs:
        value = value[..., 0]
    return (
        value,
        determinant,
        rank,
        condition,
        residual_norm,
        refinement_count,
        successful,
        status,
    )


_determinant_device = eqx.filter_jit(_determinant_kernel)
_solve_device = eqx.filter_jit(_solve_kernel)


def _small_namespace(*values: ArrayLike) -> ModuleType:
    for value in values:
        match value:
            case Array() | Tracer():
                return jnp
            case np.ndarray() | np.generic():
                # Extension dtypes require JAX's rounding and promotion rules;
                # validate their capability from metadata before transferring.
                if value.dtype.kind == "V":
                    jax.typeof(value)
                    return jnp
            case _:
                continue
    return np


@checked
def determinant_small_linear(plan: SmallLinearSolvePlan, matrix: ArrayLike, /) -> Array:
    """Evaluate a scaled batched determinant using the input's numerical backend."""
    xp = _small_namespace(matrix)
    kernel = _determinant_kernel if xp is np else _determinant_device
    if xp is jnp:
        matrix = jax.device_put(matrix)
    return jax.device_put(kernel(plan, matrix, xp))


@checked
def solve_small_linear(
    plan: SmallLinearSolvePlan, matrix: ArrayLike, right_hand_side: ArrayLike, /
) -> SmallLinearSolveResult:
    """Solve tiny batches with one plan and paired host/device numerical kernels."""
    xp = _small_namespace(matrix, right_hand_side)
    kernel = _solve_kernel if xp is np else _solve_device
    if xp is jnp:
        matrix = jax.device_put(matrix)
        right_hand_side = jax.device_put(right_hand_side)
    values = kernel(plan, matrix, right_hand_side, xp)
    return SmallLinearSolveResult(*(jax.device_put(value) for value in values))


def inverse_small_linear(
    plan: SmallLinearSolvePlan,
    matrix: ArrayLike,
    /,
) -> SmallLinearSolveResult:
    """Materialize a batched 1x1 through 4x4 inverse with solve evidence."""
    match matrix:
        case np.ndarray() | np.generic():
            xp = np
        case _:
            xp = _small_namespace(matrix)
    value = xp.asarray(matrix)
    dimension = plan.dimension
    if value.shape[-2:] != (dimension, dimension):
        raise ValueError(
            f"inverse_small_linear.matrix.shape must end in {(dimension, dimension)}; "
            f"got {value.shape}"
        )
    identity = xp.broadcast_to(xp.eye(dimension, dtype=value.dtype), value.shape)
    return solve_small_linear(plan, value, identity)


@dataclass(frozen=True, slots=True)
class ExactSmallLinearActions:
    """Store coefficient actions solved in rational arithmetic on the host.

    The requested right-hand-side columns are solved from their binary64
    coefficients. A singular preparation retains rank/status and has no
    actions. Runtime floating evaluation remains separately rounded.
    """

    actions: tuple[tuple[Fraction, ...], ...] | None
    determinant: Fraction
    rank: int
    status: Literal["exact", "singular"]
    operation_count: int
    input_id: str

    @property
    def successful(self) -> bool:
        return self.status == "exact"


def _exact_small_shape(
    matrix: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[int, int]:
    dimension = len(matrix)
    if dimension not in (1, 2, 3, 4) or any(len(row) != dimension for row in matrix):
        raise ValueError(
            "prepare_exact_small_linear_actions.matrix.shape must be square with "
            f"dimension 1 through 4; got row lengths {tuple(map(len, matrix))}"
        )
    if len(right) != dimension or any(len(row) != len(right[0]) for row in right):
        raise ValueError(
            "prepare_exact_small_linear_actions.right.shape must have "
            f"{dimension} equally sized rows; got row lengths {tuple(map(len, right))}"
        )
    if any(not isinstance(value, Fraction) for row in (*matrix, *right) for value in row):
        raise TypeError(
            "prepare_exact_small_linear_actions.coefficients must be Fraction values; "
            f"got {next(type(value).__name__ for row in (*matrix, *right) for value in row if not isinstance(value, Fraction))}"
        )
    return dimension, len(right[0])


def _exact_action_work_upper(dimension: int, count: int, /) -> int:
    triangular = dimension * (dimension - 1) // 2
    return (
        triangular
        + dimension * (dimension - 1) * (2 * dimension - 1) // 3
        + count * (4 * triangular + dimension + 2 * dimension * dimension)
    )


def exact_small_linear_action_bounds(
    matrix: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[int, int]:
    """Bound rational elimination visits and live bytes from the input coefficients."""
    dimension, count = _exact_small_shape(matrix, right)
    terms = dimension * (dimension + count)
    numerator_bits, denominator_bits = fraction_matrix_profile((matrix, right), None)
    common = 1
    for bank in (matrix, right):
        for row in bank:
            for value in row:
                common = math.lcm(common, value.denominator)
    height = numerator_bits + common.bit_length() + 1
    storage = fraction_storage_upper(4, terms * denominator_bits + 1)
    storage += fraction_storage_upper(
        8 * dimension * (dimension + count) + 16,
        16 * dimension * height + 64,
    )
    return _exact_action_work_upper(dimension, count) + 2 * terms, storage


def _admit_exact_action_work(
    matrix: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    dimension: int,
    count: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    """Admit elimination visits and live rational storage before allocating banks."""
    # Elimination visits at most T lower entries, twice sum(k**2)
    # trailing entries, and 2*T*C RHS terms. Back substitution adds
    # 2*T*C+n*C; checking against the input system adds 2*n*n*C.
    work_upper = _exact_action_work_upper(dimension, count)
    terms = dimension * (dimension + count)
    budget.admit_work_bound(work_upper + 2 * terms)
    numerator_bits, denominator_bits = fraction_matrix_profile((matrix, right), budget)
    reserve_fraction_work(budget, 0, 4, terms * denominator_bits + 1)
    budget.reserve(terms)
    common = 1
    for bank in (matrix, right):
        for row in bank:
            for value in row:
                common = math.lcm(common, value.denominator)
    height = numerator_bits + common.bit_length() + 1
    # Clearing the common denominator bounds every integer input. Gaussian
    # entries are ratios of minors of order <= n; products and remainders fit
    # 16*n*height+64 bits. Eight copies cover the live row and action banks.
    reserve_fraction_work(
        budget,
        0,
        8 * dimension * (dimension + count) + 16,
        16 * dimension * height + 64,
    )


def _exact_action_elimination(
    factors: list[list[Fraction]],
    rhs: list[list[Fraction]],
    dimension: int,
    count: int,
    /,
) -> tuple[Fraction, int, int, int]:
    """Eliminate both banks in first-nonzero pivot order, counting rational operations."""
    determinant, parity, rank, work = Fraction(1), 1, 0, 0
    for column in range(dimension):
        pivot = next(
            (row for row in range(rank, dimension) if factors[row][column]), None
        )
        if pivot is None:
            continue
        if pivot != rank:
            factors[pivot], factors[rank] = factors[rank], factors[pivot]
            rhs[pivot], rhs[rank] = rhs[rank], rhs[pivot]
            parity = -parity
        diagonal = factors[rank][column]
        determinant *= diagonal
        for row in range(rank + 1, dimension):
            if not factors[row][column]:
                continue
            multiplier = factors[row][column] / diagonal
            work += 1
            factors[row][column] = Fraction(0)
            for trailing in range(column + 1, dimension):
                factors[row][trailing] -= multiplier * factors[rank][trailing]
                work += 2
            for action in range(count):
                rhs[row][action] -= multiplier * rhs[rank][action]
                work += 2
        rank += 1
    return determinant, parity, rank, work


def _exact_action_substitution(
    factors: list[list[Fraction]],
    rhs: list[list[Fraction]],
    dimension: int,
    count: int,
    work: int,
    /,
) -> tuple[tuple[tuple[Fraction, ...], ...], int]:
    """Solve the full-rank triangular banks in reverse row order."""
    values = [[Fraction(0) for _ in range(count)] for _ in range(dimension)]
    for row in range(dimension - 1, -1, -1):
        for action in range(count):
            remainder = rhs[row][action]
            for column in range(row + 1, dimension):
                remainder -= factors[row][column] * values[column][action]
                work += 2
            values[row][action] = remainder / factors[row][row]
            work += 1
    return tuple(tuple(row) for row in values), work


def _confirm_exact_actions(
    matrix: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    actions: tuple[tuple[Fraction, ...], ...],
    dimension: int,
    count: int,
    work: int,
    /,
) -> int:
    """Confirm each action against the input system before charging work."""
    for row in range(dimension):
        for action in range(count):
            if (
                sum(
                    (
                        matrix[row][column] * actions[column][action]
                        for column in range(dimension)
                    ),
                    Fraction(0),
                )
                != right[row][action]
            ):
                raise RuntimeError(
                    "prepare_exact_small_linear_actions.actions must satisfy "
                    f"matrix @ actions == right at row {row}, column {action}; "
                    f"got action column {tuple(actions[column][action] for column in range(dimension))!r} "
                    f"for right coefficient {right[row][action]!r}"
                )
            work += 2 * dimension
    return work


def exact_small_linear_actions_id(
    matrix: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    /,
) -> str:
    """Identify one rational coefficient-action request by its exact inputs."""
    return canonical_fingerprint(
        {
            "kind": "exact-small-linear-actions",
            "matrix": tuple(
                tuple((value.numerator, value.denominator) for value in row)
                for row in matrix
            ),
            "right": tuple(
                tuple((value.numerator, value.denominator) for value in row)
                for row in right
            ),
        }
    )


def prepare_exact_small_linear_actions(
    matrix: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactSmallLinearActions:
    """Prepare requested rational coefficient actions for one 1–4 dimensional system.

    This host preparation owns the elimination; geometry consumers request
    concrete projective coefficient columns. A dimension-by-zero right bank
    requests only determinant/rank evidence, without a discarded coefficient
    solve or a generic inverse.
    """
    dimension, count = _exact_small_shape(matrix, right)
    _admit_exact_action_work(matrix, right, dimension, count, budget)
    identity = exact_small_linear_actions_id(matrix, right)
    factors = [list(row) for row in matrix]
    rhs = [list(row) for row in right]
    determinant, parity, rank, work = _exact_action_elimination(
        factors, rhs, dimension, count
    )
    if rank != dimension:
        budget.reserve(work)
        return ExactSmallLinearActions(
            None, Fraction(0), rank, "singular", work, identity
        )
    actions, work = _exact_action_substitution(factors, rhs, dimension, count, work)
    work = _confirm_exact_actions(matrix, right, actions, dimension, count, work)
    budget.reserve(work)
    return ExactSmallLinearActions(
        actions, determinant * parity, rank, "exact", work, identity
    )


type _Rational3 = tuple[Fraction, Fraction, Fraction]


def _exact_dot_3(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], /
) -> Fraction:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _exact_cross_3(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], /
) -> _Rational3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


@dataclass(frozen=True, slots=True)
class ExactAdjugate3:
    """Exact adjugate of one rational 3x3 matrix given by its three columns.

    Row ``i`` is the cross product of the two other columns in cyclic order,
    so ``rows @ columns == determinant * I``; preparation proves that identity.
    ``height`` bounds the bits of the denominator-cleared integer columns.
    """

    rows: tuple[_Rational3, _Rational3, _Rational3]
    determinant: Fraction
    height: int


def exact_column_height_3(
    columns: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> int:
    """Bound the cleared integer bits of three rational columns, one visit each entry."""
    if len(columns) != 3 or any(len(column) != 3 for column in columns):
        raise ValueError(
            "exact_column_height_3.columns.shape must be three columns of three entries; "
            f"got column lengths {tuple(map(len, columns))}"
        )
    numerator_bits, _ = fraction_matrix_profile(
        (tuple(zip(*columns, strict=True)),), budget
    )
    common = 1
    for column in columns:
        for entry in column:
            common = math.lcm(common, entry.denominator)
    return numerator_bits + common.bit_length() + 1


def prepare_exact_adjugate_3(
    columns: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactAdjugate3 | None:
    """Prepare one proved exact 3x3 adjugate; ``None`` for a singular matrix.

    Each later solve ``M^{-1} r = adj(M) r / det(M)`` is then the unique exact
    solution without another elimination (``exact_adjugate_action``).
    """
    height = exact_column_height_3(columns, budget=budget)
    first, second, third = columns
    # Three cross products (27 visits) and the nine dot products of the
    # adjugate identity (45 visits). Cleared integer inputs have height
    # bits; cofactors, determinants and checks fit four heights.
    reserve_fraction_work(budget, 72, 64, 4 * height + 64)
    rows = (
        _exact_cross_3(second, third),
        _exact_cross_3(third, first),
        _exact_cross_3(first, second),
    )
    determinant = _exact_dot_3(rows[0], first)
    if determinant == 0:
        return None
    for slot, row in enumerate(rows):
        for other, column in enumerate(columns):
            if _exact_dot_3(row, column) != (determinant if slot == other else 0):
                raise RuntimeError(
                    "prepare_exact_adjugate_3.rows must satisfy adj(M) @ M == det(M) I; "
                    f"got row {slot}, column {other}"
                )
    return ExactAdjugate3(rows, determinant, height)


def exact_adjugate_action(
    adjugate: ExactAdjugate3,
    vector: tuple[Fraction, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Rational3:
    """Return the exact solution ``adj(M) vector / det(M)`` of a proved adjugate.

    ``vector`` may carry denominators unrelated to the prepared columns.
    """
    numerator_bits, _ = fraction_matrix_profile(((tuple(vector),),), budget)
    common = 1
    for entry in vector:
        common = math.lcm(common, entry.denominator)
    height = numerator_bits + common.bit_length() + 1
    # Three three-term dot products and three exact quotients. Cleared
    # adjugate rows fit 2h bits over D**2 and det(M) is det(C)/D**3 with 3h
    # bits, so every product, sum and quotient fits 3h + v + 4 bits for a
    # cleared vector height v; 8h + 64 already covers v <= 5h + 60.
    reserve_fraction_work(
        budget, 18, 16, max(8 * adjugate.height, 3 * adjugate.height + height) + 64
    )
    first, second, third = (
        _exact_dot_3(row, vector) / adjugate.determinant for row in adjugate.rows
    )
    return first, second, third


def confirm_exact_action_3(
    columns: tuple[tuple[Fraction, ...], ...],
    height: int,
    action: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Decide ``columns @ action == right`` exactly for one 3x3 system.

    For a nonsingular matrix this proves ``action`` is the unique solution
    without an elimination or adjugate. ``height`` is the column bound from
    ``exact_column_height_3``.
    """
    numerator_bits, denominator_bits = fraction_matrix_profile(((action,),), budget)
    # Three three-term dot products and three exact comparisons.
    reserve_fraction_work(
        budget, 18, 16, numerator_bits + denominator_bits + 2 * height + 64
    )
    return all(
        columns[0][axis] * action[0]
        + columns[1][axis] * action[1]
        + columns[2][axis] * action[2]
        == right[axis]
        for axis in range(3)
    )


__all__ = [
    "SmallLinearSolvePlan",
    "SmallLinearSolveResult",
    "determinant_small_linear",
    "inverse_small_linear",
    "solve_small_linear",
    "ExactSmallLinearActions",
    "prepare_exact_small_linear_actions",
    "ExactAdjugate3",
    "exact_column_height_3",
    "prepare_exact_adjugate_3",
    "exact_adjugate_action",
    "confirm_exact_action_3",
    "cofactor_determinant_2",
    "cofactor_determinant_3",
    "cofactor_determinant",
    "exact_cofactor_determinant_3",
]
