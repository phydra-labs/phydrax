#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from fractions import Fraction
from functools import partial
from types import ModuleType
from typing import (
    assert_never,
    final,
    Generic,
    Literal,
    NamedTuple,
    TYPE_CHECKING,
    TypeVar,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..typing import checked, parse
from ._hermitian_precision import _effective_dtype, HermitianPrecisionPolicy


if TYPE_CHECKING:
    from ..discretization._coordinate_budget import CoordinateEnclosureBudget


P = TypeVar("P")
Q = TypeVar("Q")
type _FunctionKind = Literal["log", "exp", "sqrt", "inverse_sqrt"]


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _spectral_cast(
    xp: ModuleType, value: ArrayLike, requested: str | None, /, *, decision: bool = False
) -> Array | np.ndarray:
    array = xp.asarray(value)
    if xp is np:
        array = array.astype(jax.typeof(array).dtype, copy=False)
    if requested is None:
        return array
    if decision:
        return array.astype(jax.dtypes.canonicalize_dtype(requested), copy=False)
    if not jax.dtypes.issubdtype(array.dtype, jnp.inexact):
        return array
    return array.astype(
        jax.dtypes.canonicalize_dtype(_effective_dtype(requested, array.dtype.name)),
        copy=False,
    )


@eqx.filter_jit
def _spectral_boolean_device(value: Array, /) -> Array:
    return value.astype(np.bool_)


def _spectral_leaf(value: ArrayLike, /, *, boolean: bool = False) -> Array:
    match value:
        case Array() | Tracer():
            if boolean and value.dtype != np.bool_:
                return _spectral_boolean_device(value)
            return value
        case _:
            host = np.asarray(value, dtype=np.bool_ if boolean else None)
            return jax.device_put(_spectral_cast(np, host, None))


def _spectrum_kernel(
    original: ArrayLike,
    precision: HermitianPrecisionPolicy,
    tolerance: ArrayLike,
    xp: ModuleType,
    /,
) -> tuple[Array | np.ndarray, ...]:
    value = _spectral_cast(xp, original, precision.compute_dtype)
    adjoint = xp.swapaxes(xp.conj(value), -1, -2)
    hermitian = _spectral_cast(
        xp, 0.5 * value + 0.5 * adjoint, precision.factorization_dtype
    )
    eigenvalues, eigenvectors = xp.linalg.eigh(hermitian)
    differences = xp.abs(eigenvalues[..., 1:] - eigenvalues[..., :-1])
    minimum_gap = (
        xp.min(differences, axis=-1)
        if value.shape[-1] > 1
        else xp.full(value.shape[:-2], xp.inf, dtype=eigenvalues.dtype)
    )
    magnitude = _spectral_cast(
        xp, xp.max(xp.abs(eigenvalues), axis=-1), precision.decision_dtype, decision=True
    )
    threshold_tolerance = _spectral_cast(
        xp, tolerance, precision.decision_dtype, decision=True
    )
    if xp is np and precision.decision_dtype is None:
        # A Python tolerance is weakly typed in JAX; NumPy scalar tolerances
        # retain their explicitly requested component precision.
        match tolerance:
            case np.floating():
                pass
            case _:
                threshold_tolerance = threshold_tolerance.astype(magnitude.dtype)
    threshold = threshold_tolerance * xp.maximum(magnitude, 1.0)
    rank = xp.sum(xp.abs(eigenvalues) > threshold[..., None], axis=-1)
    minimum_absolute = _spectral_cast(
        xp, xp.min(xp.abs(eigenvalues), axis=-1), precision.decision_dtype, decision=True
    )
    condition = magnitude / xp.maximum(
        minimum_absolute, jax.dtypes.finfo(eigenvalues.dtype).tiny
    )
    residual = _spectral_cast(
        xp,
        xp.max(
            xp.abs(_spectral_cast(xp, value - adjoint, precision.accumulation_dtype)),
            axis=(-2, -1),
        ),
        precision.decision_dtype,
        decision=True,
    )
    minimum = _spectral_cast(
        xp, xp.min(eigenvalues, axis=-1), precision.decision_dtype, decision=True
    )
    gap = _spectral_cast(xp, minimum_gap, precision.decision_dtype, decision=True)
    condition = _spectral_cast(xp, condition, precision.decision_dtype, decision=True)
    valid = (
        xp.all(xp.isfinite(value), axis=(-2, -1))
        & (residual <= tolerance)
        & xp.all(xp.isfinite(eigenvalues), axis=-1)
    )
    return (
        hermitian,
        eigenvalues,
        eigenvectors,
        residual,
        minimum,
        gap,
        rank,
        condition,
        valid,
    )


_spectrum_device = eqx.filter_jit(_spectrum_kernel)


def _prepare_spectrum(
    matrix: ArrayLike,
    precision: HermitianPrecisionPolicy,
    tolerance: float | np.floating,
    /,
) -> tuple[Array | np.ndarray, ...]:
    match matrix:
        case Array() | Tracer():
            original = matrix
            kernel = _spectrum_device
            xp = jnp
        case _:
            original = np.asarray(matrix)
            jax.typeof(original)
            kernel = _spectrum_kernel
            xp = np
    if original.ndim < 2 or original.shape[-2] != original.shape[-1]:
        raise ValueError(
            f"HermitianSpectrum.matrix must have square trailing axes; got shape {original.shape}"
        )
    if tolerance < 0.0:
        raise ValueError(
            f"HermitianSpectrum.tolerance must be non-negative; got {tolerance!r}"
        )
    return kernel(original, precision, tolerance, xp)


class HermitianSpectrum(StrictModule):
    """Hermitian eigendecomposition and rank/conditioning evidence."""

    matrix: Array
    eigenvalues: Array
    eigenvectors: Array
    hermiticity_residual: Array
    minimum_eigenvalue: Array
    minimum_gap: Array
    numerical_rank: Array
    condition_number: Array
    valid: Array
    precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        tolerance: float | np.floating = 1e-10,
        precision: HermitianPrecisionPolicy | None = None,
    ) -> None:
        precision_ = HermitianPrecisionPolicy() if precision is None else precision
        prepared = _prepare_spectrum(matrix, precision_, tolerance)
        (
            self.matrix,
            self.eigenvalues,
            self.eigenvectors,
            self.hermiticity_residual,
            self.minimum_eigenvalue,
            self.minimum_gap,
            self.numerical_rank,
            self.condition_number,
            self.valid,
        ) = (jax.device_put(value) for value in prepared)
        self.precision = precision_
        self.precision_evidence = precision_.evidence_for(matrix)
        self.tolerance = float(tolerance)

    @eqx.filter_jit
    def reconstruct(self) -> Array:
        return (self.eigenvectors * self.eigenvalues[..., None, :]) @ _adjoint(
            self.eigenvectors
        )


@final
class HermitianFunctionResult(StrictModule):
    """Matrix value with nondifferentiable spectrum/evidence diagnostics."""

    value: Array
    spectrum: HermitianSpectrum
    valid: Array
    derivative_valid: Array
    function_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        spectrum: HermitianSpectrum,
        /,
        *,
        function_id: str,
        valid: ArrayLike,
        derivative_valid: ArrayLike,
    ) -> None:
        self.value = _spectral_leaf(value)
        self.spectrum = spectrum
        self.valid = _spectral_leaf(valid, boolean=True)
        self.derivative_valid = _spectral_leaf(derivative_valid, boolean=True)
        self.function_id = str(function_id)


def _function_values(values: Array, kind: _FunctionKind) -> Array:
    match kind:
        case "log":
            return jnp.log(values)
        case "exp":
            return jnp.exp(values)
        case "sqrt":
            return jnp.sqrt(jnp.maximum(values, 0.0))
        case "inverse_sqrt":
            return 1.0 / jnp.sqrt(values)
        case _:
            assert_never(kind)


def _divided_difference(values: Array, kind: _FunctionKind) -> Array:
    a, b = values[..., :, None], values[..., None, :]
    gap = jnp.abs(a - b)
    same = gap == 0.0
    denominator = jnp.where(same, 1.0, gap)
    match kind:
        case "log":
            return jnp.where(
                same, 1.0 / a, jnp.log1p(gap / jnp.minimum(a, b)) / denominator
            )
        case "exp":
            return jnp.exp(jnp.maximum(a, b)) * jnp.where(
                same, 1.0, -jnp.expm1(-gap) / denominator
            )
        case "sqrt":
            return 1.0 / (jnp.sqrt(a) + jnp.sqrt(b))
        case "inverse_sqrt":
            return -1.0 / (jnp.sqrt(a) * jnp.sqrt(b) * (jnp.sqrt(a) + jnp.sqrt(b)))
        case _:
            assert_never(kind)


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _function_value(
    matrix: Array, values: Array, vectors: Array, kind: _FunctionKind
) -> Array:
    # Reuse the stopped diagnostic factorization; matrix supplies its JVP.
    del matrix
    transformed = _function_values(values, kind).astype(vectors.dtype)
    result = (vectors * transformed[..., None, :]) @ _adjoint(vectors)
    return 0.5 * (result + _adjoint(result))


@_function_value.defjvp
def _function_value_jvp(
    kind: _FunctionKind,
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array, Array, Array],
) -> tuple[Array, Array]:
    _, values, vectors = primals
    direction, _, _ = tangents
    transformed = _function_values(values, kind).astype(vectors.dtype)
    value = (vectors * transformed[..., None, :]) @ _adjoint(vectors)
    local = _adjoint(vectors) @ (0.5 * (direction + _adjoint(direction))) @ vectors
    tangent = (
        vectors
        @ (_divided_difference(values, kind).astype(local.dtype) * local)
        @ _adjoint(vectors)
    )
    return 0.5 * (value + _adjoint(value)), 0.5 * (tangent + _adjoint(tangent))


def _spectral_result(
    matrix: ArrayLike,
    kind: _FunctionKind,
    /,
    *,
    function_id: str,
    tolerance: float,
    positive: bool,
    nonnegative: bool = False,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    original = jnp.asarray(matrix)
    spectrum = HermitianSpectrum(
        jax.lax.stop_gradient(original), tolerance=tolerance, precision=precision
    )
    kind = parse(kind, _FunctionKind, "kind")
    policy = spectrum.precision
    matrix_ = policy.factorization(policy.compute(original))
    value = policy.output(
        _function_value(matrix_, spectrum.eigenvalues, spectrum.eigenvectors, kind)
    )
    valid = spectrum.valid & jnp.all(jnp.isfinite(value), axis=(-2, -1))
    if positive:
        valid = valid & (spectrum.minimum_eigenvalue > tolerance)
    if nonnegative:
        valid = valid & (spectrum.minimum_eigenvalue >= -tolerance)
    derivative_valid = valid
    if positive or nonnegative:
        derivative_valid = valid & (spectrum.minimum_eigenvalue > tolerance)
    valid = jax.lax.stop_gradient(valid)
    derivative_valid = jax.lax.stop_gradient(derivative_valid)
    return HermitianFunctionResult(
        value,
        spectrum,
        function_id=function_id,
        valid=valid,
        derivative_valid=derivative_valid,
    )


def hermitian_sqrt(
    matrix: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    return _spectral_result(
        matrix,
        "sqrt",
        function_id="hermitian-sqrt",
        tolerance=tolerance,
        positive=False,
        nonnegative=True,
        precision=precision,
    )


def hermitian_inverse_sqrt(
    matrix: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    return _spectral_result(
        matrix,
        "inverse_sqrt",
        function_id="hermitian-inverse-sqrt",
        tolerance=tolerance,
        positive=True,
        precision=precision,
    )


def hermitian_log(
    matrix: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    return _spectral_result(
        matrix,
        "log",
        function_id="hermitian-log",
        tolerance=tolerance,
        positive=True,
        precision=precision,
    )


def hermitian_exp(
    matrix: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    return _spectral_result(
        matrix,
        "exp",
        function_id="hermitian-exp",
        tolerance=tolerance,
        positive=False,
        precision=precision,
    )


@final
class HermitianSpectralMapResult(StrictModule, Generic[Q]):
    """Isotropic matrix outputs and differentiable invariant PyTree.

    Spectrum, valid, and derivative_valid are nondifferentiable diagnostics.
    """

    values: Array
    invariants: Q
    spectrum: HermitianSpectrum
    valid: Array
    derivative_valid: Array
    function_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        invariants: Q,
        spectrum: HermitianSpectrum,
        /,
        *,
        valid: ArrayLike,
        derivative_valid: ArrayLike,
        function_id: str,
    ) -> None:
        self.values = jnp.asarray(values)
        self.invariants = invariants
        self.spectrum = spectrum
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.derivative_valid = jnp.asarray(derivative_valid, dtype=jnp.bool_)
        self.function_id = str(function_id)


def _principal_outputs(
    eigenvalue_map: Callable[[Array, P], tuple[Array, Q]],
    eigenvalues: Array,
    parameters: P,
) -> tuple[Array, Q]:
    principal, invariants = eigenvalue_map(eigenvalues, parameters)
    if (
        principal.ndim != 2
        or principal.shape[-1] != eigenvalues.shape[0]
        or principal.shape[0] < 1
    ):
        raise ValueError(
            f"hermitian_spectral_map.eigenvalue_map.principal must have shape (k, {eigenvalues.shape[0]}) with k >= 1; got shape {principal.shape}"
        )
    return principal, invariants


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _map_value(
    eigenvalue_map: Callable[[Array, P], tuple[Array, Q]],
    matrix: Array,
    parameters: P,
    eigenvalues: Array,
    vectors: Array,
) -> tuple[Array, Q]:
    del matrix
    principal, invariants = _principal_outputs(eigenvalue_map, eigenvalues, parameters)
    return (vectors[None] * principal.astype(vectors.dtype)[:, None, :]) @ _adjoint(
        vectors
    ), invariants


@_map_value.defjvp
def _map_value_jvp(
    eigenvalue_map: Callable[[Array, P], tuple[Array, Q]],
    primals: tuple[Array, P, Array, Array],
    tangents: tuple[Array, P, Array, Array],
) -> tuple[tuple[Array, Q], tuple[Array, Q]]:
    _, parameters, eigenvalues, vectors = primals
    direction, parameter_direction, _, _ = tangents
    local = _adjoint(vectors) @ (0.5 * (direction + _adjoint(direction))) @ vectors
    eigenvalue_direction = jnp.real(jnp.diag(local))
    (principal, invariants), (principal_direction, invariant_direction) = jax.jvp(
        eigenvalue_map,
        (eigenvalues, parameters),
        (eigenvalue_direction, parameter_direction),
    )
    jacobian = jax.jacfwd(lambda values: eigenvalue_map(values, parameters)[0])(
        eigenvalues
    )
    gap = eigenvalues[:, None] - eigenvalues[None, :]
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(eigenvalues[:, None]), jnp.abs(eigenvalues[None, :])), 1.0
    )
    confluent = jnp.abs(gap) <= jnp.sqrt(jnp.finfo(eigenvalues.dtype).eps) * scale
    limit = jnp.diagonal(jacobian, axis1=-2, axis2=-1)[:, :, None] - jacobian
    divided = (principal[:, :, None] - principal[:, None, :]) / jnp.where(
        confluent, 1.0, gap
    )[None]
    gamma = jnp.where(confluent[None], limit, divided).astype(local.dtype)
    identity = jnp.eye(eigenvalues.shape[0], dtype=local.dtype)
    tangent_local = (
        identity[None] * principal_direction.astype(local.dtype)[:, :, None]
        + gamma * (local * (1.0 - identity))[None]
    )
    values = (vectors[None] * principal.astype(vectors.dtype)[:, None, :]) @ _adjoint(
        vectors
    )
    tangent_values = vectors[None] @ tangent_local @ _adjoint(vectors)
    return (values, invariants), (tangent_values, invariant_direction)


def hermitian_spectral_map(
    matrix: ArrayLike,
    eigenvalue_map: Callable[[Array, P], tuple[Array, Q]],
    parameters: P,
    /,
    *,
    function_id: str,
    tolerance: float = 1e-10,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianSpectralMapResult[Q]:
    """Evaluate a permutation-equivariant principal map with first-order AD.

    The static, hashable callback receives (n,) eigenvalues and dynamic shared
    parameters; it returns (k,n) principal outputs and permutation-invariant
    array PyTrees. Integer/bool invariant leaves have float0 tangents. Leading
    matrix axes are mapped with shared parameters; vmap externally for varying
    parameters. No arrays may be captured by the callback. Forward-over-reverse
    second derivatives at repeated spectra are unsupported.
    """
    hash(eigenvalue_map)
    original = jnp.asarray(matrix)
    spectrum = HermitianSpectrum(
        jax.lax.stop_gradient(original), tolerance=tolerance, precision=precision
    )
    policy = spectrum.precision
    matrix_ = policy.factorization(policy.compute(original))
    n = matrix_.shape[-1]
    batch = matrix_.shape[:-2]
    if batch:
        values, invariants = jax.vmap(_map_value, in_axes=(None, 0, None, 0, 0))(
            eigenvalue_map,
            matrix_.reshape((-1, n, n)),
            parameters,
            spectrum.eigenvalues.reshape((-1, n)),
            spectrum.eigenvectors.reshape((-1, n, n)),
        )
        values = values.reshape((*batch, *values.shape[1:]))
        invariants = jax.tree.map(
            lambda leaf: leaf.reshape((*batch, *leaf.shape[1:])), invariants
        )
    else:
        values, invariants = _map_value(
            eigenvalue_map,
            matrix_,
            parameters,
            spectrum.eigenvalues,
            spectrum.eigenvectors,
        )
    values = policy.output(values)
    valid = spectrum.valid & jnp.all(jnp.isfinite(values), axis=(-3, -2, -1))
    for leaf in jax.tree.leaves(invariants):
        valid = valid & jnp.all(
            jnp.isfinite(leaf), axis=tuple(range(len(batch), leaf.ndim))
        )
    valid = jax.lax.stop_gradient(valid)
    return HermitianSpectralMapResult(
        values,
        invariants,
        spectrum,
        valid=valid,
        derivative_valid=valid,
        function_id=function_id,
    )


class SylvesterSolveResult(StrictModule):
    value: Array
    residual_norm: Array
    minimum_denominator: Array
    valid: Array
    precision_evidence: PrecisionEvidenceEnvelope

    @checked
    def __init__(
        self,
        value: ArrayLike,
        residual_norm: ArrayLike,
        minimum_denominator: ArrayLike,
        valid: ArrayLike,
        precision_evidence: PrecisionEvidenceEnvelope,
        /,
    ) -> None:
        self.value = _spectral_leaf(value)
        self.residual_norm = _spectral_leaf(residual_norm)
        self.minimum_denominator = _spectral_leaf(minimum_denominator)
        self.valid = _spectral_leaf(valid, boolean=True)
        self.precision_evidence = precision_evidence


class HermitianSylvesterOperator(StrictModule):
    """Matrix-free ``X -> rho X + X rho`` and spectral inverse action."""

    matrix: Array
    spectrum: HermitianSpectrum
    tolerance: float = eqx.field(static=True)
    precision: HermitianPrecisionPolicy

    @checked
    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
        precision: HermitianPrecisionPolicy | None = None,
    ) -> None:
        precision_ = HermitianPrecisionPolicy() if precision is None else precision
        spectrum = HermitianSpectrum(
            matrix,
            tolerance=tolerance,
            precision=precision_,
        )
        self.matrix = spectrum.reconstruct()
        self.spectrum = spectrum
        self.precision = precision_
        self.tolerance = float(tolerance)

    def mv(self, value: ArrayLike, /) -> Array:
        operand = jnp.asarray(value)
        if operand.shape != self.matrix.shape:
            raise ValueError(
                f"HermitianSylvesterOperator.mv.value must have shape {self.matrix.shape}; got shape {operand.shape}"
            )
        return self.matrix @ operand + operand @ self.matrix

    def solve(self, right_hand_side: ArrayLike, /) -> SylvesterSolveResult:
        right = self.precision.factorization(right_hand_side)
        if right.shape != self.matrix.shape:
            raise ValueError(
                f"HermitianSylvesterOperator.solve.right_hand_side must have shape {self.matrix.shape}; got shape {right.shape}"
            )
        vectors = self.spectrum.eigenvectors
        local = _adjoint(vectors) @ right @ vectors
        denominators = (
            self.spectrum.eigenvalues[..., :, None]
            + self.spectrum.eigenvalues[..., None, :]
        )
        minimum = jnp.min(jnp.abs(denominators), axis=(-2, -1))
        safe = jnp.where(jnp.abs(denominators) > self.tolerance, denominators, jnp.inf)
        solution = self.precision.output(vectors @ (local / safe) @ _adjoint(vectors))
        residual = self.precision.accumulation(self.mv(solution) - right)
        residual_norm = self.precision.decision(jnp.linalg.norm(residual, axis=(-2, -1)))
        valid = (
            self.spectrum.valid & (minimum > self.tolerance) & jnp.isfinite(residual_norm)
        )
        return SylvesterSolveResult(
            solution,
            residual_norm,
            self.precision.decision(minimum),
            valid,
            self.precision.evidence_for(right),
        )


class TracelessHermitianSpace(StrictModule):
    dimension: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /) -> None:
        dimension_ = int(dimension)
        if dimension_ < 2:
            raise ValueError(
                f"TracelessHermitianSpace.dimension must be at least two; got {dimension_!r}"
            )
        self.dimension = dimension_
        self.space_id = f"traceless-hermitian:{dimension_}"

    @property
    def shape(self) -> tuple[int, int]:
        return self.dimension, self.dimension

    def project(self, value: ArrayLike, /) -> Array:
        matrix = jnp.asarray(value)
        if matrix.shape[-2:] != self.shape:
            raise ValueError(
                f"TracelessHermitianSpace.project.value must have trailing shape {self.shape}; got shape {matrix.shape}"
            )
        hermitian = 0.5 * (matrix + _adjoint(matrix))
        trace = jnp.trace(hermitian, axis1=-2, axis2=-1) / float(self.dimension)
        identity = jnp.eye(self.dimension, dtype=hermitian.dtype)
        return hermitian - trace[..., None, None] * identity

    def inner(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return jnp.real(jnp.vdot(self.project(left), self.project(right)))


def fraction_storage_upper(count: int, bits: int, /) -> int:
    """Bound CPython Fraction objects and their temporary reference slots."""
    digits = (
        max(bits, 1) + sys.int_info.bits_per_digit - 1
    ) // sys.int_info.bits_per_digit
    fraction_upper = 256 + 2 * (sys.getsizeof(0) + digits * sys.int_info.sizeof_digit)
    return 256 + count * (fraction_upper + 16)


def reserve_fraction_work(
    budget: CoordinateEnclosureBudget | None,
    work: int,
    count: int,
    bits: int,
    /,
) -> None:
    """Admit term visits and a CPython-object storage upper bound."""
    if budget is None:
        return
    budget.reserve(work, fraction_storage_upper(count, bits))


def fraction_matrix_profile(
    matrices: tuple[tuple[tuple[Fraction, ...], ...], ...],
    budget: CoordinateEnclosureBudget | None,
    /,
) -> tuple[int, int]:
    if budget is not None:
        budget.reserve(sum(len(row) for matrix in matrices for row in matrix))
    return (
        max(
            (
                abs(value.numerator).bit_length()
                for matrix in matrices
                for row in matrix
                for value in row
            ),
            default=1,
        ),
        max(
            (
                value.denominator.bit_length()
                for matrix in matrices
                for row in matrix
                for value in row
            ),
            default=1,
        ),
    )


def fraction_sqrt_interval(
    value: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[Fraction, Fraction]:
    """Return rational square-root endpoints proved by squaring."""
    reserve_fraction_work(
        budget,
        2,
        8,
        max(abs(value.numerator).bit_length(), value.denominator.bit_length(), 1075),
    )
    numerator, denominator = math.isqrt(value.numerator), math.isqrt(value.denominator)
    if (
        numerator * numerator == value.numerator
        and denominator * denominator == value.denominator
    ):
        root = Fraction(numerator, denominator)
        return root, root
    try:
        nominal = float(value)
    except OverflowError:
        nominal = math.inf
    if not math.isfinite(nominal) or nominal < np.finfo(np.float64).tiny:
        # Scale the rational before binary64 conversion: a squared norm can
        # underflow even when its square root is an ordinary finite number.
        reserve_fraction_work(
            budget,
            6,
            12,
            2 * max(value.numerator.bit_length(), value.denominator.bit_length(), 1075),
        )
        exponent = (value.numerator.bit_length() - value.denominator.bit_length()) // 2
        scale = Fraction(2) ** exponent
        lower, upper = fraction_sqrt_interval(value / (scale * scale), budget=budget)
        return lower * scale, upper * scale
    central = math.sqrt(nominal)
    if not math.isfinite(central) or central == 0:
        raise ValueError(
            f"fraction_sqrt_interval.value must have a finite nonzero binary64 square root; got root {central!r} for {value!r}"
        )
    lower = upper = central
    while Fraction(lower) ** 2 > value:
        if budget is not None:
            budget.reserve(1)
        lower = float(np.nextafter(lower, -math.inf))
    while Fraction(upper) ** 2 < value:
        if budget is not None:
            budget.reserve(1)
        upper = float(np.nextafter(upper, math.inf))
    return Fraction(lower), Fraction(upper)


class HermitianFunctionEnclosure(NamedTuple):
    """Store a rational nominal matrix and an operator-norm error radius."""

    matrix: tuple[tuple[Fraction, ...], ...]
    error: Fraction


def _exact_matrix_product(
    first: tuple[tuple[Fraction, ...], ...],
    second: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[tuple[Fraction, ...], ...]:
    dimension = len(first)
    if budget is not None:
        numerator_bits, denominator_bits = fraction_matrix_profile(
            (first, second), budget
        )
        result_bits = (
            2 * dimension * (numerator_bits + denominator_bits)
            + dimension.bit_length()
            + 4
        )
        reserve_fraction_work(budget, 2 * dimension**3, dimension**2 + 3, result_bits)
    return tuple(
        tuple(
            sum((first[i][k] * second[k][j] for k in range(dimension)), Fraction(0))
            for j in range(dimension)
        )
        for i in range(dimension)
    )


def matrix_norm_enclosure(
    matrix: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> Fraction:
    if budget is not None:
        numerator_bits, denominator_bits = fraction_matrix_profile((matrix,), budget)
        count = sum(len(row) for row in matrix)
        reserve_fraction_work(
            budget,
            2 * count,
            4,
            2 * count * (numerator_bits + denominator_bits) + count.bit_length() + 4,
        )
    return fraction_sqrt_interval(
        sum((value * value for row in matrix for value in row), Fraction(0)),
        budget=budget,
    )[1]


def _scalar_log_enclosure(
    value: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[Fraction, Fraction]:
    """Enclose the logarithm with an atanh series after dyadic scaling."""
    if value <= 0:
        raise ValueError(f"_scalar_log_enclosure.value must be positive; got {value!r}")
    reserve_fraction_work(
        budget,
        8,
        14,
        140 * (abs(value.numerator).bit_length() + value.denominator.bit_length()) + 512,
    )
    power = value.numerator.bit_length() - value.denominator.bit_length()
    factor = Fraction(2) ** power
    while value < factor:
        if budget is not None:
            budget.reserve(1)
        power -= 1
        factor /= 2
    while value >= 2 * factor:
        if budget is not None:
            budget.reserve(1)
        power += 1
        factor *= 2
    ratio = value / factor
    z = (ratio - 1) / (ratio + 1)
    intervals = []
    for argument in (z, Fraction(1, 3)):
        squared = argument * argument
        term, total = argument, Fraction(0)
        for index in range(32):
            if budget is not None:
                budget.reserve(4)
            total += 2 * term / (2 * index + 1)
            term *= squared
        tail = 2 * term / (65 * (1 - squared))
        intervals.append((total, total + tail))
    low, high = intervals[0]
    log_two = intervals[1]
    return (
        low + power * log_two[0 if power >= 0 else 1],
        high + power * log_two[1 if power >= 0 else 0],
    )


class _HermitianSpectralEnclosure(NamedTuple):
    values: tuple[Fraction, ...]
    vectors: tuple[tuple[Fraction, ...], ...]
    vector_norm: Fraction
    polar_error: Fraction
    residual: Fraction


def _source_spectral_enclosure(
    original: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> _HermitianSpectralEnclosure:
    """Enclose residual and polar errors around the numerical eigenframe."""
    dimension = len(original)
    reserve_fraction_work(
        budget,
        dimension**2 + dimension,
        2 * dimension**2 + dimension + 3,
        1075,
    )
    floating = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(floating)):
        raise ValueError(
            f"_source_spectral_enclosure.original must convert to finite binary64 entries; got {floating!r}"
        )
    spectrum = HermitianSpectrum(jnp.asarray(floating, dtype=jnp.float64))
    if not bool(np.asarray(spectrum.valid)):
        raise ValueError(
            f"_source_spectral_enclosure.spectrum.valid must be true; got {spectrum.valid!r}"
        )
    values = tuple(Fraction(float(value)) for value in np.asarray(spectrum.eigenvalues))
    q = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(spectrum.eigenvectors)
    )
    transpose = tuple(zip(*q, strict=True))
    identity = tuple(
        tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension)
    )
    gram = _exact_matrix_product(transpose, q, budget=budget)
    orthogonality = matrix_norm_enclosure(
        tuple(
            tuple(gram[i][j] - identity[i][j] for j in range(dimension))
            for i in range(dimension)
        ),
        budget=budget,
    )
    if orthogonality >= 1:
        raise ValueError(
            f"_source_spectral_enclosure.orthogonality must be less than one for a certified polar factor; got {orthogonality!r}"
        )
    norm_q = fraction_sqrt_interval(1 + orthogonality, budget=budget)[1]
    polar_error = orthogonality / (
        1 + fraction_sqrt_interval(1 - orthogonality, budget=budget)[0]
    )
    reconstruction = _exact_matrix_product(
        tuple(
            tuple(q[i][j] * values[j] for j in range(dimension)) for i in range(dimension)
        ),
        transpose,
        budget=budget,
    )
    residual = matrix_norm_enclosure(
        tuple(
            tuple(original[i][j] - reconstruction[i][j] for j in range(dimension))
            for i in range(dimension)
        ),
        budget=budget,
    )
    residual += max(abs(value) for value in values) * polar_error * (norm_q + 1)
    return _HermitianSpectralEnclosure(values, q, norm_q, polar_error, residual)


def _scalar_exp_enclosure(
    value: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[Fraction, Fraction]:
    radius = abs(value)
    terms = max(32, 4 * math.ceil(radius) + 16)
    if terms > 512:
        raise ValueError(f"_scalar_exp_enclosure.terms must be at most 512; got {terms}")
    reserve_fraction_work(
        budget,
        3 * terms + 4,
        8,
        (terms + 2)
        * (
            abs(value.numerator).bit_length()
            + value.denominator.bit_length()
            + terms.bit_length()
            + 4
        )
        + 256,
    )
    total, term = Fraction(1), Fraction(1)
    for index in range(1, terms + 1):
        term *= value / index
        total += term
    tail = (
        Fraction(3) ** math.ceil(radius)
        * radius ** (terms + 1)
        / math.factorial(terms + 1)
    )
    return total, tail


def hermitian_log_enclosure(
    values: ArrayLike,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> HermitianFunctionEnclosure:
    """Certify the matrix logarithm using the numerical spectral residual.

    The returned rational matrix is an approximation, with an operator-norm
    enclosure covering scalar log tails, nonorthogonal numerical eigenvectors,
    and the rational residual against the input tensor values.
    """
    if np.iscomplexobj(values):
        raise TypeError(
            f"hermitian_log_enclosure.values must be real-valued; got dtype {np.asarray(values).dtype}"
        )
    metric = np.asarray(values, dtype=np.float64)
    if (
        metric.ndim != 2
        or metric.shape[0] != metric.shape[1]
        or not 1 <= metric.shape[0] <= 16
        or not np.all(np.isfinite(metric))
        or not np.array_equal(metric, metric.T)
    ):
        raise ValueError(
            f"hermitian_log_enclosure.values must be a finite symmetric square matrix of order 1–16; got {metric!r} with shape {metric.shape}"
        )
    dimension = metric.shape[0]
    reserve_fraction_work(budget, dimension**2, dimension**2 + dimension + 2, 1075)
    original = tuple(tuple(Fraction(float(value)) for value in row) for row in metric)
    enclosure = _source_spectral_enclosure(original, budget=budget)
    eigenvalues, q, norm_q, polar_error, residual = enclosure
    if min(eigenvalues) <= 0:
        raise ValueError(
            f"hermitian_log_enclosure.eigenvalues must be positive; got minimum {min(eigenvalues)!r}"
        )
    transpose = tuple(zip(*q, strict=True))
    minimum = min(eigenvalues) - residual
    if minimum <= 0:
        raise ValueError(
            f"hermitian_log_enclosure.minimum must be positive after subtracting the spectral residual; got {minimum!r}"
        )
    intervals = tuple(
        _scalar_log_enclosure(value, budget=budget) for value in eigenvalues
    )
    centers = tuple((low + high) / 2 for low, high in intervals)
    matrix = _exact_matrix_product(
        tuple(
            tuple(q[i][j] * centers[j] for j in range(dimension))
            for i in range(dimension)
        ),
        transpose,
        budget=budget,
    )
    scalar_error = max((high - low) / 2 for low, high in intervals)
    logarithm_norm = max(max(abs(low), abs(high)) for low, high in intervals)
    error = (
        residual / minimum
        + logarithm_norm * polar_error * (norm_q + 1)
        + scalar_error * norm_q * norm_q
    )
    result = HermitianFunctionEnclosure(matrix, error)
    return result


def hermitian_exp_enclosure(
    logarithm: HermitianFunctionEnclosure,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[tuple[tuple[Fraction, ...], ...], Fraction]:
    """Enclose the exponential of a Hermitian matrix ball using its native spectrum."""
    if not isinstance(logarithm, HermitianFunctionEnclosure):
        raise TypeError(
            f"hermitian_exp_enclosure.logarithm must be HermitianFunctionEnclosure; got {type(logarithm).__name__}"
        )
    dimension = len(logarithm.matrix)
    if (
        not 1 <= dimension <= 16
        or any(len(row) != dimension for row in logarithm.matrix)
        or logarithm.error < 0
    ):
        raise ValueError(
            f"hermitian_exp_enclosure.logarithm must have a square matrix of order 1–16 and a non-negative error radius; got row lengths {tuple(len(row) for row in logarithm.matrix)} and error {logarithm.error!r}"
        )
    if any(
        logarithm.matrix[i][j] != logarithm.matrix[j][i]
        for i in range(dimension)
        for j in range(dimension)
    ):
        raise ValueError(
            f"hermitian_exp_enclosure.logarithm.matrix must be symmetric; got {logarithm.matrix!r}"
        )
    spectral = _source_spectral_enclosure(logarithm.matrix, budget=budget)
    values, q, norm_q, polar_error, residual = spectral
    intervals = tuple(_scalar_exp_enclosure(value, budget=budget) for value in values)
    centers = tuple(value for value, _ in intervals)
    transpose = tuple(zip(*q, strict=True))
    result = _exact_matrix_product(
        tuple(
            tuple(q[i][j] * centers[j] for j in range(dimension))
            for i in range(dimension)
        ),
        transpose,
        budget=budget,
    )
    error = max(error for _, error in intervals) * norm_q * norm_q
    exponential_norm = max(value + error for value, error in intervals)
    error += exponential_norm * polar_error * (norm_q + 1)
    source_radius = residual + logarithm.error
    exponent = max(0, math.ceil(max(values) + source_radius))
    reserve_fraction_work(budget, exponent.bit_length() + 1, 3, 2 * exponent + 2)
    exponential_upper = Fraction(3) ** exponent
    bound = error + exponential_upper * source_radius
    return result, bound


__all__ = [
    "HermitianFunctionEnclosure",
    "hermitian_exp_enclosure",
    "hermitian_log_enclosure",
    "HermitianFunctionResult",
    "HermitianSpectralMapResult",
    "HermitianSpectrum",
    "HermitianSylvesterOperator",
    "SylvesterSolveResult",
    "TracelessHermitianSpace",
    "hermitian_exp",
    "hermitian_inverse_sqrt",
    "hermitian_log",
    "hermitian_sqrt",
    "hermitian_spectral_map",
]
