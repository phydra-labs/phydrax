#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from functools import partial
from typing import assert_never, final, Generic, Literal, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..typing import checked, parse
from ._hermitian_precision import HermitianPrecisionPolicy


P = TypeVar("P")
Q = TypeVar("Q")
type _FunctionKind = Literal["log", "exp", "sqrt", "inverse_sqrt"]


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


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

    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        tolerance: float | np.floating = 1e-10,
        precision: HermitianPrecisionPolicy | None = None,
    ) -> None:
        original = jnp.asarray(matrix)
        precision_ = HermitianPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, HermitianPrecisionPolicy):
            raise TypeError("precision must be a HermitianPrecisionPolicy or None.")
        # Measure and factorize one rounded matrix: a fused, FMA-contracted
        # recomputation would report the rounding of values never materialized.
        value = jax.lax.optimization_barrier(precision_.compute(original))
        if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
            raise ValueError("Hermitian spectrum requires square trailing matrix axes.")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")
        hermitian = precision_.factorization(0.5 * value + 0.5 * _adjoint(value))
        eigenvalues, eigenvectors = jnp.linalg.eigh(hermitian)
        differences = jnp.abs(eigenvalues[..., 1:] - eigenvalues[..., :-1])
        minimum_gap = (
            jnp.min(differences, axis=-1)
            if value.shape[-1] > 1
            else jnp.full(value.shape[:-2], jnp.inf, dtype=eigenvalues.dtype)
        )
        magnitude = precision_.decision(jnp.max(jnp.abs(eigenvalues), axis=-1))
        threshold = precision_.decision(tolerance) * jnp.maximum(magnitude, 1.0)
        rank = jnp.sum(jnp.abs(eigenvalues) > threshold[..., None], axis=-1)
        minimum_absolute = precision_.decision(jnp.min(jnp.abs(eigenvalues), axis=-1))
        condition = magnitude / jnp.maximum(
            minimum_absolute, jnp.finfo(eigenvalues.dtype).tiny
        )
        residual = precision_.decision(
            jnp.max(
                jnp.abs(precision_.accumulation(value - _adjoint(value))),
                axis=(-2, -1),
            )
        )
        self.matrix = hermitian
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.hermiticity_residual = residual
        self.minimum_eigenvalue = precision_.decision(jnp.min(eigenvalues, axis=-1))
        self.minimum_gap = precision_.decision(minimum_gap)
        self.numerical_rank = rank
        self.condition_number = precision_.decision(condition)
        self.precision = precision_
        self.precision_evidence = precision_.evidence_for(original)
        self.valid = (
            jnp.all(jnp.isfinite(value), axis=(-2, -1))
            & (residual <= tolerance)
            & jnp.all(jnp.isfinite(eigenvalues), axis=-1)
        )
        self.tolerance = float(tolerance)

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
        self.value = jnp.asarray(value)
        self.spectrum = spectrum
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.derivative_valid = jnp.asarray(derivative_valid, dtype=jnp.bool_)
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
    # Reuse the stopped diagnostic factorization; matrix owns its exact JVP.
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
            "eigenvalue_map principal output must have shape (k, n), k >= 1."
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
    """Evaluate a permutation-equivariant principal map with exact first-order AD.

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
        self.value = jnp.asarray(value)
        self.residual_norm = jnp.asarray(residual_norm)
        self.minimum_denominator = jnp.asarray(minimum_denominator)
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.precision_evidence = precision_evidence


class HermitianSylvesterOperator(StrictModule):
    """Matrix-free ``X -> rho X + X rho`` and spectral inverse action."""

    matrix: Array
    spectrum: HermitianSpectrum
    tolerance: float = eqx.field(static=True)
    precision: HermitianPrecisionPolicy

    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
        precision: HermitianPrecisionPolicy | None = None,
    ) -> None:
        precision_ = HermitianPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, HermitianPrecisionPolicy):
            raise TypeError("precision must be a HermitianPrecisionPolicy or None.")
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
            raise ValueError("Sylvester operand must match the matrix shape.")
        return self.matrix @ operand + operand @ self.matrix

    def solve(self, right_hand_side: ArrayLike, /) -> SylvesterSolveResult:
        right = self.precision.factorization(right_hand_side)
        if right.shape != self.matrix.shape:
            raise ValueError("Sylvester right-hand side must match the matrix shape.")
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
            raise ValueError("Density tangent dimension must be at least two.")
        self.dimension = dimension_
        self.space_id = f"traceless-hermitian:{dimension_}"

    @property
    def shape(self) -> tuple[int, int]:
        return self.dimension, self.dimension

    def project(self, value: ArrayLike, /) -> Array:
        matrix = jnp.asarray(value)
        if matrix.shape[-2:] != self.shape:
            raise ValueError(f"Matrix must have trailing shape {self.shape}.")
        hermitian = 0.5 * (matrix + _adjoint(matrix))
        trace = jnp.trace(hermitian, axis1=-2, axis2=-1) / float(self.dimension)
        identity = jnp.eye(self.dimension, dtype=hermitian.dtype)
        return hermitian - trace[..., None, None] * identity

    def inner(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return jnp.real(jnp.vdot(self.project(left), self.project(right)))


__all__ = [
    "HermitianFunctionEnclosure",
    "HermitianFunctionResult",
    "HermitianSpectralMapResult",
    "HermitianSpectrum",
    "HermitianSylvesterOperator",
    "SylvesterSolveResult",
    "TracelessHermitianSpace",
    "hermitian_exp",
    "hermitian_exp_enclosure",
    "hermitian_inverse_sqrt",
    "hermitian_log",
    "hermitian_log_enclosure",
    "hermitian_sqrt",
    "hermitian_spectral_map",
]


class HermitianFunctionEnclosure(NamedTuple):
    """Rational symmetric matrix and an operator-norm error radius."""

    matrix: tuple[tuple[Fraction, ...], ...]
    error: Fraction


class _HermitianSpectralEnclosure(NamedTuple):
    values: tuple[Fraction, ...]
    vectors: tuple[tuple[Fraction, ...], ...]
    vector_norm: Fraction
    polar_error: Fraction
    residual: Fraction


def fraction_sqrt_interval(value: Fraction, /) -> tuple[Fraction, Fraction]:
    """Return rational square-root endpoints verified by exact squaring."""
    if value < 0:
        raise ValueError(
            f"fraction_sqrt_interval.value must be nonnegative; got {value!r}"
        )
    if value == 0:
        return Fraction(0), Fraction(0)
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
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
        exponent = (value.numerator.bit_length() - value.denominator.bit_length()) // 2
        scale = Fraction(2) ** exponent
        lower, upper = fraction_sqrt_interval(value / (scale * scale))
        return lower * scale, upper * scale
    central = math.sqrt(nominal)
    if not math.isfinite(central) or central == 0:
        raise ValueError(
            f"fraction_sqrt_interval.value has no finite nonzero binary64 square-root estimate; got {central!r}"
        )
    # Endpoint squares compare as integers: with ``x = p / q`` (``q`` a power
    # of two) and ``value = a / b``, ``x**2 <= value`` iff ``p*p*b <= a*q*q``.
    lower = upper = central
    while _square_sign(lower, value) > 0:
        lower = math.nextafter(lower, -math.inf)
    while _square_sign(upper, value) < 0:
        upper = math.nextafter(upper, math.inf)
    return Fraction(lower), Fraction(upper)


def _square_sign(endpoint: float, value: Fraction, /) -> int:
    """Sign of ``endpoint**2 - value`` in exact integer arithmetic."""
    numerator, denominator = endpoint.as_integer_ratio()
    left = numerator * numerator * value.denominator
    right = value.numerator * denominator * denominator
    return (left > right) - (left < right)


def _exact_matrix_product(
    first: tuple[tuple[Fraction, ...], ...],
    second: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    dimension = len(first)
    return tuple(
        tuple(
            sum((first[i][k] * second[k][j] for k in range(dimension)), Fraction(0))
            for j in range(dimension)
        )
        for i in range(dimension)
    )


def matrix_norm_enclosure(matrix: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    """Bound the spectral norm by the outward-enclosed Frobenius norm."""
    squared = sum((value * value for row in matrix for value in row), Fraction(0))
    return fraction_sqrt_interval(squared)[1]


def _scalar_log_enclosure(value: Fraction, /) -> tuple[Fraction, Fraction]:
    """Enclose log(value) by an atanh series after exact dyadic scaling."""
    if value <= 0:
        raise ValueError(f"_scalar_log_enclosure.value must be positive; got {value!r}")
    power = value.numerator.bit_length() - value.denominator.bit_length()
    factor = Fraction(2) ** power
    while value < factor:
        power -= 1
        factor /= 2
    while value >= 2 * factor:
        power += 1
        factor *= 2
    ratio = value / factor
    z = (ratio - 1) / (ratio + 1)
    intervals = []
    for argument in (z, Fraction(1, 3)):
        squared = argument * argument
        term, total = argument, Fraction(0)
        for index in range(32):
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


def _source_spectral_enclosure(
    original: tuple[tuple[Fraction, ...], ...], /
) -> _HermitianSpectralEnclosure:
    """Bound eigenframe nonorthogonality and the exact residual to the source."""
    dimension = len(original)
    floating = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(floating)):
        raise ValueError(
            "_source_spectral_enclosure.original must convert to finite binary64 entries."
        )
    spectrum = HermitianSpectrum(jnp.asarray(floating, dtype=jnp.float64))
    if not bool(np.asarray(spectrum.valid)):
        raise ValueError("_source_spectral_enclosure.spectrum must be valid.")
    values = tuple(Fraction(float(value)) for value in np.asarray(spectrum.eigenvalues))
    vectors = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(spectrum.eigenvectors)
    )
    transpose = tuple(zip(*vectors, strict=True))
    identity = tuple(
        tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension)
    )
    gram = _exact_matrix_product(transpose, vectors)
    orthogonality = matrix_norm_enclosure(
        tuple(
            tuple(gram[i][j] - identity[i][j] for j in range(dimension))
            for i in range(dimension)
        )
    )
    if orthogonality >= 1:
        raise ValueError(
            "_source_spectral_enclosure.eigenvectors do not admit a certified polar factor."
        )
    vector_norm = fraction_sqrt_interval(1 + orthogonality)[1]
    polar_error = orthogonality / (1 + fraction_sqrt_interval(1 - orthogonality)[0])
    reconstruction = _exact_matrix_product(
        tuple(
            tuple(vectors[i][j] * values[j] for j in range(dimension))
            for i in range(dimension)
        ),
        transpose,
    )
    residual = matrix_norm_enclosure(
        tuple(
            tuple(original[i][j] - reconstruction[i][j] for j in range(dimension))
            for i in range(dimension)
        )
    )
    residual += max(abs(value) for value in values) * polar_error * (vector_norm + 1)
    return _HermitianSpectralEnclosure(
        values, vectors, vector_norm, polar_error, residual
    )


def _scalar_exp_enclosure(value: Fraction, /) -> tuple[Fraction, Fraction]:
    """Bound the exponential by its Taylor partial sum and a rational tail."""
    radius = abs(value)
    terms = max(32, 4 * math.ceil(radius) + 16)
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


def _real_symmetric_binary64(values: ArrayLike, name: str, /) -> np.ndarray:
    if np.iscomplexobj(values):
        raise TypeError(f"{name} must be real-valued.")
    source = np.asarray(values)
    if source.ndim != 2 or source.shape[0] != source.shape[1] or source.shape[0] == 0:
        raise ValueError(
            f"{name} must be a nonempty square matrix; got shape {source.shape}."
        )
    matrix = np.asarray(source, dtype=np.float64)
    if (
        not np.all(np.isfinite(matrix))
        or not np.array_equal(source, matrix)
        or not np.array_equal(matrix, matrix.T)
    ):
        raise ValueError(
            f"{name} must be finite, symmetric, and exactly represented in binary64."
        )
    return matrix


def hermitian_log_enclosure(values: ArrayLike, /) -> HermitianFunctionEnclosure:
    """Enclose log of a real symmetric positive-definite binary64 matrix.

    The returned rational matrix is an approximation. Its operator-norm error
    includes scalar-series tails, the numerical eigenframe residual, and
    perturbation of the input spectrum.
    """
    metric = _real_symmetric_binary64(values, "hermitian_log_enclosure.values")
    original = tuple(tuple(Fraction(float(value)) for value in row) for row in metric)
    eigenvalues, vectors, vector_norm, polar_error, residual = _source_spectral_enclosure(
        original
    )
    if min(eigenvalues) <= 0:
        raise ValueError("hermitian_log_enclosure requires positive eigenvalues.")
    minimum = min(eigenvalues) - residual
    if minimum <= 0:
        raise ValueError(
            "hermitian_log_enclosure cannot prove a positive lower spectral bound."
        )
    intervals = tuple(_scalar_log_enclosure(value) for value in eigenvalues)
    centers = tuple((low + high) / 2 for low, high in intervals)
    transpose = tuple(zip(*vectors, strict=True))
    matrix = _exact_matrix_product(
        tuple(
            tuple(vectors[i][j] * centers[j] for j in range(len(eigenvalues)))
            for i in range(len(eigenvalues))
        ),
        transpose,
    )
    scalar_error = max((high - low) / 2 for low, high in intervals)
    logarithm_norm = max(max(abs(low), abs(high)) for low, high in intervals)
    error = (
        residual / minimum
        + logarithm_norm * polar_error * (vector_norm + 1)
        + scalar_error * vector_norm * vector_norm
    )
    return HermitianFunctionEnclosure(matrix, error)


def hermitian_exp_enclosure(
    logarithm: HermitianFunctionEnclosure, /
) -> tuple[tuple[tuple[Fraction, ...], ...], Fraction]:
    """Enclose exp of a real symmetric rational matrix ball."""
    if not isinstance(logarithm, HermitianFunctionEnclosure):
        raise TypeError(
            "hermitian_exp_enclosure.logarithm must be a HermitianFunctionEnclosure."
        )
    dimension = len(logarithm.matrix)
    if (
        dimension == 0
        or any(len(row) != dimension for row in logarithm.matrix)
        or logarithm.error < 0
    ):
        raise ValueError(
            "hermitian_exp_enclosure.logarithm must contain a nonempty square matrix and a nonnegative error radius."
        )
    if any(
        logarithm.matrix[i][j] != logarithm.matrix[j][i]
        for i in range(dimension)
        for j in range(dimension)
    ):
        raise ValueError("hermitian_exp_enclosure.logarithm.matrix must be symmetric.")
    values, vectors, vector_norm, polar_error, residual = _source_spectral_enclosure(
        logarithm.matrix
    )
    intervals = tuple(_scalar_exp_enclosure(value) for value in values)
    centers = tuple(value for value, _ in intervals)
    transpose = tuple(zip(*vectors, strict=True))
    result = _exact_matrix_product(
        tuple(
            tuple(vectors[i][j] * centers[j] for j in range(dimension))
            for i in range(dimension)
        ),
        transpose,
    )
    error = max(tail for _, tail in intervals) * vector_norm * vector_norm
    exponential_norm = max(value + tail for value, tail in intervals)
    error += exponential_norm * polar_error * (vector_norm + 1)
    source_radius = residual + logarithm.error
    exponent = max(0, math.ceil(max(values) + source_radius))
    error += Fraction(3) ** exponent * source_radius
    return result, error
