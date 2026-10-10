#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from fractions import Fraction
from typing import NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..typing import checked
from ._hermitian_precision import HermitianPrecisionPolicy


if TYPE_CHECKING:
    from ..discretization._coordinate_enclosure import CoordinateEnclosureBudget


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
    precision_evidence: PrecisionEvidenceEnvelope
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
        value = precision_.compute(original)
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


class HermitianFunctionResult(StrictModule):
    value: Array
    spectrum: HermitianSpectrum
    valid: Array
    function_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        spectrum: HermitianSpectrum,
        /,
        *,
        function_id: str,
        valid: ArrayLike,
    ) -> None:
        self.value = jnp.asarray(value)
        self.spectrum = spectrum
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.function_id = str(function_id)


def _spectral_result(
    matrix: ArrayLike,
    function: Callable[[Array], Array],
    /,
    *,
    function_id: str,
    tolerance: float,
    positive: bool,
    nonnegative: bool = False,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    spectrum = HermitianSpectrum(
        matrix,
        tolerance=tolerance,
        precision=precision,
    )
    transformed = function(spectrum.eigenvalues)
    value = (spectrum.eigenvectors * transformed[..., None, :]) @ _adjoint(
        spectrum.eigenvectors
    )
    valid = spectrum.valid & jnp.all(jnp.isfinite(transformed), axis=-1)
    if positive:
        valid = valid & (spectrum.minimum_eigenvalue > tolerance)
    if nonnegative:
        valid = valid & (spectrum.minimum_eigenvalue >= -tolerance)
    return HermitianFunctionResult(
        spectrum.precision.output(0.5 * (value + _adjoint(value))),
        spectrum,
        function_id=function_id,
        valid=valid,
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
        lambda values: jnp.sqrt(jnp.maximum(values, 0.0)),
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
        lambda values: 1.0 / jnp.sqrt(values),
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
        jnp.log,
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
        jnp.exp,
        function_id="hermitian-exp",
        tolerance=tolerance,
        positive=False,
        precision=precision,
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


def _reserve_fraction_work(
    budget: CoordinateEnclosureBudget | None,
    work: int,
    count: int,
    bits: int,
    /,
) -> None:
    """Admit exact term visits and a CPython-object upper bound, never native RSS."""
    if budget is None:
        return
    digits = (
        max(bits, 1) + sys.int_info.bits_per_digit - 1
    ) // sys.int_info.bits_per_digit
    fraction_upper = 256 + 2 * (sys.getsizeof(0) + digits * sys.int_info.sizeof_digit)
    budget.reserve(work, 256 + count * (fraction_upper + 16))


def _fraction_matrix_profile(
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


def _fraction_sqrt_interval(
    value: Fraction,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> tuple[Fraction, Fraction]:
    """Rational endpoints proved by squaring, not a libm accuracy assumption."""
    _reserve_fraction_work(
        coordinate_budget,
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
    central = math.sqrt(float(value))
    if not math.isfinite(central) or central == 0:
        raise ValueError("Embedded measure exceeds certified binary64 arithmetic range.")
    lower = upper = central
    while Fraction(lower) ** 2 > value:
        if coordinate_budget is not None:
            coordinate_budget.reserve(1)
        lower = float(np.nextafter(lower, -math.inf))
    while Fraction(upper) ** 2 < value:
        if coordinate_budget is not None:
            coordinate_budget.reserve(1)
        upper = float(np.nextafter(upper, math.inf))
    return Fraction(lower), Fraction(upper)


class HermitianFunctionEnclosure(NamedTuple):
    """Immutable host certificate: exact rational nominal matrix and norm radius."""

    matrix: tuple[tuple[Fraction, ...], ...]
    error: Fraction


def _exact_matrix_product(
    first: tuple[tuple[Fraction, ...], ...],
    second: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> tuple[tuple[Fraction, ...], ...]:
    dimension = len(first)
    if coordinate_budget is not None:
        numerator_bits, denominator_bits = _fraction_matrix_profile(
            (first, second), coordinate_budget
        )
        result_bits = (
            2 * dimension * (numerator_bits + denominator_bits)
            + dimension.bit_length()
            + 4
        )
        _reserve_fraction_work(
            coordinate_budget, 2 * dimension**3, dimension**2 + 3, result_bits
        )
    return tuple(
        tuple(
            sum((first[i][k] * second[k][j] for k in range(dimension)), Fraction(0))
            for j in range(dimension)
        )
        for i in range(dimension)
    )


def _matrix_norm_enclosure(
    matrix: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> Fraction:
    if coordinate_budget is not None:
        numerator_bits, denominator_bits = _fraction_matrix_profile(
            (matrix,), coordinate_budget
        )
        count = sum(len(row) for row in matrix)
        _reserve_fraction_work(
            coordinate_budget,
            2 * count,
            4,
            2 * count * (numerator_bits + denominator_bits) + count.bit_length() + 4,
        )
    return _fraction_sqrt_interval(
        sum((value * value for row in matrix for value in row), Fraction(0)),
        coordinate_budget=coordinate_budget,
    )[1]


def _scalar_log_enclosure(
    value: Fraction,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> tuple[Fraction, Fraction]:
    """Exact atanh series after dyadic scaling; no libm accuracy premise."""
    if value <= 0:
        raise ValueError("A metric logarithm requires positive scalar eigenvalues.")
    _reserve_fraction_work(
        coordinate_budget,
        8,
        14,
        140 * (abs(value.numerator).bit_length() + value.denominator.bit_length()) + 512,
    )
    power = value.numerator.bit_length() - value.denominator.bit_length()
    factor = Fraction(2) ** power
    while value < factor:
        if coordinate_budget is not None:
            coordinate_budget.reserve(1)
        power -= 1
        factor /= 2
    while value >= 2 * factor:
        if coordinate_budget is not None:
            coordinate_budget.reserve(1)
        power += 1
        factor *= 2
    ratio = value / factor
    z = (ratio - 1) / (ratio + 1)
    intervals = []
    for argument in (z, Fraction(1, 3)):
        squared = argument * argument
        term, total = argument, Fraction(0)
        for index in range(32):
            if coordinate_budget is not None:
                coordinate_budget.reserve(4)
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
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> _HermitianSpectralEnclosure:
    """Exact residual/polar enclosure around the native numerical eigenframe."""
    dimension = len(original)
    _reserve_fraction_work(
        coordinate_budget,
        dimension**2 + dimension,
        2 * dimension**2 + dimension + 3,
        1075,
    )
    floating = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(floating)):
        raise ValueError(
            "The exact Hermitian source exceeds finite numerical spectral admission."
        )
    spectrum = HermitianSpectrum(jnp.asarray(floating, dtype=jnp.float64))
    if not bool(np.asarray(spectrum.valid)):
        raise ValueError("The native Hermitian spectrum is invalid.")
    values = tuple(Fraction(float(value)) for value in np.asarray(spectrum.eigenvalues))
    q = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(spectrum.eigenvectors)
    )
    transpose = tuple(zip(*q, strict=True))
    identity = tuple(
        tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension)
    )
    gram = _exact_matrix_product(transpose, q, coordinate_budget=coordinate_budget)
    orthogonality = _matrix_norm_enclosure(
        tuple(
            tuple(gram[i][j] - identity[i][j] for j in range(dimension))
            for i in range(dimension)
        ),
        coordinate_budget=coordinate_budget,
    )
    if orthogonality >= 1:
        raise ValueError("The native Hermitian eigenframe has no certified polar factor.")
    norm_q = _fraction_sqrt_interval(
        1 + orthogonality, coordinate_budget=coordinate_budget
    )[1]
    polar_error = orthogonality / (
        1
        + _fraction_sqrt_interval(1 - orthogonality, coordinate_budget=coordinate_budget)[
            0
        ]
    )
    reconstruction = _exact_matrix_product(
        tuple(
            tuple(q[i][j] * values[j] for j in range(dimension)) for i in range(dimension)
        ),
        transpose,
        coordinate_budget=coordinate_budget,
    )
    residual = _matrix_norm_enclosure(
        tuple(
            tuple(original[i][j] - reconstruction[i][j] for j in range(dimension))
            for i in range(dimension)
        ),
        coordinate_budget=coordinate_budget,
    )
    residual += max(abs(value) for value in values) * polar_error * (norm_q + 1)
    return _HermitianSpectralEnclosure(values, q, norm_q, polar_error, residual)


def _scalar_exp_enclosure(
    value: Fraction,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> tuple[Fraction, Fraction]:
    radius = abs(value)
    terms = max(32, 4 * math.ceil(radius) + 16)
    if terms > 512:
        raise ValueError(
            "Hermitian exponential exceeds its bounded scalar enclosure order."
        )
    _reserve_fraction_work(
        coordinate_budget,
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
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> HermitianFunctionEnclosure:
    """Certify the matrix logarithm using the native spectral owner's residual.

    The returned rational matrix is an approximation, with an operator-norm
    enclosure covering scalar log tails, nonorthogonal numerical eigenvectors,
    and the exact residual against the original tensor values.
    """
    if np.iscomplexobj(values):
        raise TypeError("This host enclosure accepts real symmetric matrices.")
    metric = np.asarray(values, dtype=np.float64)
    if (
        metric.ndim != 2
        or metric.shape[0] != metric.shape[1]
        or not 1 <= metric.shape[0] <= 16
        or not np.all(np.isfinite(metric))
        or not np.array_equal(metric, metric.T)
    ):
        raise ValueError(
            "Hermitian logarithm enclosure requires a finite symmetric matrix of order 1–16."
        )
    dimension = metric.shape[0]
    _reserve_fraction_work(
        coordinate_budget, dimension**2, dimension**2 + dimension + 2, 1075
    )
    original = tuple(tuple(Fraction(float(value)) for value in row) for row in metric)
    enclosure = _source_spectral_enclosure(original, coordinate_budget=coordinate_budget)
    eigenvalues, q, norm_q, polar_error, residual = enclosure
    if min(eigenvalues) <= 0:
        raise ValueError(
            "Hermitian logarithm enclosure requires a positive native spectrum."
        )
    transpose = tuple(zip(*q, strict=True))
    minimum = min(eigenvalues) - residual
    if minimum <= 0:
        raise ValueError(
            "Metric spectral residual does not establish a positive logarithm domain."
        )
    intervals = tuple(
        _scalar_log_enclosure(value, coordinate_budget=coordinate_budget)
        for value in eigenvalues
    )
    centers = tuple((low + high) / 2 for low, high in intervals)
    matrix = _exact_matrix_product(
        tuple(
            tuple(q[i][j] * centers[j] for j in range(dimension))
            for i in range(dimension)
        ),
        transpose,
        coordinate_budget=coordinate_budget,
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
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> tuple[tuple[tuple[Fraction, ...], ...], Fraction]:
    """Enclose the exponential of a Hermitian matrix ball using its native spectrum."""
    if not isinstance(logarithm, HermitianFunctionEnclosure):
        raise TypeError("logarithm must be HermitianFunctionEnclosure.")
    dimension = len(logarithm.matrix)
    if (
        not 1 <= dimension <= 16
        or any(len(row) != dimension for row in logarithm.matrix)
        or logarithm.error < 0
    ):
        raise ValueError(
            "Hermitian exponential enclosure requires a bounded square host certificate."
        )
    if any(
        logarithm.matrix[i][j] != logarithm.matrix[j][i]
        for i in range(dimension)
        for j in range(dimension)
    ):
        raise ValueError("The exact exponential source must be symmetric.")
    spectral = _source_spectral_enclosure(
        logarithm.matrix, coordinate_budget=coordinate_budget
    )
    values, q, norm_q, polar_error, residual = spectral
    intervals = tuple(
        _scalar_exp_enclosure(value, coordinate_budget=coordinate_budget)
        for value in values
    )
    centers = tuple(value for value, _ in intervals)
    transpose = tuple(zip(*q, strict=True))
    result = _exact_matrix_product(
        tuple(
            tuple(q[i][j] * centers[j] for j in range(dimension))
            for i in range(dimension)
        ),
        transpose,
        coordinate_budget=coordinate_budget,
    )
    error = max(error for _, error in intervals) * norm_q * norm_q
    exponential_norm = max(value + error for value, error in intervals)
    error += exponential_norm * polar_error * (norm_q + 1)
    source_radius = residual + logarithm.error
    exponent = max(0, math.ceil(max(values) + source_radius))
    _reserve_fraction_work(
        coordinate_budget, exponent.bit_length() + 1, 3, 2 * exponent + 2
    )
    exponential_upper = Fraction(3) ** exponent
    bound = error + exponential_upper * source_radius
    return result, bound


__all__ = [
    "HermitianFunctionEnclosure",
    "hermitian_exp_enclosure",
    "hermitian_log_enclosure",
    "HermitianFunctionResult",
    "HermitianSpectrum",
    "HermitianSylvesterOperator",
    "SylvesterSolveResult",
    "TracelessHermitianSpace",
    "hermitian_exp",
    "hermitian_inverse_sqrt",
    "hermitian_log",
    "hermitian_sqrt",
]
