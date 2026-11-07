#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Alpha-theory, interval Krawczyk, and exact univariate root certificates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from ..linalg._exact import solve_exact
from ._exact_univariate import isolate_real_roots
from ._system import SparsePolynomialSystem


_SMALE_ALPHA_THRESHOLD = (13.0 - 3.0 * math.sqrt(17.0)) / 4.0
_BINARY64_MAXIMUM = float(np.finfo(np.float64).max)
_RATIONAL_BINARY64_MAXIMUM = Fraction(_BINARY64_MAXIMUM)
_RATIONAL_BINARY64_TINY = Fraction(float(np.finfo(np.float64).tiny))

type _Gaussian = tuple[Fraction, Fraction]
type _Range = tuple[Fraction, Fraction]
type _MultiIndex = tuple[tuple[int, int], ...]


def _outward_float(value: Fraction, /, *, upward: bool) -> float:
    """Round a rational to binary64 toward positive or negative infinity."""
    if value > _RATIONAL_BINARY64_MAXIMUM:
        return math.inf if upward else _BINARY64_MAXIMUM
    if value < -_RATIONAL_BINARY64_MAXIMUM:
        return -_BINARY64_MAXIMUM if upward else -math.inf
    # Fraction.__float__ divides integers with one correct rounding.
    nearest = float(value)
    represented = Fraction(nearest)
    if upward and represented < value:
        return math.nextafter(nearest, math.inf)
    if not upward and represented > value:
        return math.nextafter(nearest, -math.inf)
    return nearest


def _root_upper(value: Fraction, degree: int, /) -> float:
    """Return a binary64 upper bound of ``value ** (1 / degree)`` for ``value >= 0``."""
    if value == 0:
        return 0.0
    if _RATIONAL_BINARY64_TINY <= value <= _RATIONAL_BINARY64_MAXIMUM:
        nominal = float(value)
        bound = math.sqrt(nominal) if degree == 2 else nominal ** (1.0 / degree)
    else:
        logarithm = (math.log(value.numerator) - math.log(value.denominator)) / degree
        if logarithm > 709.0:
            # exp(709) < max binary64; larger roots are reported as unbounded.
            return math.inf
        bound = math.exp(logarithm)
    # The estimate is verified exactly: ulp steps absorb the libm error, and
    # geometric steps end the rare shortfall of a large logarithm.
    steps = 0
    while math.isfinite(bound) and Fraction(bound) ** degree < value:
        steps += 1
        bound = math.nextafter(
            bound if steps <= 8 else bound * (1.0 + 2.0**-40), math.inf
        )
    return bound


def _gaussian(value: complex, /) -> _Gaussian:
    return Fraction(float(value.real)), Fraction(float(value.imag))


def _gaussian_product(left: _Gaussian, right: _Gaussian, /) -> _Gaussian:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _point_derivatives(
    system: SparsePolynomialSystem, point: tuple[_Gaussian, ...], /
) -> dict[_MultiIndex, list[_Gaussian]]:
    """Exact partial derivatives ``∂^α f(point)`` for every nonvanishing ``α``.

    A multi-index is a sorted tuple of ``(variable, order)`` pairs with positive
    orders; the empty multi-index holds the residual.
    """
    support = system.support
    zero = (Fraction(0), Fraction(0))
    exponents = np.asarray(support.exponents)
    powers = [[(Fraction(1), Fraction(0))] for _ in range(support.variable_count)]
    for variable, highest in enumerate(np.max(exponents, axis=0).tolist()):
        for _ in range(int(highest)):
            powers[variable].append(
                _gaussian_product(powers[variable][-1], point[variable])
            )
    derivatives: dict[_MultiIndex, list[_Gaussian]] = {}
    for equation, exponent, coefficient in zip(
        np.asarray(support.equation_indices).tolist(),
        exponents.tolist(),
        np.asarray(system.coefficients).tolist(),
        strict=True,
    ):
        scale = _gaussian(complex(coefficient))
        active = [
            (variable, int(power)) for variable, power in enumerate(exponent) if power
        ]
        for orders in product(*(range(power + 1) for _, power in active)):
            weight = 1
            value = scale
            for (variable, power), order in zip(active, orders, strict=True):
                weight *= math.perm(power, order)
                value = _gaussian_product(value, powers[variable][power - order])
            index = tuple(
                (variable, order)
                for (variable, _), order in zip(active, orders, strict=True)
                if order
            )
            entries = derivatives.setdefault(index, [zero] * support.equation_count)
            previous = entries[equation]
            entries[equation] = (
                previous[0] + weight * value[0],
                previous[1] + weight * value[1],
            )
    return derivatives


class _SingularJacobian(ValueError):
    """The exact Jacobian at an alpha-theory point is singular."""


def _realified_system(
    jacobian: list[list[_Gaussian]], columns: list[list[_Gaussian]], /
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """Exact real system ``M z = v`` for ``J z = v`` per column ``v``.

    Complex data are realified, so ``||z||_2`` is the complex 2-norm.
    """
    size = len(jacobian)
    real = all(entry[1] == 0 for row in jacobian for entry in row) and all(
        entry[1] == 0 for column in columns for entry in column
    )
    if real:
        matrix = [[entry[0] for entry in row] for row in jacobian]
        right = [[column[row][0] for column in columns] for row in range(size)]
        return matrix, right
    # Realify A + iB as [[A, -B], [B, A]] acting on (real, imaginary) parts.
    matrix = [
        [entry[0] for entry in row] + [-entry[1] for entry in row] for row in jacobian
    ] + [[entry[1] for entry in row] + [entry[0] for entry in row] for row in jacobian]
    right = [[column[row][0] for column in columns] for row in range(size)] + [
        [column[row][1] for column in columns] for row in range(size)
    ]
    return matrix, right


def _exact_squared_norms(
    matrix: list[list[Fraction]], right: list[list[Fraction]], /
) -> list[Fraction] | None:
    """Exact ``||M^{-1} v||_2^2`` for each column ``v``; ``None`` if ``M`` is singular."""
    try:
        solution = solve_exact(matrix, right, singular_error=_SingularJacobian)
    except _SingularJacobian:
        return None
    return [
        sum((row[column] * row[column] for row in solution), Fraction(0))
        for column in range(len(right[0]))
    ]


def _outward_array(
    values: list[list[Fraction]], /
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    lower = [[_outward_float(value, upward=False) for value in row] for row in values]
    upper = [[_outward_float(value, upward=True) for value in row] for row in values]
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _square_bounds(
    lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64], /
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Outward bounds of ``x**2`` for every ``x`` in ``[lower, upper]``."""
    high = np.maximum(np.abs(lower), np.abs(upper))
    low = np.where(
        (lower <= 0.0) & (upper >= 0.0), 0.0, np.minimum(np.abs(lower), np.abs(upper))
    )
    return (
        np.maximum(np.nextafter(low * low, -np.inf), 0.0),
        np.nextafter(high * high, np.inf),
    )


def _column_sums(
    lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64], /
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Outward bounds of the column sums of an interval matrix."""
    low, high = _point_times_interval(np.ones((1, lower.shape[0])), lower, upper)
    return low[0], high[0]


def _float_squared_norms(
    matrix: list[list[Fraction]], right: list[list[Fraction]], /
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Binary64 bounds of the exact ``||M^{-1} v||_2^2`` per column, or ``None``.

    With a binary64 inverse ``Y`` and ``q >= ||I - Y M||_F``, ``q < 1`` proves
    ``M`` nonsingular and ``M^{-1} = (I - (I - Y M))^{-1} Y``, so
    ``z = M^{-1} v`` lies within ``||Y (v - M z~)||_2 / (1 - q)`` of
    ``z~ = Y v``. ``M`` and ``v`` are rounded outward from their exact values,
    products split by sign, and every operation is rounded outward. ``None``
    means the bounds are unavailable, not that ``M`` is singular.
    """
    matrix_lower, matrix_upper = _outward_array(matrix)
    right_lower, right_upper = _outward_array(right)
    with np.errstate(invalid="ignore", over="ignore", under="ignore", divide="ignore"):
        middle = 0.5 * matrix_lower + 0.5 * matrix_upper
        if not np.all(np.isfinite(middle)) or not np.linalg.cond(middle) < 1e14:
            return None
        inverse = np.linalg.inv(middle)
        product_lower, product_upper = _point_times_interval(
            inverse, matrix_lower, matrix_upper
        )
        identity = np.eye(middle.shape[0])
        _, residual_squares = _square_bounds(
            np.nextafter(identity - product_upper, -np.inf),
            np.nextafter(identity - product_lower, np.inf),
        )
        _, residual_columns = _column_sums(residual_squares, residual_squares)
        _, residual_total = _column_sums(
            residual_columns.reshape(-1, 1), residual_columns.reshape(-1, 1)
        )
        contraction = np.nextafter(np.sqrt(residual_total[0]), np.inf)
        if not contraction < 1.0:
            return None
        center = inverse @ (0.5 * right_lower + 0.5 * right_upper)
        image_lower, image_upper = _point_times_interval(
            center.T, matrix_lower.T, matrix_upper.T
        )
        correction_lower, correction_upper = _point_times_interval(
            inverse,
            np.nextafter(right_lower - image_upper.T, -np.inf),
            np.nextafter(right_upper - image_lower.T, np.inf),
        )
        _, correction_squares = _square_bounds(correction_lower, correction_upper)
        _, correction_total = _column_sums(correction_squares, correction_squares)
        radius = np.nextafter(
            np.nextafter(np.sqrt(correction_total), np.inf)
            / np.nextafter(1.0 - contraction, -np.inf),
            np.inf,
        )
        center_lower, center_upper = _square_bounds(center, center)
        total_lower, _ = _column_sums(center_lower, center_lower)
        _, total_upper = _column_sums(center_upper, center_upper)
        norm_lower = np.maximum(np.nextafter(np.sqrt(total_lower), -np.inf), 0.0)
        norm_upper = np.nextafter(np.sqrt(total_upper), np.inf)
        squared_lower, squared_upper = _square_bounds(
            np.maximum(np.nextafter(norm_lower - radius, -np.inf), 0.0),
            np.nextafter(norm_upper + radius, np.inf),
        )
    if not (np.all(np.isfinite(squared_lower)) and np.all(np.isfinite(squared_upper))):
        return None
    return squared_lower, squared_upper


def _alpha_threshold_power(power: int, /) -> tuple[Fraction, Fraction]:
    """Return ``(p, q)`` with ``alpha_0 ** power = p + q * sqrt(17)`` exactly."""
    rational, radical = Fraction(1), Fraction(0)
    for _ in range(power):
        # (p + q sqrt 17)(13/4 - 3/4 sqrt 17)
        rational, radical = (
            Fraction(13, 4) * rational - Fraction(51, 4) * radical,
            Fraction(13, 4) * radical - Fraction(3, 4) * rational,
        )
    return rational, radical


def _below_alpha_threshold_power(value: Fraction, power: int, /) -> bool:
    """Decide ``value < alpha_0 ** power`` exactly in ``Q(sqrt 17)``."""
    rational, radical = _alpha_threshold_power(power)
    difference = rational - value
    # Sign of difference + radical * sqrt(17).
    if difference >= 0 and radical >= 0:
        return difference > 0 or radical > 0
    if difference <= 0 and radical <= 0:
        return False
    squared, scaled = difference * difference, 17 * radical * radical
    return squared > scaled if difference > 0 else scaled > squared


def _alpha_threshold_bracket() -> tuple[float, float]:
    """Binary64 ``lower < alpha_0 < upper``, verified exactly."""
    lower = upper = _SMALE_ALPHA_THRESHOLD
    while not _below_alpha_threshold_power(Fraction(lower), 1):
        lower = math.nextafter(lower, -math.inf)
    while _below_alpha_threshold_power(Fraction(upper), 1):
        upper = math.nextafter(upper, math.inf)
    return lower, upper


_ALPHA_THRESHOLD_LOWER, _ALPHA_THRESHOLD_UPPER = _alpha_threshold_bracket()


def _directed_product(left: float, right: float, /, *, upward: bool) -> float:
    """Outward-rounded product of nonnegative binary64 bounds."""
    product = left * right
    if upward:
        return math.nextafter(product, math.inf)
    return max(math.nextafter(product, -math.inf), 0.0)


def _directed_power(base: float, exponent: int, /, *, upward: bool) -> float:
    result = 1.0
    for _ in range(exponent):
        result = _directed_product(result, base, upward=upward)
    return result


def _filtered_alpha(
    squared_lower: npt.NDArray[np.float64],
    squared_upper: npt.NDArray[np.float64],
    higher: list[tuple[int, int]],
    /,
) -> tuple[bool, float, dict[int, float]] | None:
    """Decide ``alpha < alpha_0`` from binary64 bounds, or ``None`` if undecided.

    Column 0 bounds ``beta**2``; the others bound ``||J^{-1} ∂^α f||^2`` for
    ``(order, multiplicity)`` in ``higher``. The test of order ``k`` is
    ``beta**(2(k-1)) gamma_k**(2(k-1)) < alpha_0**(2(k-1))``; it is accepted
    when the value's upper bound is below the threshold power's lower bound
    and refused when the value's lower bound reaches its upper bound, so every
    decision equals the exact one. Returns the decision, an upper bound of
    ``beta**2``, and upper bounds of ``gamma_k**(2(k-1))``.
    """
    totals: dict[int, tuple[float, float]] = {}
    for (order, multiplicity), lower, upper in zip(
        higher, squared_lower[1:].tolist(), squared_upper[1:].tolist(), strict=True
    ):
        weight = Fraction(multiplicity)
        low = _directed_product(_outward_float(weight, upward=False), lower, upward=False)
        high = _directed_product(_outward_float(weight, upward=True), upper, upward=True)
        total_low, total_high = totals.get(order, (0.0, 0.0))
        totals[order] = (
            max(math.nextafter(total_low + low, -math.inf), 0.0),
            math.nextafter(total_high + high, math.inf),
        )
    beta_lower, beta_upper = float(squared_lower[0]), float(squared_upper[0])
    gamma_uppers: dict[int, float] = {}
    accepted, refused = True, False
    for order, (total_low, total_high) in totals.items():
        scale = Fraction(math.factorial(order) ** 2)
        gamma_lower = max(
            math.nextafter(total_low / _outward_float(scale, upward=True), -math.inf), 0.0
        )
        gamma_upper = math.nextafter(
            total_high / _outward_float(scale, upward=False), math.inf
        )
        gamma_uppers[order] = gamma_upper
        power = 2 * (order - 1)
        value_lower = _directed_product(
            _directed_power(beta_lower, order - 1, upward=False),
            gamma_lower,
            upward=False,
        )
        value_upper = _directed_product(
            _directed_power(beta_upper, order - 1, upward=True), gamma_upper, upward=True
        )
        refused |= value_lower >= _directed_power(
            _ALPHA_THRESHOLD_UPPER, power, upward=True
        )
        accepted &= value_upper < _directed_power(
            _ALPHA_THRESHOLD_LOWER, power, upward=False
        )
    if refused or accepted:
        return accepted and not refused, beta_upper, gamma_uppers
    return None


@dataclass(frozen=True, slots=True)
class SmaleAlphaCertificate:
    point: Array
    beta: Array
    gamma_upper: Array
    alpha_upper: Array
    approximate_root: bool
    certificate_id: str


def smale_alpha_certificate(
    system: SparsePolynomialSystem,
    point: ArrayLike,
    /,
) -> SmaleAlphaCertificate:
    """Certify a square regular polynomial point with a conservative gamma bound.

    The residual, Jacobian, and every higher derivative are evaluated exactly at
    the binary64 point. ``beta = ||J^{-1} f||`` and the Frobenius bound
    ``gamma = max_k (||J^{-1} D^k f||_F / k!) ** (1/(k-1))`` are exact
    algebraic numbers, and ``alpha = beta * gamma < (13 - 3 sqrt 17) / 4`` is
    decided exactly. An outward-rounded binary64 enclosure of the same
    quantities decides first; the exact rational solve runs only when that
    enclosure straddles the threshold, and both routes give the exact decision.
    ``beta``, ``gamma_upper``, and ``alpha_upper`` report binary64 upper bounds
    from the route taken; a singular Jacobian or non-finite input is refused.
    """

    if not isinstance(system, SparsePolynomialSystem):
        raise TypeError("system must be SparsePolynomialSystem.")
    point_ = jnp.asarray(point)
    if point_.shape != (system.support.variable_count,):
        raise ValueError("Alpha-theory point has the wrong variable dimension.")
    if system.support.equation_count != system.support.variable_count:
        raise ValueError("Alpha theory requires a square polynomial system.")
    payload = {
        "kind": "smale-alpha-certificate",
        "system_id": system.system_id,
        "point": np.asarray(point_).tolist(),
        "threshold": _SMALE_ALPHA_THRESHOLD,
    }
    certificate_id = canonical_fingerprint(payload)
    host_point = np.asarray(point_)
    unbounded = jnp.asarray(math.inf, dtype=jnp.float64)
    if not (
        np.all(np.isfinite(host_point))
        and np.all(np.isfinite(np.asarray(system.coefficients)))
    ):
        return SmaleAlphaCertificate(
            point_, unbounded, unbounded, unbounded, False, certificate_id
        )
    matrix, right, higher = _smale_system(system, host_point)
    # Binary64 bounds decide all but threshold-straddling points; the decision
    # is the exact one either way (see ``_filtered_alpha``).
    bounds = _float_squared_norms(matrix, right)
    filtered = None if bounds is None else _filtered_alpha(*bounds, higher)
    if filtered is not None:
        approximate, beta_squared_upper, gamma_uppers = filtered
        beta = _float_root_upper(beta_squared_upper, 2)
        gamma = max(
            (
                _float_root_upper(bound, 2 * (order - 1))
                for order, bound in gamma_uppers.items()
            ),
            default=0.0,
        )
    else:
        norms = _exact_alpha_norms(matrix, right, higher)
        if norms is None:
            return SmaleAlphaCertificate(
                point_, unbounded, unbounded, unbounded, False, certificate_id
            )
        beta_squared, gamma_powers = norms
        # alpha < alpha_0 iff beta**(2(k-1)) gamma_k**(2(k-1)) < alpha_0**(2(k-1))
        # for every order k.
        approximate = all(
            _below_alpha_threshold_power(
                beta_squared ** (order - 1) * bound, 2 * (order - 1)
            )
            for order, bound in gamma_powers.items()
        )
        beta = _root_upper(beta_squared, 2)
        gamma = max(
            (
                _root_upper(bound, 2 * (order - 1))
                for order, bound in gamma_powers.items()
            ),
            default=0.0,
        )
    if beta == 0.0 or gamma == 0.0:
        alpha = 0.0
    elif math.isinf(beta) or math.isinf(gamma):
        alpha = math.inf
    else:
        alpha = _outward_float(Fraction(beta) * Fraction(gamma), upward=True)
    return SmaleAlphaCertificate(
        point_,
        jnp.asarray(beta, dtype=jnp.float64),
        jnp.asarray(gamma, dtype=jnp.float64),
        jnp.asarray(alpha, dtype=jnp.float64),
        approximate,
        certificate_id,
    )


def _float_root_upper(value: float, degree: int, /) -> float:
    return _root_upper(Fraction(value), degree) if math.isfinite(value) else math.inf


def _smale_system(
    system: SparsePolynomialSystem, point: npt.NDArray[np.generic], /
) -> tuple[list[list[Fraction]], list[list[Fraction]], list[tuple[int, int]]]:
    """Exact real system for ``J z = f`` and ``J z = ∂^α f`` with ``|α| >= 2``.

    Column 0 of the right side is the residual; each later column is one
    multi-index, listed in ``higher`` as ``(order, multiplicity)`` where the
    multiplicity ``order! / prod(count!)`` counts its ordered index tuples.
    """
    size = system.support.variable_count
    derivatives = _point_derivatives(
        system, tuple(_gaussian(complex(value)) for value in point.tolist())
    )
    zero = (Fraction(0), Fraction(0))
    jacobian = [
        [
            derivatives.get(((variable, 1),), [zero] * size)[equation]
            for variable in range(size)
        ]
        for equation in range(size)
    ]
    indices = [index for index in derivatives if sum(order for _, order in index) >= 2]
    columns = [derivatives.get((), [zero] * size)] + [
        derivatives[index] for index in indices
    ]
    higher = [
        (
            sum(count for _, count in index),
            math.factorial(sum(count for _, count in index))
            // math.prod(math.factorial(count) for _, count in index),
        )
        for index in indices
    ]
    matrix, right = _realified_system(jacobian, columns)
    return matrix, right, higher


def _exact_alpha_norms(
    matrix: list[list[Fraction]],
    right: list[list[Fraction]],
    higher: list[tuple[int, int]],
    /,
) -> tuple[Fraction, dict[int, Fraction]] | None:
    """Return ``beta**2`` and ``gamma_k**(2(k-1))`` per order ``k >= 2`` exactly."""
    squared = _exact_squared_norms(matrix, right)
    if squared is None:
        return None
    frobenius: dict[int, Fraction] = {}
    for (order, multiplicity), norm in zip(higher, squared[1:], strict=True):
        frobenius[order] = frobenius.get(order, Fraction(0)) + multiplicity * norm
    gamma_powers = {
        order: total / math.factorial(order) ** 2 for order, total in frobenius.items()
    }
    return squared[0], gamma_powers


def _power_range(bounds: _Range, exponent: int, /) -> _Range:
    lower, upper = bounds
    if exponent == 0:
        return Fraction(1), Fraction(1)
    if exponent % 2 == 0 and lower <= 0 <= upper:
        return Fraction(0), max(-lower, upper) ** exponent
    values = (lower**exponent, upper**exponent)
    return min(values), max(values)


def _range_product(left: _Range, right: _Range, /) -> _Range:
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def _exact_enclosures(
    system: SparsePolynomialSystem,
    center: list[Fraction],
    box: list[_Range],
    /,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Outward binary64 enclosures of ``f(center)`` and of ``J`` over ``box``.

    Monomial ranges over a box are exact products of exact power ranges, and the
    term sums are exact rational enclosures, so only the final conversion to
    binary64 rounds, outward.
    """
    support = system.support
    size = support.variable_count
    residual = [Fraction(0)] * support.equation_count
    jacobian: dict[tuple[int, int], tuple[Fraction, Fraction]] = {}
    ranges: dict[tuple[int, int], _Range] = {}

    def power_range(variable: int, exponent: int) -> _Range:
        key = (variable, exponent)
        if key not in ranges:
            ranges[key] = _power_range(box[variable], exponent)
        return ranges[key]

    for equation, exponent, coefficient in zip(
        np.asarray(support.equation_indices).tolist(),
        np.asarray(support.exponents).tolist(),
        np.asarray(system.coefficients).tolist(),
        strict=True,
    ):
        scale = Fraction(float(coefficient))
        active = [
            (variable, int(power)) for variable, power in enumerate(exponent) if power
        ]
        value = scale
        for variable, power in active:
            value *= center[variable] ** power
        residual[equation] += value
        factors = [power_range(variable, power) for variable, power in active]
        for position, (variable, power) in enumerate(active):
            enclosure = (scale * power, scale * power)
            for other, factor in enumerate(factors):
                enclosure = _range_product(
                    enclosure,
                    power_range(variable, power - 1) if other == position else factor,
                )
            lower, upper = jacobian.get((equation, variable), (Fraction(0), Fraction(0)))
            jacobian[equation, variable] = (lower + enclosure[0], upper + enclosure[1])
    residual_lower = np.asarray(
        [_outward_float(value, upward=False) for value in residual], dtype=np.float64
    )
    residual_upper = np.asarray(
        [_outward_float(value, upward=True) for value in residual], dtype=np.float64
    )
    jacobian_lower = np.zeros((support.equation_count, size), dtype=np.float64)
    jacobian_upper = np.zeros_like(jacobian_lower)
    for (equation, variable), (lower, upper) in jacobian.items():
        jacobian_lower[equation, variable] = _outward_float(lower, upward=False)
        jacobian_upper[equation, variable] = _outward_float(upper, upward=True)
    return residual_lower, residual_upper, jacobian_lower, jacobian_upper


def _point_times_interval(
    matrix: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    /,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Enclose ``matrix @ B`` for every ``B`` in ``[lower, upper]``.

    Each weight multiplies the endpoint its sign selects, and every product and
    sum is rounded outward by one binary64 step, which dominates its
    round-to-nearest error, including underflow and overflow. Zero weights
    contribute exactly zero. Indeterminate sums widen to the whole line.
    """
    result_lower = np.zeros((matrix.shape[0], lower.shape[1]), dtype=np.float64)
    result_upper = np.zeros_like(result_lower)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        for inner in range(matrix.shape[1]):
            weights = matrix[:, inner, None]
            positive = weights > 0.0
            negative = weights < 0.0
            low = np.where(positive, lower[None, inner], upper[None, inner])
            high = np.where(positive, upper[None, inner], lower[None, inner])
            active = positive | negative
            product_lower = np.where(active, np.nextafter(weights * low, -np.inf), 0.0)
            product_upper = np.where(active, np.nextafter(weights * high, np.inf), 0.0)
            result_lower = np.nextafter(result_lower + product_lower, -np.inf)
            result_upper = np.nextafter(result_upper + product_upper, np.inf)
    return (
        np.where(np.isnan(result_lower), -np.inf, result_lower),
        np.where(np.isnan(result_upper), np.inf, result_upper),
    )


@dataclass(frozen=True, slots=True)
class KrawczykCertificate:
    center: Array
    radius: Array
    image_lower: Array
    image_upper: Array
    unique_root: bool
    certificate_id: str
    dense_workspace_bytes: int
    maximum_dense_workspace_bytes: int


def krawczyk_certificate(
    system: SparsePolynomialSystem,
    center: ArrayLike,
    radius: ArrayLike,
    /,
    *,
    maximum_dense_workspace_bytes: int = 64 * 1024 * 1024,
) -> KrawczykCertificate:
    """Certify one unique real root in an axis-aligned polynomial box.

    The box is the exact rational box ``X = [center - radius, center + radius]``.
    With ``y = center`` and a binary64 preconditioner ``Y`` close to
    ``J(y)^{-1}``, the Krawczyk operator
    ``K = y - Y f(y) + (I - Y [J](X)) (X - y)`` is enclosed rigorously: ``f(y)``
    and the Jacobian range ``[J](X)`` are exact rational enclosures, interval
    matrix products split ``Y`` by sign, and every binary64 operation is
    rounded outward. ``unique_root`` holds only if ``K`` lies in the interior of
    ``X``; then ``X`` contains exactly one real root, which lies in the reported
    image ``[image_lower, image_upper]``.

    The interval enclosure requires every entry of a dense center-Jacobian
    preconditioner. Before evaluating that Jacobian, admit a conservative
    float64 workspace estimate of ``8 * (10*n*n + 12*n)`` bytes. The default
    ceiling is 64 MiB; this accounts for dense factors, solve scratch, and
    interval matrices, not provider-private or polynomial-evaluation storage.
    Admission limits and the estimate are returned as certificate evidence
    without changing the scientific certificate identity.
    """

    center_ = np.asarray(center, dtype=np.float64)
    radius_ = np.asarray(radius, dtype=np.float64)
    if (
        center_.shape != (system.support.variable_count,)
        or radius_.shape != center_.shape
    ):
        raise ValueError("Krawczyk center/radius shapes do not match the system.")
    if np.any(radius_ <= 0.0) or not np.all(np.isfinite(center_ + radius_)):
        raise ValueError("Krawczyk radii must be finite and positive.")
    if system.support.equation_count != system.support.variable_count:
        raise ValueError("Krawczyk certification requires a square system.")
    if (
        isinstance(maximum_dense_workspace_bytes, bool)
        or not isinstance(maximum_dense_workspace_bytes, int)
        or maximum_dense_workspace_bytes <= 0
    ):
        raise ValueError("maximum_dense_workspace_bytes must be a positive integer.")
    dimension = system.support.variable_count
    dense_workspace_bytes = (10 * dimension**2 + 12 * dimension) * np.dtype(
        np.float64
    ).itemsize
    if dense_workspace_bytes > maximum_dense_workspace_bytes:
        raise ValueError(
            f"Krawczyk dense workspace requires {dense_workspace_bytes} bytes, "
            f"exceeding maximum_dense_workspace_bytes={maximum_dense_workspace_bytes}."
        )
    coefficients = np.asarray(system.coefficients)
    if np.iscomplexobj(coefficients):
        raise ValueError("Krawczyk certification currently requires real coefficients.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Krawczyk certification requires finite coefficients.")
    center_jacobian = np.asarray(system.jacobian(jnp.asarray(center_)), dtype=np.float64)
    inverse = np.linalg.solve(center_jacobian, np.eye(dimension))
    exact_center = [Fraction(value) for value in center_.tolist()]
    exact_radius = [Fraction(value) for value in radius_.tolist()]
    box = [
        (middle - half, middle + half)
        for middle, half in zip(exact_center, exact_radius, strict=True)
    ]
    residual_lower, residual_upper, jacobian_lower, jacobian_upper = _exact_enclosures(
        system, exact_center, box
    )
    shift_lower, shift_upper = _point_times_interval(
        inverse, residual_lower[:, None], residual_upper[:, None]
    )
    product_lower, product_upper = _point_times_interval(
        inverse, jacobian_lower, jacobian_upper
    )
    identity = np.eye(dimension)
    with np.errstate(invalid="ignore"):
        contraction_lower = np.nextafter(identity - product_upper, -np.inf)
        contraction_upper = np.nextafter(identity - product_lower, np.inf)
    contraction = np.where(
        np.isnan(contraction_lower) | np.isnan(contraction_upper),
        np.inf,
        np.maximum(np.abs(contraction_lower), np.abs(contraction_upper)),
    )
    # (I - Y [J]) [-r, r] is the symmetric interval of |I - Y [J]| r.
    _, spread = _point_times_interval(contraction, radius_[:, None], radius_[:, None])
    bounds = np.concatenate((shift_lower[:, 0], shift_upper[:, 0], spread[:, 0]))
    if np.all(np.isfinite(bounds)):
        image = [
            (
                middle - Fraction(high) - Fraction(width),
                middle - Fraction(low) + Fraction(width),
            )
            for middle, low, high, width in zip(
                exact_center,
                shift_lower[:, 0].tolist(),
                shift_upper[:, 0].tolist(),
                spread[:, 0].tolist(),
                strict=True,
            )
        ]
        unique = all(
            lower < image_lower and image_upper < upper
            for (lower, upper), (image_lower, image_upper) in zip(box, image, strict=True)
        )
        image_lower_ = [_outward_float(lower, upward=False) for lower, _ in image]
        image_upper_ = [_outward_float(upper, upward=True) for _, upper in image]
    else:
        unique = False
        image_lower_ = [-math.inf] * dimension
        image_upper_ = [math.inf] * dimension
    payload = {
        "kind": "krawczyk-certificate",
        "system_id": system.system_id,
        "center": center_.tolist(),
        "radius": radius_.tolist(),
    }
    return KrawczykCertificate(
        jnp.asarray(center_),
        jnp.asarray(radius_),
        jnp.asarray(image_lower_, dtype=jnp.float64),
        jnp.asarray(image_upper_, dtype=jnp.float64),
        unique,
        canonical_fingerprint(payload),
        dense_workspace_bytes,
        maximum_dense_workspace_bytes,
    )


@dataclass(frozen=True, slots=True)
class ExactRealRootInterval:
    lower: Fraction
    upper: Fraction
    multiplicity: int


def isolate_univariate_real_roots(
    coefficients: tuple[int | Fraction, ...],
    /,
    *,
    tolerance: Fraction = Fraction(1, 10**12),
) -> tuple[ExactRealRootInterval, ...]:
    """Return exact rational isolating intervals for a univariate polynomial."""

    if not coefficients or all(value == 0 for value in coefficients):
        raise ValueError("A nonzero coefficient sequence is required.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    return tuple(
        ExactRealRootInterval(lower, upper, multiplicity)
        for lower, upper, multiplicity in isolate_real_roots(
            coefficients,
            tolerance=tolerance,
        )
    )


__all__ = [
    "ExactRealRootInterval",
    "KrawczykCertificate",
    "SmaleAlphaCertificate",
    "isolate_univariate_real_roots",
    "krawczyk_certificate",
    "smale_alpha_certificate",
]
