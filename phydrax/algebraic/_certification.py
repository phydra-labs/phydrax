#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Alpha-theory, interval Krawczyk, and exact univariate root certificates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial, sqrt

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    solve,
)
from ._system import SparsePolynomialSystem


_SMALE_ALPHA_THRESHOLD = (13.0 - 3.0 * sqrt(17.0)) / 4.0


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
    """Certify a square regular polynomial point with a conservative gamma bound."""

    if not isinstance(system, SparsePolynomialSystem):
        raise TypeError("system must be SparsePolynomialSystem.")
    point_ = jnp.asarray(point)
    if point_.shape != (system.support.variable_count,):
        raise ValueError("Alpha-theory point has the wrong variable dimension.")
    if system.support.equation_count != system.support.variable_count:
        raise ValueError("Alpha theory requires a square polynomial system.")
    jacobian = system.jacobian(point_)
    residual = system.evaluate(point_)
    operator = DenseLinearOperator(
        jacobian,
        properties=OperatorProperties(square=True),
    )
    prepared = prepare(
        LinearSystem(operator),
        LinearSolvePolicy(DenseLU()),
    )
    residual_solve = solve(prepared, residual)
    beta = jnp.linalg.norm(residual_solve.value)
    maximum_degree = int(np.max(np.asarray(system.support.exponents)))
    total_degree = int(np.max(np.sum(np.asarray(system.support.exponents), axis=1)))
    degree = max(maximum_degree, total_degree)
    derivative = system.evaluate
    gamma_candidates = []
    for order in range(1, degree + 1):
        derivative = jax.jacfwd(derivative)
        if order < 2:
            continue
        tensor = derivative(point_)
        columns = tensor.reshape((tensor.shape[0], -1)).T
        solved = jax.vmap(lambda column: solve(prepared, column).value)(columns)
        transformed = solved.T.reshape(tensor.shape)
        bound = jnp.linalg.norm(transformed) / factorial(order)
        gamma_candidates.append(bound ** (1.0 / (order - 1)))
    gamma = (
        jnp.max(jnp.stack(gamma_candidates))
        if gamma_candidates
        else jnp.asarray(0.0, dtype=point_.real.dtype)
    )
    alpha = beta * gamma
    finite = (
        residual_solve.successful
        & jnp.isfinite(beta)
        & jnp.isfinite(gamma)
        & jnp.isfinite(alpha)
    )
    approximate = bool(finite & (alpha <= _SMALE_ALPHA_THRESHOLD))
    payload = {
        "kind": "smale-alpha-certificate",
        "system_id": system.system_id,
        "point": np.asarray(point_).tolist(),
        "threshold": _SMALE_ALPHA_THRESHOLD,
    }
    return SmaleAlphaCertificate(
        point_,
        beta,
        gamma,
        alpha,
        approximate,
        canonical_fingerprint(payload),
    )


def _power_interval(lower: float, upper: float, exponent: int, /) -> tuple[float, float]:
    if exponent == 0:
        return 1.0, 1.0
    if exponent % 2 == 0 and lower <= 0.0 <= upper:
        return 0.0, max(abs(lower), abs(upper)) ** exponent
    values = (lower**exponent, upper**exponent)
    return min(values), max(values)


def _multiply_interval(left, right, /):
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def _monomial_interval(lower, upper, exponents, /):
    result = (1.0, 1.0)
    for low, high, exponent in zip(lower, upper, exponents, strict=True):
        result = _multiply_interval(
            result, _power_interval(float(low), float(high), int(exponent))
        )
    return result


def _jacobian_interval(system, lower, upper, /):
    support = system.support
    coefficients = np.asarray(system.coefficients)
    if np.iscomplexobj(coefficients):
        raise ValueError("Krawczyk certification currently requires real coefficients.")
    low = np.zeros((support.equation_count, support.variable_count), dtype=np.float64)
    high = np.zeros_like(low)
    equations = np.asarray(support.equation_indices)
    exponents = np.asarray(support.exponents)
    for equation, exponent, coefficient in zip(
        equations, exponents, coefficients, strict=True
    ):
        for variable in range(support.variable_count):
            power = int(exponent[variable])
            if power == 0:
                continue
            derivative_exponent = exponent.copy()
            derivative_exponent[variable] -= 1
            interval = _monomial_interval(lower, upper, derivative_exponent)
            scaled = (
                coefficient * power * interval[0],
                coefficient * power * interval[1],
            )
            low[equation, variable] += min(scaled)
            high[equation, variable] += max(scaled)
    return low, high


@dataclass(frozen=True, slots=True)
class KrawczykCertificate:
    center: Array
    radius: Array
    image_lower: Array
    image_upper: Array
    unique_root: bool
    certificate_id: str


def krawczyk_certificate(
    system: SparsePolynomialSystem,
    center: ArrayLike,
    radius: ArrayLike,
    /,
) -> KrawczykCertificate:
    """Certify one unique real root in an axis-aligned polynomial box."""

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
    lower = center_ - radius_
    upper = center_ + radius_
    center_jacobian = np.asarray(system.jacobian(jnp.asarray(center_)), dtype=jnp.float64)
    inverse = np.linalg.solve(center_jacobian, np.eye(center_jacobian.shape[0]))
    residual = np.asarray(system.evaluate(jnp.asarray(center_)), dtype=jnp.float64)
    jacobian_low, jacobian_high = _jacobian_interval(system, lower, upper)
    matrix_low = np.eye(center_.size) - inverse @ jacobian_high
    matrix_high = np.eye(center_.size) - inverse @ jacobian_low
    matrix_lower = np.minimum(matrix_low, matrix_high)
    matrix_upper = np.maximum(matrix_low, matrix_high)
    displacement = (-radius_, radius_)
    image_center = center_ - inverse @ residual
    image_low = image_center.copy()
    image_high = image_center.copy()
    for row in range(center_.size):
        for column in range(center_.size):
            contribution = _multiply_interval(
                (matrix_lower[row, column], matrix_upper[row, column]),
                (displacement[0][column], displacement[1][column]),
            )
            image_low[row] += contribution[0]
            image_high[row] += contribution[1]
    unique = bool(np.all(image_low > lower) and np.all(image_high < upper))
    payload = {
        "kind": "krawczyk-certificate",
        "system_id": system.system_id,
        "center": center_.tolist(),
        "radius": radius_.tolist(),
    }
    return KrawczykCertificate(
        jnp.asarray(center_),
        jnp.asarray(radius_),
        jnp.asarray(image_low),
        jnp.asarray(image_high),
        unique,
        canonical_fingerprint(payload),
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
    variable = sp.Symbol("x")
    expression = sum(
        sp.Rational(value.numerator, value.denominator) * variable**index
        if isinstance(value, Fraction)
        else sp.Integer(value) * variable**index
        for index, value in enumerate(coefficients)
    )
    polynomial = sp.Poly(expression, variable, domain=sp.QQ)
    intervals = polynomial.intervals(
        eps=sp.Rational(tolerance.numerator, tolerance.denominator)
    )
    result = []
    for (lower, upper), multiplicity in intervals:
        result.append(
            ExactRealRootInterval(
                Fraction(int(lower.p), int(lower.q)),
                Fraction(int(upper.p), int(upper.q)),
                int(multiplicity),
            )
        )
    return tuple(result)


__all__ = [
    "ExactRealRootInterval",
    "KrawczykCertificate",
    "SmaleAlphaCertificate",
    "isolate_univariate_real_roots",
    "krawczyk_certificate",
    "smale_alpha_certificate",
]
