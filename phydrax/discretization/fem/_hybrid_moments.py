# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exterior-closed, zero-trace body moment tests for compatible hybrid cells."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import factorial

import numpy as np

from . import _form_elements as forms
from ._hybrid_forms import (
    _coefficient_array,
    _derivative,
    _independent_columns,
    _sum_polynomials,
    HybridPreparation,
    prepare_hybrid,
)


type _SourceSupport = tuple[
    tuple[tuple[tuple[int, ...], tuple[tuple[int, Fraction], ...]], ...], ...
]


def _body_bubble(family: forms.FormElementFamily) -> forms._Polynomial:
    zero = (0, 0, 0)
    variables: tuple[forms._Polynomial, ...] = tuple(
        {tuple(int(axis == column) for column in range(3)): 1.0} for axis in range(3)
    )
    one: forms._Polynomial = {zero: 1.0}
    x, y, z = variables
    height = forms._multiply(z, _sum_polynomials((one, z), (1.0, -1.0)))
    if family == "prism-trimmed":
        triangle = forms._multiply(
            forms._multiply(x, y), _sum_polynomials((one, x, y), (1.0, -1.0, -1.0))
        )
        return {
            alpha: 108.0 * value
            for alpha, value in forms._multiply(triangle, height).items()
        }
    base = forms._multiply(
        forms._multiply(x, _sum_polynomials((one, x), (1.0, -1.0))),
        forms._multiply(y, _sum_polynomials((one, y), (1.0, -1.0))),
    )
    return {alpha: 64.0 * value for alpha, value in forms._multiply(base, height).items()}


def _density_derivatives(
    higher: HybridPreparation, k: int
) -> tuple[tuple[forms._Polynomial, ...], ...]:
    higher_degree = k + 1
    field_blades = tuple(combinations(range(3), higher_degree))
    test_blades = tuple(combinations(range(3), 3 - higher_degree))
    output_blades = tuple(combinations(range(3), 3 - k))
    lower_blades = tuple(combinations(range(3), k))
    coefficients = higher.body_test_coefficients
    tests = []
    for column in range(coefficients.shape[-1]):
        derivative: list[forms._Polynomial] = [{} for _ in output_blades]
        for blade in test_blades:
            complement = tuple(axis for axis in range(3) if axis not in blade)
            field_component = field_blades.index(complement)
            sign = forms._wedge_sign(complement, blade)
            polynomial = {
                alpha: sign * float(coefficients[row, field_component, column])
                for row, alpha in enumerate(higher.body_test_exponents)
                if coefficients[row, field_component, column]
            }
            for axis in range(3):
                if axis in blade:
                    continue
                output = tuple(sorted((axis, *blade)))
                partial = _derivative(polynomial, axis)
                index = output_blades.index(output)
                derivative[index] = _sum_polynomials(
                    (derivative[index], partial),
                    (1.0, float(forms._wedge_sign((axis,), blade))),
                )
        density = tuple(
            _sum_polynomials(
                (
                    derivative[
                        output_blades.index(
                            tuple(axis for axis in range(3) if axis not in blade)
                        )
                    ],
                ),
                (
                    float(
                        forms._wedge_sign(
                            blade, tuple(axis for axis in range(3) if axis not in blade)
                        )
                    ),
                ),
            )
            for blade in lower_blades
        )
        if any(density):
            tests.append(density)
    if not tests:
        return ()
    _, candidate = _coefficient_array(tuple(tests))
    independent = _independent_columns(candidate.reshape(-1, len(tests)))
    return tuple(tests[index] for index in independent)


@lru_cache(maxsize=65536)
def _body_monomial_integral(
    family: forms.FormElementFamily, alpha: tuple[int, ...]
) -> Fraction:
    if family == "prism-trimmed":
        return Fraction(
            factorial(alpha[0]) * factorial(alpha[1]),
            factorial(alpha[0] + alpha[1] + 2) * (alpha[2] + 1),
        )
    return Fraction(1, (alpha[0] + 1) * (alpha[1] + 1) * (alpha[2] + 1))


def _density_moment_row(
    density: tuple[forms._Polynomial, ...],
    support: _SourceSupport,
    width: int,
    family: forms.FormElementFamily,
) -> forms._HostArray:
    # Exact source pairing avoids cancellation of expanded bubble factors
    # before the native numerical dual solve. Only the final entries round.
    row = [Fraction(0)] * width
    for component, polynomial in enumerate(density):
        for alpha, coefficient in polynomial.items():
            for beta, values in support[component]:
                integral = Fraction(coefficient) * _body_monomial_integral(
                    family, tuple(a + b for a, b in zip(alpha, beta, strict=True))
                )
                for index, value in values:
                    row[index] += integral * value
    return np.asarray(row, dtype=np.float64)


@lru_cache(maxsize=1024)
def _orthogonal_test_polynomial(alpha: tuple[int, ...]) -> forms._Polynomial:
    from .._coordinate_enclosure import _jacobi, math_product_polynomials

    modes = tuple(_jacobi(degree, 0, axis, 3) for axis, degree in enumerate(alpha))
    polynomial = math_product_polynomials(modes, 3)
    return {power: float(value) for power, value in polynomial.items()}


def _admit_moment_row(row: forms._HostArray, orthogonal: list[forms._HostArray]) -> bool:
    residual = row.copy()
    for _ in range(2):
        for basis in orthogonal:
            residual -= basis * (basis @ residual)
    norm = np.linalg.norm(residual)
    tolerance = 2048 * np.finfo(np.float64).eps * max(np.linalg.norm(row), 1.0)
    if norm <= tolerance:
        return False
    orthogonal.append(residual / norm)
    return True


def body_moments(
    family: forms.FormElementFamily,
    k: int,
    r: int,
    exponents: tuple[tuple[int, ...], ...],
    generators: forms._HostArray,
    boundary: forms._HostArray,
    source_bank: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[tuple[int, ...], ...], forms._HostArray]:
    """Construct a zero-trace, exterior-closed dual test complex.

    The sole nonzero boundary restriction is the top scalar constant, whose
    Stokes functional is total boundary flux. All remaining tests vanish on
    the boundary as differential forms. This makes d Pi = Pi d an actual
    smooth-field identity, not just nilpotence on represented polynomials.
    """
    width, count = generators.shape[-1], generators.shape[1]
    support: _SourceSupport = tuple(
        tuple(
            (
                alpha,
                tuple(
                    (index, value)
                    for index, value in enumerate(source_bank[row * count + component])
                    if value
                ),
            )
            for row, alpha in enumerate(exponents)
            if any(source_bank[row * count + component])
        )
        for component in range(count)
    )
    orthogonal: list[forms._HostArray] = []
    for row in boundary:
        if not _admit_moment_row(row, orthogonal):
            raise ValueError("Hybrid trace moments are deficient or not surjective.")
    selected: list[tuple[forms._Polynomial, ...]] = []
    required = width - len(boundary)
    initial: tuple[tuple[forms._Polynomial, ...], ...] = (
        (({(0, 0, 0): 1.0},),)
        if k == 3
        else _density_derivatives(prepare_hybrid(family, k + 1, r), k)
    )
    for density in initial:
        row = _density_moment_row(density, support, width, family)
        if not _admit_moment_row(row, orthogonal):
            raise ValueError(
                "Exterior-derived hybrid moment tests are not independently unisolvent."
            )
        selected.append(density)
    bubble = _body_bubble(family)
    for alpha in forms._exponents(3, max(sum(alpha) for alpha in exponents)):
        if len(selected) == required:
            break
        polynomial = forms._multiply(bubble, _orthogonal_test_polynomial(alpha))
        for component in range(count):
            density = tuple(
                polynomial if index == component else {} for index in range(count)
            )
            row = _density_moment_row(density, support, width, family)
            if _admit_moment_row(row, orthogonal):
                selected.append(density)
            if len(selected) == required:
                break
    if len(selected) != required:
        raise ValueError("Hybrid exterior-closed body moments are rank deficient.")
    if not selected:
        return (), np.zeros((0, count, 0), dtype=np.float64)
    test_exponents, coefficients = _coefficient_array(tuple(selected))
    return test_exponents, coefficients
