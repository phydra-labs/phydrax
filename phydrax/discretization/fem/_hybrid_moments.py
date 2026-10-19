# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exterior-closed, zero-trace body moment tests for compatible hybrid cells."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import factorial

import numpy as np

from ..._host_table_cache import host_table_cache, logical_table_bytes
from ...typing import checked
from .. import (
    _coordinate_enclosure as algebra,
    _coordinate_enclosure as imported_coordinate_enclosure,
)
from .._coordinate_budget import CoordinateEnclosureBudget
from . import _form_elements as forms, _hybrid_forms as hybrid_forms_owner
from ._form_basis_types import FormElementFamily


type _SourceSupport = tuple[
    tuple[tuple[tuple[int, ...], tuple[tuple[int, Fraction], ...]], ...], ...
]


def _body_bubble(family: FormElementFamily) -> forms.FormPolynomial:
    zero = (0, 0, 0)
    variables: tuple[forms.FormPolynomial, ...] = tuple(
        {tuple(int(axis == column) for column in range(3)): 1.0} for axis in range(3)
    )
    one: forms.FormPolynomial = {zero: 1.0}
    x, y, z = variables
    height = forms.multiply_form_polynomials(
        z, hybrid_forms_owner.sum_polynomials((one, z), (1.0, -1.0))
    )
    if family == "prism-trimmed":
        triangle = forms.multiply_form_polynomials(
            forms.multiply_form_polynomials(x, y),
            hybrid_forms_owner.sum_polynomials((one, x, y), (1.0, -1.0, -1.0)),
        )
        return {
            alpha: 108.0 * value
            for alpha, value in forms.multiply_form_polynomials(triangle, height).items()
        }
    base = forms.multiply_form_polynomials(
        forms.multiply_form_polynomials(
            x, hybrid_forms_owner.sum_polynomials((one, x), (1.0, -1.0))
        ),
        forms.multiply_form_polynomials(
            y, hybrid_forms_owner.sum_polynomials((one, y), (1.0, -1.0))
        ),
    )
    return {
        alpha: 64.0 * value
        for alpha, value in forms.multiply_form_polynomials(base, height).items()
    }


def _density_derivatives(
    higher: hybrid_forms_owner.HybridPreparation, k: int
) -> tuple[tuple[forms.FormPolynomial, ...], ...]:
    higher_degree = k + 1
    field_blades = tuple(combinations(range(3), higher_degree))
    test_blades = tuple(combinations(range(3), 3 - higher_degree))
    output_blades = tuple(combinations(range(3), 3 - k))
    lower_blades = tuple(combinations(range(3), k))
    coefficients = higher.body_test_coefficients
    tests = []
    for column in range(coefficients.shape[-1]):
        derivative: list[forms.FormPolynomial] = [{} for _ in output_blades]
        for blade in test_blades:
            complement = tuple(axis for axis in range(3) if axis not in blade)
            field_component = field_blades.index(complement)
            sign = forms.form_wedge_sign(complement, blade)
            polynomial = {
                alpha: sign * float(coefficients[row, field_component, column])
                for row, alpha in enumerate(higher.body_test_exponents)
                if coefficients[row, field_component, column]
            }
            for axis in range(3):
                if axis in blade:
                    continue
                output = tuple(sorted((axis, *blade)))
                partial = hybrid_forms_owner.derivative(polynomial, axis)
                index = output_blades.index(output)
                derivative[index] = hybrid_forms_owner.sum_polynomials(
                    (derivative[index], partial),
                    (1.0, float(forms.form_wedge_sign((axis,), blade))),
                )
        density = tuple(
            hybrid_forms_owner.sum_polynomials(
                (
                    derivative[
                        output_blades.index(
                            tuple(axis for axis in range(3) if axis not in blade)
                        )
                    ],
                ),
                (
                    float(
                        forms.form_wedge_sign(
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
    _, candidate = hybrid_forms_owner.coefficient_array(tuple(tests))
    independent = hybrid_forms_owner.independent_columns(
        candidate.reshape(-1, len(tests))
    )
    return tuple(tests[index] for index in independent)


def _body_monomial_integral_bytes(value: Fraction, /) -> int:
    return logical_table_bytes(value.numerator) + logical_table_bytes(value.denominator)


@host_table_cache(
    max_entries=65536, max_bytes=1048576, weight=_body_monomial_integral_bytes
)
def _body_monomial_integral(
    family: FormElementFamily, alpha: tuple[int, ...]
) -> Fraction:
    if family == "prism-trimmed":
        return Fraction(
            factorial(alpha[0]) * factorial(alpha[1]),
            factorial(alpha[0] + alpha[1] + 2) * (alpha[2] + 1),
        )
    return Fraction(1, (alpha[0] + 1) * (alpha[1] + 1) * (alpha[2] + 1))


def _density_moment_row(
    density: tuple[forms.FormPolynomial, ...],
    support: _SourceSupport,
    width: int,
    family: FormElementFamily,
) -> forms.HostFormArray:
    # algebraic source pairing avoids cancellation of expanded bubble factors
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


def _orthogonal_test_polynomial(
    alpha: tuple[int, ...], *, budget: CoordinateEnclosureBudget
) -> forms.FormPolynomial:

    modes = tuple(
        imported_coordinate_enclosure.jacobi(degree, 0, axis, 3, budget=budget)
        for axis, degree in enumerate(alpha)
    )
    polynomial = imported_coordinate_enclosure.math_product_polynomials(
        modes, 3, budget=budget
    )
    return {power: float(value) for power, value in polynomial.items()}


def _admit_moment_row(
    row: forms.HostFormArray, orthogonal: list[forms.HostFormArray]
) -> bool:
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


@checked
def body_moments(
    family: FormElementFamily,
    k: int,
    r: int,
    exponents: tuple[tuple[int, ...], ...],
    generators: forms.HostFormArray,
    boundary: forms.HostFormArray,
    source_bank: tuple[tuple[Fraction, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
    preparations: dict[
        tuple[FormElementFamily, int, int], hybrid_forms_owner.HybridPreparation
    ],
) -> tuple[tuple[tuple[int, ...], ...], forms.HostFormArray]:
    """Construct a zero-trace, exterior-closed dual test complex.

    The sole nonzero boundary restriction is the top scalar constant, whose
    Stokes functional is total boundary flux. All remaining tests vanish on
    the boundary as differential forms. This makes d Pi = Pi d an represented
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
    orthogonal: list[forms.HostFormArray] = []
    for row in boundary:
        if not _admit_moment_row(row, orthogonal):
            raise ValueError(
                "body_moments.configuration must pass the declared scientific certificate; got Hybrid trace moments are deficient or not surjective."
            )
    selected: list[tuple[forms.FormPolynomial, ...]] = []
    required = width - len(boundary)
    initial: tuple[tuple[forms.FormPolynomial, ...], ...] = (
        (({(0, 0, 0): 1.0},),)
        if k == 3
        else _density_derivatives(
            hybrid_forms_owner.prepare_hybrid_active(
                family, k + 1, r, budget=budget, preparations=preparations
            ),
            k,
        )
    )
    for density in initial:
        row = _density_moment_row(density, support, width, family)
        if not _admit_moment_row(row, orthogonal):
            raise ValueError(
                "body_moments.configuration must have independent unisolvent moment rows; got Exterior-derived hybrid moment tests are not independently unisolvent."
            )
        selected.append(density)
    bubble = _body_bubble(family)
    for alpha in forms.form_exponents(3, max(sum(alpha) for alpha in exponents)):
        if len(selected) == required:
            break
        polynomial = forms.multiply_form_polynomials(
            bubble, _orthogonal_test_polynomial(alpha, budget=budget)
        )
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
        raise ValueError(
            "body_moments.configuration must have full rank; got Hybrid exterior-closed body moments are rank deficient."
        )
    if not selected:
        return (), np.zeros((0, count, 0), dtype=np.float64)
    test_exponents, coefficients = hybrid_forms_owner.coefficient_array(tuple(selected))
    return test_exponents, coefficients


def body_moments_estimate(
    exponents: tuple[tuple[int, ...], ...],
) -> algebra.CoordinateEnclosureEstimate:
    """Bound every admissible Jacobi candidate, including rejected moment rows."""
    maximum_degree = max(sum(alpha) for alpha in exponents)
    return algebra.coordinate_estimate_bank(
        tuple(
            algebra.coordinate_estimate_product(
                tuple(
                    algebra.coordinate_jacobi_estimate(degree, 0, axis, 3)
                    for axis, degree in enumerate(alpha)
                ),
                3,
            )
            for alpha in forms.form_exponents(3, maximum_degree)
        )
    )
