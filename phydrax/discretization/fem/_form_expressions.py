# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Construct source expressions for form components and entity moments."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb

import numpy as np

from ..._strict import StrictModule
from ...typing import checked
from .. import _coordinate_enclosure as expression
from .._coordinate_budget import CoordinateEnclosureBudget
from . import _form_elements as forms, _hybrid_forms as hybrid_forms_owner
from ._form_basis_types import (
    AbstractReferenceFormBasis,
    AbstractReferenceHybridFactors,
    AbstractReferenceTensorFactors,
)


def _collapsed_polynomial(
    polynomial: expression.Polynomial, *, budget: CoordinateEnclosureBudget
) -> expression.Expression:
    x, y, z = expression.axes(3, budget=budget)
    s = expression.add(
        expression.constant(1, 3, budget=budget),
        expression.scale(z, -1, budget=budget),
        budget=budget,
    )
    arguments = (
        expression.add(
            x, expression.scale(z, Fraction(-1, 2), budget=budget), budget=budget
        ),
        expression.add(
            y, expression.scale(z, Fraction(-1, 2), budget=budget), budget=budget
        ),
        z,
    )
    degree = max((alpha[0] + alpha[1] for alpha in polynomial), default=0)
    numerator = expression.sum_polynomials(
        tuple(
            expression.scale(
                expression.multiply(
                    expression.math_product_polynomials(
                        tuple(
                            expression.power(variable, power, 3, budget=budget)
                            for variable, power in zip(arguments, alpha, strict=True)
                        ),
                        3,
                        budget=budget,
                    ),
                    expression.power(s, degree - alpha[0] - alpha[1], 3, budget=budget),
                    budget=budget,
                ),
                coefficient,
                budget=budget,
            )
            for alpha, coefficient in polynomial.items()
        ),
        budget=budget,
    )
    return expression.rational_expression(
        numerator, expression.power(s, degree, 3, budget=budget), budget=budget
    )


def _pyramid_component_action(
    k: int, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[expression.Expression, ...], ...]:
    x, y, z = expression.axes(3, budget=budget)
    one = expression.constant(1, 3, budget=budget)
    s = expression.add(one, expression.scale(z, -1, budget=budget), budget=budget)
    a = expression.add(
        x, expression.constant(Fraction(-1, 2), 3, budget=budget), budget=budget
    )
    b = expression.add(
        y, expression.constant(Fraction(-1, 2), 3, budget=budget), budget=budget
    )
    inv = expression.rational_expression(one, s, budget=budget)
    inv2 = expression.rational_expression(
        one, expression.power(s, 2, 3, budget=budget), budget=budget
    )
    if k == 0:
        return ((one,),)
    if k == 1:
        return (
            (inv, {}, {}),
            ({}, inv, {}),
            (
                expression.rational_expression(
                    a, expression.power(s, 2, 3, budget=budget), budget=budget
                ),
                expression.rational_expression(
                    b, expression.power(s, 2, 3, budget=budget), budget=budget
                ),
                one,
            ),
        )
    if k == 2:
        return (
            (inv2, {}, {}),
            (
                expression.rational_expression(
                    b, expression.power(s, 3, 3, budget=budget), budget=budget
                ),
                inv,
                {},
            ),
            (
                expression.rational_expression(
                    expression.scale(a, -1, budget=budget),
                    expression.power(s, 3, 3, budget=budget),
                    budget=budget,
                ),
                {},
                inv,
            ),
        )
    if k == 3:
        return ((inv2,),)
    raise ValueError(
        "pyramid_form.degree must lie between zero and three; got an unsupported degree"
    )


def _tensor_components(
    basis: AbstractReferenceFormBasis,
    factors: AbstractReferenceTensorFactors,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[expression.Expression, ...], ...]:
    n = basis.dimension
    variables = expression.axes(n, budget=budget)
    zero, one = np.asarray(factors.zero), np.asarray(factors.one)
    indices, differentials, components = (
        np.asarray(factors.indices),
        np.asarray(factors.differential_axes),
        np.asarray(factors.components),
    )
    axial = []
    for axis, variable in enumerate(variables):
        collapse = expression.add(
            expression.constant(1, n, budget=budget),
            expression.scale(variable, -1, budget=budget),
            budget=budget,
        )
        bubble = expression.multiply(variable, collapse, budget=budget)
        modes = tuple(
            expression.jacobi(order, 0, axis, n, budget=budget)
            for order in range(basis.order)
        )
        zero_modes = tuple(
            expression.add(
                expression.add(
                    expression.scale(
                        collapse, Fraction(float(zero[0, dof])), budget=budget
                    ),
                    expression.scale(
                        variable, Fraction(float(zero[1, dof])), budget=budget
                    ),
                    budget=budget,
                ),
                expression.multiply(
                    bubble,
                    expression.sum_polynomials(
                        tuple(
                            expression.scale(mode, Fraction(float(value)), budget=budget)
                            for mode, value in zip(modes[:-1], zero[2:, dof], strict=True)
                        ),
                        budget=budget,
                    ),
                    budget=budget,
                ),
                budget=budget,
            )
            for dof in range(zero.shape[1])
        )
        one_modes = tuple(
            expression.sum_polynomials(
                tuple(
                    expression.scale(mode, Fraction(float(value)), budget=budget)
                    for mode, value in zip(modes, one[:, dof], strict=True)
                ),
                budget=budget,
            )
            for dof in range(one.shape[1])
        )
        axial.append((zero_modes, one_modes))
    result = []
    for dof in range(basis.local_dof_count):
        scalar = expression.math_product_polynomials(
            tuple(
                axial[axis][int(differentials[axis, dof])][indices[axis, dof]]
                for axis in range(n)
            ),
            n,
            budget=budget,
        )
        result.append(
            tuple(
                expression.scale(scalar, Fraction(float(value)), budget=budget)
                for value in components[dof]
            )
        )
    return tuple(result)


def _admit_hybrid_source(
    basis: AbstractReferenceFormBasis,
    factors: AbstractReferenceHybridFactors,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Check numerical leaves against the declared moment-dual source."""
    with budget.temporary_scope():
        prepared = hybrid_forms_owner.prepare_hybrid(
            basis.family, basis.form_degree, basis.order, budget=budget
        )
        if (
            factors.source_bank != prepared.source_bank
            or factors.body_test_exponents != prepared.body_test_exponents
            or factors.body_test_source_bank != prepared.body_test_source_bank
            or basis.exponents != prepared.prepared.exponents
            or basis.dof_labels != prepared.prepared.labels
            or not np.array_equal(np.asarray(basis.coefficients), prepared.coefficients)
        ):
            raise ValueError(
                "form_basis.source must match its declared moment-dual identity; got different coefficients or metadata"
            )


@checked
def component_expressions(
    basis: AbstractReferenceFormBasis,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[expression.Expression, ...], ...]:
    tensor = basis.tensor_factors
    if tensor is not None:
        return _tensor_components(basis, tensor, budget=budget)
    factors = basis.hybrid_factors
    if factors is None:
        coefficients = np.asarray(basis.coefficients)
        return tuple(
            tuple(
                {
                    alpha: Fraction(float(value))
                    for alpha, value in zip(
                        basis.exponents, coefficients[:, component, dof], strict=True
                    )
                    if value
                }
                for component in range(coefficients.shape[1])
            )
            for dof in range(basis.local_dof_count)
        )
    _admit_hybrid_source(basis, factors, budget=budget)
    generators = np.asarray(factors.generators)
    expected = np.asarray(factors.source_bank, dtype=np.float64).reshape(generators.shape)
    if not np.array_equal(generators, expected):
        raise ValueError(
            "form_basis.generators must match the declared source bank; got different coefficients"
        )
    count = generators.shape[1]
    source = tuple(
        tuple(
            {
                alpha: factors.source_bank[row * count + component][generator]
                for row, alpha in enumerate(basis.exponents)
                if factors.source_bank[row * count + component][generator]
            }
            for component in range(count)
        )
        for generator in range(generators.shape[-1])
    )
    dual = np.asarray(basis.coefficients)
    # Rational linear combination precedes the collapsed chart. Summing
    # rational generators instead would repeatedly multiply identical collapse
    # denominators and consume storage unrelated to the represented source.
    action = (
        _pyramid_component_action(basis.form_degree, budget=budget)
        if basis.family == "pyramid-trimmed"
        else None
    )

    def field_expressions(dof: int) -> tuple[expression.Expression, ...]:
        combined = tuple(
            expression.sum_polynomials(
                tuple(
                    expression.scale(
                        field[component],
                        Fraction(float(dual[generator, dof])),
                        budget=budget,
                    )
                    for generator, field in enumerate(source)
                    if dual[generator, dof] != 0
                ),
                budget=budget,
            )
            for component in range(count)
        )
        if action is None:
            return combined
        collapsed = tuple(
            _collapsed_polynomial(component, budget=budget) for component in combined
        )
        return tuple(
            expression.expression_sum(
                tuple(
                    expression.expression_multiply(entry, component, budget=budget)
                    for entry, component in zip(row, collapsed, strict=True)
                    if entry and component
                ),
                budget=budget,
            )
            for row in action
        )

    result: list[tuple[expression.Expression, ...]] = []
    for dof in range(basis.local_dof_count):
        with budget.temporary_scope():
            field = field_expressions(dof)
            budget.retain_basis(field)
            result.append(field)
    return tuple(result)


def _pyramid_density_action(
    k: int, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[expression.Expression, ...], ...]:
    x, y, z = expression.axes(3, budget=budget)
    one = expression.constant(1, 3, budget=budget)
    s = expression.add(one, expression.scale(z, -1, budget=budget), budget=budget)
    a = expression.add(
        expression.constant(Fraction(1, 2), 3, budget=budget),
        expression.scale(x, -1, budget=budget),
        budget=budget,
    )
    b = expression.add(
        expression.constant(Fraction(1, 2), 3, budget=budget),
        expression.scale(y, -1, budget=budget),
        budget=budget,
    )
    inv = expression.rational_expression(one, s, budget=budget)
    inv2 = expression.rational_expression(
        one, expression.power(s, 2, 3, budget=budget), budget=budget
    )
    if k == 0:
        return ((inv2,),)
    if k == 1:
        return (
            (inv, {}, {}),
            ({}, inv, {}),
            (
                expression.rational_expression(
                    a, expression.power(s, 3, 3, budget=budget), budget=budget
                ),
                expression.rational_expression(
                    b, expression.power(s, 3, 3, budget=budget), budget=budget
                ),
                inv2,
            ),
        )
    if k == 2:
        return (
            (one, {}, {}),
            (
                expression.rational_expression(
                    b, expression.power(s, 2, 3, budget=budget), budget=budget
                ),
                inv,
                {},
            ),
            (
                expression.rational_expression(
                    expression.scale(a, -1, budget=budget),
                    expression.power(s, 2, 3, budget=budget),
                    budget=budget,
                ),
                {},
                inv,
            ),
        )
    if k == 3:
        return ((one,),)
    raise ValueError(
        "pyramid_form.degree must lie between zero and three; got an unsupported degree"
    )


class _EntityDensities(StrictModule):
    indices: tuple[int, ...]
    values: tuple[tuple[expression.Expression, ...], ...]


def _hybrid_body_densities(
    basis: AbstractReferenceFormBasis,
    factors: AbstractReferenceHybridFactors,
    indices: tuple[int, ...],
    blades: tuple[tuple[int, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> _EntityDensities:
    action = (
        _pyramid_density_action(basis.form_degree, budget=budget)
        if basis.family == "pyramid-trimmed"
        else tuple(
            tuple(
                expression.constant(int(i == j), 3, budget=budget)
                for j in range(len(blades))
            )
            for i in range(len(blades))
        )
    )
    coefficients = np.asarray(factors.body_test_coefficients)
    expected = np.asarray(factors.body_test_source_bank, dtype=np.float64).reshape(
        coefficients.shape
    )
    if not np.array_equal(coefficients, expected):
        raise ValueError(
            "form_basis.body_tests must match the declared source bank; got different coefficients"
        )
    values = []
    for column in range(len(indices)):
        polynomials: tuple[expression.Polynomial, ...] = tuple(
            {
                alpha: factors.body_test_source_bank[row * len(blades) + component][
                    column
                ]
                for row, alpha in enumerate(factors.body_test_exponents)
                if factors.body_test_source_bank[row * len(blades) + component][column]
            }
            for component in range(len(blades))
        )
        physical = (
            tuple(
                _collapsed_polynomial(polynomial, budget=budget)
                for polynomial in polynomials
            )
            if basis.family == "pyramid-trimmed"
            else polynomials
        )
        values.append(
            tuple(
                expression.expression_sum(
                    tuple(
                        expression.expression_multiply(
                            polynomial, action[component][ambient], budget=budget
                        )
                        for component, polynomial in enumerate(physical)
                    ),
                    budget=budget,
                )
                for ambient in range(len(blades))
            )
        )
    return _EntityDensities(indices, tuple(values))


def _hybrid_trace_densities(
    basis: AbstractReferenceFormBasis,
    face: tuple[int, ...],
    indices: tuple[int, ...],
    blades: tuple[tuple[int, ...], ...],
    jacobian: forms.HostFormArray,
    *,
    budget: CoordinateEnclosureBudget,
) -> _EntityDensities:
    trace = basis.entity_basis(face, budget=budget)
    interior = forms.interior_dof_indices(trace.dof_labels, len(face))
    density = functional_density_expressions(
        trace, trace.entity_vertices[-1][0], budget=budget
    )
    local_blades = tuple(combinations(range(jacobian.shape[1]), basis.form_degree))
    pullback = tuple(
        tuple(Fraction(forms.form_minor(jacobian, ambient, local)) for ambient in blades)
        for local in local_blades
    )
    values = tuple(
        tuple(
            expression.expression_sum(
                tuple(
                    expression.expression_scale(
                        value, pullback[component][ambient], budget=budget
                    )
                    for component, value in enumerate(density[trace_index])
                ),
                budget=budget,
            )
            for ambient in range(len(blades))
        )
        for _, trace_index in zip(indices, interior, strict=True)
    )
    return _EntityDensities(indices, values)


def _simplex_density_expressions(
    basis: AbstractReferenceFormBasis,
    indices: tuple[int, ...],
    dimension: int,
    blades: tuple[tuple[int, ...], ...],
    jacobian: forms.HostFormArray,
    *,
    budget: CoordinateEnclosureBudget,
) -> _EntityDensities:
    tests = forms.simplex_form_tests(
        dimension, basis.form_degree, basis.order, basis.family
    )
    if tests is None:
        raise ValueError(
            "form_basis.simplex_tests must provide polynomial moment tests; got no supported tests"
        )
    pullback, wedge = forms.simplex_form_pairing(
        jacobian, basis.dimension, basis.form_degree
    )
    pairing = wedge.T @ pullback
    values = []
    for test, _ in enumerate(indices):
        polynomials = forms.simplex_test_polynomials(tests, test)
        values.append(
            tuple(
                expression.sum_polynomials(
                    tuple(
                        expression.scale(
                            {
                                alpha: Fraction(value)
                                for alpha, value in polynomial.items()
                            },
                            Fraction(float(pairing[component, ambient])),
                            budget=budget,
                        )
                        for component, polynomial in enumerate(polynomials)
                    ),
                    budget=budget,
                )
                for ambient in range(len(blades))
            )
        )
    return _EntityDensities(indices, tuple(values))


# Entity admission and explicit family dispatch retain ordered dual rows and fail-closed scientific selectors.
@checked
def functional_density_expressions(
    basis: AbstractReferenceFormBasis,
    face: tuple[int, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[expression.Expression, ...], ...]:
    if not any(face in entities for entities in basis.entity_vertices):
        raise ValueError(
            "form_basis.entity_vertices must use a declared entity and its vertex ordering; got an undeclared ordering"
        )
    free, _, jacobian = forms.form_entity_chart(basis.dimension, basis.family, face)
    m = jacobian.shape[1]
    blades = tuple(combinations(range(basis.dimension), basis.form_degree))
    indices = forms.entity_dof_indices(basis.dof_labels, face)
    rows: list[tuple[expression.Expression, ...]] = [
        tuple({} for _ in blades) for _ in range(basis.local_dof_count)
    ]
    if not indices:
        return tuple(rows)
    factors = basis.hybrid_factors
    if factors is not None:
        _admit_hybrid_source(basis, factors, budget=budget)
        if m == 3:
            densities = _hybrid_body_densities(
                basis, factors, indices, blades, budget=budget
            )
            for index, values in zip(densities.indices, densities.values, strict=True):
                rows[index] = values
        elif m == 0:
            rows[indices[0]] = (expression.constant(1, 0, budget=budget),)
        else:
            densities = _hybrid_trace_densities(
                basis, face, indices, blades, jacobian, budget=budget
            )
            for index, values in zip(densities.indices, densities.values, strict=True):
                rows[index] = values
    elif basis.family == "tensor-trimmed":
        for index in indices:
            _, modes, blade = basis.dof_labels[index]
            alpha = tuple(modes[axis] - int(axis not in blade) for axis in free)
            rows[index] = tuple(
                {alpha: Fraction(1)} if component == blade else {} for component in blades
            )
    elif basis.family == "full" and basis.order == 0:
        for component, index in enumerate(indices):
            rows[index] = tuple(
                expression.constant(int(output == component), m, budget=budget)
                for output in range(len(blades))
            )
    else:
        densities = _simplex_density_expressions(
            basis, indices, m, blades, jacobian, budget=budget
        )
        for index, values in zip(densities.indices, densities.values, strict=True):
            rows[index] = values
    return tuple(rows)


def _estimate_axes(dimension: int) -> tuple[expression.CoordinateEnclosureEstimate, ...]:
    return tuple(
        expression.coordinate_estimate_variable(dimension) for _ in range(dimension)
    )


def _collapsed_expression_estimate(
    value: expression.CoordinateEnclosureEstimate,
) -> expression.CoordinateEnclosureEstimate:
    # Total-degree support contains every possible coefficient of
    # the combined source. Each horizontal exponent is at most total degree.
    x, y, z = _estimate_axes(3)
    s = expression.coordinate_estimate_add(
        expression.coordinate_estimate_constant(1, 3),
        expression.coordinate_estimate_scale(z, -1),
    )
    arguments = (
        expression.coordinate_estimate_add(
            x, expression.coordinate_estimate_scale(z, Fraction(-1, 2))
        ),
        expression.coordinate_estimate_add(
            y, expression.coordinate_estimate_scale(z, Fraction(-1, 2))
        ),
        z,
    )
    degree = value.degree
    coefficient = expression.coordinate_polynomial_estimate(
        terms=1,
        dimension=3,
        degree=0,
        numerator_bits=value.numerator_bits,
        denominator_bits=value.denominator_bits,
    )
    # This term dominates every monomial circuit in _collapsed_polynomial:
    # three powers and the complementary collapse power each have degree <= d.
    term = expression.coordinate_estimate_multiply(
        expression.coordinate_estimate_multiply(
            expression.coordinate_estimate_product(
                tuple(
                    expression.coordinate_estimate_power(axis, degree)
                    for axis in arguments
                ),
                3,
            ),
            expression.coordinate_estimate_power(s, degree),
        ),
        coefficient,
    )
    numerator = expression.coordinate_estimate_sum(
        tuple(term for _ in range(comb(degree + 3, 3))), 3
    )
    result = expression.coordinate_estimate_rational(
        numerator, expression.coordinate_estimate_power(s, degree)
    )
    return expression.CoordinateEnclosureEstimate(
        result.terms,
        result.dimension,
        result.degree,
        result.numerator_bits,
        result.denominator_bits,
        result.work_upper + value.work_upper,
        result.bytes_upper + value.bytes_upper,
        result.denominator,
    )


def _pyramid_action_estimates(
    k: int, *, density: bool
) -> tuple[tuple[expression.CoordinateEnclosureEstimate, ...], ...]:
    x, y, z = _estimate_axes(3)
    zero, one = (
        expression.coordinate_estimate_constant(0, 3),
        expression.coordinate_estimate_constant(1, 3),
    )
    s = expression.coordinate_estimate_add(
        one, expression.coordinate_estimate_scale(z, -1)
    )
    a = expression.coordinate_estimate_add(
        x, expression.coordinate_estimate_constant(Fraction(-1, 2), 3)
    )
    b = expression.coordinate_estimate_add(
        y, expression.coordinate_estimate_constant(Fraction(-1, 2), 3)
    )
    if density:
        a, b = (
            expression.coordinate_estimate_scale(a, -1),
            expression.coordinate_estimate_scale(b, -1),
        )
    inv = expression.coordinate_estimate_rational(one, s)
    inv2 = expression.coordinate_estimate_rational(
        one, expression.coordinate_estimate_power(s, 2)
    )
    if k == 0:
        return tuple(
            tuple(
                _estimate_with_preparation(
                    value, (x, y, z, zero, one, s, a, b, inv, inv2)
                )
                for value in row
            )
            for row in ((inv2 if density else one,),)
        )
    if k == 1:
        exponent = 3 if density else 2
        return tuple(
            tuple(
                _estimate_with_preparation(
                    value, (x, y, z, zero, one, s, a, b, inv, inv2)
                )
                for value in row
            )
            for row in (
                (inv, zero, zero),
                (zero, inv, zero),
                (
                    expression.coordinate_estimate_rational(
                        a, expression.coordinate_estimate_power(s, exponent)
                    ),
                    expression.coordinate_estimate_rational(
                        b, expression.coordinate_estimate_power(s, exponent)
                    ),
                    inv2 if density else one,
                ),
            )
        )
    if k == 2:
        exponent = 2 if density else 3
        return tuple(
            tuple(
                _estimate_with_preparation(
                    value, (x, y, z, zero, one, s, a, b, inv, inv2)
                )
                for value in row
            )
            for row in (
                (one if density else inv2, zero, zero),
                (
                    expression.coordinate_estimate_rational(
                        b, expression.coordinate_estimate_power(s, exponent)
                    ),
                    inv,
                    zero,
                ),
                (
                    expression.coordinate_estimate_rational(
                        expression.coordinate_estimate_scale(a, -1),
                        expression.coordinate_estimate_power(s, exponent),
                    ),
                    zero,
                    inv,
                ),
            )
        )
    if k == 3:
        return tuple(
            tuple(
                _estimate_with_preparation(
                    value, (x, y, z, zero, one, s, a, b, inv, inv2)
                )
                for value in row
            )
            for row in ((one if density else inv2,),)
        )
    raise ValueError(
        "pyramid_form.degree must lie between zero and three; got an unsupported degree"
    )


# Axial interpolation and its preparation bank share the ordered source circuit.
def _tensor_component_estimates(
    basis: AbstractReferenceFormBasis,
    factors: AbstractReferenceTensorFactors,
) -> tuple[tuple[expression.CoordinateEnclosureEstimate, ...], ...]:
    dimension = basis.dimension
    zero, one = np.asarray(factors.zero), np.asarray(factors.one)
    indices, differentials, components = (
        np.asarray(factors.indices),
        np.asarray(factors.differential_axes),
        np.asarray(factors.components),
    )
    axial = []
    for axis, variable in enumerate(_estimate_axes(dimension)):
        collapse = expression.coordinate_estimate_add(
            expression.coordinate_estimate_constant(1, dimension),
            expression.coordinate_estimate_scale(variable, -1),
        )
        bubble = expression.coordinate_estimate_multiply(variable, collapse)
        modes = tuple(
            expression.coordinate_jacobi_estimate(order, 0, axis, dimension)
            for order in range(basis.order)
        )
        zero_modes = tuple(
            expression.coordinate_estimate_add(
                expression.coordinate_estimate_add(
                    expression.coordinate_estimate_scale(
                        collapse, Fraction(float(zero[0, dof]))
                    ),
                    expression.coordinate_estimate_scale(
                        variable, Fraction(float(zero[1, dof]))
                    ),
                ),
                expression.coordinate_estimate_multiply(
                    bubble,
                    expression.coordinate_estimate_sum(
                        tuple(
                            expression.coordinate_estimate_scale(
                                mode, Fraction(float(value))
                            )
                            for mode, value in zip(modes[:-1], zero[2:, dof], strict=True)
                        ),
                        dimension,
                    ),
                ),
            )
            for dof in range(zero.shape[1])
        )
        one_modes = tuple(
            expression.coordinate_estimate_sum(
                tuple(
                    expression.coordinate_estimate_scale(mode, Fraction(float(value)))
                    for mode, value in zip(modes, one[:, dof], strict=True)
                ),
                dimension,
            )
            for dof in range(one.shape[1])
        )
        axial.append((zero_modes, one_modes))
    prepared_axial = expression.coordinate_estimate_bank(
        tuple(value for pair in axial for modes in pair for value in modes)
    )
    result = tuple(
        tuple(
            expression.coordinate_estimate_scale(
                expression.coordinate_estimate_product(
                    tuple(
                        axial[axis][int(differentials[axis, dof])][indices[axis, dof]]
                        for axis in range(dimension)
                    ),
                    dimension,
                ),
                Fraction(float(value)),
            )
            for value in components[dof]
        )
        for dof in range(basis.local_dof_count)
    )
    return tuple(
        tuple(_estimate_with_preparation(value, (prepared_axial,)) for value in row)
        for row in result
    )


@checked
def component_expression_estimates(
    basis: AbstractReferenceFormBasis, /
) -> tuple[tuple[expression.CoordinateEnclosureEstimate, ...], ...]:
    """Bound the declared component source circuit without preparing coefficients."""
    if basis.tensor_factors is not None:
        return _tensor_component_estimates(basis, basis.tensor_factors)
    coefficients = np.asarray(basis.coefficients)
    factors = basis.hybrid_factors
    if factors is None:
        return tuple(
            tuple(
                expression.coordinate_expression_estimate(
                    {
                        alpha: Fraction(float(value))
                        for alpha, value in zip(
                            basis.exponents, coefficients[:, component, dof], strict=True
                        )
                        if value
                    }
                )
                for component in range(coefficients.shape[1])
            )
            for dof in range(basis.local_dof_count)
        )
    return _hybrid_component_estimates(basis, factors, coefficients)


# Rational generator support, collapse, and blade-action order define source-circuit bounds.
def _hybrid_component_estimates(
    basis: AbstractReferenceFormBasis,
    factors: AbstractReferenceHybridFactors,
    coefficients: forms.HostFormArray,
) -> tuple[tuple[expression.CoordinateEnclosureEstimate, ...], ...]:
    dimension = basis.dimension
    generators = np.asarray(factors.generators)
    count = generators.shape[1]
    source = tuple(
        tuple(
            expression.coordinate_expression_estimate(
                {
                    alpha: factors.source_bank[row * count + component][generator]
                    for row, alpha in enumerate(basis.exponents)
                    if factors.source_bank[row * count + component][generator]
                }
            )
            for component in range(count)
        )
        for generator in range(generators.shape[-1])
    )
    action = (
        _pyramid_action_estimates(basis.form_degree, density=False)
        if basis.family == "pyramid-trimmed"
        else None
    )
    rows = []
    for dof in range(basis.local_dof_count):
        combined = tuple(
            expression.coordinate_estimate_sum(
                tuple(
                    expression.coordinate_estimate_scale(
                        field[component], Fraction(float(coefficients[generator, dof]))
                    )
                    for generator, field in enumerate(source)
                    if coefficients[generator, dof] != 0
                ),
                dimension,
            )
            for component in range(count)
        )
        if action is None:
            rows.append(combined)
        else:
            collapsed = tuple(
                _collapsed_expression_estimate(component) for component in combined
            )
            rows.append(
                tuple(
                    expression.coordinate_estimate_sum(
                        tuple(
                            expression.coordinate_estimate_multiply(entry, component)
                            for entry, component in zip(row, collapsed, strict=True)
                        ),
                        dimension,
                    )
                    for row in action
                )
            )

    canonical = hybrid_forms_owner.hybrid_preparation_estimate(
        basis.family, basis.form_degree, basis.order
    )
    return tuple(
        tuple(_estimate_with_preparation(value, (canonical,)) for value in row)
        for row in rows
    )


# Body-test support and rational blade action are one ordered source-faithful formula.
def _hybrid_body_density_estimates(
    basis: AbstractReferenceFormBasis,
    factors: AbstractReferenceHybridFactors,
    indices: tuple[int, ...],
    blades: tuple[tuple[int, ...], ...],
    rows: list[tuple[expression.CoordinateEnclosureEstimate, ...]],
) -> None:
    action = (
        _pyramid_action_estimates(basis.form_degree, density=True)
        if basis.family == "pyramid-trimmed"
        else tuple(
            tuple(
                expression.coordinate_estimate_constant(int(i == j), 3)
                for j in range(len(blades))
            )
            for i in range(len(blades))
        )
    )
    for column, index in enumerate(indices):
        polynomials = tuple(
            expression.coordinate_expression_estimate(
                {
                    alpha: factors.body_test_source_bank[row * len(blades) + component][
                        column
                    ]
                    for row, alpha in enumerate(factors.body_test_exponents)
                    if factors.body_test_source_bank[row * len(blades) + component][
                        column
                    ]
                }
            )
            for component in range(len(blades))
        )
        physical = (
            tuple(_collapsed_expression_estimate(value) for value in polynomials)
            if basis.family == "pyramid-trimmed"
            else polynomials
        )
        rows[index] = tuple(
            expression.coordinate_estimate_sum(
                tuple(
                    expression.coordinate_estimate_multiply(
                        value, action[component][ambient]
                    )
                    for component, value in enumerate(physical)
                ),
                3,
            )
            for ambient in range(len(blades))
        )


def _hybrid_trace_density_estimates(
    basis: AbstractReferenceFormBasis,
    face: tuple[int, ...],
    indices: tuple[int, ...],
    blades: tuple[tuple[int, ...], ...],
    jacobian: forms.HostFormArray,
    rows: list[tuple[expression.CoordinateEnclosureEstimate, ...]],
) -> None:
    dimension = jacobian.shape[1]
    trace = basis.entity_basis(face)
    interior = forms.interior_dof_indices(trace.dof_labels, len(face))
    density = functional_density_expression_estimates(trace, trace.entity_vertices[-1][0])
    local_blades = tuple(combinations(range(dimension), basis.form_degree))
    pullback = tuple(
        tuple(Fraction(forms.form_minor(jacobian, ambient, local)) for ambient in blades)
        for local in local_blades
    )
    for index, trace_index in zip(indices, interior, strict=True):
        rows[index] = tuple(
            expression.coordinate_estimate_sum(
                tuple(
                    expression.coordinate_estimate_scale(
                        value, pullback[component][ambient]
                    )
                    for component, value in enumerate(density[trace_index])
                ),
                dimension,
            )
            for ambient in range(len(blades))
        )


def _simplex_density_estimates(
    basis: AbstractReferenceFormBasis,
    indices: tuple[int, ...],
    blades: tuple[tuple[int, ...], ...],
    jacobian: forms.HostFormArray,
    rows: list[tuple[expression.CoordinateEnclosureEstimate, ...]],
) -> None:
    dimension = jacobian.shape[1]
    tests = forms.simplex_form_tests(
        dimension, basis.form_degree, basis.order, basis.family
    )
    if tests is None:
        raise ValueError(
            "form_basis.simplex_tests must provide polynomial moment tests; got no supported tests"
        )
    pullback, wedge = forms.simplex_form_pairing(
        jacobian, basis.dimension, basis.form_degree
    )
    pairing = wedge.T @ pullback
    for test, index in enumerate(indices):
        polynomials = forms.simplex_test_polynomials(tests, test)
        rows[index] = tuple(
            expression.coordinate_estimate_sum(
                tuple(
                    expression.coordinate_estimate_scale(
                        expression.coordinate_expression_estimate(
                            {
                                alpha: Fraction(value)
                                for alpha, value in polynomial.items()
                            }
                        ),
                        Fraction(float(pairing[component, ambient])),
                    )
                    for component, polynomial in enumerate(polynomials)
                ),
                dimension,
            )
            for ambient in range(len(blades))
        )


# Entity admission and explicit family dispatch retain ordered density bounds and hybrid preparation evidence.
@checked
def functional_density_expression_estimates(
    basis: AbstractReferenceFormBasis, face: tuple[int, ...], /
) -> tuple[tuple[expression.CoordinateEnclosureEstimate, ...], ...]:
    """Bound the entity density source and its rational blade action."""
    if not any(face in entities for entities in basis.entity_vertices):
        raise ValueError(
            "form_basis.entity_vertices must use a declared entity and its vertex ordering; got an undeclared ordering"
        )
    free, _, jacobian = forms.form_entity_chart(basis.dimension, basis.family, face)
    dimension = jacobian.shape[1]
    blades = tuple(combinations(range(basis.dimension), basis.form_degree))
    indices = forms.entity_dof_indices(basis.dof_labels, face)
    rows = [
        tuple(expression.coordinate_estimate_constant(0, dimension) for _ in blades)
        for _ in range(basis.local_dof_count)
    ]
    if not indices:
        return tuple(rows)
    if basis.hybrid_factors is not None:
        if dimension == 3:
            _hybrid_body_density_estimates(
                basis, basis.hybrid_factors, indices, blades, rows
            )
        elif dimension == 0:
            rows[indices[0]] = (expression.coordinate_estimate_constant(1, 0),)
        else:
            _hybrid_trace_density_estimates(basis, face, indices, blades, jacobian, rows)
    elif basis.family == "tensor-trimmed":
        for index in indices:
            _, modes, blade = basis.dof_labels[index]
            alpha = tuple(modes[axis] - int(axis not in blade) for axis in free)
            rows[index] = tuple(
                expression.coordinate_expression_estimate(
                    {alpha: Fraction(1)} if component == blade else {}
                )
                for component in blades
            )
    elif basis.family == "full" and basis.order == 0:
        for component, index in enumerate(indices):
            rows[index] = tuple(
                expression.coordinate_estimate_constant(
                    int(output == component), dimension
                )
                for output in range(len(blades))
            )
    else:
        _simplex_density_estimates(basis, indices, blades, jacobian, rows)
    if basis.hybrid_factors is not None:
        canonical = hybrid_forms_owner.hybrid_preparation_estimate(
            basis.family, basis.form_degree, basis.order
        )
        return tuple(
            tuple(_estimate_with_preparation(value, (canonical,)) for value in row)
            for row in rows
        )
    return tuple(rows)


def _estimate_with_preparation(
    value: expression.CoordinateEnclosureEstimate,
    preparations: tuple[expression.CoordinateEnclosureEstimate, ...],
) -> expression.CoordinateEnclosureEstimate:
    prepared = expression.coordinate_estimate_bank(preparations)
    return expression.CoordinateEnclosureEstimate(
        value.terms,
        value.dimension,
        value.degree,
        value.numerator_bits,
        value.denominator_bits,
        value.work_upper + prepared.work_upper,
        value.bytes_upper + prepared.bytes_upper,
        value.denominator,
    )
