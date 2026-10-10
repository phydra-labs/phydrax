# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact owning source expressions for form components and entity moments."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import numpy as np

from .. import _coordinate_enclosure as expression
from . import _form_elements as forms


def _collapsed_polynomial(polynomial: expression.Polynomial) -> expression.Expression:
    x, y, z = expression.axes(3)
    s = expression.add(expression.constant(1, 3), expression.scale(z, -1))
    arguments = (
        expression.add(x, expression.scale(z, Fraction(-1, 2))),
        expression.add(y, expression.scale(z, Fraction(-1, 2))),
        z,
    )
    degree = max((alpha[0] + alpha[1] for alpha in polynomial), default=0)
    numerator = expression.sum_polynomials(
        tuple(
            expression.scale(
                expression.multiply(
                    expression.math_product_polynomials(
                        tuple(
                            expression.power(variable, power, 3)
                            for variable, power in zip(arguments, alpha, strict=True)
                        ),
                        3,
                    ),
                    expression.power(s, degree - alpha[0] - alpha[1], 3),
                ),
                coefficient,
            )
            for alpha, coefficient in polynomial.items()
        )
    )
    return expression.rational_expression(numerator, expression.power(s, degree, 3))


def _pyramid_component_action(k: int) -> tuple[tuple[expression.Expression, ...], ...]:
    x, y, z = expression.axes(3)
    one = expression.constant(1, 3)
    s = expression.add(one, expression.scale(z, -1))
    a = expression.add(x, expression.constant(Fraction(-1, 2), 3))
    b = expression.add(y, expression.constant(Fraction(-1, 2), 3))
    inv = expression.rational_expression(one, s)
    inv2 = expression.rational_expression(one, expression.power(s, 2, 3))
    if k == 0:
        return ((one,),)
    if k == 1:
        return (
            (inv, {}, {}),
            ({}, inv, {}),
            (
                expression.rational_expression(a, expression.power(s, 2, 3)),
                expression.rational_expression(b, expression.power(s, 2, 3)),
                one,
            ),
        )
    if k == 2:
        return (
            (inv2, {}, {}),
            (expression.rational_expression(b, expression.power(s, 3, 3)), inv, {}),
            (
                expression.rational_expression(
                    expression.scale(a, -1), expression.power(s, 3, 3)
                ),
                {},
                inv,
            ),
        )
    if k == 3:
        return ((inv2,),)
    raise ValueError("Pyramid form degree must lie between zero and three.")


def _tensor_components(
    basis: forms.FormBasis,
) -> tuple[tuple[expression.Expression, ...], ...]:
    factors = basis.tensor_factors
    if factors is None:
        raise ValueError(
            "Tensor source expressions require their numerical factor owner."
        )
    n = basis.dimension
    variables = expression.axes(n)
    zero, one = np.asarray(factors.zero), np.asarray(factors.one)
    indices, differentials, components = (
        np.asarray(factors.indices),
        np.asarray(factors.differential_axes),
        np.asarray(factors.components),
    )
    axial = []
    for axis, variable in enumerate(variables):
        collapse = expression.add(
            expression.constant(1, n), expression.scale(variable, -1)
        )
        bubble = expression.multiply(variable, collapse)
        modes = tuple(
            expression._jacobi(order, 0, axis, n) for order in range(basis.order)
        )
        zero_modes = tuple(
            expression.add(
                expression.add(
                    expression.scale(collapse, Fraction(float(zero[0, dof]))),
                    expression.scale(variable, Fraction(float(zero[1, dof]))),
                ),
                expression.multiply(
                    bubble,
                    expression.sum_polynomials(
                        tuple(
                            expression.scale(mode, Fraction(float(value)))
                            for mode, value in zip(modes[:-1], zero[2:, dof], strict=True)
                        )
                    ),
                ),
            )
            for dof in range(zero.shape[1])
        )
        one_modes = tuple(
            expression.sum_polynomials(
                tuple(
                    expression.scale(mode, Fraction(float(value)))
                    for mode, value in zip(modes, one[:, dof], strict=True)
                )
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
        )
        result.append(
            tuple(
                expression.scale(scalar, Fraction(float(value)))
                for value in components[dof]
            )
        )
    return tuple(result)


def _admit_hybrid_source(basis: forms.FormBasis) -> None:
    """Check the current numerical leaves against their canonical moment owner."""
    from ._hybrid_forms import prepare_hybrid

    factors = basis.hybrid_factors
    if factors is None:
        raise ValueError(
            "A hybrid source requires its complete generator and moment owner."
        )
    prepared = prepare_hybrid(basis.family, basis.form_degree, basis.order)
    if (
        factors.source_bank != prepared.source_bank
        or factors.body_test_exponents != prepared.body_test_exponents
        or factors.body_test_source_bank != prepared.body_test_source_bank
        or basis.exponents != prepared.prepared.exponents
        or basis.dof_labels != prepared.prepared.labels
        or not np.array_equal(np.asarray(basis.coefficients), prepared.coefficients)
    ):
        raise ValueError(
            "Hybrid numerical source no longer has its declared canonical moment-dual identity."
        )


def component_expressions(
    basis: forms.FormBasis,
) -> tuple[tuple[expression.Expression, ...], ...]:
    if basis.tensor_factors is not None:
        return _tensor_components(basis)
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
    _admit_hybrid_source(basis)
    generators = np.asarray(factors.generators)
    expected = np.asarray(factors.source_bank, dtype=np.float64).reshape(generators.shape)
    if not np.array_equal(generators, expected):
        raise ValueError(
            "Hybrid numerical generator leaves no longer match their exact scientific source."
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
    # Exact linear combination precedes the collapsed rational chart. Summing
    # rational generators instead would repeatedly multiply identical collapse
    # denominators and consume storage unrelated to the represented source.
    action = (
        _pyramid_component_action(basis.form_degree)
        if basis.family == "pyramid-trimmed"
        else None
    )

    def field_expressions(dof: int) -> tuple[expression.Expression, ...]:
        combined = tuple(
            expression.sum_polynomials(
                tuple(
                    expression.scale(
                        field[component], Fraction(float(dual[generator, dof]))
                    )
                    for generator, field in enumerate(source)
                    if dual[generator, dof] != 0
                )
            )
            for component in range(count)
        )
        if action is None:
            return combined
        collapsed = tuple(_collapsed_polynomial(component) for component in combined)
        return tuple(
            expression.expression_sum(
                tuple(
                    expression.expression_multiply(entry, component)
                    for entry, component in zip(row, collapsed, strict=True)
                    if entry and component
                )
            )
            for row in action
        )

    budget = expression._COORDINATE_BUDGET.get()
    result: list[tuple[expression.Expression, ...]] = []
    for dof in range(basis.local_dof_count):
        if budget is None:
            result.append(field_expressions(dof))
        else:
            with budget.temporary_scope():
                field = field_expressions(dof)
                budget.retain_basis(field)
                result.append(field)
    return tuple(result)


def _pyramid_density_action(k: int) -> tuple[tuple[expression.Expression, ...], ...]:
    x, y, z = expression.axes(3)
    one = expression.constant(1, 3)
    s = expression.add(one, expression.scale(z, -1))
    a = expression.add(expression.constant(Fraction(1, 2), 3), expression.scale(x, -1))
    b = expression.add(expression.constant(Fraction(1, 2), 3), expression.scale(y, -1))
    inv = expression.rational_expression(one, s)
    inv2 = expression.rational_expression(one, expression.power(s, 2, 3))
    if k == 0:
        return ((inv2,),)
    if k == 1:
        return (
            (inv, {}, {}),
            ({}, inv, {}),
            (
                expression.rational_expression(a, expression.power(s, 3, 3)),
                expression.rational_expression(b, expression.power(s, 3, 3)),
                inv2,
            ),
        )
    if k == 2:
        return (
            (one, {}, {}),
            (expression.rational_expression(b, expression.power(s, 2, 3)), inv, {}),
            (
                expression.rational_expression(
                    expression.scale(a, -1), expression.power(s, 2, 3)
                ),
                {},
                inv,
            ),
        )
    if k == 3:
        return ((one,),)
    raise ValueError("Pyramid form degree must lie between zero and three.")


def functional_density_expressions(
    basis: forms.FormBasis, face: tuple[int, ...]
) -> tuple[tuple[expression.Expression, ...], ...]:
    if not any(face in entities for entities in basis.entity_vertices):
        raise ValueError("Use a declared reference entity and its vertex ordering.")
    free, _, jacobian = forms._form_entity_chart(basis.dimension, basis.family, face)
    m = jacobian.shape[1]
    blades = tuple(combinations(range(basis.dimension), basis.form_degree))
    indices = forms._entity_dof_indices(basis.dof_labels, face)
    rows: list[tuple[expression.Expression, ...]] = [
        tuple({} for _ in blades) for _ in range(basis.local_dof_count)
    ]
    if not indices:
        return tuple(rows)
    if basis.hybrid_factors is not None:
        _admit_hybrid_source(basis)
        if m == 3:
            action = (
                _pyramid_density_action(basis.form_degree)
                if basis.family == "pyramid-trimmed"
                else tuple(
                    tuple(expression.constant(int(i == j), 3) for j in range(len(blades)))
                    for i in range(len(blades))
                )
            )
            factors = basis.hybrid_factors
            coefficients = np.asarray(factors.body_test_coefficients)
            expected = np.asarray(
                factors.body_test_source_bank, dtype=np.float64
            ).reshape(coefficients.shape)
            if not np.array_equal(coefficients, expected):
                raise ValueError(
                    "Hybrid numerical moment tests no longer match their exact scientific source."
                )
            for column, index in enumerate(indices):
                polynomials = tuple(
                    {
                        alpha: factors.body_test_source_bank[
                            row * len(blades) + component
                        ][column]
                        for row, alpha in enumerate(factors.body_test_exponents)
                        if factors.body_test_source_bank[row * len(blades) + component][
                            column
                        ]
                    }
                    for component in range(len(blades))
                )
                physical = (
                    tuple(_collapsed_polynomial(polynomial) for polynomial in polynomials)
                    if basis.family == "pyramid-trimmed"
                    else polynomials
                )
                rows[index] = tuple(
                    expression.expression_sum(
                        tuple(
                            expression.expression_multiply(
                                polynomial, action[component][ambient]
                            )
                            for component, polynomial in enumerate(physical)
                        )
                    )
                    for ambient in range(len(blades))
                )
        elif m == 0:
            rows[indices[0]] = (expression.constant(1, 0),)
        else:
            trace = basis.entity_basis(face)
            interior = forms._interior_dof_indices(trace.dof_labels, len(face))
            density = functional_density_expressions(trace, trace.entity_vertices[-1][0])
            local_blades = tuple(combinations(range(m), basis.form_degree))
            pullback = tuple(
                tuple(
                    Fraction(forms._minor(jacobian, ambient, local)) for ambient in blades
                )
                for local in local_blades
            )
            for index, trace_index in zip(indices, interior, strict=True):
                rows[index] = tuple(
                    expression.expression_sum(
                        tuple(
                            expression.expression_scale(
                                value, pullback[component][ambient]
                            )
                            for component, value in enumerate(density[trace_index])
                        )
                    )
                    for ambient in range(len(blades))
                )
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
                expression.constant(int(output == component), m)
                for output in range(len(blades))
            )
    else:
        tests = forms._simplex_tests(m, basis.form_degree, basis.order, basis.family)
        if tests is None:
            raise ValueError("Declared simplex moments have no polynomial test owner.")
        pullback, wedge = forms._simplex_pairing(
            jacobian, basis.dimension, basis.form_degree
        )
        pairing = wedge.T @ pullback
        for test, index in enumerate(indices):
            polynomials = forms._simplex_test_polynomials(tests, test)
            rows[index] = tuple(
                expression.sum_polynomials(
                    tuple(
                        expression.scale(
                            {
                                alpha: Fraction(value)
                                for alpha, value in polynomial.items()
                            },
                            Fraction(float(pairing[component, ambient])),
                        )
                        for component, polynomial in enumerate(polynomials)
                    )
                )
                for ambient in range(len(blades))
            )
    return tuple(rows)
