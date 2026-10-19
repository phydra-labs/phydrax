# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""algebraic entity moment actions on collapsed/product monomial form components."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import combinations
from math import factorial, prod

import numpy as np

from phydrax.discretization.fem._hybrid_charts import (
    entity_chart,
    hybrid_kind,
    trace_basis,
)

from ...typing import checked
from .. import _coordinate_enclosure as algebra
from .._coordinate_budget import CoordinateEnclosureBudget
from .._reference_cell import reference_cell_topology
from . import (
    _form_elements as forms,
    _form_expressions as form_expressions_owner,
    _form_expressions as imported_form_expressions,
)
from ._form_basis_types import DofLabel, FormElementFamily


type ExactMomentRows = tuple[tuple[tuple[Fraction, ...], ...], ...]


def _integral(
    polynomial: Mapping[tuple[int, ...], algebra.ExactScalar],
    dimension: int,
    simplex: bool,
) -> Fraction:
    if simplex:
        return sum(
            (
                algebra.require_rational_reference_scalar(coefficient)
                * Fraction(
                    prod(factorial(power) for power in alpha),
                    factorial(sum(alpha) + dimension),
                )
                for alpha, coefficient in polynomial.items()
            ),
            Fraction(0),
        )
    return sum(
        (
            algebra.require_rational_reference_scalar(coefficient)
            / prod(power + 1 for power in alpha)
            for alpha, coefficient in polynomial.items()
        ),
        Fraction(0),
    )


def _entity_source(
    family: FormElementFamily,
    k: int,
    r: int,
    face: tuple[int, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    tuple[algebra.Polynomial, ...],
    np.ndarray,
    tuple[tuple[algebra.Polynomial, ...], ...],
    bool,
]:
    origin, jacobian = entity_chart(family, face)
    dimension = jacobian.shape[1]
    trace = trace_basis(family, k, r, face, budget=budget)
    indices = forms.interior_dof_indices(trace.dof_labels, len(face))
    raw = form_expressions_owner.functional_density_expressions(
        trace, trace.entity_vertices[-1][0], budget=budget
    )
    densities = []
    for index in indices:
        row = []
        for value in raw[index]:
            if isinstance(value, algebra.RationalPolynomial):
                raise ValueError(
                    "entity_source.configuration must have polynomial densities; got declared simplex/tensor trace moments must have polynomial densities."
                )
            row.append(value)
        densities.append(tuple(row))
    simplex = len(face) == dimension + 1
    if family != "pyramid-trimmed":
        return (
            algebra.affine_arguments(origin, jacobian, budget=budget),
            jacobian,
            tuple(densities),
            simplex,
        )
    if dimension == 1 and 4 in face:
        base = face[0] if face[1] == 4 else face[1]
        vertex = np.asarray(
            reference_cell_topology("pyramid").vertices[base], dtype=np.float64
        )
        origin = vertex.copy()
        origin[2] = float(face[0] == 4)
        jacobian = np.asarray(
            ((0.0,), (0.0,), (1.0 if face[1] == 4 else -1.0,)), dtype=np.float64
        )
    elif dimension == 2 and len(face) == 3:
        vertices = np.asarray(
            reference_cell_topology("pyramid").vertices, dtype=np.float64
        )
        origin = vertices[face[0]].copy()
        edge = vertices[face[1]] - origin
        jacobian = np.column_stack((edge, np.asarray((0.0, 0.0, 1.0), dtype=np.float64)))
        u, z = algebra.axes(2, budget=budget)
        height = algebra.add(
            algebra.constant(1, 2, budget=budget),
            algebra.scale(z, -1, budget=budget),
            budget=budget,
        )
        arguments = (algebra.multiply(u, height, budget=budget), z)
        composed = tuple(
            tuple(algebra.compose(value, arguments, budget=budget) for value in row)
            for row in densities
        )
        if k == 0:
            densities = [
                tuple(algebra.multiply(value, height, budget=budget) for value in row)
                for row in composed
            ]
        elif k == 1:
            densities = [
                (
                    algebra.add(
                        row[0], algebra.multiply(u, row[1], budget=budget), budget=budget
                    ),
                    algebra.multiply(height, row[1], budget=budget),
                )
                for row in composed
            ]
        else:
            densities = list(composed)
        simplex = False
    return (
        algebra.affine_arguments(origin, jacobian, budget=budget),
        jacobian,
        tuple(densities),
        simplex,
    )


def _vertex_functional_moments(
    family: FormElementFamily,
    face: tuple[int, ...],
    vertices: np.ndarray,
    exponents: tuple[tuple[int, ...], ...],
    indices: tuple[int, ...],
    rows: list[list[list[Fraction]]],
) -> None:
    point = vertices[face[0]]
    if family == "pyramid-trimmed" and face == (4,):
        # This extension is independent of direction on the represented
        # scalar source space, whose apex limit is unique.
        point = np.asarray((0.5, 0.5, 1.0), dtype=np.float64)
    for row, alpha in enumerate(exponents):
        rows[indices[0]][row][0] = prod(
            (
                Fraction(float(value)) ** power
                for value, power in zip(point, alpha, strict=True)
            ),
            start=Fraction(1),
        )


def _body_functional_moments(
    family: FormElementFamily,
    exponents: tuple[tuple[int, ...], ...],
    test_exponents: tuple[tuple[int, ...], ...],
    test_source_bank: tuple[tuple[Fraction, ...], ...],
    blades: tuple[tuple[int, ...], ...],
    indices: tuple[int, ...],
    rows: list[list[list[Fraction]]],
) -> None:
    for column, index in enumerate(indices):
        for row, alpha in enumerate(exponents):
            for component in range(len(blades)):
                polynomial = {
                    tuple(
                        a + b for a, b in zip(alpha, beta, strict=True)
                    ): test_source_bank[test * len(blades) + component][column]
                    for test, beta in enumerate(test_exponents)
                    if test_source_bank[test * len(blades) + component][column]
                }
                if family == "prism-trimmed":
                    rows[index][row][component] = sum(
                        (
                            value
                            * Fraction(
                                factorial(power[0]) * factorial(power[1]),
                                factorial(power[0] + power[1] + 2) * (power[2] + 1),
                            )
                            for power, value in polynomial.items()
                        ),
                        Fraction(0),
                    )
                else:
                    rows[index][row][component] = _integral(polynomial, 3, False)


def _trace_functional_moments(
    family: FormElementFamily,
    k: int,
    r: int,
    face: tuple[int, ...],
    dimension: int,
    exponents: tuple[tuple[int, ...], ...],
    blades: tuple[tuple[int, ...], ...],
    indices: tuple[int, ...],
    rows: list[list[list[Fraction]]],
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    arguments, jacobian, densities, simplex = _entity_source(
        family, k, r, face, budget=budget
    )
    local_blades = tuple(combinations(range(dimension), k))
    pullback = tuple(
        tuple(Fraction(forms.form_minor(jacobian, ambient, local)) for ambient in blades)
        for local in local_blades
    )
    for row, alpha in enumerate(exponents):
        monomial = algebra.compose({alpha: Fraction(1)}, arguments, budget=budget)
        for test, index in enumerate(indices):
            for component in range(len(blades)):
                density = algebra.sum_polynomials(
                    tuple(
                        algebra.scale(value, pullback[local][component], budget=budget)
                        for local, value in enumerate(densities[test])
                    ),
                    budget=budget,
                )
                rows[index][row][component] = _integral(
                    algebra.multiply(density, monomial, budget=budget),
                    dimension,
                    simplex,
                )


@checked
def exact_functional_moments(
    family: FormElementFamily,
    k: int,
    r: int,
    exponents: tuple[tuple[int, ...], ...],
    labels: tuple[DofLabel, ...],
    test_exponents: tuple[tuple[int, ...], ...],
    test_source_bank: tuple[tuple[Fraction, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactMomentRows:
    topology = reference_cell_topology(hybrid_kind(family))
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    blades = tuple(combinations(range(3), k))
    rows = [[[Fraction(0) for _ in blades] for _ in exponents] for _ in labels]
    for dimension, entities in enumerate(topology.entities):
        for face in entities:
            indices = forms.entity_dof_indices(labels, face)
            if not indices:
                continue
            match dimension:
                case 0:
                    _vertex_functional_moments(
                        family, face, vertices, exponents, indices, rows
                    )
                case 3:
                    _body_functional_moments(
                        family,
                        exponents,
                        test_exponents,
                        test_source_bank,
                        blades,
                        indices,
                        rows,
                    )
                case 1 | 2:
                    _trace_functional_moments(
                        family,
                        k,
                        r,
                        face,
                        dimension,
                        exponents,
                        blades,
                        indices,
                        rows,
                        budget=budget,
                    )
                case _:
                    raise ValueError(
                        f"exact_functional_moments.dimension must be between zero and three; got {dimension}"
                    )
    return tuple(tuple(tuple(component) for component in row) for row in rows)


def generator_functional_action(
    moments: ExactMomentRows,
    source_bank: tuple[tuple[Fraction, ...], ...],
    component_count: int,
) -> forms.HostFormArray:
    width = len(source_bank[0])
    support = tuple(
        tuple((column, value) for column, value in enumerate(row) if value)
        for row in source_bank
    )
    matrix = []
    for functional in moments:
        output = [Fraction(0)] * width
        for row, coefficients in enumerate(functional):
            for component, weight in enumerate(coefficients):
                if weight:
                    for column, value in support[row * component_count + component]:
                        output[column] += weight * value
        matrix.append(output)
    return np.asarray(matrix, dtype=np.float64).reshape(len(moments), width)


def _entity_source_estimates(
    family: FormElementFamily,
    k: int,
    r: int,
    face: tuple[int, ...],
) -> tuple[
    tuple[algebra.CoordinateEnclosureEstimate, ...],
    forms.HostFormArray,
    tuple[tuple[algebra.CoordinateEnclosureEstimate, ...], ...],
]:

    origin, jacobian = entity_chart(family, face)
    dimension = jacobian.shape[1]
    trace = trace_basis(family, k, r, face)
    indices = forms.interior_dof_indices(trace.dof_labels, len(face))
    raw = imported_form_expressions.functional_density_expression_estimates(
        trace, trace.entity_vertices[-1][0]
    )
    densities = tuple(raw[index] for index in indices)
    if family == "pyramid-trimmed":
        if dimension == 1 and 4 in face:
            base = face[0] if face[1] == 4 else face[1]
            origin = np.asarray(
                reference_cell_topology("pyramid").vertices[base], dtype=np.float64
            ).copy()
            origin[2] = float(face[0] == 4)
            jacobian = np.asarray(
                ((0.0,), (0.0,), (1.0 if face[1] == 4 else -1.0,)), dtype=np.float64
            )
        elif dimension == 2 and len(face) == 3:
            vertices = np.asarray(
                reference_cell_topology("pyramid").vertices, dtype=np.float64
            )
            origin = vertices[face[0]].copy()
            jacobian = np.column_stack(
                (
                    vertices[face[1]] - origin,
                    np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
                )
            )
            u, z = (algebra.coordinate_estimate_variable(2) for _ in range(2))
            height = algebra.coordinate_estimate_add(
                algebra.coordinate_estimate_constant(1, 2),
                algebra.coordinate_estimate_scale(z, -1),
            )
            arguments = (algebra.coordinate_estimate_multiply(u, height), z)
            composed = tuple(
                tuple(
                    algebra.coordinate_estimate_compose(value, arguments) for value in row
                )
                for row in densities
            )
            if k == 0:
                densities = tuple(
                    tuple(
                        algebra.coordinate_estimate_multiply(value, height)
                        for value in row
                    )
                    for row in composed
                )
            elif k == 1:
                densities = tuple(
                    (
                        algebra.coordinate_estimate_add(
                            row[0], algebra.coordinate_estimate_multiply(u, row[1])
                        ),
                        algebra.coordinate_estimate_multiply(height, row[1]),
                    )
                    for row in composed
                )
            else:
                densities = composed
    variables = tuple(
        algebra.coordinate_estimate_variable(dimension) for _ in range(dimension)
    )
    arguments = tuple(
        algebra.coordinate_estimate_sum(
            (algebra.coordinate_estimate_constant(Fraction(float(value)), dimension),)
            + tuple(
                algebra.coordinate_estimate_scale(variable, Fraction(float(coefficient)))
                for variable, coefficient in zip(variables, row, strict=True)
            ),
            dimension,
        )
        for value, row in zip(origin, jacobian, strict=True)
    )
    return arguments, jacobian, densities


def exact_functional_moments_estimate(
    family: FormElementFamily,
    k: int,
    r: int,
    exponents: tuple[tuple[int, ...], ...],
) -> algebra.CoordinateEnclosureEstimate:
    """Bound all boundary moment source/composition circuits before execution."""
    topology = reference_cell_topology(hybrid_kind(family))
    blades = tuple(combinations(range(3), k))
    estimates = []
    for dimension, entities in enumerate(topology.entities[:-1]):
        if dimension == 0 or dimension < k:
            continue
        for face in entities:
            arguments, jacobian, densities = _entity_source_estimates(family, k, r, face)
            estimates.extend(arguments)
            estimates.extend(value for row in densities for value in row)
            local_blades = tuple(combinations(range(dimension), k))
            pullback = tuple(
                tuple(
                    Fraction(forms.form_minor(jacobian, ambient, local))
                    for ambient in blades
                )
                for local in local_blades
            )
            for alpha in exponents:
                monomial = algebra.coordinate_estimate_compose(
                    algebra.coordinate_expression_estimate({alpha: Fraction(1)}),
                    arguments,
                )
                estimates.append(monomial)
                for row in densities:
                    for component in range(len(blades)):
                        density = algebra.coordinate_estimate_sum(
                            tuple(
                                algebra.coordinate_estimate_scale(
                                    value, pullback[local][component]
                                )
                                for local, value in enumerate(row)
                            ),
                            dimension,
                        )
                        estimates.append(
                            algebra.coordinate_estimate_multiply(density, monomial)
                        )
    return algebra.coordinate_estimate_bank(tuple(estimates))
