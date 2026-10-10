# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact entity moment actions on collapsed/product monomial form components."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import factorial, prod

import numpy as np

from .. import _coordinate_enclosure as algebra
from .._reference_cell import reference_cell_topology
from . import _form_elements as forms
from ._form_expressions import functional_density_expressions
from ._hybrid_forms import entity_chart, hybrid_kind, trace_basis


type ExactMomentRows = tuple[tuple[tuple[Fraction, ...], ...], ...]


def _integral(polynomial: algebra.Polynomial, dimension: int, simplex: bool) -> Fraction:
    if simplex:
        return sum(
            (
                coefficient
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
            coefficient / prod(power + 1 for power in alpha)
            for alpha, coefficient in polynomial.items()
        ),
        Fraction(0),
    )


def _entity_source(
    family: forms.FormElementFamily,
    k: int,
    r: int,
    face: tuple[int, ...],
) -> tuple[
    tuple[algebra.Polynomial, ...],
    np.ndarray,
    tuple[tuple[algebra.Polynomial, ...], ...],
    bool,
]:
    origin, jacobian = entity_chart(family, face)
    dimension = jacobian.shape[1]
    trace = trace_basis(family, k, r, face)
    indices = forms._interior_dof_indices(trace.dof_labels, len(face))
    raw = functional_density_expressions(trace, trace.entity_vertices[-1][0])
    densities = []
    for index in indices:
        row = []
        for value in raw[index]:
            if isinstance(value, algebra.RationalPolynomial):
                raise ValueError(
                    "Canonical simplex/tensor trace moments must have polynomial densities."
                )
            row.append(value)
        densities.append(tuple(row))
    simplex = len(face) == dimension + 1
    if family != "pyramid-trimmed":
        return (
            algebra.affine_arguments(origin, jacobian),
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
        u, z = algebra.axes(2)
        height = algebra.add(algebra.constant(1, 2), algebra.scale(z, -1))
        arguments = (algebra.multiply(u, height), z)
        composed = tuple(
            tuple(algebra.compose(value, arguments) for value in row) for row in densities
        )
        if k == 0:
            densities = [
                tuple(algebra.multiply(value, height) for value in row)
                for row in composed
            ]
        elif k == 1:
            densities = [
                (
                    algebra.add(row[0], algebra.multiply(u, row[1])),
                    algebra.multiply(height, row[1]),
                )
                for row in composed
            ]
        else:
            densities = list(composed)
        simplex = False
    return algebra.affine_arguments(origin, jacobian), jacobian, tuple(densities), simplex


def exact_functional_moments(
    family: forms.FormElementFamily,
    k: int,
    r: int,
    exponents: tuple[tuple[int, ...], ...],
    labels: tuple[forms.DofLabel, ...],
    test_exponents: tuple[tuple[int, ...], ...],
    test_source_bank: tuple[tuple[Fraction, ...], ...],
) -> ExactMomentRows:
    topology = reference_cell_topology(hybrid_kind(family))
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    blades = tuple(combinations(range(3), k))
    rows = [[[Fraction(0) for _ in blades] for _ in exponents] for _ in labels]
    for dimension, entities in enumerate(topology.entities):
        for face in entities:
            indices = forms._entity_dof_indices(labels, face)
            if not indices:
                continue
            if dimension == 0:
                point = vertices[face[0]]
                if family == "pyramid-trimmed" and face == (4,):
                    # This extension is independent of direction on the actual
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
                continue
            if dimension == 3:
                for column, index in enumerate(indices):
                    for row, alpha in enumerate(exponents):
                        for component in range(len(blades)):
                            polynomial = {
                                tuple(
                                    a + b for a, b in zip(alpha, beta, strict=True)
                                ): test_source_bank[test * len(blades) + component][
                                    column
                                ]
                                for test, beta in enumerate(test_exponents)
                                if test_source_bank[test * len(blades) + component][
                                    column
                                ]
                            }
                            if family == "prism-trimmed":
                                rows[index][row][component] = sum(
                                    (
                                        value
                                        * Fraction(
                                            factorial(power[0]) * factorial(power[1]),
                                            factorial(power[0] + power[1] + 2)
                                            * (power[2] + 1),
                                        )
                                        for power, value in polynomial.items()
                                    ),
                                    Fraction(0),
                                )
                            else:
                                rows[index][row][component] = _integral(
                                    polynomial, 3, False
                                )
                continue
            arguments, jacobian, densities, simplex = _entity_source(family, k, r, face)
            local_blades = tuple(combinations(range(dimension), k))
            pullback = tuple(
                tuple(
                    Fraction(forms._minor(jacobian, ambient, local)) for ambient in blades
                )
                for local in local_blades
            )
            for row, alpha in enumerate(exponents):
                monomial = algebra.compose({alpha: Fraction(1)}, arguments)
                for test, index in enumerate(indices):
                    for component in range(len(blades)):
                        density = algebra.sum_polynomials(
                            tuple(
                                algebra.scale(value, pullback[local][component])
                                for local, value in enumerate(densities[test])
                            )
                        )
                        rows[index][row][component] = _integral(
                            algebra.multiply(density, monomial), dimension, simplex
                        )
    return tuple(tuple(tuple(component) for component in row) for row in rows)


def generator_functional_action(
    moments: ExactMomentRows,
    source_bank: tuple[tuple[Fraction, ...], ...],
    component_count: int,
) -> forms._HostArray:
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
