from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

import phydrax.discretization._mapped_geometry_measure as mapped_measure
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import (
    PerCellRestrictedCellGeometryElement,
    RationalComposedCellGeometryElement,
)
from phydrax.discretization._coordinate_enclosure import (
    ExpressionInput,
    RationalPolynomial,
)
from phydrax.discretization._mapped_geometry_measure import (
    certified_cell_measures,
    certified_cell_moments,
)
from phydrax.discretization.fem import lagrange_element


def test_restricted_polynomial_measures_across_distinct_charts() -> None:
    source = lagrange_element("interval", 2)
    first: tuple[ExpressionInput, ...] = ({(0,): Fraction(0), (1,): Fraction(1, 2)},)
    second: tuple[ExpressionInput, ...] = ({(0,): Fraction(1, 3), (1,): Fraction(1, 2)},)
    element = PerCellRestrictedCellGeometryElement(
        source, "interval", (1, 2, 3), (4, 4, 4), (first, first, second)
    )
    mesh = CellMesh(
        np.asarray(((0.0,), (0.75,), (0.0,), (0.75,), (4 / 9,), (55 / 36,))),
        (
            CellBlock(
                "cells",
                "interval",
                np.arange(6).reshape(3, 2),
                global_ids=np.asarray((1, 2, 3)),
            ),
        ),
    )
    nodes = np.asarray(source.reference_nodes, dtype=np.float64).reshape(-1, 1)
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.tile(np.arange(3), (3, 1))},
        nodes + nodes * nodes,
    )
    validity = mapped_measure.certify_cell_geometry_validity(geometry, mesh=mesh)
    measures = certified_cell_measures(mesh, geometry, validity=validity)
    np.testing.assert_allclose(measures.values, (0.75, 0.75, 13 / 12))


def _rational_box(offset: float = 0.0) -> tuple[CellMesh, CellGeometrySpec]:
    source_element = lagrange_element("hexahedron", 1)
    reference_nodes = np.asarray(source_element.reference_nodes, dtype=np.float64)
    unit: dict[tuple[int, ...], Fraction] = {(0, 0, 0): Fraction(1)}
    element = RationalComposedCellGeometryElement(
        source_element,
        "hexahedron",
        (
            RationalPolynomial(
                {(1, 0, 0): Fraction(1)},
                {(0, 0, 0): Fraction(1), (1, 0, 0): Fraction(1)},
            ),
            RationalPolynomial({(0, 1, 0): Fraction(1)}, unit),
            RationalPolynomial({(0, 0, 1): Fraction(1)}, unit),
        ),
    )
    physical_vertices = np.array(reference_nodes, copy=True)
    physical_vertices[:, 0] /= 1.0 + physical_vertices[:, 0]
    physical_vertices[:, 0] += offset
    mesh = CellMesh(
        physical_vertices,
        (
            CellBlock(
                "mapped",
                "hexahedron",
                np.arange(reference_nodes.shape[0], dtype=np.int32)[None, :],
            ),
        ),
    )
    coordinate_values = np.array(reference_nodes, copy=True)
    coordinate_values[:, 0] += offset
    geometry = CellGeometrySpec(
        {"mapped": element},
        {"mapped": np.arange(reference_nodes.shape[0], dtype=np.int64)[None, :]},
        coordinate_values,
    )
    return mesh, geometry


def test_rational_mapped_cell_measure_matches_independent_gauss_quadrature() -> None:
    mesh, geometry = _rational_box()

    measures = certified_cell_measures(mesh, geometry, absolute_tolerance=1e-13)

    nodes, weights = np.polynomial.legendre.leggauss(96)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    independent = float(np.dot(unit_weights, 1.0 / np.square(1.0 + unit_nodes)))
    assert measures.values.shape == (1,)
    assert abs(float(measures.values[0]) - independent) <= (
        float(measures.absolute_errors[0]) + 5e-14
    )
    np.testing.assert_allclose(measures.values, (0.5,), rtol=0.0, atol=1e-12)


def test_exact_physical_volume_moments_preserve_signed_coordinates() -> None:
    points = np.asarray(
        ((-1, 2, -3), (1, 2, -3), (-1, 5, -3), (-1, 2, 1)), dtype=np.float64
    )
    mesh = CellMesh.from_tetrahedra(points, np.asarray(((0, 1, 2, 3),), dtype=np.int32))
    moments = certified_cell_moments(
        mesh, CellGeometrySpec.affine(mesh), second_moments=True
    )
    # Uniform tetrahedral Dirichlet moments on lengths (2,3,4), translated by
    # (-1,2,-3), give these independent physical integrals.
    np.testing.assert_array_equal(moments.measures, (4.0,))
    np.testing.assert_array_equal(moments.first_moments, ((-2.0, 11.0, -8.0),))
    expected = (
        np.asarray(((8, -29, 18), (-29, 158, -113), (18, -113, 92)), dtype=np.float64)
        / 5.0
    )
    assert moments.second_moments is not None
    np.testing.assert_array_equal(moments.second_moments[0], expected)
    assert moments.exact


def _assert_integral_enclosures(
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    exact_values: tuple[Fraction, ...],
    tolerance: float,
) -> None:
    for index, (value, error, exact) in enumerate(
        zip(values.flat, errors.flat, exact_values, strict=True)
    ):
        assert abs(Fraction(float(value)) - exact) <= Fraction(float(error)), index
        budget = tolerance + 4.0 * np.spacing(abs(float(exact)))
        assert 0.0 <= error <= budget, index


def test_signed_rational_volume_moments_enclose_independent_box_integrals() -> None:
    mesh, geometry = _rational_box(-1.0)
    tolerance = 1e-13
    moments = certified_cell_moments(
        mesh,
        geometry,
        second_moments=True,
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
    )
    # The physical image is [-1,-1/2] x [0,1] x [0,1], independently of its
    # rational parametrization.
    _assert_integral_enclosures(
        moments.measures, moments.measure_errors, (Fraction(1, 2),), tolerance
    )
    _assert_integral_enclosures(
        moments.first_moments,
        moments.first_moment_errors,
        (Fraction(-3, 8), Fraction(1, 4), Fraction(1, 4)),
        tolerance,
    )
    assert moments.second_moments is not None
    assert moments.second_moment_errors is not None
    _assert_integral_enclosures(
        moments.second_moments,
        moments.second_moment_errors,
        (
            Fraction(7, 24),
            Fraction(-3, 16),
            Fraction(-3, 16),
            Fraction(-3, 16),
            Fraction(1, 6),
            Fraction(1, 8),
            Fraction(-3, 16),
            Fraction(1, 8),
            Fraction(1, 6),
        ),
        tolerance,
    )
