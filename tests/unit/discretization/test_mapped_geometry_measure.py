from __future__ import annotations

from fractions import Fraction

import numpy as np

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
from phydrax.discretization._mapped_geometry_measure import certified_cell_measures
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


def test_rational_mapped_cell_measure_matches_independent_gauss_quadrature() -> None:
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
    geometry = CellGeometrySpec(
        {"mapped": element},
        {"mapped": np.arange(reference_nodes.shape[0], dtype=np.int64)[None, :]},
        reference_nodes,
    )

    measures = certified_cell_measures(mesh, geometry, absolute_tolerance=1e-13)

    nodes, weights = np.polynomial.legendre.leggauss(96)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    independent = float(np.dot(unit_weights, 1.0 / np.square(1.0 + unit_nodes)))
    assert not measures.exact
    assert measures.values.shape == (1,)
    assert abs(float(measures.values[0]) - independent) <= (
        float(measures.absolute_errors[0]) + 5e-14
    )
    np.testing.assert_allclose(measures.values, (0.5,), rtol=0.0, atol=1e-12)
