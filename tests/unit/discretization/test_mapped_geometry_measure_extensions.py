from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import RationalComposedCellGeometryElement
from phydrax.discretization._coordinate_enclosure import RationalPolynomial
from phydrax.discretization._mapped_geometry_measure import certified_cell_measures
from phydrax.discretization.fem import lagrange_element


def test_rational_embedded_curve_measure_matches_independent_gauss_rule() -> None:
    source = lagrange_element("interval", 1)
    chart = RationalPolynomial(
        {(1,): Fraction(1)},
        {(0,): Fraction(1), (1,): Fraction(1)},
    )
    element = RationalComposedCellGeometryElement(source, "interval", (chart,))
    local_coordinates = np.asarray(((0.0, 0.0), (1.0, 2.0)), dtype=np.float64)
    mesh = CellMesh(
        np.asarray(((0.0, 0.0), (0.5, 1.0)), dtype=np.float64),
        (
            CellBlock(
                "curve",
                "interval",
                np.asarray(((0, 1),), dtype=np.int32),
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"curve": element},
        {"curve": np.asarray(((0, 1),), dtype=np.int64)},
        local_coordinates,
    )

    measures = certified_cell_measures(mesh, geometry, absolute_tolerance=2e-12)

    nodes, weights = np.polynomial.legendre.leggauss(128)
    reference = 0.5 * (nodes + 1.0)
    independent = math.sqrt(5.0) * float(
        np.dot(0.5 * weights, 1.0 / np.square(1.0 + reference))
    )
    discrepancy = abs(float(measures.values[0]) - independent)
    assert measures.values.shape == (1,)
    assert np.isfinite(measures.values[0]) and measures.values[0] > 0.0
    assert np.isfinite(measures.absolute_errors[0]) and measures.absolute_errors[0] > 0.0
    assert discrepancy <= float(measures.absolute_errors[0]) + 2e-14
    np.testing.assert_allclose(
        measures.values,
        (math.sqrt(5.0) / 2.0,),
        rtol=0.0,
        atol=2e-12,
    )


def test_rational_pyramid_volume_matches_independent_gauss_rule() -> None:
    source = lagrange_element("hexahedron", 1)
    unit: dict[tuple[int, int, int], Fraction] = {
        (0, 0, 0): Fraction(1)
    }
    denominator = {
        (0, 0, 0): Fraction(1),
        (1, 0, 0): Fraction(1),
    }
    x = RationalPolynomial(
        {
            (1, 0, 0): Fraction(11, 10),
            (2, 0, 0): Fraction(9, 10),
            (1, 0, 1): Fraction(-1, 10),
            (2, 0, 1): Fraction(1, 10),
        },
        denominator,
    )
    y = RationalPolynomial({(0, 1, 0): Fraction(1)}, unit)
    z = RationalPolynomial({(0, 0, 1): Fraction(1)}, unit)
    element = RationalComposedCellGeometryElement(
        source, "pyramid", (x, y, z)
    )
    pyramid_vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 1.0),
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        pyramid_vertices,
        (
            CellBlock(
                "pyramid",
                "pyramid",
                np.asarray(((0, 1, 2, 3, 4),), dtype=np.int32),
            ),
        ),
    )
    source_coordinates = np.asarray(source.reference_nodes, dtype=np.float64)
    geometry = CellGeometrySpec(
        {"pyramid": element},
        {"pyramid": np.arange(source_coordinates.shape[0], dtype=np.int64)[None, :]},
        source_coordinates,
    )

    measures = certified_cell_measures(mesh, geometry, absolute_tolerance=1e-11)

    nodes, weights = np.polynomial.legendre.leggauss(64)
    reference_nodes = 0.5 * (nodes + 1.0)
    reference_weights = 0.5 * weights
    u, v, w = np.meshgrid(
        reference_nodes, reference_nodes, reference_nodes, indexing="ij"
    )
    x_reference = (1.0 - w) * u + 0.5 * w
    determinant = 1.0 + 0.1 * (1.0 - w) * (
        1.0 - 2.0 * x_reference - x_reference**2
    ) / (1.0 + x_reference) ** 2
    tensor_weights = (
        reference_weights[:, None, None]
        * reference_weights[None, :, None]
        * reference_weights[None, None, :]
    )
    independent = float(
        np.sum(tensor_weights * (1.0 - w) ** 2 * determinant)
    )
    discrepancy = abs(float(measures.values[0]) - independent)
    assert measures.values.shape == (1,)
    assert np.isfinite(measures.values[0]) and measures.values[0] > 0.0
    assert np.isfinite(measures.absolute_errors[0]) and measures.absolute_errors[0] > 0.0
    assert discrepancy <= float(measures.absolute_errors[0]) + 5e-14
