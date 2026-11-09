from __future__ import annotations

from fractions import Fraction

import numpy as np

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import RationalComposedCellGeometryElement
from phydrax.discretization._coordinate_enclosure import RationalPolynomial
from phydrax.discretization.fem import lagrange_element


def _rational_box(
    offset: float = 0.0, scale: float = 1.0
) -> tuple[CellMesh, CellGeometrySpec]:
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
    physical_vertices *= scale
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
    coordinate_values *= scale
    coordinate_values[:, 0] += offset
    geometry = CellGeometrySpec(
        {"mapped": element},
        {"mapped": np.arange(reference_nodes.shape[0], dtype=np.int64)[None, :]},
        coordinate_values,
    )
    return mesh, geometry
