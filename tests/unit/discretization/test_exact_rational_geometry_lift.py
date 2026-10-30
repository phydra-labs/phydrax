from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._exact_rational_geometry import (
    ExactRationalCellGeometrySource,
    rational_geometry_lift,
)
from phydrax.discretization.fem._reference import lagrange_element


def _tetrahedron() -> CellMesh:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    block = CellBlock(
        "solid",
        "tetrahedron",
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    )
    return CellMesh(coordinates, (block,))


def _barycentric_actions(
    reference_nodes: np.ndarray,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    actions = []
    for node in reference_nodes:
        point = tuple(Fraction.from_float(float(value)) for value in node)
        barycentric = (Fraction(1) - sum(point, Fraction(0)), *point)
        actions.append(
            tuple(
                (index, value.numerator, value.denominator)
                for index, value in enumerate(barycentric)
                if value
            )
        )
    return tuple(actions)


def test_rational_csr_lift_preserves_exact_parent_interpolation() -> None:
    mesh = _tetrahedron()
    parent = CellGeometrySpec.affine(mesh)
    element = lagrange_element("tetrahedron", 2)
    actions = _barycentric_actions(np.asarray(element.reference_nodes))
    geometry = rational_geometry_lift(
        parent,
        mesh,
        {"solid": element},
        {"solid": np.arange(element.local_dof_count, dtype=np.int32)[None, :]},
        actions,
        source_id="rational-lift-source",
        source_revision="revision-1",
    )

    exact_parent = parent.rational_source_coordinates()
    expected = tuple(
        tuple(
            sum(
                (
                    Fraction(numerator, denominator) * exact_parent[index][axis]
                    for index, numerator, denominator in row
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        for row in actions
    )
    assert geometry.rational_source_coordinates() == expected
    source = geometry.source_map
    assert isinstance(source, ExactRationalCellGeometrySource)
    assert source.source_coordinates() == expected
    np.testing.assert_array_equal(
        np.asarray(geometry.coordinates),
        np.asarray([[float(value) for value in row] for row in expected]),
    )


def test_rational_csr_lift_rejects_nonaffine_coordinate_rows() -> None:
    mesh = _tetrahedron()
    parent = CellGeometrySpec.affine(mesh)
    element = lagrange_element("tetrahedron", 2)
    actions = list(_barycentric_actions(np.asarray(element.reference_nodes)))
    actions[0] = ((0, 1, 3),)
    with pytest.raises(ValueError, match="sum exactly to one"):
        rational_geometry_lift(
            parent,
            mesh,
            {"solid": element},
            {"solid": np.arange(element.local_dof_count, dtype=np.int32)[None, :]},
            tuple(actions),
            source_id="rational-lift-source",
            source_revision="revision-1",
        )
