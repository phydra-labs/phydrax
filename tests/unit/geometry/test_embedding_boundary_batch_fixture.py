from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import (
    add,
    axes,
    constant,
    evaluate_expression,
    Expression,
    multiply,
    polynomial_bounds,
    scale,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._boundary_degree_embedding import build_oriented_boundary
from phydrax.geometry._embedding_cells import EmbeddingCell


def _cell(
    global_id: int, vertex_ids: tuple[int, ...], coordinates: tuple[Expression, ...],
    *, affine: bool = False,
) -> EmbeddingCell:
    topology = reference_cell_topology("tetrahedron")
    references = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    corners = tuple(
        tuple(evaluate_expression(expression, point) for expression in coordinates)
        for point in references
    )
    bounds = tuple(polynomial_bounds(expression, "simplex", 3) for expression in coordinates)
    return EmbeddingCell(
        global_id, "mapped", 0, "tetrahedron", 3, "simplex", tuple(range(4)),
        vertex_ids, references, coordinates, corners,
        (Fraction(0), Fraction(0), Fraction(0)) if affine else None,
        ((Fraction(0), Fraction(1), Fraction(0)),
         (Fraction(1), Fraction(0), Fraction(0)),
         (Fraction(0), Fraction(0), Fraction(-1))) if affine else None,
        tuple(lower for lower, _ in bounds), tuple(upper for _, upper in bounds),
        tuple(tuple(face) for face in topology.entities[2]),
    )


def _mesh(extra: bool) -> CellMesh:
    points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1),
         (3, 0, 0), (4, 0, 0), (3, 1, 0), (3, 0, 1)), dtype=np.float64
    )
    connectivity = ((0, 1, 2, 3), (0, 2, 1, 4))
    if extra:
        connectivity += ((5, 6, 7, 8),)
    else:
        points = points[:5]
    return CellMesh(
        points,
        (CellBlock("cells", "tetrahedron", np.asarray(connectivity, dtype=np.int32)),),
        vertex_global_ids=np.asarray(tuple(range(10, 10 + len(points))), dtype=np.int64),
    )


@pytest.mark.parametrize("fault", (False, True))
def test_boundary_evidence_is_independent_of_other_facets(fault: bool) -> None:
    x, y, z = axes(3)
    curved = add(z, scale(multiply(x, z), Fraction(1, 8)))
    first = _cell(1, (10, 11, 12, 13), (x, y, curved))
    bump = Fraction(1, 4096) if fault else Fraction(0)
    extra = _cell(3, (15, 16, 17, 18), (add(x, constant(3, 3)), y, curved))

    second = _cell(
        2, (10, 12, 11, 14),
        (y, x, add(scale(z, -1), scale(multiply(x, y), bump))),
        affine=not fault,
    )
    isolated = build_oriented_boundary(_mesh(False), (first, second))
    combined = build_oriented_boundary(_mesh(True), (first, second, extra))
    isolated_faces = {frozenset(face.vertex_ids): face.map_evidence for face in isolated.faces}
    combined_faces = {frozenset(face.vertex_ids): face.map_evidence for face in combined.faces}
    assert all(combined_faces[key] == evidence for key, evidence in isolated_faces.items())
    isolated_mismatch = tuple(finding for finding in isolated.findings if finding.check == "mismatched_interior_trace")
    combined_mismatch = tuple(finding for finding in combined.findings if finding.check == "mismatched_interior_trace")
    assert combined_mismatch == isolated_mismatch
    assert len(isolated_mismatch) == int(fault)
