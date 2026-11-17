from __future__ import annotations

from itertools import combinations
from random import Random

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.geometry._boundary_degree_embedding import (
    build_oriented_boundary,
    OrientedBoundary,
)
from phydrax.geometry._embedding_cells import prepare_embedding_cells
from phydrax.geometry._exact_polyhedral_geometry import triangle_contact_points3


_FIRST = ((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2))


def _boundary(second: tuple[tuple[float, float, float], ...]) -> OrientedBoundary:
    coordinates = np.asarray((*_FIRST, *second), dtype=np.float64)
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "tetrahedra",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3), (4, 5, 6, 7)), dtype=np.int32),
            ),
        ),
    )
    return build_oriented_boundary(
        mesh, prepare_embedding_cells(mesh, CellGeometrySpec.affine(mesh))
    )


def _reference_findings(boundary: OrientedBoundary) -> tuple[tuple[int, ...], ...]:
    # Both tetrahedra use distinct declared vertex IDs; within each tetrahedron,
    # adjacent triangles share an allowed edge, and across them no contact is allowed.
    return tuple(
        tuple(sorted((first.facet_id, second.facet_id)))
        for first, second in combinations(boundary.triangles, 2)
        if not set(first.vertex_ids).intersection(second.vertex_ids)
        and triangle_contact_points3(first.points, second.points)
    )


def _contact_findings(boundary: OrientedBoundary) -> tuple[tuple[int, ...], ...]:
    return tuple(
        finding.entity_ids
        for finding in boundary.findings
        if finding.check == "exterior_boundary_self_contact"
    )


@pytest.mark.parametrize(
    ("second", "has_contact"),
    (
        (((1, 0, 0), (3, 0, 0), (1, 2, 0), (1, 0, 2)), True),  # coplanar overlap
        (((2, 0, 0), (4, 0, 0), (2, 2, 0), (2, 0, 2)), True),  # vertex touch
        (((2, 0, 0), (0, 2, 0), (2, 2, 0), (2, 0, -2)), True),  # edge touch
        (
            ((0.5, 0.5, 0.5), (2.5, 0.5, 0.5), (0.5, 2.5, 0.5), (0.5, 0.5, 2.5)),
            True,
        ),  # crossing
        (((3, 0, 0), (5, 0, 0), (3, 2, 0), (3, 0, 2)), False),  # separated
    ),
    ids=("coplanar-overlap", "vertex-touch", "edge-touch", "crossing", "separated"),
)
def test_boundary_contact_findings_match_fraction_reference(
    second: tuple[tuple[float, float, float], ...], has_contact: bool
) -> None:
    boundary = _boundary(second)
    expected = _reference_findings(boundary)
    assert bool(expected) is has_contact
    assert _contact_findings(boundary) == expected


def test_boundary_contact_is_not_lost_at_exact_touch() -> None:
    touching = _boundary(((2, 0, 0), (4, 0, 0), (2, 2, 0), (2, 0, 2)))
    separated = _boundary(
        ((2.015625, 0, 0), (4.015625, 0, 0), (2.015625, 2, 0), (2.015625, 0, 2))
    )
    assert _contact_findings(touching) == _reference_findings(touching)
    assert _contact_findings(touching)
    assert _contact_findings(separated) == _reference_findings(separated) == ()


def test_random_translated_boundaries_match_fraction_reference() -> None:
    rng = Random(91742)
    for _ in range(20):
        shift = tuple(rng.randint(-5, 9) / 4 for _ in range(3))
        second = tuple(
            (point[0] + shift[0], point[1] + shift[1], point[2] + shift[2])
            for point in _FIRST
        )
        boundary = _boundary(second)
        assert _contact_findings(boundary) == _reference_findings(boundary)
