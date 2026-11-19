"""The cell-mesh audit consuming a certified native tetrahedral degree theorem.

Every report decided with the route-selected theorem must carry the same
issues, counts and quality gates as the independent welded-topology and exact
gate evaluation, and record the selection in its identity.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import (
    certify_cell_geometry_validity,
)
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.meshing import audit_cell_mesh, CellMeshAuditPolicy, CellMeshAuditReport


pytestmark = pytest.mark.meshcore

_CORNER = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)


def _positive(points: np.ndarray, cell: tuple[int, int, int, int], /) -> list[int]:
    rows = list(cell)
    corners = points[rows]
    if np.linalg.det(corners[1:] - corners[0]) < 0.0:
        rows[0], rows[1] = rows[1], rows[0]
    return rows


def _two_tetrahedra(offset: np.ndarray, shared: bool, /) -> CellMesh:
    """A corner tetrahedron and its point reflection through ``offset``.

    ``shared`` makes the two meet in the one vertex at the origin.
    """
    second = offset - _CORNER
    if shared:
        points = np.concatenate((_CORNER, second[1:]))
        cells = ((0, 1, 2, 3), (0, 4, 5, 6))
    else:
        points = np.concatenate((_CORNER, second))
        cells = ((0, 1, 2, 3), (4, 5, 6, 7))
    return CellMesh.from_simplices(
        points,
        np.asarray([_positive(points, cell) for cell in cells], dtype=np.int64),
        dimension=3,
    )


def _cube() -> CellMesh:
    """The unit cube split into six tetrahedra around its main diagonal."""
    points = np.asarray(
        [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
        dtype=np.float64,
    )
    paths = ((4, 6), (4, 5), (2, 6), (2, 3), (1, 5), (1, 3))
    cells = [_positive(points, (0, first, second, 7)) for first, second in paths]
    return CellMesh.from_simplices(points, np.asarray(cells, dtype=np.int64), dimension=3)


def _audits(mesh: CellMesh, /) -> tuple[CellMeshAuditReport, CellMeshAuditReport]:
    geometry = CellGeometrySpec.affine(mesh)
    policy = CellMeshAuditPolicy()
    validity = certify_cell_geometry_validity(
        geometry, mesh=mesh, policy=policy.validity_policy
    )
    embedding = certify_global_embedding(
        mesh, geometry, validity, tetrahedral_degree=True
    )
    assert embedding.status == "certified"
    assert "tetrahedral_degree" in embedding.evaluated_checks

    def audit(selected: bool, /) -> CellMeshAuditReport:
        return audit_cell_mesh(
            mesh,
            geometry,
            policy=policy,
            validity=validity,
            global_embedding=embedding,
            tetrahedral_degree=selected,
        )

    return audit(False), audit(True)


@pytest.mark.parametrize(
    ("mesh", "issue"),
    (
        (_cube, None),
        (lambda: _two_tetrahedra(np.zeros((3,)), True), "nonmanifold_vertices"),
        (
            lambda: _two_tetrahedra(np.full((3,), -1.0e-14), False),
            "coincident_vertices",
        ),
    ),
    ids=("cube", "pinched-vertex", "coincident-vertices"),
)
def test_degree_selected_audit_matches_the_evaluated_audit(
    mesh: Callable[[], CellMesh], issue: str | None
) -> None:
    evaluated, selected = _audits(mesh())
    assert not evaluated.tetrahedral_degree and selected.tetrahedral_degree
    for field in ("passed", "issues", "unresolved", "evaluated_checks", "check_counts"):
        assert getattr(selected, field) == getattr(evaluated, field), field
    assert selected.quality.report_id == evaluated.quality.report_id
    assert selected.report_id != evaluated.report_id
    if issue is None:
        assert selected.passed
    else:
        assert issue in selected.issues


def test_degree_selection_requires_the_embedding_certificate() -> None:
    mesh = _cube()
    with pytest.raises(ValueError, match="global embedding certificate"):
        audit_cell_mesh(mesh, CellGeometrySpec.affine(mesh), tetrahedral_degree=True)
