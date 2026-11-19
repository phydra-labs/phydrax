#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact native degree theorems of affine tetrahedral meshes.

Global embedding and region coverage of a mesh of affine tetrahedra are
decided by the exact native verifiers ``phx_mc_verify_tetrahedra`` and
``phx_mc_verify_volume`` (theorems in ``native/meshcore/src/verify_volume.cpp``):
positive cells whose faces pair with opposite orientations form a chain whose
covering degree is the winding number of its boundary, so an embedded boundary
whose shells have winding number zero in front tiles its solid exactly once,
and labels whose interfaces and boundary are exactly the declared oriented
source triangles tile exactly the declared regions. Both proofs run on the
stored binary64 coordinates; nothing is supplied by the construction.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import VerdictStatus, verify_tetrahedra, verify_volume
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ._domain_coverage import PiecewiseLinearDomain
from ._embedding_state import (
    MeshCertificateEntityKind,
    MeshCertificateFinding,
    MeshFindingStatus,
)
from ._mesh_certificates import MeshCertificateStatus


_TETRAHEDRA = frozenset(("tetrahedron", "simplex:3"))


class AffineTetrahedra(NamedTuple):
    """Exact vertex coordinates and cells (mesh block order) of an affine map."""

    points: np.ndarray
    cells: np.ndarray
    cell_ids: np.ndarray


class TetrahedralDegreeProof(NamedTuple):
    """The decided native global-embedding theorem of one mesh."""

    status: MeshCertificateStatus
    findings: tuple[MeshCertificateFinding, ...]
    boundary_facet_count: int
    shell_count: int


def affine_tetrahedra(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> AffineTetrahedra | None:
    """The mesh's tetrahedra when ``geometry`` is exactly its affine vertex map.

    The theorems apply to nonperiodic three-dimensional meshes of tetrahedra
    whose coordinate map is the affine interpolation of the stored vertex
    coordinates (``CellGeometrySpec.affine``), with every reference vertex at
    its mesh vertex; otherwise ``None``.
    """
    if (
        mesh.periodic_topology is not None
        or mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or any(
            block.cell_kind not in _TETRAHEDRA or block.arity != 4
            for block in mesh.blocks
        )
        or cell_geometry_id(geometry) != cell_geometry_id(CellGeometrySpec.affine(mesh))
    ):
        return None
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
    )
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    return AffineTetrahedra(
        np.asarray(mesh.coordinates, dtype=np.float64), cells, cell_ids
    )


def _finding(
    check: str,
    status: MeshFindingStatus,
    kind: MeshCertificateEntityKind,
    identifiers: tuple[int, ...],
    count: int,
    detail: str,
    /,
) -> MeshCertificateFinding:
    return MeshCertificateFinding(
        check,
        status,
        kind,
        identifiers,
        detail=f"{detail} ({count} occurrences; the first is named).",
    )


def prove_tetrahedral_embedding(
    mesh: CellMesh, tetrahedra: AffineTetrahedra, /
) -> TetrahedralDegreeProof:
    """Decide global embedding by the exact native degree theorem."""
    verdict = verify_tetrahedra(tetrahedra.points, tetrahedra.cells)
    cell_ids = tetrahedra.cell_ids
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    findings: list[MeshCertificateFinding] = []
    if verdict.nonpositive_cells:
        findings.append(
            _finding(
                "tetrahedral_orientation",
                "violated",
                "cell",
                (int(cell_ids[verdict.first_nonpositive_cell]),),
                verdict.nonpositive_cells,
                "A tetrahedron is not positively oriented by the exact orientation predicate",
            )
        )
    if verdict.nonmanifold_faces:
        findings.append(
            _finding(
                "tetrahedral_face_pairing",
                "violated",
                "cell",
                (int(cell_ids[verdict.first_nonmanifold_cell]),),
                verdict.nonmanifold_faces,
                "A face lies in more than two cells or two cells induce one orientation on it",
            )
        )
    if verdict.boundary_contacts:
        findings.append(
            _finding(
                "boundary_contact",
                "violated",
                "cell",
                tuple(
                    sorted(
                        {
                            int(cell_ids[verdict.first_contact_cell]),
                            int(cell_ids[verdict.second_contact_cell]),
                        }
                    )
                ),
                verdict.boundary_contacts,
                "Two boundary triangles meet beyond a shared vertex or edge",
            )
        )
    if verdict.misnested_shells:
        findings.append(
            _finding(
                "boundary_shell_degree",
                "violated",
                "cell",
                (int(cell_ids[verdict.first_misnested_shell_cell]),),
                verdict.misnested_shells,
                "A boundary shell has nonzero winding number in front of it, so cells overlap",
            )
        )
    if verdict.outside_domain_points:
        findings.append(
            _finding(
                "exact_predicate_domain",
                "unresolved",
                "vertex",
                (int(vertex_ids[verdict.first_outside_domain_point]),),
                verdict.outside_domain_points,
                "A vertex coordinate lies outside the exact predicate domain",
            )
        )
    if verdict.nonmanifold_boundary_edges:
        findings.append(
            _finding(
                "boundary_edge_manifold",
                "unresolved",
                "cell",
                (int(cell_ids[verdict.first_nonmanifold_boundary_cell]),),
                verdict.nonmanifold_boundary_edges,
                "A boundary edge lies in other than two boundary triangles",
            )
        )
    if verdict.unresolved_shells:
        findings.append(
            _finding(
                "boundary_shell_degree",
                "unresolved",
                "cell",
                (int(cell_ids[verdict.first_unresolved_shell_cell]),),
                verdict.unresolved_shells,
                "A boundary shell has no vertex on no other shell",
            )
        )
    match verdict.status:
        case VerdictStatus.CERTIFIED:
            status: MeshCertificateStatus = "certified"
        case VerdictStatus.VIOLATED:
            status = "violated"
        case VerdictStatus.UNRESOLVED:
            status = "unresolved"
        case _:
            raise ValueError("Unknown native verdict status.")
    return TetrahedralDegreeProof(
        status, tuple(findings), verdict.boundary_faces, verdict.boundary_shells
    )


def _domain_vertex_rows(points: np.ndarray, vertices: np.ndarray, /) -> np.ndarray | None:
    """Rows of ``points`` at exactly the coordinates of ``vertices``, or ``None``.

    Each source vertex must coincide with exactly one mesh vertex; signed zeros
    compare equal.
    """
    mesh_count = points.shape[0]
    combined = np.concatenate((points, vertices)) + 0.0
    order = np.lexsort((combined[:, 2], combined[:, 1], combined[:, 0]))
    ordered = combined[order]
    starts = np.concatenate(
        (np.ones((1,), dtype=np.bool_), np.any(ordered[1:] != ordered[:-1], axis=1))
    )
    groups = np.empty((combined.shape[0],), dtype=np.int64)
    groups[order] = np.cumsum(starts) - 1
    group_count = int(np.count_nonzero(starts))
    mesh_groups = groups[:mesh_count]
    source_groups = groups[mesh_count:]
    if np.any(np.bincount(mesh_groups, minlength=group_count)[source_groups] != 1):
        return None
    rows = np.empty((group_count,), dtype=np.int64)
    rows[mesh_groups] = np.arange(mesh_count, dtype=np.int64)
    return rows[source_groups]


def prove_tetrahedral_coverage(
    mesh: CellMesh,
    tetrahedra: AffineTetrahedra,
    domain: PiecewiseLinearDomain,
    cell_regions: np.ndarray,
    /,
) -> str | None:
    """Identity of the exact native region-coverage proof, or ``None``.

    The theorem applies when every declared source vertex is exactly one mesh
    vertex; it certifies when the mesh's boundary and region interfaces are
    exactly the declared oriented source triangles with the cells' regions on
    their sides. Premise: the mesh is globally embedded. ``None`` decides
    nothing (a mesh whose boundary subdivides the source triangles needs the
    exact partition proof).
    """
    if domain.ambient_dimension != 3:
        return None
    used = np.unique(domain.facets)
    rows = _domain_vertex_rows(tetrahedra.points, np.asarray(domain.vertices)[used])
    if rows is None:
        return None
    vertex_rows = np.full((domain.vertices.shape[0],), -1, dtype=np.int64)
    vertex_rows[used] = rows
    surface = vertex_rows[np.asarray(domain.facets, dtype=np.int64)]
    verdict = verify_volume(
        tetrahedra.points,
        tetrahedra.cells,
        surface,
        cell_regions=cell_regions,
        surface_regions=np.asarray(domain.facet_regions, dtype=np.int64),
    )
    if not verdict.certified:
        return None
    return canonical_fingerprint(
        {
            "kind": "tetrahedral-region-coverage",
            "mesh": mesh.mesh_id,
            "domain": domain.domain_id,
            "regions": array_tree_fingerprint(cell_regions),
            "boundary_faces": verdict.boundary_faces,
        }
    )


__all__ = [
    "AffineTetrahedra",
    "TetrahedralDegreeProof",
    "affine_tetrahedra",
    "prove_tetrahedral_coverage",
    "prove_tetrahedral_embedding",
]
