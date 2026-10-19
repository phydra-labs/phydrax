#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Inherited embedding evidence preserves decisions and logical contact counts."""

from dataclasses import replace

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode, SegmentIntersectionResult
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_predecessor import (
    CellGeometryPredecessor,
    prepare_cell_geometry_predecessor,
)
from phydrax.discretization._cell_geometry_validity import (
    CellValidityCertificate,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import _global_embedding as embedding_owner
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._global_embedding import (
    certify_global_embedding,
    EmbeddingCounts,
    EmbeddingExpressionWork,
    GlobalEmbeddingCertificate,
)
from phydrax.typing import ConvertibleToArray
from tests._support.mesh_certificate_fixtures import (
    coordinate_budget as coordinate_budget,
    reference_budget as reference_budget,
)


def _certificates(
    mesh: CellMesh,
) -> tuple[CellGeometrySpec, CellValidityCertificate, GlobalEmbeddingCertificate]:
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)
    return geometry, validity, certificate


def _contact_counter(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    counts: list[int] = []
    predicate = embedding_owner.segment_intersections_2d

    def counted(
        a: ConvertibleToArray,
        b: ConvertibleToArray,
        c: ConvertibleToArray,
        d: ConvertibleToArray,
        /,
        *,
        mode: PredicateMode,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> SegmentIntersectionResult:
        counts.append(np.asarray(a).shape[0])
        return predicate(a, b, c, d, mode=mode, budget=budget)

    monkeypatch.setattr(embedding_owner, "segment_intersections_2d", counted)
    return counts


def _interval_pair(
    crossing: bool, budget: CoordinateEnclosureBudget
) -> tuple[CellMesh, CellGeometrySpec, CellValidityCertificate, CellGeometryPredecessor]:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)), dtype=np.float64
    )
    source = CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "segments",
                "interval",
                np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int32),
                global_ids=np.asarray((10, 20, 30), dtype=np.int64),
            ),
        ),
    )
    source_geometry, source_validity, source_embedding = _certificates(source)
    assert source_embedding.status == "certified"
    target_coordinates = coordinates.copy()
    target_coordinates[3] = (0.0, 1.0) if crossing else (3.0, 1.25)
    target = CellMesh(
        target_coordinates,
        (
            phx.discretization.CellBlock(
                "segments",
                "interval",
                np.asarray(((1, 2), (2, 3), (0, 1)), dtype=np.int32),
                global_ids=np.asarray((20, 30, 10), dtype=np.int64),
            ),
        ),
    )
    geometry = CellGeometrySpec.affine(target)
    validity = certify_cell_geometry_validity(geometry, mesh=target)
    predecessor = prepare_cell_geometry_predecessor(
        source,
        source_geometry,
        source_validity,
        source_embedding,
        target,
        geometry,
        budget=budget,
    )
    assert predecessor.source_rows == (1, -1, 0)
    return target, geometry, validity, predecessor


@pytest.mark.parametrize("crossing", (False, True), ids=("disjoint", "changed-crossing"))
def test_inherited_cell_contacts_preserve_full_proof_and_reduce_predicates(
    crossing: bool,
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, geometry, validity, predecessor = _interval_pair(crossing, reference_budget)
    counts = _contact_counter(monkeypatch)
    full = certify_global_embedding(mesh, geometry, validity)
    full_work = sum(counts)
    counts.clear()
    reused = certify_global_embedding(mesh, geometry, validity, predecessor=predecessor)

    assert reused.status == ("violated" if crossing else "certified")
    assert reused.certificate_id == full.certificate_id
    assert reused.evaluated_checks == full.evaluated_checks
    assert tuple(f.finding_id for f in reused.findings) == tuple(
        f.finding_id for f in full.findings
    )
    assert reused.candidate_pair_count == full.candidate_pair_count
    assert sum(counts) == full_work - 1
    assert sum(counts) > 0
    if crossing:
        assert tuple(f.check for f in reused.findings) == ("cell_contact",)


def test_newly_exposed_facet_keeps_changed_boundary_contacts(
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), dtype=np.float64
    )
    source = CellMesh.from_triangles(
        coordinates,
        np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32),
        cell_global_ids=np.asarray((10, 20), dtype=np.int64),
    )
    source_geometry, source_validity, source_embedding = _certificates(source)
    assert source_embedding.status == "certified"
    # Renumber vertices: source interior edge (1, 2) becomes target boundary.
    order = np.asarray((2, 0, 3, 1), dtype=np.int64)
    target = CellMesh.from_triangles(
        coordinates[order],
        np.asarray(((1, 3, 0),), dtype=np.int32),
        vertex_global_ids=order,
        cell_global_ids=np.asarray((10,), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(target)
    validity = certify_cell_geometry_validity(geometry, mesh=target)
    predecessor = prepare_cell_geometry_predecessor(
        source,
        source_geometry,
        source_validity,
        source_embedding,
        target,
        geometry,
        budget=reference_budget,
    )
    assert predecessor.source_rows == (0,)
    counts = _contact_counter(monkeypatch)
    full = certify_global_embedding(target, geometry, validity)
    full_work = sum(counts)
    counts.clear()
    reused = certify_global_embedding(target, geometry, validity, predecessor=predecessor)

    assert reused.status == "certified"
    assert reused.certificate_id == full.certificate_id
    assert reused.candidate_pair_count == full.candidate_pair_count == 3
    assert full_work == 3
    assert sum(counts) == 2
    assert reused.evaluated_checks == full.evaluated_checks
    assert reused.ray_test_count == full.ray_test_count
    assert "exterior_degree" in reused.evaluated_checks


def test_incompatible_policy_keeps_full_contact_route(
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, geometry, validity, predecessor = _interval_pair(False, reference_budget)
    limits = MeshCertificateLimits(maximum_candidate_pairs=12)
    counts = _contact_counter(monkeypatch)
    full = certify_global_embedding(mesh, geometry, validity, limits=limits)
    full_work = sum(counts)
    counts.clear()
    reused = certify_global_embedding(
        mesh, geometry, validity, limits=limits, predecessor=predecessor
    )

    assert reused.certificate_id == full.certificate_id
    assert sum(counts) == full_work


def test_tampered_imported_embedding_payload_refuses_before_predicates(
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, geometry, validity, predecessor = _interval_pair(False, reference_budget)
    source = predecessor.source_embedding
    assert source is not None
    altered = GlobalEmbeddingCertificate(
        source.binding,
        source.validity_certificate_id,
        source.findings,
        source.evaluated_checks,
        counts=EmbeddingCounts(
            source.cell_count,
            source.boundary_facet_count,
            source.shell_count,
            source.candidate_pair_count,
            source.ray_test_count,
            source.subdivision_piece_count,
            source.periodic_image_count,
        ),
        expression_work=EmbeddingExpressionWork(
            source.source_expression_work_units + 1,
            source.source_expression_peak_bytes,
        ),
        boundary_degree=source.boundary_degree,
    )
    # Invocation work is outside certificate identity, but inside import digest.
    assert altered.certificate_id == source.certificate_id
    tampered = replace(predecessor, source_embedding=altered)
    counts = _contact_counter(monkeypatch)

    with pytest.raises(ValueError, match="content_id"):
        certify_global_embedding(mesh, geometry, validity, predecessor=tampered)
    assert counts == []


def test_inherited_surface_cells_remove_triangle_contact_predicates(
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = CellMesh.from_triangles(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32),
    )
    geometry, validity, source = _certificates(mesh)
    assert source.status == "certified"
    predecessor = prepare_cell_geometry_predecessor(
        mesh, geometry, validity, source, mesh, geometry, budget=reference_budget
    )
    counts: list[int] = []
    predicate = embedding_owner.audit_topology_owner.triangle_pairs_intersect

    def counted(
        points: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        counts.append(first.shape[0])
        return predicate(points, first, second, budget=budget)

    monkeypatch.setattr(
        embedding_owner.audit_topology_owner, "triangle_pairs_intersect", counted
    )
    full = certify_global_embedding(mesh, geometry, validity)
    full_work = sum(counts)
    counts.clear()
    reused = certify_global_embedding(mesh, geometry, validity, predecessor=predecessor)

    assert reused.status == "certified"
    assert reused.certificate_id == full.certificate_id
    assert reused.candidate_pair_count == full.candidate_pair_count == 1
    assert reused.evaluated_checks == full.evaluated_checks
    assert full_work == 1
    assert counts == []


def test_changed_contained_component_keeps_exterior_degree_checks(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    coordinates = np.asarray(
        ((0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (8.0, 0.0), (9.0, 0.0), (8.0, 1.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
    source = CellMesh.from_triangles(coordinates, cells)
    source_geometry, source_validity, source_embedding = _certificates(source)
    assert source_embedding.status == "certified"
    target_coordinates = coordinates.copy()
    target_coordinates[3:] = ((0.5, 0.5), (1.0, 0.5), (0.5, 1.0))
    target = CellMesh.from_triangles(target_coordinates, cells)
    geometry = CellGeometrySpec.affine(target)
    validity = certify_cell_geometry_validity(geometry, mesh=target)
    predecessor = prepare_cell_geometry_predecessor(
        source,
        source_geometry,
        source_validity,
        source_embedding,
        target,
        geometry,
        budget=reference_budget,
    )
    assert predecessor.source_rows == (0, -1)
    full = certify_global_embedding(target, geometry, validity)
    reused = certify_global_embedding(target, geometry, validity, predecessor=predecessor)

    assert reused.status == "violated"
    assert reused.certificate_id == full.certificate_id
    assert reused.evaluated_checks == full.evaluated_checks
    assert reused.ray_test_count == full.ray_test_count
    assert reused.ray_test_count > 0
    assert reused.candidate_pair_count == full.candidate_pair_count
    assert tuple(f.check for f in reused.findings) == ("nested_component",)
