#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer-visible global certification contracts."""

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax._resource_ticket import HostBatchEstimateError, HostBatchTicket
from phydrax.discretization import _coordinate_enclosure as algebra
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.geometry import _planar_coverage as planar_coverage
from phydrax.geometry._embedding_state import EmbeddingState, MeshCertificateFinding
from phydrax.geometry._source_fidelity import (
    SourceFidelityBounds,
    SourceFidelityCertificate,
)
from phydrax.meshing._certification import (
    MeshCertificationOutcome,
    MeshCertificationPremiseFailure,
    MeshCertificationReport,
    MeshCertificationReportFailure,
)
from phydrax.meshing._certification_inputs import MeshCertificationInputs
from phydrax.meshing._lineage_records import EntityLineage, EntityLineageKind, MeshLineage
from phydrax.meshing._plc_mapped_support import (
    _cell_maps,
    _require_support_work_bounds,
    bind_mapped_plc_support,
    MappedPlcSupport,
    MappedPlcSupportProof,
    prepare_mapped_plc_support,
)
from phydrax.meshing.providers._native_publication import (
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from phydrax.meshing.providers._native_route_policy import NativePublicationPolicy
from phydrax.meshing.qualification import sphere_surface
from phydrax.units import METER
from tests._support.domain_coverage import split_square_domain, SPLIT_SQUARE_POINTS
from tests._support.mesh_certificate_fixtures import (
    coordinate_budget as coordinate_budget,
    octahedron,
    plc_audit,
    prepared_publication_authority,
    prepared_tetrahedral_acceptance,
    reference_budget as reference_budget,
    SPLIT_SQUARE_CELLS,
    TETRAHEDRON,
)
from tests._support.meshing_fixtures import finding_checks


def test_volume_route_refuses_an_audit_that_skipped_closure() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    geometry, audit = plc_audit(mesh)

    with pytest.raises(ValueError, match="open_boundary"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
            domain=split_square_domain(),
            cell_regions=np.asarray((0, 0, 1, 1)),
        )


@pytest.mark.parametrize(
    ("regions", "stage_status"),
    [
        pytest.param(
            (0, 0, 1, 1), phx.meshing.MeshingStageStatus.PASSED, id="conforming"
        ),
        pytest.param((0, 0, 0, 0), phx.meshing.MeshingStageStatus.FAILED, id="omitted"),
    ],
)
def test_plc_acceptance_schedule_composes_certificates(
    regions: tuple[int, ...], stage_status: Any
) -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    geometry, audit = plc_audit(
        mesh, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=split_square_domain(),
        cell_regions=np.asarray(regions),
    )
    stage = phx.meshing.acceptance_stage_report(report)

    assert stage.status == stage_status
    assert {value.check for value in report.outcomes} == set(
        report.schedule.required_checks
    )
    assert ("region_measure[left]", 0.5) in report.requested
    if report.passed:
        report.require_passed()
        return
    failing = {value.check: value for value in report.failing_outcomes}
    assert set(failing) == {"domain_coverage"}
    with pytest.raises(phx.meshing.MeshingFailure) as raised:
        report.require_passed()
    assert raised.value.stage == "certification"
    assert ("region_measure[left]", 1.0) in raised.value.evidence.achieved


def test_certificate_trace_keeps_independent_resource_quantities() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    geometry, audit = plc_audit(
        mesh,
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=split_square_domain(),
        cell_regions=np.asarray((0, 0, 1, 1), dtype=np.int64),
        limits=phx.geometry.MeshCertificateLimits(maximum_work_units=1),
    )
    stage = phx.meshing.acceptance_stage_report(report)
    assert stage.status is phx.meshing.MeshingStageStatus.UNRESOLVED
    assert report.embedding is not None
    refusal = next(
        finding
        for finding in report.embedding.findings
        if finding.resource == "coefficient_work"
    )
    assert dict(refusal.requested)["limit"] == 1
    assert dict(refusal.requested)["requested"] > 1
    assert dict(refusal.achieved)["completed"] <= 1
    with pytest.raises(phx.meshing.MeshingFailure) as raised:
        report.require_passed()
    assert raised.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert any(
        name.endswith(":coefficient_work:limit") and value == 1
        for name, value in raised.value.evidence.requested
    )
    assert any(
        name.endswith(":coefficient_work:completed") and value <= 1
        for name, value in raised.value.evidence.achieved
    )


def test_route_refuses_inputs_its_checks_do_not_use() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    geometry, audit = plc_audit(mesh)

    with pytest.raises(ValueError, match="exactly the inputs"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=phx.meshing.MeshCertificationSchedule("curve"),
            domain=split_square_domain(),
            cell_regions=np.asarray((0, 0, 1, 1)),
        )


def test_retained_certification_renews_source_coverage_on_actual_refinement() -> None:

    source = phx.discretization.CellMesh.from_tetrahedra(
        TETRAHEDRON, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    geometry, audit = plc_audit(
        source, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    facets = np.asarray(
        phx.discretization.reference_cell_topology("tetrahedron").entities[2],
        dtype=np.int64,
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        TETRAHEDRON,
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("body",),
        source_id="independent-refinement-source",
    )
    original = phx.meshing.certify_meshing_acceptance(
        source,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
    )
    target = phx.discretization.CellMesh.from_tetrahedra(
        np.concatenate((TETRAHEDRON, np.asarray(((0.5, 0.0, 0.0),), dtype=np.float64))),
        np.asarray(((0, 4, 2, 3), (4, 1, 2, 3)), dtype=np.int32),
    )
    target_geometry, target_audit = plc_audit(
        target, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    lineage = MeshLineage(
        source.topology_id,
        target.topology_id,
        (
            EntityLineage(
                3,
                source.entity_set(3).entity_set_id,
                target.entity_set(3).entity_set_id,
                np.asarray((0, 0), dtype=np.int64),
                np.asarray((0, 1), dtype=np.int64),
                np.asarray((int(EntityLineageKind.REFINED_FROM),) * 2, dtype=np.int32),
            ),
        ),
    )
    renewed = original.request.recertify_transition(
        target, target_geometry, target_audit, lineage=lineage
    )

    assert original.passed and renewed.passed
    assert renewed.mesh_id == target.mesh_id
    assert renewed.geometry_id != original.geometry_id
    assert isinstance(renewed.coverage, phx.geometry.DomainCoverageCertificate)
    assert renewed.coverage.domain_id == domain.domain_id
    assert renewed.coverage.achieved_region_measures == (float(Fraction(1, 6)),)
    assert np.array_equal(
        np.asarray(renewed.request.cell_regions), np.asarray((0, 0), dtype=np.int64)
    )


def test_retained_source_integrity_rejects_changed_region_assignments() -> None:

    mesh = phx.discretization.CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS, SPLIT_SQUARE_CELLS
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    request = MeshCertificationInputs(
        mesh,
        geometry,
        phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=split_square_domain(),
        cell_regions=np.asarray((0, 0, 1, 1), dtype=np.int64),
    )
    changed = eqx.tree_at(
        lambda value: value.cell_regions, request, jnp.zeros((4,), dtype=jnp.int64)
    )

    request.validate_source_integrity()
    with pytest.raises(ValueError, match="assignments"):
        changed.validate_source_integrity()
    cell_ids = request.cell_global_ids
    assert type(cell_ids) is np.ndarray and cell_ids.dtype == np.int64
    assert not cell_ids.flags.writeable
    np.testing.assert_array_equal(
        cell_ids,
        np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks]),
    )
    reordered = eqx.tree_at(lambda value: value.cell_global_ids, request, cell_ids[::-1])
    with pytest.raises(ValueError, match="assignments"):
        reordered.validate_source_integrity()
    narrowed = eqx.tree_at(
        lambda value: value.cell_global_ids, request, cell_ids.astype(np.int32)
    )
    with pytest.raises(TypeError, match="cell_global_ids"):
        narrowed.validate_source_integrity()


def test_prepared_acceptance_preserves_scientific_report_and_work() -> None:
    mesh, geometry, audit, prepared = prepared_tetrahedral_acceptance()
    request = prepared.request
    cold = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=request.schedule,
        domain=request.domain,
        cell_regions=request.cell_regions,
    )
    reused = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=request.schedule,
        domain=request.domain,
        cell_regions=request.cell_regions,
        prepared=phx.meshing.PreparedVolumeAcceptance(prepared),
    )
    assert cold.passed and reused.passed
    assert reused.report_id == cold.report_id
    assert reused.embedding is not None and cold.embedding is not None
    assert isinstance(reused.coverage, phx.geometry.DomainCoverageCertificate)
    assert isinstance(cold.coverage, phx.geometry.DomainCoverageCertificate)
    assert reused.embedding.candidate_pair_count == cold.embedding.candidate_pair_count
    assert reused.coverage.candidate_pair_count == cold.coverage.candidate_pair_count
    assert (
        reused.coverage.achieved_region_measure_bounds
        == cold.coverage.achieved_region_measure_bounds
    )


@pytest.mark.parametrize("mismatch", ("labels", "source", "limits", "coordinates"))
def test_prepared_acceptance_refuses_changed_scientific_bindings(mismatch: str) -> None:
    mesh, geometry, audit, prepared = prepared_tetrahedral_acceptance()
    request = prepared.request
    domain, regions, limits = request.domain, request.cell_regions, request.limits
    if mismatch == "labels":
        regions = np.asarray((-1,), dtype=np.int64)
    elif mismatch == "source":
        domain = phx.geometry.PiecewiseLinearDomain(
            domain.vertices,
            domain.facets,
            domain.facet_regions,
            domain.region_ids,
            source_id="foreign-source",
        )
    elif mismatch == "limits":
        limits = phx.geometry.MeshCertificateLimits(maximum_candidate_pairs=1)
    else:
        mesh = mesh.with_coordinates(2.0 * TETRAHEDRON, numeric_version="changed")
        geometry, audit = plc_audit(
            mesh,
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
        )
    with pytest.raises(ValueError, match=r"MeshCertificationPreparedEvidence\.request"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=request.schedule,
            domain=domain,
            cell_regions=regions,
            limits=limits,
            prepared=phx.meshing.PreparedVolumeAcceptance(prepared),
        )


def test_mapped_prepared_acceptance_keeps_source_limits_and_region_measure(
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, audit, initial = prepared_tetrahedral_acceptance()
    domain = initial.request.domain
    assert isinstance(domain, phx.geometry.PiecewiseLinearDomain)
    limits = phx.geometry.MeshCertificateLimits(
        maximum_candidate_pairs=4096,
        maximum_ray_tests=8192,
        maximum_subdivision_pieces=16384,
    )
    request = phx.meshing.MeshCertificationInputs(
        mesh,
        geometry,
        initial.request.schedule,
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
        limits=limits,
        fidelity_sample_order=3,
    )
    original_coordinates = geometry.source_coordinates(budget=reference_budget)
    support = prepare_mapped_plc_support(
        mesh,
        geometry,
        domain,
        np.asarray((17,), dtype=np.int64),
        np.asarray((17,), dtype=np.int64),
        maximum_support_queries=32768,
        validity=audit.validity,
        certification_request=request,
        budget=reference_budget,
    )
    reused = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=request.schedule,
        domain=domain,
        cell_regions=request.cell_regions,
        limits=limits,
        fidelity_sample_order=3,
        prepared=phx.meshing.PreparedVolumeAcceptance(support.prepared),
    )
    assert reused.passed
    assert reused.embedding is not None
    assert isinstance(reused.coverage, phx.geometry.DomainCoverageCertificate)
    for certificate in (reused.embedding, reused.coverage):
        certificate.binding.require(mesh, geometry)
        assert certificate.binding.limits_id == limits.limits_id
        assert certificate.status == "certified"
        assert certificate.candidate_pair_count <= limits.maximum_candidate_pairs
        assert certificate.source_expression_work_units is not None
        assert certificate.source_expression_work_units <= limits.maximum_work_units
        assert certificate.source_expression_peak_bytes is not None
        assert certificate.source_expression_peak_bytes <= limits.maximum_scratch_bytes
    assert reused.coverage.binding.source_id == domain.source_id
    assert reused.coverage.binding.source_revision == domain.source_revision
    assert reused.coverage.region_ids == domain.region_ids
    assert geometry.source_coordinates(budget=reference_budget) == original_coordinates
    bounds = reused.coverage.achieved_region_measure_bounds[0]
    assert bounds is not None
    lower, upper = bounds
    assert lower <= 1.0 / 6.0 <= upper


@pytest.fixture
def mapped_source_support(
    reference_budget: CoordinateEnclosureBudget,
) -> MappedPlcSupport:
    mesh, geometry, audit, initial = prepared_tetrahedral_acceptance()
    return prepare_mapped_plc_support(
        mesh,
        geometry,
        initial.request.domain,
        np.asarray((0,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        maximum_support_queries=32768,
        validity=audit.validity,
        budget=reference_budget,
    )


def test_mapped_cardinal_images_reuse_proved_corner_rows(
    mapped_source_support: MappedPlcSupport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = mapped_source_support
    original = algebra.coordinate_map_images
    calls: list[bool] = []

    def counted(
        element: CellGeometryElement,
        local: algebra.CoordinateCoefficients,
        *,
        include_expressions: bool = True,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[tuple[algebra.Expression, ...] | None, algebra.CoordinateSourceBank | None]:
        calls.append(include_expressions)
        return original(
            element, local, include_expressions=include_expressions, budget=budget
        )

    monkeypatch.setattr(algebra, "coordinate_map_images", counted)
    general = _cell_maps(
        support.mesh, support.geometry, prepare_expressions=True, budget=support.ledger
    )
    assert calls
    calls.clear()
    cardinal = _cell_maps(
        support.mesh, support.geometry, prepare_expressions=False, budget=support.ledger
    )
    assert not calls
    assert cardinal.vertex_images == general.vertex_images
    assert cardinal.degree_one_exactness_work_units == general.degree_one_exactness_work_units
    assert tuple(
        (cell.kind, cell.vertices, cell.controls, cell.root_cell_id, cell.root_vertex_ids)
        for cell in cardinal.cells
    ) == tuple(
        (cell.kind, cell.vertices, cell.controls, cell.root_cell_id, cell.root_vertex_ids)
        for cell in general.cells
    )


@pytest.mark.parametrize("mixed_regions", (False, True))
def test_mapped_region_batch_matches_scalar_predicates_and_charges(
    reference_budget: CoordinateEnclosureBudget,
    mixed_regions: bool,
) -> None:
    cells = np.asarray(
        ((0, 1, 2, 4), (0, 3, 1, 4), (0, 2, 3, 4), (1, 3, 2, 4)),
        dtype=np.int64,
    )
    vertices = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.25, 0.25, 0.25)),
        dtype=np.float64,
    )
    mesh = phx.discretization.CellMesh.from_tetrahedra(vertices, cells)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    facets: list[tuple[int, ...]] = []
    pairs: list[list[int]] = []
    owners: dict[tuple[int, ...], int] = {}
    labels = np.arange(4, dtype=np.int64) if mixed_regions else np.zeros(4, dtype=np.int64)
    reference_faces = phx.discretization.reference_cell_topology("tetrahedron").entities[2]
    for cell, region in zip(cells, labels.tolist(), strict=True):
        for face in reference_faces:
            corners = tuple(int(cell[index]) for index in face)
            key = tuple(sorted(corners))
            if key in owners:
                pairs[owners[key]][1] = region
            else:
                owners[key] = len(facets)
                facets.append(corners)
                pairs.append([region, -1])
    keep = np.asarray([own != other for own, other in pairs], dtype=np.bool_)
    domain = phx.geometry.PiecewiseLinearDomain(
        vertices,
        np.asarray(facets, dtype=np.int64)[keep],
        np.asarray(pairs, dtype=np.int64)[keep],
        tuple(f"body-{index}" for index in range(4 if mixed_regions else 1)),
        source_id="batched-region-source",
    )
    support = prepare_mapped_plc_support(
        mesh,
        geometry,
        domain,
        labels,
        np.arange(4 if mixed_regions else 1, dtype=np.int64),
        maximum_support_queries=32768,
        budget=reference_budget,
    )
    empty_edges = np.empty((0, 2), dtype=np.int64)
    empty_triangles = np.empty((0, 3), dtype=np.int64)
    empty_facets = np.empty((0,), dtype=np.int64)
    for dimension in range(4):
        ordered = np.arange(mesh.entity_set(dimension).count, dtype=np.int64)
        noncontiguous = ordered[::-2]
        for rows in (
            ordered[:0],
            ordered[::-1],
            np.concatenate((noncontiguous, noncontiguous[:1])),
        ):
            regions = np.asarray(
                [support.cell_regions[support.parents(dimension, int(row))[0]] for row in rows],
                dtype=np.int64,
            )
            if regions.size:
                regions[-1] = 99
            scalar_work = [32768]
            queries = reference_budget.ticket.geometry_queries
            coordinate_work = reference_budget.work_units
            expected = np.asarray(
                [
                    support.entity_support(
                        dimension,
                        int(row),
                        3,
                        int(region),
                        edge_vertices=empty_edges,
                        triangle_vertices=empty_triangles,
                        triangle_facets=empty_facets,
                        work=scalar_work,
                    )[0]
                    for row, region in zip(rows, regions, strict=True)
                ],
                dtype=np.bool_,
            )
            scalar_queries = reference_budget.ticket.geometry_queries - queries
            scalar_coordinate_work = reference_budget.work_units - coordinate_work
            batch_work = [32768]
            queries = reference_budget.ticket.geometry_queries
            coordinate_work = reference_budget.work_units
            actual = support.region_support(dimension, rows, regions, batch_work)
            np.testing.assert_array_equal(actual, expected)
            assert batch_work == scalar_work
            assert reference_budget.ticket.geometry_queries - queries == scalar_queries
            assert reference_budget.work_units - coordinate_work == scalar_coordinate_work


def test_mapped_region_batch_refuses_before_completing_unadmitted_queries(
    mapped_source_support: MappedPlcSupport,
) -> None:
    support = mapped_source_support
    rows = np.zeros((2,), dtype=np.int64)
    regions = np.zeros((2,), dtype=np.int64)
    work = [1]
    before = (support.ledger.work_units, support.ledger.ticket.geometry_queries)
    with pytest.raises(ValueError, match="support query allowance"):
        support.region_support(3, rows, regions, work)
    assert work == [1]
    assert (support.ledger.work_units, support.ledger.ticket.geometry_queries) == before

    limits = support.prepared.request.limits
    with HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=support.proof.preparation_queries + 1,
        bytes_upper=limits.maximum_scratch_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        rebound = bind_mapped_plc_support(
            support.proof,
            support.mesh,
            support.geometry,
            support.domain,
            support.cell_regions,
            maximum_support_queries=32768,
            certification_request=support.prepared.request,
            budget=budget,
        )
        before = (budget.work_units, ticket.geometry_queries)
        work = [2]
        with pytest.raises(HostBatchEstimateError):
            rebound.region_support(3, rows, regions, work)
        assert work == [2]
        assert (budget.work_units, ticket.geometry_queries) == before


def test_mapped_source_simplex_reuse_authenticates_content_and_ledger(
    mapped_source_support: MappedPlcSupport,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = mapped_source_support
    original = planar_coverage.prepare_affine_exact_source_simplex
    preparations: list[CoordinateEnclosureBudget] = []

    def counted(
        points: tuple[tuple[algebra.ExactScalar, ...], ...],
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> planar_coverage.AffineSourceSimplex:
        preparations.append(budget)
        return original(points, budget=budget)

    monkeypatch.setattr(planar_coverage, "prepare_affine_exact_source_simplex", counted)
    vertices = np.asarray(support.mesh.coordinates).copy()
    vertices.setflags(write=False)
    triangles = support.entity_vertices[2][:1].copy()
    work = [32768]

    def query(
        view: MappedPlcSupport, bank: np.ndarray, rows: np.ndarray
    ) -> tuple[bool, int]:
        return view.entity_support(
            2,
            0,
            2,
            17,
            edge_vertices=np.empty((0, 2), dtype=np.int64),
            triangle_vertices=rows,
            triangle_facets=np.asarray((17,), dtype=np.int64),
            source_vertices=bank,
            work=work,
        )

    first = query(support, vertices, triangles)
    assert first[0] and first[1] in (-1, 1)
    before = (work[0], support.ledger.ticket.geometry_queries, support.ledger.work_units)
    assert query(support, vertices, triangles) == first
    assert len(preparations) == 1
    assert work[0] == before[0] - 2
    assert support.ledger.ticket.geometry_queries == before[1] + 1
    assert support.ledger.work_units > before[2]
    assert query(support, vertices, triangles[:, ::-1]) == (True, -first[1])
    assert len(preparations) == 2
    other_bank = vertices.copy()
    assert query(support, other_bank, triangles) == first
    assert len(preparations) == 3

    limits = support.prepared.request.limits
    other_ledger = coordinate_budget(
        limits.maximum_work_units, limits.maximum_scratch_bytes
    )

    def bind(proof: MappedPlcSupportProof) -> MappedPlcSupport:
        return bind_mapped_plc_support(
            proof,
            support.mesh,
            support.geometry,
            support.domain,
            support.cell_regions,
            maximum_support_queries=32768,
            certification_request=support.prepared.request,
            budget=other_ledger,
        )

    rebound = bind(support.proof)
    assert query(rebound, other_bank, triangles) == first
    assert preparations == [support.ledger] * 3 + [other_ledger]
    assert len(rebound._source_simplices) == 1
    with pytest.raises(ValueError, match="stale or foreign proof"):
        bind(replace(support.proof))

    # Read-only flags are not an authentication boundary.
    vertices.setflags(write=True)
    vertices += 3.0
    vertices.setflags(write=False)
    assert not query(support, vertices, triangles)[0]
    assert len(preparations) == 5
    vertices.setflags(write=True)
    vertices[:] = 0.0
    with pytest.raises(ValueError, match="nondegenerate source triangle"):
        query(support, vertices, triangles)


@pytest.mark.parametrize("prepared_first", (False, True))
def test_mapped_source_simplex_rejects_unselected_nonfinite_bank_rows(
    mapped_source_support: MappedPlcSupport,
    prepared_first: bool,
) -> None:
    support = mapped_source_support
    triangle = np.asarray((0, 1, 2), dtype=np.int64)
    vertices = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.float64,
    )
    vertices.setflags(write=False)
    if prepared_first:
        support._source_simplex(triangle, vertices)
    vertices.setflags(write=True)
    vertices[3, 0] = np.nan
    vertices.setflags(write=False)
    with pytest.raises(ValueError, match="NaN"):
        support._source_simplex(triangle, vertices)


@pytest.mark.parametrize("scalar", ("fraction", "radical"))
def test_mapped_source_simplex_rebuilds_mutated_exact_object_banks(
    mapped_source_support: MappedPlcSupport,
    scalar: str,
) -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    support = mapped_source_support
    triangle = np.asarray((0, 1, 2), dtype=np.int64)
    vertices = np.asarray(
        [
            [Fraction(value) for value in row]
            for row in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        ],
        dtype=object,
    )
    value = (
        Fraction(1)
        if scalar == "fraction"
        else RadicalElement(terms=((2, Fraction(1)),))
    )
    vertices[2, 2] = value
    vertices.setflags(write=False)
    first = support._source_simplex(triangle, vertices)
    original_plane = repr(first.plane)
    pointer_bytes = vertices.tobytes()
    if scalar == "fraction":
        object.__setattr__(value, "_numerator", 2)
    else:
        object.__setattr__(value, "terms", ((3, Fraction(1)),))
    assert vertices.tobytes() == pointer_bytes
    second = support._source_simplex(triangle, vertices)
    assert second is not first
    assert repr(second.plane) != original_plane
    second_plane = repr(second.plane)
    vertices.setflags(write=True)
    vertices[2, 2] = (
        Fraction(3)
        if scalar == "fraction"
        else RadicalElement(terms=((5, Fraction(1)),))
    )
    vertices.setflags(write=False)
    third = support._source_simplex(triangle, vertices)
    assert third is not second
    assert repr(third.plane) != second_plane
    assert not support._source_simplices
    assert not support._source_vertex_banks
    assert all(
        sum(
            (weight * coordinate for weight, coordinate in zip(
                third.plane[:-1], point, strict=True
            )),
            third.plane[-1],
        ) == 0
        for point in vertices[triangle]
    )


@pytest.mark.parametrize("union", ("disconnected", "nonconvex"))
def test_mapped_affine_facet_union_preserves_gaps_and_orientation(
    mapped_source_support: MappedPlcSupport,
    union: str,
) -> None:
    support = mapped_source_support
    if union == "disconnected":
        vertices = np.asarray(
            (
                (0, 0, 0), (1, 0, 0), (0, 1, 0),
                (2, 0, 0), (3, 0, 0), (2, 1, 0),
            ),
            dtype=np.float64,
        )
        triangles = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
        good = ((0.125, 0.125, 0), (0.5, 0.125, 0), (0.125, 0.5, 0))
        bad = ((0.25, 0.25, 0), (2.25, 0.25, 0), (2.125, 0.5, 0))
    else:
        # Three unit squares form an L; the upper-right square is absent.
        vertices = np.asarray(
            (
                (0, 0, 0), (1, 0, 0), (2, 0, 0),
                (0, 1, 0), (1, 1, 0), (2, 1, 0),
                (0, 2, 0), (1, 2, 0),
            ),
            dtype=np.float64,
        )
        triangles = np.asarray(
            (
                (0, 1, 4), (0, 4, 3), (1, 2, 5),
                (1, 5, 4), (3, 4, 7), (3, 7, 6),
            ),
            dtype=np.int64,
        )
        good = ((0.25, 0.25, 0), (1.75, 0.25, 0), (0.25, 1.75, 0))
        bad = ((0.75, 1.75, 0), (1.75, 0.75, 0), (0.25, 0.25, 0))
    groups = support._source_facet_groups(17, triangles, vertices, [32768])
    for dimension in (1, 2):
        for accepted, corners in ((True, good), (False, bad)):
            for reverse in (False, True):
                values = np.asarray(corners[: dimension + 1], dtype=np.float64)
                if reverse:
                    values = values[::-1]
                points = tuple(tuple(Fraction(float(v)) for v in row) for row in values)
                direct_work = [32768]
                queries = support.ledger.ticket.geometry_queries
                direct = support._affine_facet_support(points, groups, direct_work)
                assert direct.supported is accepted
                assert direct_work[0] < 32768
                assert support.ledger.ticket.geometry_queries > queries
                if dimension == 1:
                    assert direct.orientation == 0
                # Compare the retained exact union predicate with the original
                # full-chart route, including orientation and candidate work.
                chart = algebra.affine_arguments(
                    values[0], (values[1:] - values[0]).T, budget=support.ledger
                )
                chart_work = [32768]
                general = support._facet_chart_support(
                    dimension,
                    ((chart, "box" if dimension == 1 else "simplex"),),
                    groups,
                    chart_work,
                )
                assert (direct.supported, direct.orientation) == (
                    general.supported,
                    general.orientation,
                )
                if dimension == 2 or accepted:
                    assert direct_work == chart_work


def test_mapped_support_preserves_its_stricter_source_quota_across_queries(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, audit, initial = prepared_tetrahedral_acceptance()
    domain = initial.request.domain
    assert isinstance(domain, phx.geometry.PiecewiseLinearDomain)
    labels = np.asarray((0,), dtype=np.int64)
    # The acceptance fixture still owns a coverage proof issued under default
    # limits for this mesh. A distinct policy makes the calibrating request
    # prove coverage itself, as the limited request below must.
    calibration = phx.geometry.MeshCertificateLimits(
        maximum_work_units=phx.geometry.MeshCertificateLimits().maximum_work_units + 1
    )
    measured = prepare_mapped_plc_support(
        mesh,
        geometry,
        domain,
        labels,
        labels,
        maximum_support_queries=32768,
        validity=audit.validity,
        certificate_limits=calibration,
        budget=reference_budget,
    )
    limits = phx.geometry.MeshCertificateLimits(
        maximum_work_units=measured.ledger.work_units + 96,
    )
    ambient = coordinate_budget(
        4 * limits.maximum_work_units + 1024,
        limits.maximum_scratch_bytes,
    )
    empty_edges = np.empty((0, 2), dtype=np.int64)
    empty_triangles = np.empty((0, 3), dtype=np.int64)
    empty_facets = np.empty((0,), dtype=np.int64)
    query_work = [limits.maximum_work_units]
    with ambient.activate():
        algebra.expression_bernstein_coefficients(
            {(2, 0, 0): Fraction(1)},
            "simplex",
            3,
            budget=ambient,
        )
        original_start = ambient.work_units
        support = prepare_mapped_plc_support(
            mesh,
            geometry,
            domain,
            labels,
            labels,
            maximum_support_queries=32768,
            validity=audit.validity,
            certificate_limits=limits,
            budget=ambient,
        )
        assert ambient.work_units > original_start
        with pytest.raises(CoordinateEnclosureResourceError) as refused:
            # Every source-point query visits at least three rational
            # coordinates. The source quota must refuse before the larger
            # ambient request is exhausted, even across separate activations.
            for _ in range(limits.maximum_work_units // 3 + 1):
                assert support.vertex_support(
                    0,
                    0,
                    0,
                    edge_vertices=empty_edges,
                    triangle_vertices=empty_triangles,
                    triangle_facets=empty_facets,
                    work=query_work,
                )
        assert refused.value.resource == "coefficient_work"
        assert refused.value.limit == original_start + limits.maximum_work_units
        assert refused.value.completed == ambient.work_units
        assert ambient.work_units < ambient.maximum_work_units


@pytest.mark.parametrize("mismatch", ("facets", "tolerance"))
def test_mapped_prepared_acceptance_refuses_changed_declared_source_scope(
    mismatch: str,
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, audit, initial = prepared_tetrahedral_acceptance()
    domain = initial.request.domain
    assert isinstance(domain, phx.geometry.PiecewiseLinearDomain)
    source = phx.geometry.MeshingDomainBoundarySource(
        sphere_surface().domain, (0,), resolution=4
    )
    facets = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    scoped = ((source, facets[:1], 2.0),)
    request = phx.meshing.MeshCertificationInputs(
        mesh,
        geometry,
        phx.meshing.MeshCertificationSchedule("volume_implicit"),
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
        source=source,
        fidelity_tolerance=2.0,
        scoped_fidelity=scoped,
    )
    # Preparation owns positive volume premises, not a source-fidelity verdict.
    support = prepare_mapped_plc_support(
        mesh,
        geometry,
        domain,
        np.asarray((0,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        maximum_support_queries=32768,
        validity=audit.validity,
        certification_request=request,
        budget=reference_budget,
    )
    changed = (
        ((source, facets[1:2], 2.0),)
        if mismatch == "facets"
        else ((source, facets[:1], 1.5),)
    )
    with pytest.raises(ValueError, match=r"MeshCertificationPreparedEvidence\.request"):
        phx.meshing.certify_meshing_acceptance(
            mesh,
            geometry,
            audit,
            schedule=request.schedule,
            domain=domain,
            cell_regions=request.cell_regions,
            source=source,
            fidelity_tolerance=2.0,
            scoped_fidelity=changed,
            prepared=phx.meshing.PreparedVolumeAcceptance(support.prepared),
        )


def test_prepared_audit_refuses_a_different_validity_policy() -> None:
    mesh, geometry, _, prepared = prepared_tetrahedral_acceptance()
    policy = phx.meshing.CellMeshAuditPolicy(
        validity_policy=phx.discretization.CellValidityPolicy(
            maximum_subdivision_depth=1
        ),
    )
    with pytest.raises(ValueError, match="prepared_validity"):
        phx.meshing.audit_cell_mesh(
            mesh,
            geometry,
            policy=policy,
            prepared_validity=prepared.validity,
        )


def test_prepared_native_publication_refuses_authored_revision_drift(
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, geometry, request, proof = prepared_publication_authority(
        reference_budget=reference_budget
    )
    request = replace(request, prepared=proof.support.prepared)
    assert request.source_revision != request.domain.source_revision
    policy = phx.meshing.CellMeshAuditPolicy(
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
    )
    with (
        NativeExecutionScope(request.limits.native_limits()) as execution_scope,
        execution_scope.host_workspace() as workspace,
    ):
        published = publish_native_result(
            mesh,
            phx.SpatialCoordinateContract.si(),
            phx.meshing.MeshingComplianceReport("authoritative-publication"),
            (),
            phx.meshing.NativeMeshingProvider.info(),
            {"kind": "authoritative-publication"},
            request,
            policy=NativePublicationPolicy(
                policy, phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE, (), ()
            ),
            organization=NativePublicationOrganization(associations=proof.associations),
            execution=NativePublicationExecution(execution_scope, workspace, None, None),
            geometry=geometry,
        )
    assert published.certification is not None and published.certification.passed
    with pytest.raises(ValueError, match="publication.associations"):
        with (
            NativeExecutionScope(request.limits.native_limits()) as execution_scope,
            execution_scope.host_workspace() as workspace,
        ):
            publish_native_result(
                mesh,
                phx.SpatialCoordinateContract.si(),
                phx.meshing.MeshingComplianceReport("authoritative-publication"),
                (),
                phx.meshing.NativeMeshingProvider.info(),
                {"kind": "authoritative-publication"},
                replace(request, source_revision="foreign-authored-revision"),
                policy=NativePublicationPolicy(
                    policy, phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE, (), ()
                ),
                organization=NativePublicationOrganization(
                    associations=proof.associations
                ),
                execution=NativePublicationExecution(
                    execution_scope, workspace, None, None
                ),
                geometry=geometry,
            )


def test_prepared_coverage_refusal_transports_the_negative_scientific_certificate(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    mesh, geometry, audit, positive = prepared_tetrahedral_acceptance()
    original = positive.request.domain
    assert original is not None
    domain = phx.geometry.PiecewiseLinearDomain(
        TETRAHEDRON + np.asarray((2.0, 0.0, 0.0)),
        original.facets,
        original.facet_regions,
        original.region_ids,
        source_id="independent-shifted-coverage-source",
    )
    negative = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        domain,
        np.asarray((0,), dtype=np.int64),
        embedding=positive.embedding,
        limits=positive.request.limits,
    )
    ledger = coordinate_budget(2_000_000, 81_920_000)
    with ledger.activate(), pytest.raises(MeshCertificationPremiseFailure) as caught:
        phx.meshing.MeshCertificationPreparedEvidence(
            mesh,
            geometry,
            audit.validity,
            schedule=positive.request.schedule,
            domain=domain,
            cell_regions=np.asarray((0,), dtype=np.int64),
            embedding=positive.embedding,
            limits=positive.request.limits,
            budget=ledger,
        )
    failure = caught.value
    certificate = failure.coverage
    assert certificate.certificate_id == negative.certificate_id
    assert tuple(finding.finding_id for finding in certificate.findings) == tuple(
        finding.finding_id for finding in negative.findings
    )
    assert certificate.status == "violated" and certificate.findings
    assert certificate.binding.source_id == domain.source_id
    failure_domain = failure.request.domain
    if failure_domain is None or failure.ledger is None:
        raise RuntimeError(
            "fixture.coverage_failure must retain domain and active ledger; got missing evidence"
        )
    assert failure_domain.domain_id == domain.domain_id
    assert failure.geometry.geometry_layout_id == geometry.geometry_layout_id
    assert failure.mesh.topology_id == mesh.topology_id
    assert failure.ledger.work_units == ledger.work_units
    assert failure.evidence.provider_code == certificate.certificate_id
    assert failure.evidence.category is phx.meshing.MeshingFailureCategory.AUDIT_FAILED
    assert failure.evidence.stage == "certification"
    assert all(
        finding.check in failure.evidence.message
        for finding in certificate.findings
        if finding.status == certificate.status
    )
    for finding in certificate.findings:
        prefix = f"certificate:{certificate.certificate_id}:finding:{finding.finding_id}:{finding.check}"
        if finding.resource is not None:
            prefix += f":resource:{finding.resource}"
        assert all(
            dict(failure.evidence.requested)[f"{prefix}:{key}"] == value
            for key, value in finding.requested
        )
        assert all(
            dict(failure.evidence.achieved)[f"{prefix}:{key}"] == value
            for key, value in finding.achieved
        )


def test_uncertain_prepared_refusal_keeps_unbounded_values_out_of_finite_evidence(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    mesh, geometry, audit, positive = prepared_tetrahedral_acceptance()
    request = positive.request
    assert request.domain is not None and request.cell_regions is not None
    ledger = coordinate_budget(0, 81_920_000)
    with ledger.activate():
        coverage = phx.geometry.certify_domain_coverage(
            mesh,
            geometry,
            request.domain,
            request.cell_regions,
            embedding=positive.embedding,
            limits=request.limits,
            budget=ledger,
        )
        assert coverage.status == "unresolved"
        assert any(
            value is None or not np.isfinite(value)
            for value in coverage.achieved_region_measures
        )
        with pytest.raises(MeshCertificationPremiseFailure) as caught:
            raise MeshCertificationPremiseFailure(
                mesh,
                geometry,
                request,
                audit.validity,
                positive.embedding,
                coverage,
                budget=ledger,
            )
    failure = caught.value
    assert isinstance(failure.coverage, phx.geometry.DomainCoverageCertificate)
    assert failure.coverage.certificate_id == coverage.certificate_id
    assert failure.coverage.achieved_region_measures == coverage.achieved_region_measures
    assert failure.unbounded_region_measures == tuple(
        (region, "achieved", measured)
        for region, measured in zip(
            coverage.region_ids, coverage.achieved_region_measures, strict=True
        )
        if measured is None or not np.isfinite(measured)
    )
    assert all(
        np.isfinite(value)
        for _, value in (*failure.evidence.requested, *failure.evidence.achieved)
    )
    assert all(
        f"region_measure[{region}]" not in dict(failure.evidence.achieved)
        for region, measured in zip(
            coverage.region_ids, coverage.achieved_region_measures, strict=True
        )
        if measured is None or not np.isfinite(measured)
    )
    assert failure.evidence.evidence_id
    for finding in coverage.findings:
        prefix = f"certificate:{coverage.certificate_id}:finding:{finding.finding_id}:{finding.check}"
        if finding.resource is not None:
            prefix += f":resource:{finding.resource}"
        assert all(
            dict(failure.evidence.requested)[f"{prefix}:{key}"] == value
            for key, value in finding.requested
            if np.isfinite(value)
        )
        assert all(
            dict(failure.evidence.achieved)[f"{prefix}:{key}"] == value
            for key, value in finding.achieved
            if np.isfinite(value)
        )


def test_public_failure_preserves_actual_fidelity_resource_quantities() -> None:

    mesh = octahedron()
    geometry, audit = plc_audit(
        mesh, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    sphere_surface = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 1.0, feature_id="sphere"
    ).compile(coordinate_unit=METER)
    source = phx.geometry.ImplicitBoundarySource(
        sphere_surface, source_id="limited-fidelity-sphere", spacing=0.2
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("surface"),
        source=source,
        fidelity_tolerance=0.8,
        limits=phx.geometry.MeshCertificateLimits(maximum_distance_evaluations=1),
    )
    assert report.fidelity is not None and report.fidelity.status == "unresolved"
    assert "distance_capacity" in finding_checks(report.fidelity, "unresolved")
    with pytest.raises(MeshCertificationReportFailure) as caught:
        report.require_passed()
    failure = caught.value
    assert failure.report.report_id == report.report_id
    assert failure.report.request.request_id == report.request.request_id
    failed_fidelity = failure.report.fidelity
    if failed_fidelity is None:
        raise RuntimeError(
            "fixture.fidelity_failure must retain source certificate; got None"
        )
    assert failed_fidelity.binding.source_id == source.source_id
    assert all(
        np.isfinite(value)
        for _, value in (*failure.evidence.requested, *failure.evidence.achieved)
    )
    for certificate in (
        report.embedding,
        report.coverage,
        report.fidelity,
        *report.scoped_fidelity,
    ):
        if certificate is None:
            continue
        for finding in certificate.findings:
            prefix = f"certificate:{certificate.certificate_id}:finding:{finding.finding_id}:{finding.check}"
            if finding.resource is not None:
                prefix += f":resource:{finding.resource}"
            assert all(
                dict(failure.evidence.requested)[f"{prefix}:{key}"] == value
                for key, value in finding.requested
                if np.isfinite(value)
            )
            assert all(
                dict(failure.evidence.achieved)[f"{prefix}:{key}"] == value
                for key, value in finding.achieved
                if np.isfinite(value)
            )
            assert all(
                dict(failure.evidence.achieved)[f"{prefix}:observation:{key}"] == value
                for key, value in finding.observations
                if np.isfinite(value)
            )
    assert len(dict(failure.evidence.requested)) == len(failure.evidence.requested)
    assert len(dict(failure.evidence.achieved)) == len(failure.evidence.achieved)


def test_native_refusal_survives_fidelity_certificate_and_report_consumer() -> None:
    mesh = octahedron()
    geometry, audit = plc_audit(
        mesh, watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    compiled = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    source = phx.geometry.ImplicitBoundarySource(
        compiled, source_id="native-refusal-fidelity-sphere", spacing=0.2
    )
    scientific = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("surface"),
        source=source,
        fidelity_tolerance=0.1,
    )
    original = scientific.fidelity
    assert original is not None and original.status == "violated"
    with pytest.raises(MeshCertificationReportFailure) as violated:
        scientific.require_passed()
    assert violated.value.category is phx.meshing.MeshingFailureCategory.AUDIT_FAILED
    assert violated.value.report.fidelity is original
    assert all(finding.native_refusal is None for finding in original.findings)

    scope = NativeExecutionScope(NativeExecutionLimits(20, 10, 8, 65536, 120))
    with pytest.raises(MeshcoreError) as caught:
        with scope:
            with scope.batch(
                12, queries_upper=6, cavity_upper=4, bytes_upper=4096
            ) as ticket:
                ticket.consume(3, geometry_queries=2, cavity_cells=2, managed_bytes=512)
                ticket.refuse(18, queries_upper=9, cavity_upper=9, bytes_upper=65537)
    error = caught.value
    assert error.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert scope.evidence is not None
    state = EmbeddingState([], [])
    state.add("distance_capacity", "unresolved", "mesh", resource_error=error)
    finding = state.findings[0]
    refusal = finding.native_refusal
    assert refusal is not None
    assert refusal.status is error.status
    assert refusal.admitted_upper == (12, 6, 4, 4096)
    assert refusal.requested_upper == (18, 9, 9, 65537)
    assert error.batch_refusal_evidence is not None
    assert refusal.root_receipt == tuple(
        int(value) for value in error.batch_refusal_evidence
    )
    assert refusal.root_receipt is not None
    assert refusal.root_receipt[4:6] == (3, 2)
    assert finding.resource is None
    assert finding.requested == finding.achieved == ()
    repeated = EmbeddingState([], [])
    repeated.add("distance_capacity", "unresolved", "mesh", resource_error=error)
    assert repeated.findings[0].finding_id == finding.finding_id
    assert (
        finding.finding_id
        != MeshCertificateFinding("distance_capacity", "unresolved", "mesh").finding_id
    )

    certificate = SourceFidelityCertificate(
        original.binding,
        tuple(state.findings),
        tolerance=original.tolerance,
        bounds=SourceFidelityBounds(
            semantics=("sampled", "sampled"),
            mesh_to_source=(np.inf, 0.0),
            source_to_mesh=(np.inf, 0.0),
            sample_counts=(0, 0),
        ),
        sample_order=original.sample_order,
    )
    outcomes = tuple(
        MeshCertificationOutcome(
            "source_fidelity",
            certificate.status,
            certificate.certificate_id,
            reasons=tuple(value.check for value in certificate.findings),
        )
        if outcome.check == "source_fidelity"
        else outcome
        for outcome in scientific.outcomes
    )
    report = MeshCertificationReport(
        scientific.schedule,
        mesh,
        geometry,
        audit,
        outcomes,
        embedding=scientific.embedding,
        coverage=scientific.coverage,
        fidelity=certificate,
        request=scientific.request,
    )
    assert certificate.status == "unresolved"
    assert certificate.findings[0].native_refusal is refusal
    assert report.failing_outcomes[0].evidence_id == certificate.certificate_id
    with pytest.raises(MeshCertificationReportFailure) as failed:
        report.require_passed()
    failure = failed.value
    assert failure.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert failure.report is report
    assert failure.report.fidelity is certificate
    assert failure.report.fidelity.findings[0].finding_id == finding.finding_id
    prefix = (
        f"certificate:{certificate.certificate_id}:finding:{finding.finding_id}"
        f":{finding.check}:native_batch"
    )
    requested = dict(failure.evidence.requested)
    achieved = dict(failure.evidence.achieved)
    assert achieved[f"{prefix}:status"] == int(error.status)
    dimensions = ("work_units", "geometry_queries", "cavity_cells", "managed_bytes")
    for name, admitted, wanted in zip(
        dimensions, refusal.admitted_upper, refusal.requested_upper, strict=True
    ):
        assert achieved[f"{prefix}:admitted_{name}"] == admitted
        assert requested[f"{prefix}:requested_{name}"] == wanted
    receipt_names = (
        *(f"remaining_{name}" for name in dimensions),
        "consumed_work_units",
        "consumed_geometry_queries",
        "peak_cavity_cells",
        "peak_managed_bytes",
    )
    for name, value in zip(receipt_names, refusal.root_receipt, strict=True):
        assert achieved[f"{prefix}:{name}"] == value
    with pytest.raises(MeshCertificationReportFailure) as again:
        report.require_passed()
    assert again.value.evidence.evidence_id == failure.evidence.evidence_id
    with pytest.raises(MeshCertificationReportFailure) as scientific_again:
        scientific.require_passed()
    assert (
        scientific_again.value.category is phx.meshing.MeshingFailureCategory.AUDIT_FAILED
    )
    retained_scientific = scientific_again.value.report.fidelity
    assert retained_scientific is not None
    assert original.certificate_id == retained_scientific.certificate_id
    assert tuple(value.finding_id for value in original.findings) == tuple(
        value.finding_id for value in retained_scientific.findings
    )


type _CanonicalAcceptance = tuple[
    phx.discretization.CellMesh,
    phx.discretization.CellGeometrySpec,
    phx.meshing.MeshCertificationPreparedEvidence,
    phx.meshing.MeshLineage,
    phx.meshing.CellMeshAuditPolicy,
]


@pytest.fixture(scope="module")
def canonical_tetrahedral_acceptance() -> _CanonicalAcceptance:
    mesh, geometry, _, prepared = prepared_tetrahedral_acceptance()
    lineage = phx.meshing.MeshLineage(
        mesh.topology_id,
        mesh.topology_id,
        tuple(
            phx.meshing.EntityLineage(
                entities.intrinsic_dimension,
                entities.entity_set_id,
                entities.entity_set_id,
                entities.entity_ids,
                entities.entity_ids,
                np.full(
                    entities.entity_ids.shape,
                    phx.meshing.EntityLineageKind.PRESERVED,
                    dtype=np.int32,
                ),
            )
            for entities in mesh.topology.entity_sets
        ),
    )
    policy = phx.meshing.CellMeshAuditPolicy(
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    return mesh, geometry, prepared, lineage, policy


@pytest.mark.parametrize("kind", ("volume", "boundary"))
def test_canonical_certification_prepared_variants_preserve_acceptance_identity(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance, kind: str
) -> None:
    mesh, geometry, evidence, lineage, policy = canonical_tetrahedral_acceptance
    prepared = (
        phx.meshing.PreparedVolumeAcceptance(evidence)
        if kind == "volume"
        else phx.meshing.PreparedBoundaryAcceptance(embedding=evidence.embedding)
    )
    cold = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
        audit_policy=policy,
        certification_inputs=evidence.request,
        lineage=lineage,
    )
    reused = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
        audit_policy=policy,
        certification_inputs=evidence.request,
        lineage=lineage,
        certification_prepared=prepared,
    )
    assert reused.certification is not None and cold.certification is not None
    assert reused.certification.report_id == cold.certification.report_id
    assert reused.audit.report_id == cold.audit.report_id
    assert reused.trace.trace_id == cold.trace.trace_id
    assert reused.result_id == cold.result_id


def test_canonical_volume_preparation_owns_audit_validity(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance,
) -> None:
    mesh, geometry, evidence, lineage, policy = canonical_tetrahedral_acceptance
    result = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
        audit_policy=policy,
        certification_inputs=evidence.request,
        lineage=lineage,
        certification_prepared=phx.meshing.PreparedVolumeAcceptance(evidence),
        audit_prepared_validity=evidence.validity,
    )
    assert result.audit.validity is evidence.validity
    assert result.certification is not None
    assert result.certification.embedding is evidence.embedding
    assert result.certification.coverage is evidence.coverage


@pytest.mark.parametrize("policy", (None, phx.meshing.CellMeshAuditPolicy()))
def test_canonical_publication_executes_retained_volume_closure(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance,
    policy: phx.meshing.CellMeshAuditPolicy | None,
) -> None:
    mesh, geometry, evidence, lineage, required_policy = canonical_tetrahedral_acceptance
    result = phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
        audit_policy=policy,
        certification_inputs=evidence.request,
        lineage=lineage,
        certification_prepared=phx.meshing.PreparedVolumeAcceptance(evidence),
    )
    assert result.certification is not None and result.certification.passed
    assert result.audit.policy_id == required_policy.policy_id
    assert "open_boundary" in result.audit.evaluated_checks


def test_canonical_retained_volume_still_refuses_a_tampered_skipped_audit(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance,
) -> None:
    mesh, geometry, evidence, lineage, _ = canonical_tetrahedral_acceptance
    skipped = phx.meshing.audit_cell_mesh(mesh, geometry)
    with pytest.raises(ValueError, match="open_boundary"):
        evidence.request.recertify_transition(
            mesh,
            geometry,
            skipped,
            lineage=lineage,
            prepared=phx.meshing.PreparedVolumeAcceptance(evidence),
        )


@pytest.mark.parametrize("kind", ("volume", "boundary"))
def test_canonical_prepared_evidence_requires_its_transition_request(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance, kind: str
) -> None:
    mesh, geometry, evidence, _, policy = canonical_tetrahedral_acceptance
    prepared = (
        phx.meshing.PreparedVolumeAcceptance(evidence)
        if kind == "volume"
        else phx.meshing.PreparedBoundaryAcceptance(embedding=evidence.embedding)
    )
    with pytest.raises(ValueError, match="certification_prepared"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
            audit_policy=policy,
            certification_prepared=prepared,
        )


def test_canonical_prepared_validity_must_agree_with_volume_evidence(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance,
) -> None:
    mesh, geometry, evidence, lineage, policy = canonical_tetrahedral_acceptance
    different = phx.discretization.certify_cell_geometry_validity(
        geometry,
        mesh=mesh,
        policy=phx.discretization.CellValidityPolicy(maximum_subdivision_depth=0),
    )
    assert different.certificate_id != evidence.validity.certificate_id
    with pytest.raises(ValueError, match="audit_prepared_validity"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
            audit_policy=policy,
            certification_inputs=evidence.request,
            lineage=lineage,
            certification_prepared=phx.meshing.PreparedVolumeAcceptance(evidence),
            audit_prepared_validity=different,
        )


def test_canonical_checked_preparation_refuses_the_unwrapped_volume_evidence(
    canonical_tetrahedral_acceptance: _CanonicalAcceptance,
) -> None:
    mesh, geometry, evidence, lineage, policy = canonical_tetrahedral_acceptance
    with pytest.raises(TypeError, match="certification_prepared"):
        phx.meshing.certify_cell_mesh(
            mesh,
            phx.SpatialCoordinateContract.si(),
            geometry=geometry,
            audit_policy=policy,
            certification_inputs=evidence.request,
            lineage=lineage,
            # ty: ignore[invalid-argument-type]
            certification_prepared=evidence,
        )


def _planar_audit_square() -> phx.discretization.CellMesh:
    return phx.discretization.CellMesh.from_triangles(
        np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
    )


def test_planar_audit_imports_completed_embedding_without_skipping_contact(
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._audit_topology as owner

    mesh = _planar_audit_square()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    policy = phx.meshing.CellMeshAuditPolicy(
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    validity, embedding = owner.prepare_planar_audit_embedding(
        mesh,
        geometry,
        policy=policy,
        limits=phx.geometry.MeshCertificateLimits(),
        junction_vertices=None,
        budget=reference_budget,
    )
    assert validity is not None and embedding is not None
    assert embedding.status == "certified"

    def unexpected_pair_sweep(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("A completed bound planar embedding proof was not imported")

    monkeypatch.setattr(owner, "_self_intersections", unexpected_pair_sweep)
    audit = phx.meshing.audit_cell_mesh(
        mesh, geometry, policy=policy, prepared_validity=validity, budget=reference_budget
    )
    assert audit.passed
    assert "self_intersection" in audit.evaluated_checks
    assert "self_intersection" not in audit.skipped_checks
    assert dict(audit.check_counts)["self_intersection"] == 0


@pytest.mark.parametrize("drift", ("coordinates", "triangles", "geometry", "welding"))
def test_planar_embedding_audit_proof_rejects_changed_carrier_or_welding(
    drift: str,
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._audit_topology as owner

    mesh = _planar_audit_square()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    policy = phx.meshing.CellMeshAuditPolicy(
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
    )
    validity, embedding = owner.prepare_planar_audit_embedding(
        mesh,
        geometry,
        policy=policy,
        limits=phx.geometry.MeshCertificateLimits(),
        junction_vertices=None,
        budget=reference_budget,
    )
    assert validity is not None and embedding is not None
    if drift == "coordinates":
        mesh = mesh.with_coordinates(
            np.asarray(mesh.coordinates) + (2, 3), numeric_version="translated-test"
        )
    elif drift == "triangles":
        mesh = phx.discretization.CellMesh.from_triangles(
            mesh.coordinates, np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int64)
        )
    elif drift == "geometry":
        geometry = eqx.tree_at(
            lambda value: value.coordinates,
            geometry,
            geometry.coordinates + jnp.asarray((2, 3)),
        )
    original = owner._self_intersections
    calls = 0

    def check_pairs(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "_self_intersections", check_pairs)
    owner.audit_welded_topology(
        mesh,
        np.asarray(mesh.coordinates),
        coincident_tolerance=2.0 if drift == "welding" else None,
        candidate_capacity=policy.maximum_coincidence_candidates,
        intersection_candidate_capacity=policy.maximum_intersection_candidates,
        check_manifold=True,
        check_watertight=True,
        check_self_intersection=True,
        budget=reference_budget,
        audit_policy_id=policy.policy_id,
        audit_geometry=geometry,
    )
    assert calls == 1


def test_mapped_prepared_capacity_exhaustion_retains_resource_classification(
    mapped_source_support: MappedPlcSupport,
) -> None:
    proof = mapped_source_support.proof
    with pytest.raises(phx.meshing.MeshingFailure) as refusal:
        _require_support_work_bounds(
            proof.prepared,
            phx.geometry.MeshCertificateLimits(maximum_ray_tests=1),
            proof.entity_vertices,
            proof.boundary_entities,
            False,
        )
    assert refusal.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert refusal.value.evidence.provider_code == "mapped_support_resource_bound"
