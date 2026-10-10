# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import exact_orient3d, MeshcoreError, MeshcoreStatus
from phydrax.geometry.surface._contracts import SurfaceMetadata
from phydrax.geometry.surface._model import SurfaceModel
from phydrax.lifecycle._meshing_sources import write_meshing_source_closure
from phydrax.meshing._surface_envelope import (
    EnvelopeFeaturePermission,
    EnvelopeTopologyPermission,
    execute_surface_envelope_route,
    PreparedSurfaceEnvelopeVolume,
    RawTriangleSoup,
    surface_envelope_source_entities,
    SurfaceEnvelope,
    SurfaceEnvelopePolicy,
    wrap_surface_envelope,
)
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native import NativeMeshingProvider
from phydrax.meshing.providers._native_sources import NativeSurfaceEnvelopeSource


M = phx.meshing
_SI = phx.SpatialCoordinateContract.si()


def _metadata() -> SurfaceMetadata:
    return SurfaceMetadata(
        source_id="sheet-source",
        source_revision="scan-17",
        coordinate_contract=_SI,
        provenance=("synthetic-open-sheet",),
    )


def _sheet() -> RawTriangleSoup:
    return RawTriangleSoup(
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float64),
        np.array([[0, 1, 2]], dtype=np.int64),
        _metadata(),
        feature_edges=np.array([[0, 1]], dtype=np.int64),
    )


def _policy(
    *,
    max_samples: int = 100_000,
    max_work_units: int = 10_000_000,
    max_vertices: int = 10_000,
    max_triangles: int = 20_000,
    max_volume_vertices: int | None = None,
    max_tetrahedra: int | None = None,
    feature_permission: EnvelopeFeaturePermission = EnvelopeFeaturePermission.ALLOW_ROUNDING,
    topology_permission: EnvelopeTopologyPermission = EnvelopeTopologyPermission.ALLOW_CHANGE,
) -> SurfaceEnvelopePolicy:
    return SurfaceEnvelopePolicy(
        offset=0.263,
        maximum_deviation=0.55,
        spacing=0.11,
        certificate_spacing=0.05,
        feature_permission=feature_permission,
        topology_permission=topology_permission,
        max_samples=max_samples,
        max_work_units=max_work_units,
        max_vertices=max_vertices,
        max_triangles=max_triangles,
        max_volume_vertices=max_volume_vertices,
        max_tetrahedra=max_tetrahedra,
    )


def _specification(
    source: NativeSurfaceEnvelopeSource,
    *,
    original_identity: bool = False,
    maximum_wall_seconds: float = 3600.0,
    maximum_scratch_bytes: int = 8_000_000_000,
) -> M.VolumeMeshingSpec:
    scope = M.MeshingScope(
        source.envelope.source.metadata.source_id
        if original_identity
        else source.source_id,
        source.envelope.source.metadata.source_revision
        if original_identity
        else source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "envelope-facets",
        np.arange(source.plc_source.complex.facet_count, dtype=np.int64),
    )
    return M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
        limits=M.MeshingLimits(
            maximum_vertices=20_000,
            maximum_cells=100_000,
            maximum_work_units=20_000_000,
            maximum_wall_seconds=maximum_wall_seconds,
            maximum_scratch_bytes=maximum_scratch_bytes,
        ),
    )


def test_open_sheet_wrap_is_closed_and_has_two_directed_bounds() -> None:
    soup = _sheet()
    envelope = wrap_surface_envelope(soup, _policy())
    evidence = envelope.evidence
    assert evidence.source_topology.boundary_edges == 3
    assert evidence.repaired_topology.boundary_edges == 0
    assert evidence.repaired_topology.nonmanifold_edges == 0
    assert evidence.source_topology.euler_characteristic == 1
    assert evidence.repaired_topology.euler_characteristic == 2
    assert evidence.source_inclusion_margin > 0
    assert 0 < evidence.source_to_repaired_upper <= envelope.policy.maximum_deviation
    assert 0 < evidence.repaired_to_source_upper <= envelope.policy.maximum_deviation
    assert evidence.rounded_feature_edges == 1
    assert (
        envelope.repaired.model.metadata.source_revision != soup.metadata.source_revision
    )
    # Independent distance-to-sheet lower bound: every boundary vertex must
    # stay within the declared two-sided envelope, including the rounded rim.
    points = np.asarray(envelope.repaired.mesh.coordinates, dtype=np.float64)
    assert np.max(np.abs(points[:, 2])) <= evidence.repaired_to_source_upper
    assert np.min(points[:, 2]) < 0 < np.max(points[:, 2])
    assert np.array_equal(soup.triangles, np.array([[0, 1, 2]], dtype=np.int64))
    receipt = envelope.execution_evidence
    assert receipt is not None
    receipt.require_valid()
    assert receipt.owner_id == envelope.envelope_id
    assert int(np.asarray(receipt.total_geometry_queries)) > 0
    assert int(np.asarray(receipt.total_work_units)) >= evidence.work_units
    assert int(np.asarray(receipt.memory)[2]) > 0


@pytest.mark.parametrize("forgery", ("directed-bound", "topology-count"))
def test_envelope_archive_refuses_self_reported_repair_theorem(
    tmp_path: Path,
    forgery: str,
) -> None:
    envelope = wrap_surface_envelope(_sheet(), _policy())
    if forgery == "directed-bound":
        evidence = replace(envelope.evidence, source_to_repaired_upper=0.0)
    else:
        evidence = replace(
            envelope.evidence,
            repaired_topology=replace(envelope.evidence.repaired_topology, faces=0),
        )
    forged = SurfaceEnvelope(
        envelope.source,
        envelope.policy,
        envelope.repaired,
        evidence,
        envelope.carrier_vertices,
        envelope.carrier_tetrahedra,
    )
    source = NativeSurfaceEnvelopeSource(forged, "solid")
    with pytest.raises(ValueError):
        write_meshing_source_closure(tmp_path / "forged-source", source)


def test_dirty_duplicate_reversed_and_degenerate_faces_remain_visible() -> None:
    clean = _sheet()
    faces = np.array([[0, 1, 2], [2, 1, 0], [0, 0, 1]], dtype=np.int64)
    dirty = RawTriangleSoup(clean.vertices, faces, _metadata())
    envelope = wrap_surface_envelope(dirty, _policy())
    assert envelope.evidence.source_topology.duplicate_faces == 1
    assert envelope.evidence.source_topology.degenerate_faces == 1
    assert envelope.evidence.source_topology.nonmanifold_edges > 0
    assert envelope.evidence.repaired_topology.degenerate_faces == 0
    assert envelope.evidence.repaired_topology.duplicate_faces == 0
    assert np.array_equal(dirty.triangles, faces)


@pytest.mark.parametrize(
    ("coordinates", "degenerate_faces"),
    [
        (((0.0, 0.0, 0.0), (1e-200, 0.0, 0.0), (0.0, 1e-200, 0.0)), 0),
        (((0.0, 0.0, 0.0), (1e-200, 0.0, 0.0), (2e-200, 0.0, 0.0)), 1),
    ],
    ids=["nonzero-dyadic-area-underflows", "exactly-collinear"],
)
def test_tiny_source_face_defects_are_exact_not_floating_underflow(
    coordinates: tuple[tuple[float, float, float], ...], degenerate_faces: int
) -> None:
    points = np.array(coordinates, dtype=np.float64)
    soup = RawTriangleSoup(points, np.array([[0, 1, 2]], dtype=np.int64), _metadata())
    envelope = wrap_surface_envelope(soup, _policy())
    assert envelope.evidence.source_topology.degenerate_faces == degenerate_faces
    assert envelope.evidence.repaired_topology.degenerate_faces == 0
    assert np.array_equal(soup.vertices, points)


@pytest.mark.parametrize(
    "budgets",
    [
        (1, 10_000_000, 10_000, 20_000),
        (100_000, 1, 10_000, 20_000),
        (100_000, 10_000_000, 1, 20_000),
        (100_000, 10_000_000, 10_000, 1),
    ],
    ids=["samples", "work", "vertices", "triangles"],
)
def test_finite_native_budget_refuses_without_a_surface(
    budgets: tuple[int, int, int, int],
) -> None:
    samples, work, vertices, triangles = budgets
    with pytest.raises(MeshcoreError) as error:
        wrap_surface_envelope(
            _sheet(),
            _policy(
                max_samples=samples,
                max_work_units=work,
                max_vertices=vertices,
                max_triangles=triangles,
            ),
        )
    assert error.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_preserved_cutcell_carrier_keeps_every_authoritative_skin_triangle() -> None:
    envelope = wrap_surface_envelope(_sheet(), _policy())
    surface_points = np.asarray(envelope.repaired.mesh.coordinates, dtype=np.float64)
    skin = np.asarray(envelope.repaired.mesh.blocks[0].vertices, dtype=np.int64)
    assert np.array_equal(
        envelope.carrier_vertices[: surface_points.shape[0]], surface_points
    )
    cells = envelope.carrier_tetrahedra
    corners = envelope.carrier_vertices[cells]
    assert np.all(
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]) > 0
    )
    faces = np.concatenate(
        (
            cells[:, [1, 3, 2]],
            cells[:, [0, 2, 3]],
            cells[:, [0, 3, 1]],
            cells[:, [0, 1, 2]],
        )
    )
    unique, incidence = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    assert np.all(incidence <= 2)
    assert np.array_equal(
        unique[incidence == 1], np.unique(np.sort(skin, axis=1), axis=0)
    )


def test_finite_volume_vertex_budget_refuses_the_complete_carrier() -> None:
    with pytest.raises(MeshcoreError) as error:
        wrap_surface_envelope(_sheet(), _policy(max_volume_vertices=1))
    assert error.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_finite_tetrahedron_budget_refuses_the_complete_carrier() -> None:
    with pytest.raises(MeshcoreError) as error:
        wrap_surface_envelope(_sheet(), _policy(max_tetrahedra=1))
    assert error.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_subnormal_sampling_refuses_an_unrepresentable_inclusion_certificate() -> None:
    unit = np.finfo(np.float64).smallest_subnormal
    soup = RawTriangleSoup(
        np.zeros((1, 3), dtype=np.float64), np.zeros((1, 3), dtype=np.int64), _metadata()
    )
    policy = SurfaceEnvelopePolicy(
        offset=float(10 * unit),
        maximum_deviation=float(20 * unit),
        spacing=float(unit),
        certificate_spacing=float(unit),
        feature_permission=EnvelopeFeaturePermission.ALLOW_ROUNDING,
        topology_permission=EnvelopeTopologyPermission.ALLOW_CHANGE,
        max_samples=100_000,
        max_work_units=100_000,
        max_vertices=1_000,
        max_triangles=1_000,
    )
    with pytest.raises(ValueError, match="certificate"):
        wrap_surface_envelope(soup, policy)


def test_exact_feature_preservation_is_not_silently_rounded() -> None:
    with pytest.raises(ValueError, match="feature preservation"):
        wrap_surface_envelope(
            _sheet(), _policy(feature_permission=EnvelopeFeaturePermission.PRESERVE)
        )


def test_topology_preservation_requires_a_real_certificate() -> None:
    with pytest.raises(ValueError, match="homeomorphism certificate"):
        wrap_surface_envelope(
            _sheet(), _policy(topology_permission=EnvelopeTopologyPermission.PRESERVE)
        )


def test_certified_surface_is_distinct_from_dirty_soup_and_loses_no_hidden_selection() -> (
    None
):
    soup = _sheet()
    model = SurfaceModel.from_triangles(soup.vertices, soup.triangles, soup.metadata)
    selection = model.bind_selection("sheet", np.array([0], dtype=np.int64))
    model = model.with_selection(selection)
    envelope = wrap_surface_envelope(model, _policy())
    assert envelope.evidence.source_geometry_id == model.model_id
    assert envelope.evidence.lost_selection_ids == (selection.selection_id,)
    assert envelope.repaired.model.selections == ()
    assert envelope.evidence.rounded_feature_edges == 0


def test_original_soup_identity_cannot_bind_repaired_volume_constraints() -> None:
    source = NativeSurfaceEnvelopeSource(
        wrap_surface_envelope(_sheet(), _policy()), "wrapped-solid"
    )
    with pytest.raises(ValueError):
        PreparedSurfaceEnvelopeVolume(
            source, _specification(source, original_identity=True), NativeVolumeSchedule()
        )


def test_whole_envelope_volume_route_refuses_an_exhausted_wall_budget() -> None:
    source = NativeSurfaceEnvelopeSource(
        wrap_surface_envelope(_sheet(), _policy()), "wrapped-solid"
    )
    specification = _specification(source, maximum_wall_seconds=1e-12)
    schedule = NativeVolumeSchedule(
        refinement_rounds=1, improvement_passes=1, minimum_dihedral_degrees=0.0
    )
    prepared = PreparedSurfaceEnvelopeVolume(source, specification, schedule)
    with pytest.raises(M.MeshingFailure) as error:
        execute_surface_envelope_route(
            source,
            specification,
            prepared,
            _SI,
            NativeMeshingProvider.info(),
            "envelope-wall-refusal",
        )
    assert error.value.category is M.MeshingFailureCategory.TIMED_OUT


def test_envelope_native_initial_allocation_refuses_and_preserves_repaired_source() -> (
    None
):
    soup = _sheet()
    source = NativeSurfaceEnvelopeSource(
        wrap_surface_envelope(soup, _policy()), "wrapped-solid"
    )
    envelope = source.envelope
    points, cells = envelope.carrier_vertices.copy(), envelope.carrier_tetrahedra.copy()
    specification = _specification(source, maximum_scratch_bytes=1)
    schedule = NativeVolumeSchedule(
        refinement_rounds=1, improvement_passes=1, minimum_dihedral_degrees=0.0
    )
    prepared = PreparedSurfaceEnvelopeVolume(source, specification, schedule)
    with pytest.raises(M.MeshingFailure) as error:
        execute_surface_envelope_route(
            source,
            specification,
            prepared,
            _SI,
            NativeMeshingProvider.info(),
            "envelope-allocation-refusal",
        )
    assert error.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(error.value.evidence.requested)["maximum_scratch_bytes"] == 1
    assert np.array_equal(envelope.carrier_vertices, points)
    assert np.array_equal(envelope.carrier_tetrahedra, cells)
    assert envelope.evidence.source_geometry_id == soup.soup_id
    assert envelope.evidence.repaired_model_id == envelope.repaired.model.model_id


@pytest.mark.parametrize(
    ("source_kind", "original_boundary_edges", "repaired_components"),
    (("sheet", 3, 1), ("dirty", 1, 1), ("disconnected", 6, 2)),
)
def test_envelope_volume_publishes_exact_fill_and_original_repair_evidence(
    source_kind: str,
    original_boundary_edges: int,
    repaired_components: int,
    tmp_path: Path,
) -> None:
    soup = _sheet()
    if source_kind == "dirty":
        soup = RawTriangleSoup(
            soup.vertices,
            np.array([[0, 1, 2], [2, 1, 0], [0, 0, 1]], dtype=np.int64),
            soup.metadata,
            feature_edges=soup.feature_edges,
        )
    elif source_kind == "disconnected":
        soup = RawTriangleSoup(
            np.concatenate((soup.vertices, soup.vertices + np.array([0.0, 0.0, 1.0]))),
            np.concatenate((soup.triangles, soup.triangles + 3)),
            soup.metadata,
            feature_edges=np.concatenate((soup.feature_edges, soup.feature_edges + 3)),
        )
    source = NativeSurfaceEnvelopeSource(
        wrap_surface_envelope(soup, _policy()), "wrapped-solid"
    )
    specification = _specification(source)
    triangles, polygon_ids, edge_vertices = surface_envelope_source_entities(source)
    assert np.array_equal(polygon_ids, np.arange(triangles.shape[0]))
    edge_oracle = np.unique(
        np.sort(
            np.concatenate(
                (
                    triangles[:, (0, 1)],
                    triangles[:, (1, 2)],
                    triangles[:, (2, 0)],
                )
            ),
            axis=1,
        ),
        axis=0,
    )
    assert np.array_equal(edge_vertices, edge_oracle)
    schedule = NativeVolumeSchedule(
        refinement_rounds=1, improvement_passes=1, minimum_dihedral_degrees=0.0
    )
    prepared = PreparedSurfaceEnvelopeVolume(source, specification, schedule)
    try:
        result = execute_surface_envelope_route(
            source,
            specification,
            prepared,
            _SI,
            NativeMeshingProvider.info(),
            "envelope-positive-smoke",
        )
    except M.MeshingFailure as failure:
        evidence = failure.evidence
        path = tmp_path / f"envelope-{source_kind}-failure.json"
        path.write_text(
            json.dumps(
                {
                    "category": failure.category.value,
                    "message": str(failure),
                    "stage": evidence.stage,
                    "provider_code": evidence.provider_code,
                    "evidence_id": evidence.evidence_id,
                    "entity_ids": evidence.entity_ids,
                    "locations": evidence.locations,
                    "checkpoint_id": evidence.checkpoint_id,
                    "requested": evidence.requested,
                    "achieved": evidence.achieved,
                    "logical_findings": [
                        {
                            "name": name,
                            "dtype": str(np.asarray(value).dtype),
                            "shape": np.asarray(value).shape,
                            "bits": np.asarray(value).tobytes().hex(),
                        }
                        for name, value in evidence.logical_findings
                    ],
                    "original_source_geometry_id": soup.soup_id,
                    "source_binding_id": source.binding_id,
                    "policy_id": source.envelope.policy.policy_id,
                    "specification_id": specification.specification_id,
                    "limits_id": specification.limits.limits_id,
                },
                indent=2,
            )
        )
        print(f"Retained envelope failure evidence: {path}")
        raise
    corners = np.asarray(result.mesh.coordinates, dtype=np.float64)[
        np.asarray(result.mesh.blocks[0].vertices, dtype=np.int64)
    ]
    assert np.all(
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]) > 0
    )
    skin_points = np.asarray(source.envelope.repaired.mesh.coordinates, dtype=np.float64)
    skin = np.asarray(source.envelope.repaired.mesh.blocks[0].vertices, dtype=np.int64)
    triangles = skin_points[skin]
    expected_volume = (
        np.sum(
            np.sum(triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2]), axis=1)
        )
        / 6.0
    )
    actual_volume = np.sum(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0
    assert actual_volume == pytest.approx(expected_volume, rel=1e-12, abs=1e-14)
    assert result.certification is not None
    assert result.certification.passed
    assert result.audit.passed
    achieved = dict(result.compliance.achieved)
    assert (
        achieved["envelope:source_to_repaired_upper"]
        <= source.envelope.policy.maximum_deviation
    )
    assert (
        achieved["envelope:repaired_to_source_upper"]
        <= source.envelope.policy.maximum_deviation
    )
    assert achieved["envelope:source_inclusion_margin"] > 0
    transition = result.trace.stages[0]
    assert transition.stage is M.MeshingStageKind.TOPOLOGY_REPAIR
    assert source.envelope.source.metadata.source_revision in transition.input_ids
    assert source.source_revision in transition.output_ids
    payload = dict(json.loads(result.provenance.content_json)["items"])
    policy = dict(payload["wrapping_policy"]["items"])
    evidence = dict(payload["wrapping_evidence"]["items"])
    original_topology = dict(evidence["source_topology"]["items"])
    repaired_topology = dict(evidence["repaired_topology"]["items"])
    assert policy["interpretation"] == "sampled_unsigned_distance_sublevel"
    assert original_topology["boundary_edges"] == original_boundary_edges
    assert repaired_topology["boundary_edges"] == 0
    assert repaired_topology["components"] == repaired_components
    assert payload["preserved_carrier"] == source.envelope.carrier_id
    assert {zone.name for zone in result.zones} == {"wrapped-solid"}
    from tools.meshing_qualification import _envelope_qualification_acceptance

    assert _envelope_qualification_acceptance(source, result)["passed"]
    missing_bound = M.MeshingComplianceReport(
        result.compliance.specification_id,
        issues=result.compliance.issues,
        requested=result.compliance.requested,
        achieved=tuple(
            item
            for item in result.compliance.achieved
            if item[0] != "envelope:source_to_repaired_upper"
        ),
    )
    stripped_bound = eqx.tree_at(
        lambda value: value.compliance,
        result,
        missing_bound,
    )
    with pytest.raises(RuntimeError, match="original directed"):
        _envelope_qualification_acceptance(source, stripped_bound)
    stripped_ancestry = eqx.tree_at(
        lambda value: value.trace,
        result,
        M.MeshingTrace(result.trace.stages[1:], binding=result.trace.binding),
    )
    with pytest.raises(RuntimeError, match="raw-source repair ancestry"):
        _envelope_qualification_acceptance(source, stripped_ancestry)
    from tools.meshing_qualification import (
        _envelope_qualification_association_transfer,
        _envelope_qualification_prepare_solver,
        _envelope_qualification_state_transition,
    )

    association_transfer = _envelope_qualification_association_transfer(
        source,
        result,
        specification.limits,
    )
    from dataclasses import asdict

    from phydrax.meshing._certification import _MeshCertificationPremiseFailure

    try:
        adaptation = M.execute_mesh_adaptation(
            M.prepare_mesh_adaptation(
                result,
                M.MarkedMeshAdaptation(
                    np.concatenate(
                        [np.asarray(block.global_ids) for block in result.mesh.blocks]
                    ),
                    np.empty((0,), dtype=np.int64),
                ),
                policy=M.MeshAdaptationPolicy(
                    M.MeshAdaptationRoute.NATIVE_MIXED,
                    limits=specification.limits,
                    association_transfer=association_transfer,
                    audit_policy=M.CellMeshAuditPolicy(
                        require_complete_association=True,
                        watertight_boundary=M.CellMeshAuditDisposition.REJECT,
                    ),
                ),
            )
        )
    except _MeshCertificationPremiseFailure as failure:
        ledger = failure.ledger
        evidence = failure.evidence
        path = tmp_path / f"envelope-{source_kind}-adaptation-coverage-failure.json"
        path.write_text(
            json.dumps(
                {
                    "category": failure.category.value,
                    "message": str(failure),
                    "stage": evidence.stage,
                    "provider_code": evidence.provider_code,
                    "evidence_id": evidence.evidence_id,
                    "requested": evidence.requested,
                    "achieved": evidence.achieved,
                    "coverage": asdict(failure.coverage),
                    "embedding": asdict(failure.embedding),
                    "unbounded_region_measures": failure.unbounded_region_measures,
                    "certificate_request": {
                        "request_id": failure.request.request_id,
                        "mesh_id": failure.request.mesh_id,
                        "topology_id": failure.request.topology_id,
                        "geometry_id": failure.request.geometry_id,
                        "source_id": failure.request.source_id,
                        "source_revision": failure.request.source_revision,
                        "domain_id": failure.request.domain_id,
                        "schedule_id": failure.request.schedule.schedule_id,
                        "limits": asdict(failure.request.limits),
                    },
                    "original_ledger_at_catch": None
                    if ledger is None
                    else {
                        "maximum_work_units": ledger.maximum_work_units,
                        "maximum_memory_bytes": ledger.maximum_memory_bytes,
                        "work_units": ledger.work_units,
                        "native_charged_work_units": ledger.native_charged_work_units,
                        "retained_basis_bytes": ledger.retained_basis_bytes,
                        "temporary_bytes_upper": ledger.temporary_bytes_upper,
                        "peak_bytes_upper": ledger.peak_bytes_upper,
                    },
                    "original_source_geometry_id": soup.soup_id,
                    "source_binding_id": source.binding_id,
                    "policy_id": source.envelope.policy.policy_id,
                    "specification_id": specification.specification_id,
                    "limits_id": specification.limits.limits_id,
                },
                indent=2,
            )
        )
        print(f"Retained envelope adaptation coverage evidence: {path}")
        raise
    assert adaptation.status.converged
    space, problem, _ = _envelope_qualification_prepare_solver(result)
    values = space.dof_maps[0].dof_coordinates[:, 0]
    _, auxiliary, transfer = _envelope_qualification_state_transition(
        result,
        adaptation,
        problem,
        values,
        0,
    )
    assert transfer["field_names"] == ["u", "density", "history"]
    rollback = transfer["rollback"]
    if not isinstance(rollback, dict):
        raise TypeError("Envelope transfer evidence requires a rollback record.")
    assert rollback["passed"]
    assert np.all(np.asarray(auxiliary[0]) == 2.0)
    history_defect = transfer["history_max_defect"]
    if not isinstance(history_defect, (int, float)):
        raise TypeError("Envelope transfer evidence requires a numeric history defect.")
    assert history_defect <= 1.0e-12


@pytest.mark.parametrize("bound", ("work", "queries", "bytes", "clock"))
def test_original_source_preparation_receipt_refuses_tighter_volume_request_without_mutation(
    bound: str,
) -> None:
    from dataclasses import fields as dataclass_fields

    from phydrax._fingerprint import array_tree_fingerprint

    soup = _sheet()
    envelope = wrap_surface_envelope(soup, _policy())
    source = NativeSurfaceEnvelopeSource(envelope, "wrapped-solid")
    assert envelope.execution_evidence is not None
    before = array_tree_fingerprint(source)
    controls = {
        "work": {"maximum_work_units": 1},
        "queries": {"maximum_geometry_queries": 1},
        "bytes": {"maximum_scratch_bytes": 1},
        "clock": {"maximum_wall_seconds": 1.0e-12},
    }
    original = _specification(source)
    requested = {
        field.name: getattr(original.limits, field.name)
        for field in dataclass_fields(original.limits)
        if field.name != "limits_id"
    }
    requested.update(controls[bound])
    specification = eqx.tree_at(
        lambda value: value.limits,
        original,
        M.MeshingLimits(**requested),
    )
    schedule = NativeVolumeSchedule(
        refinement_rounds=1, improvement_passes=1, minimum_dihedral_degrees=0.0
    )
    prepared = PreparedSurfaceEnvelopeVolume(source, specification, schedule)
    with pytest.raises(M.MeshingFailure) as error:
        execute_surface_envelope_route(
            source,
            specification,
            prepared,
            _SI,
            NativeMeshingProvider.info(),
            f"envelope-original-source-{bound}-refusal",
        )
    expected = (
        M.MeshingFailureCategory.TIMED_OUT
        if bound == "clock"
        else M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    )
    assert error.value.category is expected
    assert array_tree_fingerprint(source) == before
    assert array_tree_fingerprint(source.envelope.source) == array_tree_fingerprint(soup)
    assert source.envelope.evidence.source_geometry_id == soup.soup_id
    assert not soup.vertices.flags.writeable
    assert not soup.triangles.flags.writeable
    assert not soup.feature_edges.flags.writeable
