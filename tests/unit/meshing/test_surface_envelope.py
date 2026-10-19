# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import math
from dataclasses import (
    asdict,
    dataclass,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
import pytest
from jax import Array, tree_util

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint, canonical_json
from phydrax._identity import SemanticProvenance
from phydrax._meshcore import exact_orient3d, MeshcoreError, MeshcoreStatus
from phydrax._model._structure import _restore_dataclass, restore_registered_fields
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry.surface._model import SurfaceModel
from phydrax.lifecycle._meshing_source_records import register_meshing_source_types
from phydrax.lifecycle._meshing_sources import (
    write_meshing_source_closure,
)
from phydrax.meshing._surface_envelope import (
    _admit_surface_envelope_geometry_replay,
    _admit_surface_envelope_record,
    _authenticate_surface_envelope_repair,
    _carry_surface_envelope_repair,
    _certified_surface_envelope_domain,
    _decode_surface_envelope_repair,
    _repair_provenance_mapping,
    _require_surface_envelope_repair_metrics,
    _surface_envelope_achieved,
    _surface_envelope_evidence_id,
    _surface_envelope_policy_payload,
    _surface_envelope_repair_payload,
    _surface_envelope_repair_stage,
    _topology,
    _validate_surface_envelope_evidence_fields,
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
from phydrax.meshing._volume_generation import (
    native_volume_execution_budget,
    NativeVolumeSchedule,
)
from phydrax.meshing.providers._native import NativeMeshingProvider
from phydrax.meshing.providers._native_sources import NativeSurfaceEnvelopeSource
from phydrax.meshing.qualification import (
    prepare_envelope_association_transfer,
    prepare_envelope_solver,
    transition_envelope_state,
    validate_envelope_acceptance,
)
from tests._support.envelope_meshing import (
    envelope_failure,
    envelope_metadata,
    envelope_policy,
    envelope_sheet,
    envelope_specification,
    quadratic_l2_error,
    quadratic_source,
    quadratic_value,
    tetrahedron_measures,
)


M = phx.meshing
_SI = phx.SpatialCoordinateContract.si()

# Empirical bounds retained from the native repaired-carrier regression fixture
# (offset 0.263, grid spacing 0.11, certificate spacing 0.05). The volume
# reference sums oriented skin triangles independently of tetrahedron maps.
_VOLUME_RELATIVE_ACCEPTANCE = 1.0e-12
_VOLUME_ABSOLUTE_ACCEPTANCE = 1.0e-14
# The affine density/history contract uses the same 1e-12 acceptance bound as
# the moved envelope state-transition qualification.
_AFFINE_FIELD_ACCEPTANCE = 1.0e-12


def _changed_stored_fields[T](value: T, /, **changes: Any) -> T:
    """Stage corrupt persisted fields; leave scientific authentication to consumers."""
    owner: type[T] = type(value)
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(
            "stored_fields.value must be a dataclass instance; got another object kind"
        )
    register_meshing_source_types()
    stored = {
        member.name: getattr(value, member.name) for member in dataclass_fields(value)
    }
    stored.update(changes)
    if getattr(owner, "_strict_contract_", False):
        # Opted-in owners need their field checks and initialized Strict guard.
        return restore_registered_fields(owner, stored)
    # Legacy dataclasses have no Strict field contract. Use the same registered
    # archive staging boundary as persisted transfer-evidence corruption tests;
    # replaying their owner here would reject the fault before the
    # authentication/resource-admission boundary under test can consume it.
    ordered = list(stored.values())
    restored = _restore_dataclass(type(value), list(range(len(ordered))), ordered)
    if not isinstance(restored, owner) or type(restored) is not owner:
        raise TypeError(
            "stored_fields.owner must restore the same declared dataclass; got another owner"
        )
    return restored


def _carry_repair_with_request_budget(
    source: M.CellMeshingResult,
    target: M.CellMeshingResult,
    /,
    *,
    limits: M.MeshingLimits,
) -> M.CellMeshingResult:
    with HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=limits.maximum_geometry_queries,
        bytes_upper=limits.maximum_scratch_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with budget.measurement(), budget.activate():
            return _carry_surface_envelope_repair(source, target, budget=budget)


@pytest.mark.parametrize("change", ("duplicate", "unordered"))
def test_envelope_repair_records_require_canonical_named_fields(change: str) -> None:
    encoded = json.loads(
        SemanticProvenance({"source": "source", "policy": "policy"}).content_json
    )
    if change == "duplicate":
        encoded["items"].append(encoded["items"][0])
    else:
        encoded["items"].reverse()
    with pytest.raises(ValueError, match=r"repair_provenance.records"):
        _repair_provenance_mapping(encoded)


def test_open_sheet_wrap_is_closed_and_has_two_directed_bounds() -> None:
    soup = envelope_sheet()
    envelope = wrap_surface_envelope(soup, envelope_policy())
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
    assert int(np.asarray(receipt.memory_counters.peak_bytes)) > 0


@pytest.mark.parametrize("forgery", ("directed-bound", "topology-count"))
def test_envelope_archive_refuses_self_reported_repair_theorem(
    tmp_path: Path,
    forgery: str,
) -> None:
    envelope = wrap_surface_envelope(envelope_sheet(), envelope_policy())
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
    with pytest.raises(
        ValueError, match="stored_fields must match its reconstructed source content"
    ):
        write_meshing_source_closure(tmp_path / "forged-source", source)


def test_dirty_duplicate_reversed_and_degenerate_faces_remain_visible() -> None:
    clean = envelope_sheet()
    faces = np.array([[0, 1, 2], [2, 1, 0], [0, 0, 1]], dtype=np.int64)
    dirty = RawTriangleSoup(clean.vertices, faces, envelope_metadata())
    envelope = wrap_surface_envelope(dirty, envelope_policy())
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
    soup = RawTriangleSoup(
        points, np.array([[0, 1, 2]], dtype=np.int64), envelope_metadata()
    )
    envelope = wrap_surface_envelope(soup, envelope_policy())
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
            envelope_sheet(),
            envelope_policy(
                max_samples=samples,
                max_work_units=work,
                max_vertices=vertices,
                max_triangles=triangles,
            ),
        )
    assert error.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_preserved_cutcell_carrier_keeps_every_authoritative_skin_triangle() -> None:
    envelope = wrap_surface_envelope(envelope_sheet(), envelope_policy())
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
        wrap_surface_envelope(envelope_sheet(), envelope_policy(max_volume_vertices=1))
    assert error.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_finite_tetrahedron_budget_refuses_the_complete_carrier() -> None:
    with pytest.raises(MeshcoreError) as error:
        wrap_surface_envelope(envelope_sheet(), envelope_policy(max_tetrahedra=1))
    assert error.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_subnormal_sampling_refuses_an_unrepresentable_inclusion_certificate() -> None:
    unit = np.finfo(np.float64).smallest_subnormal
    soup = RawTriangleSoup(
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.int64),
        envelope_metadata(),
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
            envelope_sheet(),
            envelope_policy(feature_permission=EnvelopeFeaturePermission.PRESERVE),
        )


def test_topology_preservation_requires_a_real_certificate() -> None:
    with pytest.raises(ValueError, match="homeomorphism certificate"):
        wrap_surface_envelope(
            envelope_sheet(),
            envelope_policy(topology_permission=EnvelopeTopologyPermission.PRESERVE),
        )


def test_certified_surface_is_distinct_from_dirty_soup_and_loses_no_hidden_selection() -> (
    None
):
    soup = envelope_sheet()
    model = SurfaceModel.from_triangles(soup.vertices, soup.triangles, soup.metadata)
    selection = model.bind_selection("sheet", np.array([0], dtype=np.int64))
    model = model.with_selection(selection)
    envelope = wrap_surface_envelope(model, envelope_policy())
    assert envelope.evidence.source_geometry_id == model.model_id
    assert envelope.evidence.lost_selection_ids == (selection.selection_id,)
    assert envelope.repaired.model.selections == ()
    assert envelope.evidence.rounded_feature_edges == 0


def test_original_soup_identity_cannot_bind_repaired_volume_constraints() -> None:
    source = NativeSurfaceEnvelopeSource(
        wrap_surface_envelope(envelope_sheet(), envelope_policy()), "wrapped-solid"
    )
    with pytest.raises(ValueError, match=r"PreparedSurfaceEnvelopeVolume\.constraints"):
        PreparedSurfaceEnvelopeVolume(
            source,
            envelope_specification(source, original_identity=True),
            NativeVolumeSchedule(),
        )


def test_whole_envelope_volume_route_refuses_an_exhausted_wall_budget() -> None:
    source = NativeSurfaceEnvelopeSource(
        wrap_surface_envelope(envelope_sheet(), envelope_policy()), "wrapped-solid"
    )
    specification = envelope_specification(source, maximum_wall_seconds=1e-12)
    schedule = NativeVolumeSchedule(
        refinement_rounds=1, improvement_passes=1, minimum_dihedral_degrees=0.0
    )
    prepared = PreparedSurfaceEnvelopeVolume(source, specification, schedule)
    with pytest.raises(M.MeshingFailure) as error:
        with native_volume_execution_budget(specification.limits) as scope:
            with scope.host_workspace() as workspace:
                execute_surface_envelope_route(
                    source,
                    specification,
                    prepared,
                    _SI,
                    NativeMeshingProvider.info(),
                    "envelope-wall-refusal",
                    scope=scope,
                    workspace=workspace,
                )
    assert error.value.category is M.MeshingFailureCategory.TIMED_OUT


def test_envelope_native_initial_allocation_refuses_and_preserves_repaired_source() -> (
    None
):
    soup = envelope_sheet()
    source = NativeSurfaceEnvelopeSource(
        wrap_surface_envelope(soup, envelope_policy()), "wrapped-solid"
    )
    envelope = source.envelope
    points, cells = envelope.carrier_vertices.copy(), envelope.carrier_tetrahedra.copy()
    specification = envelope_specification(source, maximum_scratch_bytes=1)
    schedule = NativeVolumeSchedule(
        refinement_rounds=1, improvement_passes=1, minimum_dihedral_degrees=0.0
    )
    prepared = PreparedSurfaceEnvelopeVolume(source, specification, schedule)
    with pytest.raises(M.MeshingFailure) as error:
        with native_volume_execution_budget(specification.limits) as scope:
            with scope.host_workspace() as workspace:
                execute_surface_envelope_route(
                    source,
                    specification,
                    prepared,
                    _SI,
                    NativeMeshingProvider.info(),
                    "envelope-allocation-refusal",
                    scope=scope,
                    workspace=workspace,
                )
    assert error.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(error.value.evidence.requested)["maximum_scratch_bytes"] == 1
    assert np.array_equal(envelope.carrier_vertices, points)
    assert np.array_equal(envelope.carrier_tetrahedra, cells)
    assert envelope.evidence.source_geometry_id == soup.soup_id
    assert envelope.evidence.repaired_model_id == envelope.repaired.model.model_id


@dataclass(frozen=True)
class _EnvelopePublication:
    source: NativeSurfaceEnvelopeSource
    specification: M.VolumeMeshingSpec
    result: M.CellMeshingResult
    original_boundary_edges: int
    repaired_components: int


_ENVELOPE_PUBLICATION_CASES = (("sheet", 3, 1), ("dirty", 1, 1), ("disconnected", 6, 2))
_REPAIR_AUTHENTICATION_CASE = pytest.mark.parametrize(
    "envelope_publication",
    (_ENVELOPE_PUBLICATION_CASES[0],),
    indirect=True,
    ids=("sheet-3-1",),
    scope="module",
)


@pytest.fixture(
    scope="module",
    params=_ENVELOPE_PUBLICATION_CASES,
    ids=("sheet-3-1", "dirty-1-1", "disconnected-6-2"),
)
def envelope_publication(request: pytest.FixtureRequest) -> _EnvelopePublication:
    source_kind, original_boundary_edges, repaired_components = request.param
    soup = envelope_sheet()
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
        wrap_surface_envelope(soup, envelope_policy()), "wrapped-solid"
    )
    specification = envelope_specification(source)
    schedule = NativeVolumeSchedule(
        refinement_rounds=1, improvement_passes=1, minimum_dihedral_degrees=0.0
    )
    result = (
        NativeMeshingProvider(
            M.NativeMeshingOptions(
                "surface_envelope_tetrahedral", volume_schedule=schedule
            )
        )
        .plan(source, specification, coordinate_contract=_SI)
        .execute()
    )
    return _EnvelopePublication(
        source=source,
        specification=specification,
        result=result,
        original_boundary_edges=original_boundary_edges,
        repaired_components=repaired_components,
    )


def test_envelope_specification_binds_repaired_identity_and_exact_requested_limits(
    envelope_publication: _EnvelopePublication,
) -> None:
    source = envelope_publication.source
    specification = envelope_publication.specification

    def require_repaired_request(requested: M.VolumeMeshingSpec) -> None:
        assert requested.boundary_scope.source_id == source.source_id
        assert requested.boundary_scope.source_revision == source.source_revision
        np.testing.assert_array_equal(
            requested.boundary_scope.entity_ids,
            np.arange(source.plc_source.complex.facet_count, dtype=np.int64),
        )
        limits = requested.limits
        assert (limits.maximum_vertices, limits.maximum_cells) == (20_000, 100_000)
        assert limits.maximum_work_units == 20_000_000
        assert limits.maximum_wall_seconds == 3600.0
        assert limits.maximum_scratch_bytes == 8_000_000_000

    require_repaired_request(specification)
    with pytest.raises(AssertionError):
        require_repaired_request(envelope_specification(source, original_identity=True))
    requested = envelope_specification(
        source, maximum_wall_seconds=1e-12, maximum_scratch_bytes=1
    )

    def require_override(request: M.VolumeMeshingSpec) -> None:
        assert request.limits.maximum_wall_seconds == 1e-12
        assert request.limits.maximum_scratch_bytes == 1

    require_override(requested)
    with pytest.raises(AssertionError):
        require_override(specification)


def test_envelope_source_entities_preserve_polygon_and_edge_identity(
    envelope_publication: _EnvelopePublication,
) -> None:
    source = envelope_publication.source
    with native_volume_execution_budget(
        envelope_publication.specification.limits
    ) as scope:
        triangles, polygon_ids, edge_vertices = surface_envelope_source_entities(
            source, scope=scope
        )
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


def test_envelope_volume_publishes_exact_fill_and_original_repair_evidence(
    envelope_publication: _EnvelopePublication,
) -> None:
    source, result = envelope_publication.source, envelope_publication.result
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
    assert actual_volume == pytest.approx(
        expected_volume, rel=_VOLUME_RELATIVE_ACCEPTANCE, abs=_VOLUME_ABSOLUTE_ACCEPTANCE
    )
    assert result.certification is not None
    assert result.certification.passed
    assert result.audit.passed


def test_envelope_publication_retains_repair_provenance(
    envelope_publication: _EnvelopePublication,
) -> None:
    source, result = envelope_publication.source, envelope_publication.result
    original_boundary_edges = envelope_publication.original_boundary_edges
    repaired_components = envelope_publication.repaired_components
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
    assert validate_envelope_acceptance(
        source,
        result,
        limits=envelope_publication.specification.limits,
        failure_factory=envelope_failure,
        measure_cells=tetrahedron_measures,
    )["passed"]


def test_envelope_stored_fields_revalidate_round_trip(
    envelope_publication: _EnvelopePublication,
) -> None:
    envelope = envelope_publication.source.envelope
    restored = SurfaceEnvelope.from_stored_fields(envelope)
    assert restored.envelope_id == envelope.envelope_id
    assert restored.carrier_id == envelope.carrier_id
    assert restored.evidence == envelope.evidence
    assert np.array_equal(restored.carrier_vertices, envelope.carrier_vertices)
    assert np.array_equal(restored.carrier_tetrahedra, envelope.carrier_tetrahedra)
    assert restored.execution_evidence is envelope.execution_evidence


def test_envelope_stored_fields_refuse_corrupt_execution_counters(
    envelope_publication: _EnvelopePublication,
) -> None:
    envelope = envelope_publication.source.envelope
    receipt = envelope.execution_evidence
    assert receipt is not None
    forged = eqx.tree_at(
        lambda value: value.execution_evidence,
        envelope,
        eqx.tree_at(
            lambda value: value.work_counters.construction_work,
            receipt,
            receipt.work_counters.construction_work.astype(np.float64),
        ),
    )
    with pytest.raises(
        ValueError, match="construction_work must be a scalar uint64 leaf"
    ):
        SurfaceEnvelope.from_stored_fields(forged)


def test_envelope_acceptance_refuses_missing_directed_bound(
    envelope_publication: _EnvelopePublication,
) -> None:
    source, result = envelope_publication.source, envelope_publication.result
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
    with pytest.raises(
        RuntimeError, match="envelope.acceptance.source_to_repaired_upper"
    ):
        validate_envelope_acceptance(
            source,
            stripped_bound,
            limits=envelope_publication.specification.limits,
            failure_factory=envelope_failure,
            measure_cells=tetrahedron_measures,
        )


def test_envelope_acceptance_refuses_missing_repair_ancestry(
    envelope_publication: _EnvelopePublication,
) -> None:
    source, result = envelope_publication.source, envelope_publication.result
    stripped_ancestry = eqx.tree_at(
        lambda value: value.trace,
        result,
        M.MeshingTrace(result.trace.stages[1:], binding=result.trace.binding),
    )
    with pytest.raises(RuntimeError, match="envelope.acceptance.repair_ancestry"):
        validate_envelope_acceptance(
            source,
            stripped_ancestry,
            limits=envelope_publication.specification.limits,
            failure_factory=envelope_failure,
            measure_cells=tetrahedron_measures,
        )


@pytest.fixture(scope="module")
def envelope_adaptation(
    envelope_publication: _EnvelopePublication,
) -> M.MeshAdaptationResult:
    source, result, specification = (
        envelope_publication.source,
        envelope_publication.result,
        envelope_publication.specification,
    )
    association_transfer = prepare_envelope_association_transfer(
        source,
        result,
        specification.limits,
    )
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
    assert adaptation.status.converged
    return adaptation


@_REPAIR_AUTHENTICATION_CASE
def test_envelope_adaptation_retains_authenticated_repair_not_mesh_statistics(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
) -> None:
    source = envelope_publication.source
    initial, target = envelope_publication.result, envelope_adaptation.target
    evidence = source.envelope.evidence
    assert dict(target.compliance.achieved) == {
        "envelope:source_to_repaired_upper": evidence.source_to_repaired_upper,
        "envelope:repaired_to_source_upper": evidence.repaired_to_source_upper,
        "envelope:source_inclusion_margin": evidence.source_inclusion_margin,
        "envelope:interpolation_error": evidence.interpolation_error,
        "envelope:original_components": evidence.source_topology.components,
        "envelope:repaired_components": evidence.repaired_topology.components,
        "envelope:original_boundary_edges": evidence.source_topology.boundary_edges,
        "envelope:repaired_boundary_edges": evidence.repaired_topology.boundary_edges,
        "envelope:rounded_feature_edges": evidence.rounded_feature_edges,
        "envelope:grid_samples": evidence.grid_samples,
        "envelope:certificate_samples": evidence.certificate_samples,
        "envelope:work_units": evidence.work_units,
        "envelope:carrier_vertices": evidence.carrier_vertices,
        "envelope:carrier_tetrahedra": evidence.carrier_tetrahedra,
    }
    assert target.quality.report_id == target.audit.quality.report_id
    assert target.quality.report_id != initial.quality.report_id
    assert target.trace.stages[0] == initial.trace.stages[0]
    assert envelope_adaptation.transition is not None
    assert envelope_adaptation.transition.target.result_id == target.result_id
    payload = dict(json.loads(target.provenance.content_json)["items"])
    assert payload["kind"] == "native-adapted-surface-envelope-volume"
    assert payload["source"] == initial.result_id
    assert payload["repair_provenance"] == initial.provenance.semantic_id
    assert payload["preserved_carrier"] == source.envelope.carrier_id
    assert validate_envelope_acceptance(
        source,
        target,
        limits=envelope_publication.specification.limits,
        failure_factory=envelope_failure,
        measure_cells=tetrahedron_measures,
    )["passed"]


def _repair_provenance_content(
    source: NativeSurfaceEnvelopeSource, result: M.CellMeshingResult, /
) -> dict[str, Any]:
    retained = dict(json.loads(result.provenance.content_json)["items"])
    content = {
        "kind": retained["kind"],
        "envelope": retained["envelope"],
        "preserved_carrier": retained["preserved_carrier"],
        "wrapping_evidence": asdict(source.envelope.evidence),
        "wrapping_policy": _surface_envelope_policy_payload(source.envelope.policy),
        "exact_repaired_volume": retained["exact_repaired_volume"],
    }
    for name in ("plan", "source", "repair_provenance"):
        if name in retained:
            content[name] = retained[name]
    return content


@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_carry_refuses_tampered_semantic_payload(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
) -> None:
    initial = envelope_publication.result
    provenance = _changed_stored_fields(
        initial.provenance,
        content_json=initial.provenance.content_json.replace(
            "sampled_unsigned_distance_sublevel", "foreign_unsigned_distance_sublevel"
        ),
    )
    tampered = eqx.tree_at(lambda value: value.provenance, initial, provenance)
    with pytest.raises(ValueError, match=r"SemanticProvenance.semantic_id"):
        _carry_repair_with_request_budget(
            tampered,
            envelope_adaptation.target,
            limits=envelope_publication.specification.limits,
        )


@pytest.mark.parametrize("change", ("source", "evidence", "policy"))
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_carry_refuses_rehashed_foreign_provenance(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
    change: str,
) -> None:
    source, initial = envelope_publication.source, envelope_publication.result
    evidence = source.envelope.evidence
    policy = _surface_envelope_policy_payload(source.envelope.policy)
    if change == "source":
        evidence = replace(evidence, source_id="foreign-authored-soup")
    elif change == "evidence":
        evidence = replace(
            evidence, source_to_repaired_upper=evidence.source_to_repaired_upper / 2.0
        )
    else:
        policy["offset"] = source.envelope.policy.offset + 0.001
    content = _repair_provenance_content(source, initial)
    content["wrapping_evidence"] = asdict(evidence)
    content["wrapping_policy"] = policy
    provenance = SemanticProvenance(content)
    foreign = eqx.tree_at(lambda value: value.provenance, initial, provenance)
    with pytest.raises(ValueError, match=r"SurfaceEnvelope.repair_provenance"):
        _carry_repair_with_request_budget(
            foreign,
            envelope_adaptation.target,
            limits=envelope_publication.specification.limits,
        )
    with pytest.raises(RuntimeError, match="envelope.acceptance.repair_provenance"):
        validate_envelope_acceptance(
            source,
            foreign,
            limits=envelope_publication.specification.limits,
            failure_factory=envelope_failure,
            measure_cells=tetrahedron_measures,
        )


@pytest.mark.parametrize("change", ("certificate", "coverage", "association"))
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_carry_refuses_uncertified_or_foreign_target(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
    change: str,
) -> None:
    target = envelope_adaptation.target
    if change == "certificate":
        target = eqx.tree_at(lambda value: value.certification, target, None)
    elif change == "coverage":
        report = target.certification
        assert report is not None and report.coverage is not None
        coverage = _changed_stored_fields(
            report.coverage, domain_id="foreign-repaired-domain"
        )
        report = eqx.tree_at(lambda value: value.coverage, report, coverage)
        target = eqx.tree_at(lambda value: value.certification, target, report)
    else:
        association = _changed_stored_fields(
            target.associations[0], source_revision="foreign-repaired-revision"
        )
        target = eqx.tree_at(
            lambda value: value.associations,
            target,
            (association, *target.associations[1:]),
        )
    with pytest.raises(ValueError, match=r"repair_provenance|GeometryAssociation"):
        _carry_repair_with_request_budget(
            envelope_publication.result,
            target,
            limits=envelope_publication.specification.limits,
        )


@pytest.mark.parametrize("replacement", ("corrupt", "stripped", "nonmapping", "foreign"))
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_adaptation_preparation_refuses_replaced_repair_owner(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
    replacement: str,
) -> None:
    initial = envelope_publication.result
    match replacement:
        case "corrupt":
            provenance = _changed_stored_fields(
                initial.provenance, content_json=json.dumps("foreign-owner")
            )
        case "stripped":
            provenance = SemanticProvenance({})
        case "nonmapping":
            provenance = SemanticProvenance("foreign-owner")
        case _:
            provenance = SemanticProvenance(
                {
                    "kind": "native-cell-mesh-certification",
                    "mesh_id": initial.mesh.mesh_id,
                }
            )
    changed = eqx.tree_at(lambda value: value.provenance, initial, provenance)
    with pytest.raises(
        ValueError, match=r"SemanticProvenance.semantic_id|repair_provenance.owner"
    ):
        M.prepare_mesh_adaptation(
            changed,
            envelope_adaptation.request,
            policy=envelope_adaptation.policy,
        )


@pytest.mark.parametrize(
    "owner,name,change",
    (
        ("initial", "plan", "missing"),
        ("initial", "exact_repaired_volume", "missing"),
        ("adapted", "source", "missing"),
        ("adapted", "repair_provenance", "missing"),
        ("adapted", "exact_repaired_volume", "missing"),
        ("initial", "envelope", "empty"),
        ("adapted", "preserved_carrier", "nonstring"),
        ("initial", "plan", "noncanonical"),
        ("adapted", "source", "nonstring"),
    ),
)
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_carry_requires_complete_owner_history(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
    owner: str,
    name: str,
    change: str,
) -> None:
    result = (
        envelope_publication.result if owner == "initial" else envelope_adaptation.target
    )
    content = _repair_provenance_content(envelope_publication.source, result)
    match change:
        case "missing":
            del content[name]
        case "empty":
            content[name] = ""
        case "noncanonical":
            content[name] = " foreign-owner "
        case _:
            content[name] = 17
    changed = eqx.tree_at(
        lambda value: value.provenance, result, SemanticProvenance(content)
    )
    with pytest.raises((TypeError, ValueError), match=r"repair_provenance"):
        _carry_repair_with_request_budget(
            changed,
            envelope_adaptation.target,
            limits=envelope_publication.specification.limits,
        )


@_REPAIR_AUTHENTICATION_CASE
def test_non_envelope_dispatch_preserves_unrelated_repair_records(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
) -> None:
    initial = envelope_publication.result
    generic = eqx.tree_at(
        lambda value: (value.provenance, value.compliance, value.trace),
        initial,
        (
            SemanticProvenance(
                {
                    "kind": "unrelated-native-source",
                    "wrapping_evidence": "unrelated-record",
                    "preserved_carrier": "unrelated-carrier",
                }
            ),
            M.MeshingComplianceReport(
                initial.compliance.specification_id,
                achieved=(("envelope:unrelated_custom_quality", 1.0),),
            ),
            M.MeshingTrace(
                (
                    M.MeshingStageReport(
                        M.MeshingStageKind.TOPOLOGY_REPAIR,
                        M.MeshingStageStatus.PASSED,
                    ),
                    *initial.trace.stages[1:],
                ),
                binding=initial.trace.binding,
            ),
        ),
    )
    assert (
        _carry_repair_with_request_budget(
            generic,
            envelope_adaptation.target,
            limits=envelope_publication.specification.limits,
        )
        is envelope_adaptation.target
    )


@pytest.mark.parametrize("side", ("source", "target"))
@pytest.mark.parametrize(
    "change",
    ("embedding-id", "coverage-id", "coverage-measures", "coverage-premises"),
)
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_carry_authenticates_retained_certificates(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
    side: str,
    change: str,
) -> None:
    source, target = envelope_publication.result, envelope_adaptation.target
    result = source if side == "source" else target
    report = result.certification
    assert report is not None
    assert report.embedding is not None and report.coverage is not None
    if change == "embedding-id":
        embedding = _changed_stored_fields(
            report.embedding, certificate_id="foreign-embedding-certificate"
        )
        report = eqx.tree_at(lambda value: value.embedding, report, embedding)
    else:
        coverage = report.coverage
        match change:
            case "coverage-id":
                coverage = _changed_stored_fields(
                    coverage, certificate_id="foreign-coverage-certificate"
                )
            case "coverage-measures":
                coverage = _changed_stored_fields(
                    coverage,
                    achieved_region_measures=tuple(
                        None if value is None else value + 0.125
                        for value in coverage.achieved_region_measures
                    ),
                )
            case _:
                coverage = _changed_stored_fields(
                    coverage,
                    premise_certificate_ids=(
                        *coverage.premise_certificate_ids,
                        "foreign-coverage-premise",
                    ),
                )
        report = eqx.tree_at(lambda value: value.coverage, report, coverage)
    changed = eqx.tree_at(lambda value: value.certification, result, report)
    if side == "source":
        source = changed
    else:
        target = changed
    with pytest.raises(
        ValueError, match=r"GlobalEmbeddingCertificate|DomainCoverage|repair_provenance"
    ):
        _carry_repair_with_request_budget(
            source, target, limits=envelope_publication.specification.limits
        )


@pytest.mark.parametrize("side", ("source", "target"))
@pytest.mark.parametrize("change", ("identity", "residuals"))
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_carry_authenticates_association_content(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
    side: str,
    change: str,
) -> None:
    source, target = envelope_publication.result, envelope_adaptation.target
    result = source if side == "source" else target
    association = result.associations[0]
    association = (
        _changed_stored_fields(
            association, association_id="foreign-association-certificate"
        )
        if change == "identity"
        else _changed_stored_fields(association, residuals=association.residuals + 0.125)
    )
    changed = eqx.tree_at(
        lambda value: value.associations,
        result,
        (association, *result.associations[1:]),
    )
    if side == "source":
        source = changed
    else:
        target = changed
    with pytest.raises(ValueError, match=r"GeometryAssociation"):
        _carry_repair_with_request_budget(
            source, target, limits=envelope_publication.specification.limits
        )


@pytest.mark.parametrize("count", ("vertices", "edges", "faces", "components"))
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_carry_refuses_rehashed_false_repaired_topology(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
    count: str,
) -> None:
    source, initial = envelope_publication.source, envelope_publication.result
    original = source.envelope.evidence
    evidence = replace(
        original,
        repaired_topology=replace(
            original.repaired_topology,
            **{count: getattr(original.repaired_topology, count) + 1},
        ),
    )
    topology = evidence.repaired_topology
    evidence = replace(
        evidence,
        repaired_topology=replace(
            topology,
            euler_characteristic=topology.vertices - topology.edges + topology.faces,
        ),
    )
    evidence = replace(
        evidence,
        evidence_id=_surface_envelope_evidence_id(
            evidence, source.envelope.repaired.mesh.coordinates.shape[0]
        ),
    )
    content = _repair_provenance_content(source, initial)
    content["wrapping_evidence"] = asdict(evidence)
    changed = eqx.tree_at(
        lambda value: (value.provenance, value.compliance, value.trace),
        initial,
        (
            SemanticProvenance(content),
            M.MeshingComplianceReport(
                initial.compliance.specification_id,
                issues=initial.compliance.issues,
                requested=initial.compliance.requested,
                achieved=_surface_envelope_achieved(evidence),
            ),
            M.MeshingTrace(
                (
                    _surface_envelope_repair_stage(
                        evidence, source.source_id, source.source_revision
                    ),
                    *initial.trace.stages[1:],
                ),
                binding=initial.trace.binding,
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"repair_provenance.evidence"):
        _carry_repair_with_request_budget(
            changed,
            envelope_adaptation.target,
            limits=envelope_publication.specification.limits,
        )


@pytest.mark.parametrize(
    "change",
    ("policy-string", "counter-float", "topology-float", "tuple-extra"),
)
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_carry_refuses_noncanonical_owner_records(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
    change: str,
) -> None:
    source, initial = envelope_publication.source, envelope_publication.result
    content = _repair_provenance_content(source, initial)
    if change == "policy-string":
        content["wrapping_policy"]["offset"] = str(source.envelope.policy.offset)
    elif change == "counter-float":
        content["wrapping_evidence"]["grid_samples"] = float(
            source.envelope.evidence.grid_samples
        )
    elif change == "topology-float":
        content["wrapping_evidence"]["repaired_topology"]["vertices"] = float(
            source.envelope.evidence.repaired_topology.vertices
        )
    provenance = SemanticProvenance(content)
    if change == "tuple-extra":
        encoded = json.loads(provenance.content_json)
        payload = dict(encoded["items"])
        evidence = dict(payload["wrapping_evidence"]["items"])
        evidence["lost_selection_ids"]["extra"] = True
        content_id = canonical_fingerprint(
            {"kind": "semantic-content", "payload": encoded}
        )
        provenance = _changed_stored_fields(
            provenance,
            content_json=canonical_json(encoded),
            content_id=content_id,
            semantic_id=canonical_fingerprint(
                {
                    "kind": "semantic-provenance",
                    "content_id": content_id,
                    "resource_ids": [list(record) for record in provenance.resource_ids],
                }
            ),
        )
    SemanticProvenance.from_stored_fields(provenance)
    changed = eqx.tree_at(lambda value: value.provenance, initial, provenance)
    with pytest.raises(ValueError, match=r"repair_provenance.representation"):
        _carry_repair_with_request_budget(
            changed,
            envelope_adaptation.target,
            limits=envelope_publication.specification.limits,
        )
    with pytest.raises(RuntimeError, match="envelope.acceptance.repair_provenance"):
        validate_envelope_acceptance(
            source,
            changed,
            limits=envelope_publication.specification.limits,
            failure_factory=envelope_failure,
            measure_cells=tetrahedron_measures,
        )


@pytest.mark.parametrize("resource", ("work", "bytes"))
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_provenance_refuses_before_unadmitted_parsing(
    envelope_publication: _EnvelopePublication,
    resource: str,
) -> None:
    initial = envelope_publication.result
    provenance = SemanticProvenance(
        {"kind": "unrelated-native-source", "padding": "x" * 4096}
    )
    # Parsing would fail before owner dispatch. Resource admission must precede
    # either owner parse, rather than surfacing this JSON error.
    provenance = _changed_stored_fields(
        provenance, content_json=provenance.content_json + "["
    )
    changed = eqx.tree_at(lambda value: value.provenance, initial, provenance)
    work = 100 if resource == "work" else 20_000_000
    storage = 1024 if resource == "bytes" else 128 * 1024**2
    with HostBatchTicket(work, bytes_upper=storage) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=storage
        )
        with pytest.raises(CoordinateEnclosureResourceError) as error:
            with budget.measurement():
                with _surface_envelope_repair_payload(changed, budget=budget):
                    pass
        assert error.value.phase_evidence is not None
        assert error.value.phase_evidence.work_units == budget.work_units
        assert error.value.requested > error.value.limit
        assert budget.temporary_bytes_upper == 0


@pytest.mark.parametrize("resource", ("work", "bytes"))
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_preparation_governs_provenance_before_owner_dispatch(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
    resource: str,
) -> None:
    initial = envelope_publication.result
    changed = eqx.tree_at(
        lambda value: value.provenance,
        initial,
        SemanticProvenance({"kind": "unrelated-native-source", "padding": "x" * 4096}),
    )
    values = {
        member.name: getattr(envelope_publication.specification.limits, member.name)
        for member in dataclass_fields(envelope_publication.specification.limits)
        if member.name != "limits_id"
    }
    values["maximum_work_units" if resource == "work" else "maximum_scratch_bytes"] = (
        100 if resource == "work" else 1024
    )
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.NATIVE_MIXED,
        limits=M.MeshingLimits(**values),
        association_transfer=envelope_adaptation.policy.association_transfer,
        audit_policy=envelope_adaptation.policy.audit_policy,
    )
    with pytest.raises(CoordinateEnclosureResourceError):
        M.prepare_mesh_adaptation(changed, envelope_adaptation.request, policy=policy)


@_REPAIR_AUTHENTICATION_CASE
def test_envelope_acceptance_preserves_authentication_resource_refusal(
    envelope_publication: _EnvelopePublication,
) -> None:
    initial = envelope_publication.result
    values = {
        member.name: getattr(envelope_publication.specification.limits, member.name)
        for member in dataclass_fields(envelope_publication.specification.limits)
        if member.name != "limits_id"
    }
    values["maximum_work_units"] = 100
    validation_limits = M.MeshingLimits(**values)
    with pytest.raises(CoordinateEnclosureResourceError) as error:
        validate_envelope_acceptance(
            envelope_publication.source,
            initial,
            limits=validation_limits,
            failure_factory=envelope_failure,
            measure_cells=tetrahedron_measures,
        )
    assert error.value.phase_evidence is not None
    assert error.value.requested > error.value.limit


@_REPAIR_AUTHENTICATION_CASE
def test_non_envelope_provenance_authentication_releases_decoded_storage(
    envelope_publication: _EnvelopePublication,
) -> None:
    initial = envelope_publication.result
    changed = eqx.tree_at(
        lambda value: (value.provenance, value.compliance),
        initial,
        (
            SemanticProvenance(
                {"kind": "unrelated-native-source", "padding": "x" * 4096}
            ),
            M.MeshingComplianceReport(initial.compliance.specification_id),
        ),
    )
    with HostBatchTicket(20_000_000, bytes_upper=128 * 1024**2) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=20_000_000,
            maximum_memory_bytes=128 * 1024**2,
        )
        with _surface_envelope_repair_payload(changed, budget=budget) as payload:
            assert payload is None
            assert budget.temporary_bytes_upper > 0
        assert budget.temporary_bytes_upper == 0
        assert budget.work_units > 0


@pytest.mark.parametrize("resource", ("work", "bytes"))
def test_envelope_topology_refuses_before_unadmitted_reconstruction(
    resource: str,
) -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    # One unit per visited vertex, face and edge record: 4 + 4 * 4 = 20.
    work = 19 if resource == "work" else 100_000
    storage = 1024 if resource == "bytes" else 4 * 1024**2
    with HostBatchTicket(work, bytes_upper=storage) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=storage
        )
        with pytest.raises(CoordinateEnclosureResourceError) as error:
            with budget.measurement():
                _topology(points, faces, budget=budget)
        assert error.value.phase_evidence is not None
        assert error.value.phase_evidence.work_units == 0
        assert budget.work_units == 0
        assert budget.temporary_bytes_upper == 0


def test_envelope_topology_governed_reconstruction_preserves_exact_facts() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    with HostBatchTicket(100_000, bytes_upper=4 * 1024**2) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=4 * 1024**2
        )
        topology = _topology(points, faces, budget=budget)
        assert topology == _topology(points, faces)
        assert budget.work_units > 0
        assert budget.peak_bytes_upper > 0
        assert budget.temporary_bytes_upper == 0


@pytest.mark.parametrize(
    "change",
    (
        "limit-fields",
        "limit-object",
        "request-id",
        "report-id",
        "embedding-limit",
        "coverage-limit",
    ),
)
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_acceptance_authenticates_request_limits_and_bindings(
    envelope_publication: _EnvelopePublication,
    change: str,
) -> None:
    initial = envelope_publication.result
    report = initial.certification
    assert (
        report is not None
        and report.embedding is not None
        and report.coverage is not None
    )
    if change in ("limit-fields", "limit-object"):
        original = report.request.limits
        limits = (
            _changed_stored_fields(
                original, maximum_work_units=2 * original.maximum_work_units
            )
            if change == "limit-fields"
            else replace(original, maximum_work_units=2 * original.maximum_work_units)
        )
        report = eqx.tree_at(lambda value: value.request.limits, report, limits)
    elif change == "request-id":
        request = _changed_stored_fields(report.request, request_id="foreign-request")
        report = eqx.tree_at(lambda value: value.request, report, request)
    elif change == "report-id":
        report = _changed_stored_fields(report, report_id="foreign-report")
    elif change == "embedding-limit":
        binding = _changed_stored_fields(
            report.embedding.binding, limits_id="foreign-limits"
        )
        embedding = eqx.tree_at(lambda value: value.binding, report.embedding, binding)
        report = eqx.tree_at(lambda value: value.embedding, report, embedding)
    else:
        binding = _changed_stored_fields(
            report.coverage.binding, limits_id="foreign-limits"
        )
        coverage = eqx.tree_at(lambda value: value.binding, report.coverage, binding)
        report = eqx.tree_at(lambda value: value.coverage, report, coverage)
    changed = eqx.tree_at(lambda value: value.certification, initial, report)
    with pytest.raises(RuntimeError, match="envelope.acceptance.repair_provenance"):
        validate_envelope_acceptance(
            envelope_publication.source,
            changed,
            limits=envelope_publication.specification.limits,
            failure_factory=envelope_failure,
            measure_cells=tetrahedron_measures,
        )


@pytest.mark.parametrize("record", ("domain", "embedding", "coverage"))
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_reconstruction_admits_size_before_canonical_owner_replay(
    envelope_publication: _EnvelopePublication,
    record: str,
) -> None:
    initial = envelope_publication.result
    report = initial.certification
    assert (
        report is not None
        and report.embedding is not None
        and report.coverage is not None
    )
    if record == "domain":
        domain = _changed_stored_fields(
            report.request.domain, vertices=np.zeros((50_000, 3), dtype=np.float64)
        )
        report = eqx.tree_at(lambda value: value.request.domain, report, domain)
    elif record == "embedding":
        embedding = _changed_stored_fields(
            report.embedding, certificate_id="foreign-certificate-" * 50_000
        )
        report = eqx.tree_at(lambda value: value.embedding, report, embedding)
    else:
        coverage = _changed_stored_fields(
            report.coverage,
            premise_certificate_ids=tuple(
                f"foreign-premise-{index}" for index in range(50_000)
            ),
        )
        report = eqx.tree_at(lambda value: value.coverage, report, coverage)
    changed = eqx.tree_at(lambda value: value.certification, initial, report)
    with HostBatchTicket(20_000_000, bytes_upper=1024**2) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=20_000_000, maximum_memory_bytes=1024**2
        )
        with pytest.raises(CoordinateEnclosureResourceError) as error:
            with budget.measurement():
                _certified_surface_envelope_domain(changed, budget=budget)
        assert error.value.phase_evidence is not None
        assert error.value.requested > error.value.limit
        assert budget.temporary_bytes_upper == 0


@pytest.mark.parametrize("section", ("requested", "achieved"))
@pytest.mark.parametrize(
    "change", ("duplicate", "swapped", "reordered", "missing", "wrong-section")
)
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_metrics_require_ordered_reserved_inventories(
    envelope_publication: _EnvelopePublication,
    section: str,
    change: str,
) -> None:
    initial = envelope_publication.result
    envelope = envelope_publication.source.envelope
    inventories = {
        "requested": list(initial.compliance.requested),
        "achieved": list(initial.compliance.achieved),
    }
    reserved = {
        "envelope:maximum_deviation",
        "envelope:offset",
        "envelope:minimum_inclusion_margin",
        *(name for name, _ in _surface_envelope_achieved(envelope.evidence)),
    }
    records = inventories[section]
    positions = [index for index, (name, _) in enumerate(records) if name in reserved]
    assert len(positions) >= 3
    first, second = positions[:2]
    if change == "duplicate":
        records.insert(first + 1, records[first])
    elif change == "swapped":
        records[first], records[second] = records[second], records[first]
    elif change == "reordered":
        reversed_records = [records[index] for index in reversed(positions)]
        for index, record in zip(positions, reversed_records, strict=True):
            records[index] = record
    elif change == "missing":
        records.pop(first)
    else:
        other = "achieved" if section == "requested" else "requested"
        # Keep the proper section populated so only wrong-section placement is
        # responsible for refusal, rather than an incidental missing metric.
        inventories[other].append(records[first])
    compliance = M.MeshingComplianceReport(
        initial.compliance.specification_id,
        issues=initial.compliance.issues,
        requested=tuple(inventories["requested"]),
        achieved=tuple(inventories["achieved"]),
    )
    changed = eqx.tree_at(lambda value: value.compliance, initial, compliance)
    with pytest.raises(ValueError, match=r"repair_provenance.metrics"):
        _require_surface_envelope_repair_metrics(
            changed, envelope.evidence, envelope.policy
        )


@pytest.mark.parametrize(
    "metric",
    (
        "envelope:maximum_deviation",
        "envelope:offset",
        "envelope:minimum_inclusion_margin",
    ),
)
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_metrics_authenticate_requested_values(
    envelope_publication: _EnvelopePublication,
    metric: str,
) -> None:
    initial = envelope_publication.result
    envelope = envelope_publication.source.envelope
    compliance = M.MeshingComplianceReport(
        initial.compliance.specification_id,
        issues=initial.compliance.issues,
        requested=tuple(
            (name, math.nextafter(value, math.inf) if name == metric else value)
            for name, value in initial.compliance.requested
        ),
        achieved=initial.compliance.achieved,
    )
    changed = eqx.tree_at(lambda value: value.compliance, initial, compliance)
    with pytest.raises(ValueError, match=r"repair_provenance.metrics"):
        _require_surface_envelope_repair_metrics(
            changed, envelope.evidence, envelope.policy
        )


@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_metrics_preserve_unrelated_records(
    envelope_publication: _EnvelopePublication,
) -> None:
    initial = envelope_publication.result
    envelope = envelope_publication.source.envelope
    compliance = M.MeshingComplianceReport(
        initial.compliance.specification_id,
        issues=initial.compliance.issues,
        requested=(
            ("unrelated:requested", 0.125),
            initial.compliance.requested[0],
            ("envelope:unrelated_requested", 0.25),
            *initial.compliance.requested[1:],
        ),
        achieved=(
            ("unrelated:achieved", 0.5),
            initial.compliance.achieved[0],
            ("envelope:unrelated_achieved", 1.0),
            *initial.compliance.achieved[1:],
        ),
    )
    changed = eqx.tree_at(lambda value: value.compliance, initial, compliance)
    _require_surface_envelope_repair_metrics(changed, envelope.evidence, envelope.policy)
    assert changed.compliance.requested == compliance.requested
    assert changed.compliance.achieved == compliance.achieved


@pytest.mark.parametrize(
    "change",
    (
        "negative-grid",
        "negative-work",
        "negative-features",
        "negative-carrier",
        "joint-samples",
        "excess-work",
        "excess-vertices",
        "excess-tetrahedra",
        "excess-source-features",
        "selection-kind",
        "interface-empty",
        "tag-kind",
        "tag-empty",
        "tag-count",
    ),
)
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_decoder_requires_native_counter_and_history_semantics(
    envelope_publication: _EnvelopePublication,
    change: str,
) -> None:
    source, initial = envelope_publication.source, envelope_publication.result
    policy = source.envelope.policy
    content = _repair_provenance_content(source, initial)
    fields = content["wrapping_evidence"]
    if change == "negative-grid":
        fields["grid_samples"] = -1
    elif change == "negative-work":
        fields["work_units"] = -1
    elif change == "negative-features":
        fields["rounded_feature_edges"] = -1
    elif change == "negative-carrier":
        fields["carrier_vertices"] = -1
    elif change == "joint-samples":
        fields["grid_samples"] = policy.max_samples
    elif change == "excess-work":
        fields["work_units"] = policy.max_work_units + 1
    elif change == "excess-vertices":
        fields["carrier_vertices"] = policy.max_volume_vertices + 1
    elif change == "excess-tetrahedra":
        fields["carrier_tetrahedra"] = policy.max_tetrahedra + 1
    elif change == "excess-source-features":
        fields["rounded_feature_edges"] = (
            source.envelope.evidence.source_topology.edges + 1
        )
    elif change == "selection-kind":
        fields["lost_selection_ids"] = (17,)
    elif change == "interface-empty":
        fields["lost_interface_ids"] = ("",)
    elif change == "tag-kind":
        fields["discarded_cell_tags"] = (False,)
    elif change == "tag-count":
        fields["discarded_cell_tags"] = ("material",) * (
            source.envelope.evidence.source_topology.faces + 1
        )
    else:
        fields["discarded_cell_tags"] = ("",)
    encoded = dict(json.loads(SemanticProvenance(content).content_json)["items"])
    with pytest.raises((TypeError, ValueError)):
        _decode_surface_envelope_repair(encoded)


@_REPAIR_AUTHENTICATION_CASE
def test_envelope_counter_rules_preserve_distinct_policy_axes_and_tag_data(
    envelope_publication: _EnvelopePublication,
) -> None:
    source = envelope_publication.source
    evidence, policy = source.envelope.evidence, source.envelope.policy
    fields = {
        member.name: getattr(evidence, member.name)
        for member in dataclass_fields(evidence)
    }
    # Feature permission counts represented input edges; tag values are not
    # normalized identifiers, but a nonempty bank is face-aligned.
    fields["rounded_feature_edges"] = evidence.source_topology.edges
    fields["discarded_cell_tags"] = (" material ",) * evidence.source_topology.faces
    _validate_surface_envelope_evidence_fields(fields, policy)


@pytest.mark.parametrize(
    "change",
    (
        "feature-selection-history",
        "feature-interface-history",
        "preserve-features",
        "preserve-topology",
        "zero-source-bound",
        "zero-interpolation-error",
        "unrounded-repaired-bound",
        "unrounded-inclusion-margin",
        "implausibly-small-interpolation-error",
        "grid-only-interpolation-error",
        "below-native-source-floor",
    ),
)
@_REPAIR_AUTHENTICATION_CASE
def test_envelope_decoder_refuses_impossible_native_source_and_bounds(
    envelope_publication: _EnvelopePublication,
    change: str,
) -> None:
    source, initial = envelope_publication.source, envelope_publication.result
    content = _repair_provenance_content(source, initial)
    fields, policy = content["wrapping_evidence"], source.envelope.policy
    if change == "feature-selection-history":
        fields["rounded_feature_edges"] = 1
        fields["lost_selection_ids"] = ("selection",)
    elif change == "feature-interface-history":
        fields["rounded_feature_edges"] = 1
        fields["lost_interface_ids"] = ("interface",)
    elif change in ("preserve-features", "preserve-topology"):
        policy_fields = {
            member.name: getattr(policy, member.name)
            for member in dataclass_fields(policy)
            if member.name != "policy_id"
        }
        if change == "preserve-features":
            policy_fields["feature_permission"] = EnvelopeFeaturePermission.PRESERVE
        else:
            policy_fields["topology_permission"] = EnvelopeTopologyPermission.PRESERVE
        policy = SurfaceEnvelopePolicy(**policy_fields)
    elif change == "zero-source-bound":
        fields["source_to_repaired_upper"] = 0.0
    elif change == "zero-interpolation-error":
        fields["interpolation_error"] = 0.0
    elif change in (
        "implausibly-small-interpolation-error",
        "grid-only-interpolation-error",
    ):
        fields["interpolation_error"] = (
            policy.spacing / 2
            if change == "implausibly-small-interpolation-error"
            else math.nextafter(
                math.nextafter(math.sqrt(3.0), math.inf) * policy.spacing, math.inf
            )
        )
        # Keep both defining equations unchanged: only the native positive error
        # floor, not a stale equation or evidence identity, may reject this.
        fields["repaired_to_source_upper"] = math.nextafter(
            policy.offset + fields["interpolation_error"], math.inf
        )
        fields["source_inclusion_margin"] = math.nextafter(
            policy.offset - fields["interpolation_error"], -math.inf
        )
    elif change == "below-native-source-floor":
        fields["source_to_repaired_upper"] = math.nextafter(0.0, math.inf)
    elif change == "unrounded-repaired-bound":
        fields["repaired_to_source_upper"] = policy.offset + fields["interpolation_error"]
    else:
        fields["source_inclusion_margin"] = policy.offset - fields["interpolation_error"]
    fields["policy_id"] = policy.policy_id
    content["wrapping_policy"] = _surface_envelope_policy_payload(policy)
    evidence = replace(
        source.envelope.evidence,
        **{
            name: value
            for name, value in fields.items()
            if name not in ("source_topology", "repaired_topology")
        },
    )
    fields["evidence_id"] = _surface_envelope_evidence_id(
        evidence, source.envelope.repaired.mesh.coordinates.shape[0]
    )
    encoded = dict(json.loads(SemanticProvenance(content).content_json)["items"])
    expected = (
        r"repair_provenance.bounds"
        if change
        in (
            "implausibly-small-interpolation-error",
            "grid-only-interpolation-error",
            "below-native-source-floor",
        )
        else None
    )
    with pytest.raises(ValueError, match=expected):
        _decode_surface_envelope_repair(encoded)


@_REPAIR_AUTHENTICATION_CASE
def test_envelope_repair_authentication_refuses_unused_domain_vertices(
    envelope_publication: _EnvelopePublication,
) -> None:
    publication = envelope_publication
    report = publication.result.certification
    assert report is not None and isinstance(report.request.domain, PiecewiseLinearDomain)
    original = report.request.domain
    domain = PiecewiseLinearDomain(
        np.concatenate((original.vertices, original.vertices[:1])),
        original.facets,
        original.facet_regions,
        original.region_ids,
        source_id=original.source_id,
    )
    evidence = publication.source.envelope.evidence
    evidence = replace(
        evidence,
        evidence_id=_surface_envelope_evidence_id(evidence, domain.vertices.shape[0]),
    )
    limits = publication.specification.limits
    with HostBatchTicket(
        limits.maximum_work_units, bytes_upper=limits.maximum_scratch_bytes
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with pytest.raises(ValueError, match="repair_provenance.evidence"):
            _authenticate_surface_envelope_repair(
                evidence, publication.source.envelope.policy, domain, budget=budget
            )


@_REPAIR_AUTHENTICATION_CASE
def test_envelope_carry_refuses_rehashed_counter_with_matching_trace_and_metrics(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
) -> None:
    source, initial = envelope_publication.source, envelope_publication.result
    evidence = replace(source.envelope.evidence, grid_samples=-1)
    evidence = replace(
        evidence,
        evidence_id=_surface_envelope_evidence_id(
            evidence, source.envelope.repaired.mesh.coordinates.shape[0]
        ),
    )
    content = _repair_provenance_content(source, initial)
    content["wrapping_evidence"] = asdict(evidence)
    changed = eqx.tree_at(
        lambda value: (value.provenance, value.compliance, value.trace),
        initial,
        (
            SemanticProvenance(content),
            M.MeshingComplianceReport(
                initial.compliance.specification_id,
                requested=initial.compliance.requested,
                achieved=_surface_envelope_achieved(evidence),
            ),
            M.MeshingTrace(
                (
                    _surface_envelope_repair_stage(
                        evidence, source.source_id, source.source_revision
                    ),
                    *initial.trace.stages[1:],
                ),
                binding=initial.trace.binding,
            ),
        ),
    )
    with pytest.raises(ValueError, match="repair_provenance.grid_samples"):
        _carry_repair_with_request_budget(
            changed,
            envelope_adaptation.target,
            limits=envelope_publication.specification.limits,
        )


def test_envelope_adaptation_transfers_fields_and_rolls_back_rejection(
    envelope_publication: _EnvelopePublication,
    envelope_adaptation: M.MeshAdaptationResult,
) -> None:
    result = envelope_publication.result
    space, problem, _ = prepare_envelope_solver(
        result,
        failure_factory=envelope_failure,
        source_coefficient=quadratic_source,
        dirichlet_values=quadratic_value,
    )
    values = space.dof_maps[0].dof_coordinates[:, 0]
    # Linear x cannot equal |x|^2 on a positive-volume simplex; the analytic L2
    # moment oracle establishes a nonzero error before testing the zero gate.
    assert quadratic_l2_error(result.mesh, values) > 0.0
    _, auxiliary, transfer = transition_envelope_state(
        result,
        envelope_adaptation,
        problem,
        values,
        0,
        failure_factory=envelope_failure,
        physical_error=quadratic_l2_error,
    )
    assert transfer["field_names"] == ["u", "density", "history"]
    rollback = transfer["rollback"]
    if not isinstance(rollback, dict):
        raise TypeError(
            "transfer.rollback must be a dictionary record; got a non-dictionary value."
        )
    assert rollback["passed"]
    assert np.all(np.asarray(auxiliary[0]) == 2.0)
    history_defect = transfer["history_max_defect"]
    if not isinstance(history_defect, (int, float)):
        raise TypeError(
            "transfer.history_max_defect must be numeric; got a nonnumeric value."
        )
    assert history_defect <= _AFFINE_FIELD_ACCEPTANCE


@pytest.mark.parametrize("bound", ("work", "queries", "bytes", "clock"))
def test_original_source_preparation_receipt_refuses_tighter_volume_request_without_mutation(
    bound: str,
) -> None:
    from dataclasses import fields as dataclass_fields

    from phydrax._fingerprint import array_tree_fingerprint

    soup = envelope_sheet()
    envelope = wrap_surface_envelope(soup, envelope_policy())
    source = NativeSurfaceEnvelopeSource(envelope, "wrapped-solid")
    assert envelope.execution_evidence is not None
    before = array_tree_fingerprint(source)
    controls = {
        "work": {"maximum_work_units": 1},
        "queries": {"maximum_geometry_queries": 1},
        "bytes": {"maximum_scratch_bytes": 1},
        "clock": {"maximum_wall_seconds": 1.0e-12},
    }
    original = envelope_specification(source)
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
    with (
        pytest.raises(M.MeshingFailure) as error,
        native_volume_execution_budget(specification.limits) as scope,
        scope.host_workspace() as workspace,
    ):
        execute_surface_envelope_route(
            source,
            specification,
            prepared,
            _SI,
            NativeMeshingProvider.info(),
            f"envelope-original-source-{bound}-refusal",
            scope=scope,
            workspace=workspace,
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


@pytest.mark.parametrize("region_id", (17, None), ids=("integer", "none"))
def test_envelope_source_region_identity_rejects_nonstring_kind(
    region_id: int | None,
) -> None:
    envelope = wrap_surface_envelope(envelope_sheet(), envelope_policy())
    with pytest.raises(TypeError, match="NativeSurfaceEnvelopeSource.region_id"):
        NativeSurfaceEnvelopeSource(envelope, region_id)  # ty: ignore[invalid-argument-type]


def test_envelope_source_region_identity_rejects_empty_string_value() -> None:
    envelope = wrap_surface_envelope(envelope_sheet(), envelope_policy())
    with pytest.raises(ValueError, match="NativeSurfaceEnvelopeSource.region_id"):
        NativeSurfaceEnvelopeSource(envelope, "")


def test_retained_certificate_sequence_admission_is_linear_in_scalar_ids() -> None:
    identifiers = tuple(range(4096))
    with HostBatchTicket(100_000, bytes_upper=8 * 1024**2) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=8 * 1024**2
        )
        with budget.measurement(), budget.temporary_scope():
            _admit_surface_envelope_record(identifiers, budget=budget)
            assert budget.work_units <= 100_000
            assert budget.temporary_bytes_upper > 0
        assert budget.temporary_bytes_upper == 0


def test_retained_record_admission_reuses_shared_text_not_expanded_buffers() -> None:
    text = "physical"
    records = (text,) * 1024
    with HostBatchTicket(20_000, bytes_upper=2 * 1024**2) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=20_000, maximum_memory_bytes=2 * 1024**2
        )
        with budget.measurement(), budget.temporary_scope():
            _admit_surface_envelope_record(records, budget=budget)
            assert budget.work_units < 20_000
            assert budget.temporary_bytes_upper >= 24 * len(text) * len(records)
        assert budget.temporary_bytes_upper == 0


def test_retained_record_shared_text_still_admits_every_serialized_occurrence() -> None:
    records = ("x" * 2048,) * 1024
    with HostBatchTicket(100_000, bytes_upper=4 * 1024**2) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=4 * 1024**2
        )
        with pytest.raises(CoordinateEnclosureResourceError) as error:
            with budget.measurement(), budget.temporary_scope():
                _admit_surface_envelope_record(records, budget=budget)
        assert error.value.resource == "polynomial_storage"
        assert error.value.requested > error.value.limit
        assert budget.work_units < 100_000
        assert budget.temporary_bytes_upper == 0


@_REPAIR_AUTHENTICATION_CASE
def test_envelope_geometry_replay_holds_metadata_and_maximum_bank_buffer(
    envelope_publication: _EnvelopePublication,
) -> None:
    result = envelope_publication.result
    geometry = result.geometry
    assert geometry.storage is None
    banks = (geometry.elements, geometry.geometry_dofs, geometry.coordinates)
    numeric_buffers = tuple(
        8
        * sum(
            max(int(leaf.nbytes), 8 * int(leaf.size))
            for leaf in tree_util.tree_leaves(bank)
            if isinstance(leaf, (Array, np.ndarray))
        )
        for bank in banks
    )
    limits = envelope_publication.specification.limits
    with HostBatchTicket(
        limits.maximum_work_units, bytes_upper=limits.maximum_scratch_bytes
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with budget.measurement(), budget.temporary_scope():
            budget.reserve(1, 64)
            prefix = budget.temporary_bytes_upper
            with _admit_surface_envelope_geometry_replay(result, budget=budget) as bounds:
                metadata, maximum_buffer = bounds
                assert metadata > 0
                assert max(numeric_buffers) <= maximum_buffer < sum(numeric_buffers)
                assert budget.temporary_bytes_upper == prefix + metadata + maximum_buffer
                assert budget.peak_bytes_upper == budget.temporary_bytes_upper
                with budget.temporary_scope():
                    budget.reserve(1, 32)
                    assert budget.temporary_bytes_upper == (
                        prefix + metadata + maximum_buffer + 32
                    )
                    assert geometry.coordinates.shape[0] > 0
                assert budget.temporary_bytes_upper == prefix + metadata + maximum_buffer
            assert budget.temporary_bytes_upper == prefix
        assert budget.temporary_bytes_upper == 0
