from dataclasses import fields, replace
from pathlib import Path
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    cad_mesh_solid,
    CadSolidResult,
    SurfaceFindingKind,
    VerdictStatus,
    verify_chart_partition,
    verify_surface_embedding,
    verify_tetrahedra,
    verify_volume,
)
from phydrax._strict import reconstruct_strict, register_strict_reconstruction
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry._surface_source_support import SurfaceSourceCharts
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._projection_contracts import brep_entity_id
from phydrax.geometry.multiregion_surface import MultiRegionSurfaceTopology
from phydrax.meshing._association import GeometrySourceEntityRole
from phydrax.meshing._certification import (
    _audit_findings,
    _multiregion_proof,
    MeshCertificationReport,
)
from phydrax.meshing._model_codec import (
    model_archive_content_id,
    MODEL_ARCHIVE_LIMITS,
    read_model_artifact,
    restore_model_archive_value,
    write_model_artifact,
)
from phydrax.meshing._native_cad_certificate import NativeCadSurfaceWitness
from phydrax.meshing._native_cad_records import (
    _native_cad_regional_surface,
    _RegionalSurfaceRows,
    native_cad_records,
)
from phydrax.meshing._native_construction import NativeMeshOrganization
from phydrax.meshing._plc_source import (
    _require_unambiguous_cad_copies,
    PiecewiseLinearComplex,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing.providers._native_cad_bundle import cad_geometry_bundle
from phydrax.meshing.providers._native_cad_volume import _cad_plc_data
from phydrax.meshing.providers._native_plc_geometry import simplex_volume_entities
from phydrax.meshing.providers._native_volume import _face_order_sign


@pytest.mark.parametrize(
    ("vertices", "expected"),
    (
        ((3, 5, 7), 1),
        ((5, 7, 3), 1),
        ((7, 3, 5), 1),
        ((3, 7, 5), -1),
        ((7, 5, 3), -1),
        ((5, 3, 7), -1),
    ),
)
def test_recovered_face_orientation_transports_exact_vertex_parity(
    vertices: tuple[int, int, int],
    expected: int,
) -> None:
    assert _face_order_sign((3, 5, 7), vertices) == expected


def test_recovered_face_orientation_rejects_other_vertex_identity() -> None:
    with pytest.raises(ValueError, match="vertex identities"):
        _face_order_sign((3, 5, 7), (3, 5, 8))


@pytest.fixture(scope="module")
def split_box_model() -> BRepModel:
    pytest.importorskip("OCP")
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    from phydrax._physical import SpatialCoordinateContract
    from phydrax.interchange import import_occt_shape

    splitter = BOPAlgo_Splitter()
    for x in (0.0, 1.0):
        splitter.AddArgument(
            BRepPrimAPI_MakeBox(gp_Pnt(x, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
        )
    splitter.Perform()
    assert not splitter.HasErrors()
    return import_occt_shape(
        splitter.Shape(), coordinate_contract=SpatialCoordinateContract.si()
    ).model


@pytest.fixture(scope="module")
def authored_split_box_model() -> BRepModel:
    from tests.unit.geometry.test_source_shell_admissibility import (
        _partition_region_boxes,
    )

    return _partition_region_boxes(
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
    )


def _split_box_volume_plan(model: BRepModel, /) -> phx.meshing.NativeMeshingPlan:
    boundary = phx.meshing.MeshingScope.whole(model, 2)
    spec = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        boundary,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(boundary, 0.4),),
        maximum_deviation=0.05,
    )
    return phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions(
            "cad_tetrahedral",
            volume_schedule=phx.meshing.NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
                shape_strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        )
    ).plan(model, spec)


class _AuthoredSplitBox(NamedTuple):
    plan: phx.meshing.NativeMeshingPlan
    candidate: phx.meshing.CellMeshingCandidate
    result: CellMeshingResult


@pytest.fixture(scope="module")
def authored_split_box_volume(
    authored_split_box_model: BRepModel,
) -> _AuthoredSplitBox:
    plan = _split_box_volume_plan(authored_split_box_model)
    candidate = plan.execute(prove=False)
    return _AuthoredSplitBox(plan, candidate, candidate.prove())


def _assert_split_box_regions(result: CellMeshingResult, /) -> None:
    report = result.certification
    assert report is not None and report.passed
    inputs = report.inputs
    domain, regions = inputs.domain, inputs.cell_regions
    assert isinstance(domain, MeshingDomain) and regions is not None
    model = domain.brep_authority
    assert isinstance(model, BRepModel)
    np.testing.assert_array_equal(np.unique(regions), [0, 1])
    (source_face,) = (
        face for face, owners in enumerate(model.topology.face_solids) if len(owners) == 2
    )
    (patch_row,) = (
        row for row, face in enumerate(domain.source_indices[2]) if face == source_face
    )
    face_path = tuple(domain.source_occurrences[2][patch_row])
    face_id = brep_entity_id(
        domain.source_id,
        domain.source_revision,
        2,
        source_face,
        occurrence_path=face_path,
    )
    region_ids = tuple(domain.entity_id(3, row) for row in range(2))
    region_paths = tuple(tuple(path) for path in domain.region_source_occurrences)
    signs = tuple(
        model.topology.solid_face_orientations[solid][
            model.topology.solid_faces[solid].index(source_face)
        ]
        for solid in domain.region_source_indices
    )
    assert set(signs) == {-1, 1}
    # The organization and the BRep associations both use (positive, negative).
    side_order = (signs.index(-1), signs.index(1))
    sides = (region_ids[side_order[0]], region_ids[side_order[1]])
    paths = (region_paths[side_order[0]], region_paths[side_order[1]])
    mesh = result.mesh
    _, faces = simplex_volume_entities(mesh)
    interface = np.asarray(mesh.entity_set(2).entity_ids)[
        np.all(np.asarray(mesh.coordinates)[faces, 0] == 1.0, axis=1)
    ]
    assert interface.size > 1
    (label,) = (item for item in result.labels if item.name == "material_interface")
    np.testing.assert_array_equal(label.scope.entity_ids, interface)
    (patch,) = (
        item for item in result.patches if item.source_adjacent_region_ids == sides
    )
    np.testing.assert_array_equal(patch.scope.entity_ids, interface)
    assert patch.source_adjacent_region_occurrence_paths == paths
    assert set(patch.adjacent_zone_ids) == {zone.zone_id for zone in result.zones}

    records = {
        dimension: next(
            item
            for item in result.associations
            if item.association_kind is phx.meshing.GeometryAssociationKind.BREP
            and item.target_entity_set_id == mesh.entity_set(dimension).entity_set_id
        )
        for dimension in (2, 3)
    }
    facet = records[2]
    face_rows = np.flatnonzero(np.isin(facet.target_global_ids, interface))
    assert face_rows.size == interface.size
    np.testing.assert_array_equal(facet.source_dimensions[face_rows], 2)
    np.testing.assert_array_equal(facet.source_indices[face_rows], source_face)
    assert {facet.source_entity_ids[row] for row in face_rows} == {face_id}
    assert {facet.source_occurrence_paths[row] for row in face_rows} == {face_path}
    assert {facet.source_region_sides[row] for row in face_rows} == {sides}

    cells = records[3]
    np.testing.assert_array_equal(cells.target_global_ids, inputs.cell_global_ids)
    np.testing.assert_array_equal(cells.source_dimensions, 3)
    assert set(cells.source_roles) == {GeometrySourceEntityRole.REGION}
    for row, (region_id, path, solid) in enumerate(
        zip(region_ids, region_paths, domain.region_source_indices, strict=True)
    ):
        cell_rows = np.flatnonzero(regions == row)
        solid_id = brep_entity_id(
            domain.source_id, domain.source_revision, 3, solid, occurrence_path=path
        )
        np.testing.assert_array_equal(cells.source_indices[cell_rows], solid)
        assert {cells.source_entity_ids[cell] for cell in cell_rows} == {solid_id}
        assert {cells.source_occurrence_paths[cell] for cell in cell_rows} == {path}
        (zone,) = (item for item in result.zones if item.name == region_id)
        np.testing.assert_array_equal(
            zone.scope.entity_ids, inputs.cell_global_ids[cell_rows]
        )
        (boundary,) = (
            evidence
            for evidence in result.region_boundary_evidence
            if evidence.source_region_id == region_id
        )
        assert boundary.source_occurrence_path == path
        assert dict(boundary.patch_sides)[patch.patch_id] == -signs[row]
        assert np.all(np.isin(interface, boundary.boundary_label.scope.entity_ids))


def test_authored_regional_volume_deferred_and_eager_agree(
    authored_split_box_volume: _AuthoredSplitBox,
) -> None:
    plan, candidate, deferred = authored_split_box_volume
    assert isinstance(candidate, phx.meshing.CellMeshingCandidate)
    assert candidate.route == "cad_tetrahedral"
    request = candidate._construction.certification
    assert deferred.mesh is candidate.mesh and deferred.geometry is candidate.geometry
    report = deferred.certification
    assert report is not None and report.passed
    native, records = report.native_cad, report.inputs.native_cad_records
    assert native is not None and records is not None and native.source_shell is not None
    assert native.records_id == records.records_id
    assert native.witness.witness_id == records.witness_id
    assert request.native_cad_records is not None
    assert request.native_cad_records.records_id == records.records_id
    assert request.native_cad_surface is not None
    assert request.native_cad_surface.witness_id == native.witness.witness_id
    assert request.cell_regions is not None and report.inputs.cell_regions is not None
    np.testing.assert_array_equal(report.inputs.cell_regions, request.cell_regions)
    assert tuple(item.label_id for item in deferred.labels) == tuple(
        item.label_id for item in candidate._construction.organization.labels
    )
    report.require_current(deferred.mesh, deferred.geometry)
    _assert_split_box_regions(deferred)

    eager = plan.execute(prove=True)
    eager_report = eager.certification
    assert eager_report is not None and eager_report.native_cad is not None
    assert eager.mesh.mesh_id == candidate.mesh.mesh_id
    assert eager.result_id == deferred.result_id
    assert eager_report.report_id == report.report_id
    assert eager_report.native_cad.certificate_id == native.certificate_id
    np.testing.assert_array_equal(eager_report.inputs.cell_regions, request.cell_regions)
    assert tuple(item.label_id for item in eager.labels) == tuple(
        item.label_id for item in deferred.labels
    )


@pytest.mark.parametrize("remove_bounds", (False, True), ids=("same-maximum", "missing"))
def test_authored_regional_volume_binds_every_fidelity_bound(
    authored_split_box_volume: _AuthoredSplitBox, remove_bounds: bool
) -> None:
    _, _, result = authored_split_box_volume
    report = result.certification
    assert report is not None and report.native_cad is not None
    native = report.native_cad
    bounds = native.triangle_deviation
    assert bounds is not None and native.maximum_deviation is not None
    replacement = None
    if not remove_bounds:
        row = np.flatnonzero((bounds > 0.0) & (bounds < native.maximum_deviation))[0]
        replacement = bounds.copy()
        replacement[row] = 0.0
        assert np.max(replacement) == native.maximum_deviation
    corrupted = eqx.tree_at(lambda value: value.triangle_deviation, native, replacement)
    forged = eqx.tree_at(lambda value: value.native_cad, report, corrupted)
    with pytest.raises(ValueError, match="Native CAD certificate evidence differs"):
        forged.require_current(result.mesh, result.geometry)


def test_authored_regional_volume_refuses_changed_certificate_chart(
    authored_split_box_volume: _AuthoredSplitBox,
) -> None:
    _, _, result = authored_split_box_volume
    report = result.certification
    assert report is not None and report.native_cad is not None
    native = report.native_cad
    uv = native.witness.chart_uv.copy()
    uv[0, 0, 0] += 0.125
    altered = eqx.tree_at(lambda value: value.chart_uv, native.witness, uv)
    corrupted = eqx.tree_at(lambda value: value.witness, native, altered)
    forged = eqx.tree_at(lambda value: value.native_cad, report, corrupted)
    assert altered.witness_id == native.witness.witness_id
    retained = forged.inputs.native_cad_surface
    assert retained is not None
    np.testing.assert_array_equal(retained.chart_uv, native.witness.chart_uv)
    with pytest.raises(ValueError, match="Native CAD certificate binds another witness"):
        forged.require_current(result.mesh, result.geometry)


def test_authored_regional_volume_refuses_swapped_currentness(
    authored_split_box_volume: _AuthoredSplitBox,
) -> None:
    _, _, result = authored_split_box_volume
    report = result.certification
    assert report is not None and report.inputs.cell_regions is not None
    swapped = 1 - report.inputs.cell_regions
    swapped.setflags(write=False)
    forged = eqx.tree_at(lambda value: value.inputs.cell_regions, report, swapped)
    assert forged.inputs.request_id == report.inputs.request_id
    assert forged.report_id == report.report_id
    with pytest.raises(ValueError, match="(?i)(region|material)"):
        forged.require_current(result.mesh, result.geometry)


def test_authored_regional_volume_refuses_translated_coordinate_map(
    authored_split_box_volume: _AuthoredSplitBox,
) -> None:
    _, _, result = authored_split_box_volume
    report = result.certification
    assert report is not None
    inputs = report.inputs
    translated = eqx.tree_at(
        lambda value: value.coordinates,
        result.geometry,
        result.geometry.coordinates + 2.0,
    )
    changed = phx.meshing.MeshCertificationInputs(
        result.mesh,
        translated,
        inputs.schedule,
        source_id=inputs.source_id,
        source_revision=inputs.source_revision,
        domain=inputs.domain,
        cell_regions=inputs.cell_regions,
        source=inputs.source,
        fidelity_tolerance=inputs.fidelity_tolerance,
        feature_source=inputs.feature_source,
        native_cad_surface=inputs.native_cad_surface,
        native_cad_records=inputs.native_cad_records,
    )
    acceptance = MeshCertificationReport(
        changed, organization=NativeMeshOrganization(associations=result.associations)
    )
    assert acceptance.embedding.status == "certified"
    assert acceptance.native_cad is not None
    assert acceptance.native_cad.maximum_deviation is None
    with pytest.raises(
        phx.meshing.MeshingFailure,
        match="cad_fidelity: unresolved.*affine tetrahedral map",
    ):
        acceptance.require_passed()


def test_authored_regional_volume_refuses_coherent_material_swap(
    authored_split_box_volume: _AuthoredSplitBox,
) -> None:
    _, _, result = authored_split_box_volume
    report = result.certification
    assert report is not None
    inputs = report.inputs
    domain, regions = inputs.domain, inputs.cell_regions
    witness, tolerance = inputs.native_cad_surface, inputs.fidelity_tolerance
    assert isinstance(domain, MeshingDomain) and regions is not None
    assert witness is not None and tolerance is not None
    swapped = 1 - regions
    # Rebuild the records and request, instead of invalidating a cached ID.
    records, tables = native_cad_records(result.mesh, domain, swapped, witness, tolerance)
    forged = phx.meshing.MeshCertificationInputs(
        result.mesh,
        result.geometry,
        inputs.schedule,
        source_id=inputs.source_id,
        source_revision=inputs.source_revision,
        domain=domain,
        cell_regions=swapped,
        source=inputs.source,
        fidelity_tolerance=tolerance,
        feature_source=inputs.feature_source,
        native_cad_surface=witness,
        native_cad_records=records,
    )
    assert forged.request_id != inputs.request_id

    # A matching outer fingerprint is not enough to reuse the original proof
    # with new, internally consistent associations and dense labels.
    assert report.native_cad is not None
    assert records.records_id != report.native_cad.records_id
    substituted = eqx.tree_at(
        lambda value: (value.inputs, value.associations),
        report,
        (forged, tables),
    )
    register_strict_reconstruction(MeshCertificationReport)
    forged_report = reconstruct_strict(
        MeshCertificationReport,
        {
            **{
                item.name: getattr(substituted, item.name) for item in fields(substituted)
            },
            "report_id": substituted._identity(),
        },
    )
    assert forged_report.report_id != report.report_id
    with pytest.raises(ValueError, match="Native CAD certificate binds another"):
        forged_report.require_passed()
    with pytest.raises(ValueError, match="Native CAD certificate binds another"):
        forged_report.require_current(result.mesh, result.geometry)

    bad_report = MeshCertificationReport(
        forged, organization=NativeMeshOrganization(associations=tables)
    )
    assert bad_report.native_cad is not None
    assert not bad_report.passed
    (finding,) = (
        item for item in bad_report.native_cad.findings if item.check == "cad_volume"
    )
    assert finding.status == "violated" and finding.entity_ids
    with pytest.raises(
        phx.meshing.MeshingFailure, match=r"cad_volume.*[1-9][0-9]* region mismatches"
    ):
        bad_report.require_current(result.mesh, result.geometry)


def test_authored_regional_volume_adaptation_and_archive_replay(
    authored_split_box_volume: _AuthoredSplitBox,
    authored_split_box_model: BRepModel,
    tmp_path: Path,
) -> None:
    _, _, source = authored_split_box_volume
    transfer = phx.meshing.BRepAssociationTransfer(
        phx.geometry.prepare_brep_projection(authored_split_box_model)
    )
    adapted = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(
                np.asarray(source.mesh.blocks[0].global_ids)
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                association_transfer=transfer,
            ),
        )
    )
    target = adapted.target
    assert target.mesh.blocks[0].cell_count > source.mesh.blocks[0].cell_count
    _assert_split_box_regions(target)
    original, refined = source.certification, target.certification
    assert original is not None and refined is not None
    assert original.native_cad is not None and refined.native_cad is not None
    assert original.native_cad.source_shell is not None
    assert refined.native_cad.source_shell is not None
    assert refined.native_cad.source_shell != original.native_cad.source_shell
    assert refined.native_cad.records_id != original.native_cad.records_id
    old_labels, new_labels = original.inputs.cell_regions, refined.inputs.cell_regions
    assert old_labels is not None and new_labels is not None
    old_by_id = dict(
        zip(original.inputs.cell_global_ids.tolist(), old_labels, strict=True)
    )
    new_by_id = dict(
        zip(refined.inputs.cell_global_ids.tolist(), new_labels, strict=True)
    )
    assert adapted.lineage is not None
    relation = adapted.lineage.entity_lineage(3)
    assert set(np.asarray(relation.target_global_ids).tolist()) == set(new_by_id)
    for old, new in zip(
        np.asarray(relation.source_global_ids).tolist(),
        np.asarray(relation.target_global_ids).tolist(),
        strict=True,
    ):
        assert new_by_id[new] == old_by_id[old]

    path = tmp_path / "regional-cad-refined.phx"
    write_model_artifact(path, target, limits=MODEL_ARCHIVE_LIMITS)
    restored = read_model_artifact(
        path,
        expected_type=CellMeshingResult,
        restore=restore_model_archive_value,
        expected_content_id=model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert restored.result_id == target.result_id
    replay = restored.certification
    assert replay is not None and replay.native_cad is not None
    assert replay.native_cad.certificate_id == refined.native_cad.certificate_id
    np.testing.assert_array_equal(replay.inputs.cell_regions, new_labels)
    _assert_split_box_regions(restored)

    # The retained origin labels are scientific operands, not lineage hints.
    inputs = replay.inputs
    records, witness = inputs.native_cad_records, inputs.native_cad_surface
    assert records is not None and records.origin_regions is not None
    assert witness is not None and inputs.fidelity_tolerance is not None
    assert isinstance(inputs.domain, MeshingDomain)
    tampered = eqx.tree_at(
        lambda value: value.origin_regions, records, 1 - records.origin_regions
    )
    with pytest.raises(
        ValueError, match=r"original chart: cad_volume.*region mismatches"
    ):
        type(records).from_stored_fields(
            tampered,
            restored.mesh,
            inputs.domain,
            new_labels,
            witness,
            inputs.fidelity_tolerance,
        )


def test_imported_regional_volume_keeps_chart_plc_authority(
    split_box_model: BRepModel,
) -> None:
    result = _split_box_volume_plan(split_box_model).execute(prove=True)
    report = result.certification
    assert report is not None and report.passed
    assert report.native_cad is None
    assert report.fidelity is not None and report.coverage is not None
    assert result.surface_source is not None
    assert report.inputs.cell_regions is not None
    np.testing.assert_array_equal(np.unique(report.inputs.cell_regions), [0, 1])
    assert any(item.name == "material_interface" for item in result.labels)


@pytest.fixture(scope="module")
def multiregion_surface(
    split_box_model: BRepModel,
) -> tuple[MeshCertificationReport, NativeMeshOrganization, CellMeshingResult]:
    model = split_box_model
    scope = phx.meshing.MeshingScope.whole(model, 2)
    spec = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.5),),
        maximum_deviation=0.05,
    )
    result = phx.meshing.NativeMeshingProvider().plan(model, spec).execute(prove=True)
    report = result.certification
    assert report is not None and report.multiregion_surface is not None
    organization = NativeMeshOrganization(
        boundary=report.boundary,
        patches=report.patches,
        zones=report.zones,
        labels=report.labels,
        attributes=report.attributes,
        associations=report.associations,
        multiregion_surface=report.multiregion_surface,
    )
    return report, organization, result


def test_missing_multiregion_evidence_does_not_excuse_junction(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
) -> None:
    report, organization, _ = multiregion_surface
    assert report.multiregion_evidence is not None
    assert report.multiregion_evidence.accepted
    assert (
        _multiregion_proof(report.inputs, replace(organization, multiregion_surface=None))
        is None
    )
    without = _audit_findings(report.inputs, report.audit, None)
    assert any(finding.check == "nonmanifold_facets" for finding in without)
    qualified = _audit_findings(report.inputs, report.audit, report.multiregion_evidence)
    assert {f.check for f in without} - {
        "nonmanifold_facets",
        "nonmanifold_edges",
        "nonmanifold_vertices",
    } == {f.check for f in qualified}
    assert report.embedding.status == "certified"
    report.revalidate(report.inputs.mesh, report.inputs.geometry)


def test_multiregion_proof_refuses_missing_source_association(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
) -> None:
    report, organization, _ = multiregion_surface
    with pytest.raises(ValueError, match="unique source association"):
        _multiregion_proof(report.inputs, replace(organization, associations=()))


@pytest.mark.parametrize("defect", ("inverted", "open", "wrong-region"))
def test_multiregion_proof_refuses_altered_region_topology(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
    defect: Literal["inverted", "open", "wrong-region"],
) -> None:
    report, organization, _ = multiregion_surface
    original = report.multiregion_surface
    assert original is not None
    faces = original.host_faces().copy()
    labels = original.host_face_labels().copy()
    face_ids = np.asarray(original.face_global_ids[: original.face_count], dtype=np.int64)
    if defect == "inverted":
        faces[0] = faces[0, ::-1]
    elif defect == "open":
        faces, labels, face_ids = faces[:-1], labels[:-1], face_ids[:-1]
    else:
        labels[0] = labels[0, ::-1]
    with pytest.raises(ValueError):
        malformed = MultiRegionSurfaceTopology(
            original.plan,
            faces,
            labels,
            original.region_ids,
            original.region_kinds,
            vertex_count=original.vertex_count,
            vertex_global_ids=np.asarray(
                original.vertex_global_ids[: original.vertex_count]
            ),
            face_global_ids=face_ids,
            epoch=original.epoch,
            domain=original.domain,
        )
        _multiregion_proof(
            report.inputs, replace(organization, multiregion_surface=malformed)
        )


@pytest.fixture(scope="module")
def cad_plc(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
) -> PiecewiseLinearComplex:
    report, _, result = multiregion_surface
    domain = report.inputs.feature_source
    charts = result.surface_source
    assert isinstance(domain, MeshingDomain) and isinstance(charts, SurfaceSourceCharts)
    return _cad_plc_data(domain, result, charts, "conforming", 0.05).complex


def _rebuild_plc(
    plc: PiecewiseLinearComplex,
    /,
    *,
    charts: SurfaceSourceCharts | None,
    vertex_rows: np.ndarray | None,
    segment_edge_ids: np.ndarray | None,
    polygon_cell_ids: np.ndarray | None,
    tolerance: float | None,
    facet_regions: np.ndarray | None = None,
    drop_proof: bool = False,
) -> PiecewiseLinearComplex:
    offsets = np.asarray(plc.polygon_offsets)
    loops = tuple(
        np.asarray(plc.polygon_vertices)[first:last]
        for first, last in zip(offsets[:-1], offsets[1:], strict=True)
    )
    return PiecewiseLinearComplex(
        plc.vertices,
        loops,
        plc.polygon_facets,
        plc.facet_regions if facet_regions is None else facet_regions,
        plc.region_ids,
        segments=plc.segments,
        vertex_ids=plc.vertex_ids,
        vertex_occurrence_paths=plc.vertex_occurrence_paths,
        facet_ids=plc.facet_ids,
        facet_occurrence_paths=plc.facet_occurrence_paths,
        segment_ids=plc.segment_ids,
        segment_occurrence_paths=plc.segment_occurrence_paths,
        region_occurrence_paths=plc.region_occurrence_paths,
        boundary=plc.boundary,
        geometry=plc.geometry,
        facet_planes=plc.facet_planes,
        facet_plane_tolerances=plc.facet_plane_tolerances,
        cad_source_charts=charts,
        cad_fidelity_tolerance=tolerance,
        cad_embedding=plc.cad_embedding if charts is not None else None,
        cad_fidelity=plc.cad_fidelity if charts is not None and not drop_proof else None,
        cad_maximum_normal_deviation=plc.cad_maximum_normal_deviation
        if charts is not None
        else None,
        cad_vertex_rows=vertex_rows,
        cad_segment_edge_ids=segment_edge_ids,
        cad_polygon_cell_ids=polygon_cell_ids,
    )


def test_generic_plc_still_refuses_unproven_occurrence_join(
    cad_plc: PiecewiseLinearComplex,
) -> None:
    with pytest.raises(ValueError, match="occurrence path"):
        _rebuild_plc(
            cad_plc,
            charts=None,
            vertex_rows=None,
            segment_edge_ids=None,
            polygon_cell_ids=None,
            tolerance=None,
        )


@pytest.mark.parametrize(
    "defect",
    (
        "edge",
        "vertex",
        "polygon",
        "region",
        "stale-charts",
        "association",
        "no-fidelity",
        "no-proof",
    ),
)
def test_cad_plc_refuses_forged_ancestry(
    cad_plc: PiecewiseLinearComplex,
    defect: Literal[
        "edge",
        "vertex",
        "polygon",
        "region",
        "stale-charts",
        "association",
        "no-fidelity",
        "no-proof",
    ],
) -> None:
    charts = cad_plc.cad_source_charts
    vertices = cad_plc.cad_vertex_rows
    edges = cad_plc.cad_segment_edge_ids
    polygons = cad_plc.cad_polygon_cell_ids
    tolerance = cad_plc.cad_fidelity_tolerance
    facet_regions = None
    assert charts is not None and vertices is not None and edges is not None
    assert polygons is not None and tolerance is not None
    if defect == "edge":
        edges = edges.copy()
        edges[0] = int(edges.max()) + 1
    elif defect == "vertex":
        vertices = vertices.copy()
        vertices[0] = (vertices[0] + 1) % len(cad_plc.vertex_ids)
    elif defect == "polygon":
        polygons = polygons.copy()
        polygons[0] = int(polygons.max()) + 1
    elif defect == "region":
        facet_regions = np.array(cad_plc.facet_regions, copy=True)
        facet_regions[0] = facet_regions[0, ::-1]
    elif defect == "stale-charts":
        patches = np.array(charts.root_patches, copy=True)
        patches[0] = (patches[0] + 1) % len(charts.domain.patches)
        charts = eqx.tree_at(lambda value: value.root_patches, charts, patches)
    elif defect == "association":
        charts = eqx.tree_at(
            lambda value: value.associations, charts, charts.associations[:-1]
        )
    elif defect == "no-fidelity":
        tolerance = None
    with pytest.raises((ValueError, TypeError)):
        _rebuild_plc(
            cad_plc,
            charts=charts,
            vertex_rows=vertices,
            segment_edge_ids=edges,
            polygon_cell_ids=polygons,
            tolerance=tolerance,
            facet_regions=facet_regions,
            drop_proof=defect == "no-proof",
        )


def test_cad_plc_restores_exact_scientific_authority(
    cad_plc: PiecewiseLinearComplex,
) -> None:
    rebuilt = PiecewiseLinearComplex.from_stored_fields(cad_plc)
    assert rebuilt.complex_id == cad_plc.complex_id
    assert rebuilt.cad_source_charts is not None
    assert rebuilt.cad_segment_edge_ids is not None
    assert not rebuilt.cad_segment_edge_ids.flags.writeable


def test_cad_plc_rejects_ambiguous_repeated_shared_copy(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
) -> None:
    from phydrax.geometry.brep._model import (
        BRepAssemblyGraph,
        BRepAssemblyUse,
        BRepGeometry,
    )

    report, _, _ = multiregion_surface
    domain = report.inputs.feature_source
    assert isinstance(domain, MeshingDomain) and domain.brep_authority is not None
    source = domain.brep_authority
    geometry = source.geometry
    graph = BRepAssemblyGraph(
        roots=(
            BRepAssemblyUse("first", solid_index=0),
            BRepAssemblyUse("repeated", solid_index=0),
            BRepAssemblyUse("neighbor", solid_index=1),
        ),
        definitions=(),
        solid_count=2,
    )
    repeated = BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=geometry.edge_face_gap_bounds,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=graph,
    )
    with pytest.raises(ValueError, match="Repeated coincident"):
        _require_unambiguous_cad_copies(
            repeated, source.source_id, source.source_revision
        )


def test_cad_plc_handoffs_fresh_surface_proof(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
    cad_plc: PiecewiseLinearComplex,
) -> None:
    import copy
    import pickle

    from phydrax.meshing import _plc_source
    from phydrax.meshing._model_codec import _same_model_content

    report, _, result = multiregion_surface
    charts = result.surface_source
    domain = report.inputs.feature_source
    assert isinstance(charts, SurfaceSourceCharts)
    assert isinstance(domain, MeshingDomain)
    # Exercise the producer path that must compute a missing fidelity proof.
    without_fidelity = eqx.tree_at(lambda value: value.fidelity, report, None)
    source = eqx.tree_at(lambda value: value.certification, result, without_fidelity)
    rebuilt = _cad_plc_data(domain, source, charts, "conforming", 0.05).complex
    assert _same_model_content(rebuilt.cad_embedding, cad_plc.cad_embedding)
    assert _same_model_content(rebuilt.cad_fidelity, cad_plc.cad_fidelity)
    assert rebuilt.complex_id == cad_plc.complex_id
    assert rebuilt.cad_embedding is not None
    assert rebuilt.cad_fidelity_tolerance is not None
    junctions = np.asarray(
        rebuilt.cad_embedding.binding.junction_vertices, dtype=np.int64
    )
    artifact = _plc_source._ValidatedCadSurface(
        charts,
        rebuilt.cad_fidelity_tolerance,
        rebuilt.cad_maximum_normal_deviation,
        junctions,
    )
    assert all(value is not artifact for value in vars(rebuilt).values())
    with pytest.raises(ValueError, match="different proof operands"):
        artifact.require(
            charts,
            rebuilt.cad_fidelity_tolerance + 1.0,
            rebuilt.cad_maximum_normal_deviation,
            junctions,
        )
    with pytest.raises(ValueError, match="different proof operands"):
        artifact.require(
            charts,
            rebuilt.cad_fidelity_tolerance,
            rebuilt.cad_maximum_normal_deviation,
            np.append(junctions, -1),
        )
    with pytest.raises(ValueError, match="different proof operands"):
        artifact.require(
            copy.copy(charts),
            rebuilt.cad_fidelity_tolerance,
            rebuilt.cad_maximum_normal_deviation,
            junctions,
        )
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(artifact)
    # The constructor's verification binds only this exact, unchanged object.
    assert PiecewiseLinearComplex.from_stored_fields(rebuilt) is rebuilt
    # A copy or pytree rebuild is untrusted even in this process and must replay.
    for untrusted in (
        jax.tree_util.tree_map(lambda leaf: leaf, rebuilt),
        copy.copy(rebuilt),
    ):
        assert untrusted is not rebuilt
        restored = PiecewiseLinearComplex.from_stored_fields(untrusted)
        assert restored is not untrusted
        assert restored.complex_id == rebuilt.complex_id
        assert restored.cad_embedding is not rebuilt.cad_embedding
        assert _same_model_content(restored.cad_embedding, rebuilt.cad_embedding)
        # That full replay reproduced this live value exactly, so it is now
        # verified and is not proved a second time.
        assert PiecewiseLinearComplex.from_stored_fields(untrusted) is untrusted
    # An in-place content change voids the reuse: the full replay refuses it.
    vertices = rebuilt.vertices
    chain = [vertices]
    while isinstance(chain[-1].base, np.ndarray):
        chain.append(chain[-1].base)
    original = float(vertices[0, 0])
    for array in reversed(chain):
        array.setflags(write=True)
    vertices[0, 0] = original + 0.125
    try:
        with pytest.raises(ValueError):
            PiecewiseLinearComplex.from_stored_fields(rebuilt)
    finally:
        vertices[0, 0] = original
        for array in chain:
            array.setflags(write=False)
    assert PiecewiseLinearComplex.from_stored_fields(rebuilt) is rebuilt


def test_cad_curve_ancestry_replay_binds_its_exact_live_charts(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copy

    import phydrax.geometry._surface_source_support as support

    _, _, result = multiregion_surface
    charts = result.surface_source
    assert isinstance(charts, SurfaceSourceCharts)
    replays: list[None] = []
    original = support._cad_root_edges

    def counted(value: SurfaceSourceCharts, /) -> object:
        replays.append(None)
        return original(value)

    monkeypatch.setattr(support, "_cad_root_edges", counted)
    first = support.validate_cad_curve_ancestry(charts, 0.05)
    count = len(replays)
    first[-1] = ()
    again = support.validate_cad_curve_ancestry(charts, 0.05)
    # The same live, unchanged charts at the same tolerance reuse the replay,
    # and callers own their result mappings.
    assert len(replays) == count
    assert -1 not in again
    # Another tolerance or a copy of the charts replays in full.
    support.validate_cad_curve_ancestry(charts, 0.1)
    assert len(replays) == count + 1
    copied = support.validate_cad_curve_ancestry(copy.copy(charts), 0.05)
    assert len(replays) == count + 2
    assert copied == again


def test_native_surface_proofs_revalidate_scientific_content(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
) -> None:
    from phydrax._meshcore import meshcore_identity
    from phydrax.meshing._canonical import _accept_native_construction, _NATIVE_PROVIDER
    from phydrax.meshing._certification import _ComputedSurfaceAcceptance
    from phydrax.meshing._model_codec import _same_model_content
    from phydrax.meshing._native_construction import NativeMeshConstruction

    report, _, _ = multiregion_surface
    domain = report.inputs.feature_source
    assert isinstance(domain, MeshingDomain)
    model = domain.brep_authority
    assert model is not None
    scope = phx.meshing.MeshingScope.whole(model, 2)
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.5),),
        maximum_deviation=0.05,
    )
    plan = phx.meshing.NativeMeshingProvider().plan(model, specification)
    construction = plan.prepared.execute()
    assert isinstance(construction, NativeMeshConstruction)
    computed: list[_ComputedSurfaceAcceptance] = []
    result = _accept_native_construction(
        construction,
        specification,
        _NATIVE_PROVIDER,
        meshcore_identity(),
        plan.plan_id,
        plan.route,
        computed_surface_out=computed,
    )
    assert len(computed) == 1
    charts = result.surface_source
    assert isinstance(charts, SurfaceSourceCharts)
    accepted_domain = charts.domain
    live = _cad_plc_data(
        accepted_domain,
        result,
        charts,
        "conforming",
        0.05,
        computed_surface=computed[0],
    ).complex
    renewed = PiecewiseLinearComplex.from_stored_fields(
        jax.tree_util.tree_map(lambda leaf: leaf, live)
    )
    assert renewed.complex_id == live.complex_id
    assert _same_model_content(renewed.cad_embedding, live.cad_embedding)
    assert _same_model_content(renewed.cad_fidelity, live.cad_fidelity)
    # The live handoff is process-local and bound to its actual scientific owners.
    import copy
    import pickle

    from phydrax.discretization._surface_chart_records import SurfaceChartWitness

    acceptance = computed[0]
    changed = eqx.tree_at(
        lambda value: value.mesh, acceptance.inputs, copy.copy(acceptance.inputs.mesh)
    )
    assert not acceptance.matches(changed)
    changed = eqx.tree_at(
        lambda value: value.geometry,
        acceptance.inputs,
        copy.copy(acceptance.inputs.geometry),
    )
    assert not acceptance.matches(changed)
    assert acceptance.inputs.source is not None
    changed = eqx.tree_at(
        lambda value: value.source,
        acceptance.inputs,
        copy.copy(acceptance.inputs.source),
    )
    assert not acceptance.matches(changed)
    witness = acceptance.inputs.fidelity_witness
    assert isinstance(witness, SurfaceChartWitness)
    altered_charts = np.asarray(witness.charts).copy()
    altered_charts[0, 0, 0] += 0.25
    altered_witness = eqx.tree_at(lambda value: value.charts, witness, altered_charts)
    changed = eqx.tree_at(
        lambda value: value.fidelity_witness, acceptance.inputs, altered_witness
    )
    assert not acceptance.matches(changed)
    assert acceptance.inputs.fidelity_tolerance is not None
    changed = eqx.tree_at(
        lambda value: value.fidelity_tolerance,
        acceptance.inputs,
        acceptance.inputs.fidelity_tolerance + 1.0,
    )
    assert not acceptance.matches(changed)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(acceptance)
    _, _, another_result = multiregion_surface
    assert another_result.surface_source is not None
    with pytest.raises(ValueError, match="another report"):
        _cad_plc_data(
            domain,
            another_result,
            another_result.surface_source,
            "conforming",
            0.05,
            computed_surface=acceptance,
        )


class _NativeSplitBox(NamedTuple):
    domain: MeshingDomain
    mesh: CellMesh
    solid: CadSolidResult
    witness: NativeCadSurfaceWitness
    rows: _RegionalSurfaceRows
    interface: np.ndarray


@pytest.fixture(scope="module")
def native_split_box(split_box_model: BRepModel) -> _NativeSplitBox:
    domain = MeshingDomain.from_brep(split_box_model)
    model = cad_geometry_bundle(domain).model
    # Exercise the native construction itself without changing provider
    # admission: this component does not yet prove BRep material occupancy.
    solid = cad_mesh_solid(
        model,
        np.full(model.surface_nodes.size, 0.4),
        np.full(model.surface_nodes.size, 0.05),
        np.full(model.curve_nodes.size, 0.05),
        radius_edge_bound=8.0,
        min_dihedral_degrees=0.0,
    )
    assert len(solid.volume.tetrahedra) > 0, solid.volume.stats
    mesh = CellMesh.from_simplices(
        solid.volume.points,
        solid.volume.tetrahedra,
        dimension=3,
        numeric_version=domain.source_revision,
    )
    np.testing.assert_array_equal(mesh.coordinates, solid.volume.points)
    np.testing.assert_array_equal(mesh.blocks[0].vertices, solid.volume.tetrahedra)
    witness = NativeCadSurfaceWitness(mesh, solid.surface, model.loop_surfaces.size)
    rows = _native_cad_regional_surface(
        mesh, domain, witness, native_surface=solid.surface
    )
    interface = np.flatnonzero(np.all(rows.regions >= 0, axis=1))
    assert interface.size > 1
    return _NativeSplitBox(domain, mesh, solid, witness, rows, interface)


def test_native_split_box_regional_boundary_and_exact_volume(
    native_split_box: _NativeSplitBox,
) -> None:
    domain, mesh, solid, witness, rows, interface = native_split_box
    points, cells = np.asarray(mesh.coordinates), np.asarray(mesh.blocks[0].vertices)
    cell_regions = solid.volume.tetrahedron_regions
    assert len(domain.regions) == 2
    np.testing.assert_array_equal(np.unique(cell_regions), [0, 1])
    assert not verify_chart_partition(
        len(domain.patches), witness.chart_patches, witness.chart_uv, witness.chords()
    ).size
    replay = _native_cad_regional_surface(mesh, domain, witness)
    np.testing.assert_array_equal(replay.patches, rows.patches)
    np.testing.assert_array_equal(replay.regions, rows.regions)

    # The shared patch is one actual BRep face occurrence. Its sign in each
    # solid, not the labels on incident tetrahedra, fixes the declaration.
    (patch,) = np.unique(rows.patches[interface])
    source, identifier, occurrence = rows.patch_occurrences[patch]
    assert source == domain.source_indices[2][patch]
    assert occurrence == domain.source_occurrences[2][patch]
    assert identifier == brep_entity_id(
        domain.source_id, domain.source_revision, 2, source, occurrence_path=occurrence
    )
    authority = domain.brep_authority
    assert isinstance(authority, BRepModel)
    topo = authority.topology
    for region, solid_id in enumerate(domain.region_source_indices):
        source_slot = topo.solid_faces[solid_id].index(source)
        sign = topo.solid_face_orientations[solid_id][source_slot]
        assert np.all(rows.regions[interface, 0 if sign == 1 else 1] == region)

    aggregate = verify_surface_embedding(points, rows.triangles)
    assert aggregate.status is VerdictStatus.VIOLATED
    assert SurfaceFindingKind.EDGE in aggregate.findings["kind"]
    exterior_rows, exterior = rows.outward(None)
    assert not np.intersect1d(exterior_rows, interface).size
    assert len(exterior_rows) + len(interface) == len(rows.triangles)
    assert verify_surface_embedding(points, exterior).status is VerdictStatus.CERTIFIED

    interface_uses = []
    for region in range(len(domain.regions)):
        original, outward = rows.outward(region)
        verdict = verify_surface_embedding(points, outward)
        assert verdict.status is VerdictStatus.CERTIFIED, verdict.findings
        offsets = np.searchsorted(original, interface)
        np.testing.assert_array_equal(original[offsets], interface)
        interface_uses.append(outward[offsets])
    np.testing.assert_array_equal(interface_uses[0], interface_uses[1][:, ::-1])

    tiling = verify_tetrahedra(points, cells)
    assert tiling.status is VerdictStatus.CERTIFIED, tiling
    assert tiling.boundary_faces == len(exterior_rows)
    coverage = verify_volume(
        points,
        cells,
        rows.triangles,
        cell_regions=cell_regions,
        surface_regions=rows.regions,
    )
    assert coverage.certified, coverage
    assert coverage.boundary_faces == len(exterior_rows)


@pytest.mark.parametrize(
    "defect",
    (
        "swapped-sides",
        "reversed-triangle",
        "reversed-chart-and-triangle",
        "wrong-chart-patch",
        "wrong-occurrence",
    ),
)
def test_native_split_box_refuses_invalid_chart_incidence(
    native_split_box: _NativeSplitBox, defect: str
) -> None:
    domain, mesh, solid, witness, rows, interface = native_split_box
    surface = solid.surface
    triangle = int(interface[0])
    chart_row = int(np.flatnonzero(witness.chart_to_mesh == triangle)[0])
    patch = int(rows.patches[triangle])
    match = rf"triangle row {triangle}\."
    if defect == "swapped-sides":
        sides = surface.facet_regions.copy()
        sides[triangle] = sides[triangle, ::-1]
        surface = surface._replace(facet_regions=sides)
    elif defect in ("reversed-triangle", "reversed-chart-and-triangle"):
        triangles = surface.triangles.copy()
        triangles[triangle] = triangles[triangle, ::-1]
        surface = surface._replace(triangles=triangles)
        match = rf"chart row {chart_row}.*oriented boundary triangle row {triangle}\."
        if defect == "reversed-chart-and-triangle":
            chart = surface.chart
            chart_triangles, uv = chart.triangles.copy(), chart.uv.copy()
            chart_triangles[chart_row] = chart_triangles[chart_row, ::-1]
            uv[chart_row] = uv[chart_row, ::-1]
            surface = surface._replace(
                chart=chart._replace(triangles=chart_triangles, uv=uv)
            )
            match = (
                rf"chart row {chart_row} reverses or degenerates source patch row {patch}"
            )
        witness = NativeCadSurfaceWitness(mesh, surface, witness.loop_count)
    elif defect == "wrong-chart-patch":
        patches = surface.chart.patches.copy()
        wrong_patch = next(
            row
            for row, face in enumerate(domain.patches)
            if row != patch and face.orientation == domain.patches[patch].orientation
        )
        patches[chart_row] = wrong_patch
        witness = NativeCadSurfaceWitness(
            mesh,
            surface._replace(chart=surface.chart._replace(patches=patches)),
            witness.loop_count,
        )
        findings = verify_chart_partition(
            len(domain.patches), witness.chart_patches, witness.chart_uv, witness.chords()
        )
        assert np.any(np.isin(findings["patch"], [patch, wrong_patch]))
    else:
        vertices, edges, faces = domain.source_occurrences
        wrong_faces = (*faces[:patch], ("wrong-occurrence",), *faces[patch + 1 :])
        domain = eqx.tree_at(
            lambda value: value.source_occurrences, domain, (vertices, edges, wrong_faces)
        )
        match = "MeshingDomain content changed"
    with pytest.raises(ValueError, match=match):
        _native_cad_regional_surface(mesh, domain, witness, native_surface=surface)


@pytest.mark.parametrize("defect", ("missing", "duplicate"))
def test_native_split_box_refuses_invalid_physical_facet_rows(
    native_split_box: _NativeSplitBox, defect: str
) -> None:
    domain, mesh, solid, witness, rows, interface = native_split_box
    surface = solid.surface
    triangle = int(interface[0])
    chart_row = int(np.flatnonzero(witness.chart_to_mesh == triangle)[0])
    chart = surface.chart
    if defect == "missing":
        mapping = chart.chart_to_mesh.copy()
        mapping[chart_row] = -1
        surface = surface._replace(chart=chart._replace(chart_to_mesh=mapping))
        match = rf"triangle row {triangle} has 0 chart images"
    else:
        duplicate = len(rows.triangles)
        chart = chart._replace(
            triangles=np.concatenate((chart.triangles, chart.triangles[[chart_row]])),
            patches=np.append(chart.patches, chart.patches[chart_row]),
            uv=np.concatenate((chart.uv, chart.uv[[chart_row]])),
            chart_to_mesh=np.append(chart.chart_to_mesh, duplicate),
            edge_chord=np.concatenate((chart.edge_chord, chart.edge_chord[[chart_row]])),
        )
        surface = surface._replace(
            triangles=np.concatenate((surface.triangles, surface.triangles[[triangle]])),
            chart=chart,
        )
        match = rf"triangle row {duplicate} duplicates a facet"
    changed = NativeCadSurfaceWitness(mesh, surface, witness.loop_count)
    with pytest.raises(ValueError, match=match):
        _native_cad_regional_surface(mesh, domain, changed)


@pytest.mark.parametrize(
    "defect",
    (
        "swapped",
        "reversed",
        "undeclared",
        "duplicate",
        "missing-exterior",
        "wrong-cell-label",
    ),
)
def test_native_split_box_rejects_wrong_exact_regional_coverage(
    native_split_box: _NativeSplitBox, defect: str
) -> None:
    _, mesh, solid, _, rows, interface = native_split_box
    points, cells = np.asarray(mesh.coordinates), np.asarray(mesh.blocks[0].vertices)
    cell_regions = solid.volume.tetrahedron_regions.copy()
    triangles, sides = rows.triangles.copy(), rows.regions.copy()
    row = int(rows.outward(None)[0][0] if defect == "missing-exterior" else interface[0])
    # Only for checking the exact diagnostic row, not for declaring any facet.
    incident = np.flatnonzero(
        np.count_nonzero(np.isin(cells, triangles[row]), axis=1) == 3
    )
    assert incident.size == (1 if defect == "missing-exterior" else 2)
    if defect == "swapped":
        sides[row] = sides[row, ::-1]
    elif defect == "reversed":
        triangles[row] = triangles[row, ::-1]
    elif defect in ("undeclared", "missing-exterior"):
        triangles = np.delete(triangles, row, axis=0)
        sides = np.delete(sides, row, axis=0)
    elif defect == "duplicate":
        triangles = np.concatenate((triangles, triangles[[row]]))
        sides = np.concatenate((sides, sides[[row]]))
    else:
        cell_regions[0] = 1 - cell_regions[0]

    coverage = verify_volume(
        points, cells, triangles, cell_regions=cell_regions, surface_regions=sides
    )
    assert not coverage.certified
    if defect == "duplicate":
        assert coverage.unmatched_surface_triangles == 1
        # The last declaration in the key group is used; the original row
        # is the extra one in the native verifier's ordering.
        assert coverage.first_unmatched_surface_triangle == row
    elif defect == "missing-exterior":
        assert coverage.unmatched_boundary_faces == 1
        assert coverage.first_unmatched_cell == incident[0]
    elif defect == "wrong-cell-label":
        assert coverage.region_mismatches == 4
        assert coverage.first_region_mismatch_cell == 0
    else:
        assert coverage.region_mismatches == 1
        assert coverage.first_region_mismatch_cell == incident[0]


def test_native_regional_v2_reports_original_facet_row(
    native_split_box: _NativeSplitBox,
) -> None:
    from phydrax.meshing._native_cad_certificate import _surface_embedding_findings

    _, mesh, _, _, rows, _ = native_split_box
    original, outward = rows.outward(0)
    local = np.flatnonzero(original != np.arange(original.size))[0]
    outward[local, 0] = outward[local, 1]
    (finding,) = _surface_embedding_findings(
        np.asarray(mesh.coordinates), outward, "region 0 surface", original_rows=original
    )
    assert finding.entity_kind == "facet"
    assert finding.entity_ids == (int(original[local]),)
    assert "DEGENERATE" in finding.detail


def test_native_split_box_v2_reports_edges_of_missing_interface_facet(
    native_split_box: _NativeSplitBox,
) -> None:
    from phydrax.meshing._native_cad_certificate import _surface_embedding_findings

    _, mesh, _, _, rows, interface = native_split_box
    points = np.asarray(mesh.coordinates)
    omitted = interface[0]
    a, b, c = (int(vertex) for vertex in rows.triangles[omitted])
    open_edges = sorted(tuple(sorted(edge)) for edge in ((a, b), (b, c), (c, a)))
    for region in (0, 1):
        original, outward = rows.outward(region)
        verdict = verify_surface_embedding(points, outward[original != omitted])
        assert verdict.status is VerdictStatus.VIOLATED
        findings = verdict.findings[verdict.findings["kind"] == SurfaceFindingKind.EDGE]
        assert list(zip(findings["first"], findings["second"], strict=True)) == open_edges
        (finding,) = _surface_embedding_findings(
            points,
            outward[original != omitted],
            "region surface",
            original_rows=original[original != omitted],
        )
        assert finding.entity_kind == "vertex"
        assert finding.entity_ids == tuple(sorted((a, b, c)))


def test_native_split_box_rejects_inverted_tetrahedron_in_union(
    native_split_box: _NativeSplitBox,
) -> None:
    _, mesh, _, _, _, _ = native_split_box
    points, cells = np.asarray(mesh.coordinates), np.array(mesh.blocks[0].vertices)
    cells[0, [0, 1]] = cells[0, [1, 0]]
    verdict = verify_tetrahedra(points, cells)
    assert verdict.status is VerdictStatus.VIOLATED
    assert verdict.nonpositive_cells == 1
    assert verdict.first_nonpositive_cell == 0
