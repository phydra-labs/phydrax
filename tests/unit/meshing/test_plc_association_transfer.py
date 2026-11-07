"""Authored PLC source strata survive native tetrahedral adaptation exactly."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from functools import cache
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry._domain_certification import certify_domain_coverage
from phydrax.geometry._planar_coverage import PlcSourcePointWitness
from phydrax.meshing import (
    GeometryAssociation,
    GeometryAssociationKind,
    PlcAssociationTransfer,
)
from phydrax.meshing._association import GeometrySourceEntityRole
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers import NativeMeshingProvider, NativePlcSource


_CORNERS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float64,
)
_QUADS = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


def _source(revision: str = "r1") -> NativePlcSource:
    complex_ = phx.meshing.PiecewiseLinearComplex(
        _CORNERS,
        _QUADS,
        np.arange(6, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (6, 1)),
        ("solid",),
        vertex_ids=tuple(f"corner-{index}" for index in range(8)),
        vertex_occurrence_paths=((),) * 8,
        facet_ids=tuple(f"side-{index}" for index in range(6)),
        facet_occurrence_paths=((),) * 6,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="conforming",
    )
    return NativePlcSource(complex_, "plc-transfer-cube", revision)


@cache
def _accepted(
    revision: str = "r1",
) -> tuple[NativePlcSource, phx.meshing.CellMeshingResult]:
    source = _source(revision)
    scope = phx.meshing.MeshingScope.whole(source, 2)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 0.6, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    options = phx.meshing.NativeMeshingOptions(
        "plc_tetrahedral",
        volume_schedule=NativeVolumeSchedule(
            radius_edge_bound=2.0, minimum_dihedral_degrees=0.0
        ),
    )
    result = (
        NativeMeshingProvider(options)
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    return source, result


def _refine_all(
    result: phx.meshing.CellMeshingResult,
    transfer: PlcAssociationTransfer,
    route: phx.meshing.MeshAdaptationRoute,
) -> phx.meshing.MeshAdaptationResult:
    marks = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )
    prepared = phx.meshing.prepare_mesh_adaptation(
        result,
        phx.meshing.MarkedMeshAdaptation(marks, np.empty((0,), dtype=np.int64)),
        policy=phx.meshing.MeshAdaptationPolicy(
            route,
            association_transfer=transfer,
            audit_policy=phx.meshing.CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
            ),
        ),
    )
    return phx.meshing.execute_mesh_adaptation(prepared)


def _table(result: phx.meshing.CellMeshingResult, dimension: int) -> GeometryAssociation:
    entity_set_id = result.mesh.entity_set(dimension).entity_set_id
    (table,) = (
        value
        for value in result.associations
        if value.target_entity_set_id == entity_set_id
        and value.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
    )
    return table


@pytest.mark.parametrize(
    "route",
    (
        phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
    ),
)
def test_refinement_retains_every_authored_plc_stratum(
    route: phx.meshing.MeshAdaptationRoute,
) -> None:
    source, result = _accepted()
    transfer = PlcAssociationTransfer(source)
    transfer.validate_source(result)
    adaptation = _refine_all(result, transfer, route)
    target = adaptation.target
    assert adaptation.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert target.audit.passed and target.certification is not None
    assert target.certification.passed
    assert sum(block.cell_count for block in target.mesh.blocks) > sum(
        block.cell_count for block in result.mesh.blocks
    )
    # The target is itself an admissible transfer source: the adapted tables
    # were re-verified against the authored strata, not merely copied.
    transfer.validate_source(target)
    coordinates = np.asarray(target.mesh.coordinates)
    vertices = _table(target, 0)
    rows = np.searchsorted(
        np.asarray(target.mesh.vertex_global_ids), np.asarray(vertices.target_global_ids)
    )
    for row, role, key in zip(
        rows.tolist(), vertices.source_roles, vertices.source_entity_ids, strict=True
    ):
        point = coordinates[row]
        on_boundary = bool(np.any(point == 0.0) or np.any(point == 1.0))
        if role is GeometrySourceEntityRole.VERTEX:
            index = int(key.removeprefix("corner-"))
            np.testing.assert_array_equal(point, _CORNERS[index])
        elif role is GeometrySourceEntityRole.REGION:
            assert not on_boundary and key == "solid"
        else:
            assert on_boundary
    faces = _table(target, 2)
    boundary = [role is GeometrySourceEntityRole.FACET for role in faces.source_roles]
    assert sum(boundary) > 12
    cell_regions = {key for key in _table(target, 3).source_entity_ids}
    assert cell_regions == {"solid"}
    classes = transfer.classes(target)
    assert np.all(classes[0].dimensions[classes[0].dimensions == 0] == 0)
    assert set(np.unique(classes[2].dimensions).tolist()) == {2, 3}


_COLD_READER = """
import json, sys
import numpy as np
from phydrax.lifecycle import Composition
from phydrax.meshing import PlcAssociationTransfer
from phydrax.meshing._model_codec import (
    MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_model_archive_value,
)
archive = read_model_artifact(
    sys.argv[1], expected_type=Composition, restore=restore_model_archive_value,
    limits=MODEL_ARCHIVE_LIMITS, expected_content_id=sys.argv[2],
)
source = archive.value("source")
target = archive.value("result/refined")
transfer = PlcAssociationTransfer(source)
transfer.validate_source(target)
print(json.dumps({
    "result_id": target.result_id,
    "associations": [value.association_id for value in target.associations],
    "classes": [
        [level.dimensions.tolist(), level.indices.tolist(), level.codes.tolist()]
        for level in transfer.classes(target)
    ],
}))
"""


def test_adapted_plc_associations_survive_an_independent_cold_restore(
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        write_model_artifact,
    )

    source, result = _accepted()
    transfer = PlcAssociationTransfer(source)
    target = _refine_all(
        result, transfer, phx.meshing.MeshAdaptationRoute.NATIVE_MIXED
    ).target
    source_entry = CompositionEntry(
        source,
        entry_id="source",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=source.complex.complex_id,
        revision_id=source.binding_id,
        semantics_id="authored-plc-source",
    )
    target_entry = CompositionEntry(
        target,
        entry_id="result/refined",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=target.mesh.topology_id,
        revision_id=target.result_id,
        semantics_id="plc-refined-result",
        dependencies=(source_entry.binding("revision"),),
    )
    path = tmp_path / "plc-refined.phx"
    write_model_artifact(
        path,
        Composition((source_entry, target_entry), boundary_id="plc-transfer-cold"),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            _COLD_READER,
            str(path),
            model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS),
        ),
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    cold = json.loads(completed.stdout.strip().splitlines()[-1])
    assert cold["result_id"] == target.result_id
    assert cold["associations"] == [value.association_id for value in target.associations]
    assert cold["classes"] == [
        [level.dimensions.tolist(), level.indices.tolist(), level.codes.tolist()]
        for level in transfer.classes(target)
    ]


def test_transfer_refuses_a_stale_source_revision() -> None:
    _, result = _accepted()
    stale = PlcAssociationTransfer(_source("r2"))
    with pytest.raises(ValueError, match="namespace|declare"):
        stale.validate_source(result)


def test_transfer_refuses_a_forged_facet_association() -> None:
    source, result = _accepted()
    faces = _table(result, 2)
    facets = [
        row
        for row, role in enumerate(faces.source_roles)
        if role is GeometrySourceEntityRole.FACET
    ]
    row = facets[0]
    wrong = next(
        f"side-{index}"
        for index in range(6)
        if f"side-{index}" != faces.source_entity_ids[row]
    )
    forged = GeometryAssociation(
        faces.association_kind,
        faces.source_id,
        faces.source_revision,
        faces.target_entity_set_id,
        faces.target_global_ids,
        tuple(
            wrong if index == row else value
            for index, value in enumerate(faces.source_entity_ids)
        ),
        faces.residuals,
        exact=faces.exact,
        source_roles=faces.source_roles,
        source_occurrence_paths=faces.source_occurrence_paths,
        source_region_sides=faces.source_region_sides,
        orientations=faces.orientations,
    )
    tampered = eqx.tree_at(
        lambda value: value.associations,
        result,
        tuple(forged if value is faces else value for value in result.associations),
    )
    with pytest.raises(ValueError, match="off the source strata|orientations"):
        PlcAssociationTransfer(source).validate_source(tampered)


def test_transfer_refuses_an_entity_outside_the_source_complex() -> None:
    source, result = _accepted()
    cells = _table(result, 3)
    forged = GeometryAssociation(
        cells.association_kind,
        cells.source_id,
        cells.source_revision,
        cells.target_entity_set_id,
        cells.target_global_ids,
        ("void",) + cells.source_entity_ids[1:],
        cells.residuals,
        exact=cells.exact,
        source_roles=cells.source_roles,
        source_occurrence_paths=cells.source_occurrence_paths,
    )
    tampered = eqx.tree_at(
        lambda value: value.associations,
        result,
        tuple(forged if value is cells else value for value in result.associations),
    )
    with pytest.raises(ValueError, match="outside its source complex"):
        PlcAssociationTransfer(source).validate_source(tampered)


def _tilted_source() -> NativePlcSource:
    """A small rotated octahedron far from the origin, like a reconstructed PLC.

    Its binary64 vertices are not dyadic in any frame, so a floating facet
    plane through them misplaces their own corners by more than a bounding
    scale rounding slack.
    """
    w, x, y, z = np.asarray((0.3, -0.5, 0.7, 0.4)) / np.linalg.norm((0.3, -0.5, 0.7, 0.4))
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    axes = np.concatenate([np.eye(3), -np.eye(3)])[[0, 3, 1, 4, 2, 5]]
    vertices = np.asarray((0.61, 0.73, 0.89)) + _TILTED_RADIUS * axes @ rotation.T
    faces = []
    for sx, sy, sz in itertools.product((1, -1), repeat=3):
        face = (0 if sx > 0 else 1, 2 if sy > 0 else 3, 4 if sz > 0 else 5)
        faces.append(face if sx * sy * sz > 0 else face[::-1])
    complex_ = phx.meshing.PiecewiseLinearComplex(
        vertices,
        tuple(faces),
        np.arange(8, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (8, 1)),
        ("solid",),
        vertex_ids=tuple(f"apex-{index}" for index in range(6)),
        vertex_occurrence_paths=((),) * 6,
        facet_ids=tuple(f"face-{index}" for index in range(8)),
        facet_occurrence_paths=((),) * 8,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="conforming",
    )
    return NativePlcSource(complex_, "plc-transfer-tilted", "r1")


_TILTED_RADIUS = 0.004


@cache
def _tilted_accepted() -> tuple[NativePlcSource, phx.meshing.CellMeshingResult]:
    source = _tilted_source()
    scope = phx.meshing.MeshingScope.whole(source, 2)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                0.5 * _TILTED_RADIUS,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
    )
    options = phx.meshing.NativeMeshingOptions(
        "plc_tetrahedral",
        volume_schedule=NativeVolumeSchedule(
            radius_edge_bound=2.0, minimum_dihedral_degrees=0.0
        ),
    )
    result = (
        NativeMeshingProvider(options)
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    return source, result


def _point_witness(result: phx.meshing.CellMeshingResult) -> PlcSourcePointWitness:
    assert result.certification is not None
    witness = result.certification.inputs.source_coverage_witness
    assert isinstance(witness, PlcSourcePointWitness)
    return witness


@cache
def _tilted_refined() -> phx.meshing.CellMeshingResult:
    source, result = _tilted_accepted()
    adaptation = _refine_all(
        result,
        PlcAssociationTransfer(source),
        phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
    )
    assert adaptation.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    return adaptation.target


def _rewitnessed(
    witness: PlcSourcePointWitness,
    *,
    parameters: np.ndarray | None = None,
    deviations: np.ndarray | None = None,
) -> PlcSourcePointWitness:
    return PlcSourcePointWitness(
        witness.vertex_global_ids,
        witness.vertex_kinds,
        witness.vertex_entities,
        witness.vertex_parameters if parameters is None else parameters,
        segments=witness.segments,
        faces=witness.faces,
        face_groups=witness.face_groups,
        face_planes=witness.face_planes,
        face_plane_tolerances=witness.face_plane_tolerances,
        domain_face_rows=witness.domain_face_rows,
        domain_id=witness.domain_id,
        vertex_deviations=witness.vertex_deviations if deviations is None else deviations,
    )


def _coverage_findings(
    result: phx.meshing.CellMeshingResult, witness: PlcSourcePointWitness
) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    assert result.certification is not None
    inputs = result.certification.inputs
    assert inputs.domain is not None and inputs.cell_regions is not None
    coverage = certify_domain_coverage(
        inputs.mesh,
        inputs.geometry,
        inputs.domain,
        inputs.cell_regions,
        embedding=result.certification.embedding,
        source_coverage_witness=witness,
    )
    return coverage.status, tuple(
        (finding.check, tuple(finding.entity_ids)) for finding in coverage.findings
    )


def _bounded_row(witness: PlcSourcePointWitness, kind: int) -> int:
    """A created row whose bound exceeds its carrier rounding: nested between rounded parents."""
    assert witness.vertex_deviations is not None
    deviations = np.asarray(witness.vertex_deviations)
    kinds = np.asarray(witness.vertex_kinds)
    return int(
        np.flatnonzero((kinds == kind) & (deviations == deviations[kinds == kind].max()))[
            0
        ]
    )


def test_tilted_facets_classify_by_exact_source_points() -> None:
    source, result = _tilted_accepted()
    assert result.certification is not None and result.certification.passed
    # Native boundary insertion produced correctly rounded segment/facet points.
    witness = _point_witness(result)
    assert witness.vertex_deviations is None
    assert np.any(np.asarray(witness.vertex_kinds) >= 1)
    transfer = PlcAssociationTransfer(source)
    transfer.validate_source(result)
    target = _tilted_refined()
    assert target.certification is not None and target.certification.passed
    # Midpoints between rounded source points carry exact source points in
    # their strata with recorded, outward-rounded carrier deviations.
    renewed = _point_witness(target)
    assert renewed.vertex_deviations is not None
    pointed = np.asarray(renewed.vertex_kinds) >= 0
    assert np.all(np.asarray(renewed.vertex_deviations)[pointed] > 0.0)
    assert np.all(np.asarray(renewed.vertex_deviations)[~pointed] == 0.0)
    transfer.validate_source(target)
    for dimension in range(4):
        residuals = np.asarray(_table(target, dimension).residuals)
        assert np.all(residuals <= 4.0 * np.finfo(np.float64).eps)
    classes = transfer.classes(target)
    assert set(np.unique(classes[2].dimensions).tolist()) == {2, 3}


def test_tilted_carrier_beyond_its_recorded_bound_is_refused() -> None:
    target = _tilted_refined()
    witness = _point_witness(target)
    row = _bounded_row(witness, 1)
    deviations = np.array(witness.vertex_deviations, copy=True)
    deviations[row] = np.finfo(np.float64).tiny
    status, findings = _coverage_findings(
        target, _rewitnessed(witness, deviations=deviations)
    )
    vertex = int(witness.vertex_global_ids[row])
    assert status != "certified"
    assert ("source_point_deviation", (vertex,)) in findings
    # Without a bank, the carrier must be the correctly rounded source point.
    with pytest.raises(ValueError, match="bound every source point|finite"):
        _rewitnessed(witness, deviations=np.where(deviations > 0, 0.0, deviations))


def test_tilted_source_point_off_its_stratum_is_refused() -> None:
    target = _tilted_refined()
    witness = _point_witness(target)
    row = _bounded_row(witness, 1)
    parameters = np.array(witness.vertex_parameters, copy=True)
    parameters[row, 0] = 1.25
    status, findings = _coverage_findings(
        target, _rewitnessed(witness, parameters=parameters)
    )
    assert status != "certified"
    assert ("source_point_parameter", (int(witness.vertex_global_ids[row]),)) in findings


def test_tilted_bounded_witness_archive_round_trip(tmp_path: Path) -> None:
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )

    source, _ = _tilted_accepted()
    target = _tilted_refined()
    witness = _point_witness(target)
    source_entry = CompositionEntry(
        source,
        entry_id="source",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=source.complex.complex_id,
        revision_id=source.binding_id,
        semantics_id="authored-plc-source",
    )
    target_entry = CompositionEntry(
        target,
        entry_id="result/refined",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=target.mesh.topology_id,
        revision_id=target.result_id,
        semantics_id="plc-refined-result",
        dependencies=(source_entry.binding("revision"),),
    )
    path = tmp_path / "tilted-refined.phx"
    write_model_artifact(
        path,
        Composition((source_entry, target_entry), boundary_id="tilted-bounded"),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    archive = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
        expected_content_id=model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS),
    )
    restored = archive.value("result/refined")
    restored_witness = _point_witness(restored)
    assert restored.result_id == target.result_id
    assert restored_witness.witness_id == witness.witness_id
    np.testing.assert_array_equal(
        np.asarray(restored_witness.vertex_deviations),
        np.asarray(witness.vertex_deviations),
    )
    PlcAssociationTransfer(source).validate_source(restored)
    deviations = np.array(witness.vertex_deviations, copy=True)
    deviations[_bounded_row(witness, 1)] *= 2.0
    tampered = eqx.tree_at(lambda value: value.vertex_deviations, witness, deviations)
    with pytest.raises(ValueError, match="identity changed"):
        restore_model_archive_value(tampered)


def test_tilted_transfer_refuses_a_face_named_on_an_adjacent_facet() -> None:
    source, result = _tilted_accepted()
    faces = _table(result, 2)
    complex_ = source.complex
    facet_rows = {key: row for row, key in enumerate(complex_.facet_ids)}
    loops = [
        set(np.asarray(complex_.polygon_vertices).reshape((-1, 3))[facet].tolist())
        for facet in range(8)
    ]
    row = next(
        row
        for row, role in enumerate(faces.source_roles)
        if role is GeometrySourceEntityRole.FACET
    )
    named = facet_rows[faces.source_entity_ids[row]]
    # The adjacent facet shares an authored edge: two corners may lie on it.
    adjacent = next(
        facet
        for facet in range(8)
        if facet != named and len(loops[facet] & loops[named]) == 2
    )
    forged = GeometryAssociation(
        faces.association_kind,
        faces.source_id,
        faces.source_revision,
        faces.target_entity_set_id,
        faces.target_global_ids,
        tuple(
            complex_.facet_ids[adjacent] if index == row else value
            for index, value in enumerate(faces.source_entity_ids)
        ),
        faces.residuals,
        exact=faces.exact,
        source_roles=faces.source_roles,
        source_occurrence_paths=faces.source_occurrence_paths,
        source_region_sides=faces.source_region_sides,
        orientations=faces.orientations,
    )
    tampered = eqx.tree_at(
        lambda value: value.associations,
        result,
        tuple(forged if value is faces else value for value in result.associations),
    )
    with pytest.raises(ValueError, match="off the source strata"):
        PlcAssociationTransfer(source).validate_source(tampered)


def test_tilted_transfer_refuses_a_carrier_off_its_source_point() -> None:
    source, result = _tilted_accepted()
    witness = _point_witness(result)
    rows = np.flatnonzero(np.asarray(witness.vertex_kinds) >= 1)
    vertex = int(
        np.searchsorted(
            np.asarray(result.mesh.vertex_global_ids),
            np.asarray(witness.vertex_global_ids)[rows[0]],
        )
    )
    coordinates = np.array(result.mesh.coordinates, dtype=np.float64, copy=True)
    # Sixteen units in the last place off the rounded source point: inside the
    # former bounding-scale slack, outside every exact admission.
    for _ in range(16):
        coordinates[vertex, 0] = np.nextafter(coordinates[vertex, 0], np.inf)
        coordinates[vertex, 1] = np.nextafter(coordinates[vertex, 1], -np.inf)
    tampered = eqx.tree_at(lambda value: value.mesh.coordinates, result, coordinates)
    with pytest.raises(ValueError, match="off the source strata"):
        PlcAssociationTransfer(source).validate_source(tampered)


@cache
def _envelope_accepted() -> tuple[
    phx.meshing.SurfaceEnvelopeSource, phx.meshing.CellMeshingResult
]:
    from tests.unit.meshing.test_native_surface_envelope_route import _request

    _, source, specification, options, contract = _request()
    result = (
        NativeMeshingProvider(options)
        .plan(source, specification, coordinate_contract=contract)
        .execute()
    )
    return source, result


def test_envelope_boundary_model_follows_refinement_lineage() -> None:
    source, result = _envelope_accepted()
    assert result.boundary is not None
    transfer = PlcAssociationTransfer(source.plc_source)
    adaptation = _refine_all(
        result, transfer, phx.meshing.MeshAdaptationRoute.NATIVE_MIXED
    )
    assert adaptation.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    target = adaptation.target
    transfer.validate_source(target)
    boundary = target.boundary
    assert boundary is not None
    metadata = boundary.metadata
    assert (metadata.source_id, metadata.source_revision) == (
        source.source_id,
        source.source_revision,
    )
    assert metadata.provenance[:-1] == result.boundary.metadata.provenance
    exterior = np.asarray(target.mesh.entity_set(2).entity_ids, dtype=np.int64)[
        np.asarray(target.mesh.boundary_masks[2], dtype=np.bool_)
    ]
    surface_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in boundary.mesh.blocks]
    )
    np.testing.assert_array_equal(np.sort(surface_ids), np.sort(exterior))
    assert exterior.size > np.count_nonzero(np.asarray(result.mesh.boundary_masks[2]))
    assert set(metadata.cell_tags) == {"unsigned-distance-envelope"}
    # Outward orientation: the closed boundary encloses exactly the cell volume.
    points = np.asarray(boundary.mesh.coordinates, dtype=np.float64)
    triangles = points[
        np.concatenate([np.asarray(block.vertices) for block in boundary.mesh.blocks])
    ]
    enclosed = np.sum(triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2])) / 6.0
    corners = np.asarray(target.mesh.coordinates, dtype=np.float64)[
        np.concatenate([np.asarray(block.vertices) for block in target.mesh.blocks])
    ]
    volume = np.sum(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0
    np.testing.assert_allclose(enclosed, volume, rtol=1e-12)


def test_envelope_boundary_remap_refuses_a_forged_namespace() -> None:
    source, result = _envelope_accepted()
    assert result.boundary is not None
    transfer = PlcAssociationTransfer(source.plc_source)
    adaptation = _refine_all(
        result, transfer, phx.meshing.MeshAdaptationRoute.NATIVE_MIXED
    )
    lineage = adaptation.lineage
    assert lineage is not None
    metadata = result.boundary.metadata
    forged = eqx.tree_at(
        lambda value: value.boundary.metadata,
        result,
        phx.geometry.surface.SurfaceMetadata(
            source_id=metadata.source_id,
            source_revision="forged-revision",
            coordinate_contract=metadata.coordinate_contract,
            provenance=metadata.provenance,
            cell_tags=metadata.cell_tags,
        ),
    )
    with pytest.raises(ValueError, match="association namespace"):
        transfer.remap_boundary(forged, lineage, adaptation.target.mesh)
