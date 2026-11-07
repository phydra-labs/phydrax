"""Authored PLC source strata survive native tetrahedral adaptation exactly."""

from __future__ import annotations

import json
import subprocess
import sys
from functools import cache
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
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
