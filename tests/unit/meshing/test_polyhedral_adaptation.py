#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingScope,
    RegionRole,
    RegionSeed,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._adaptation import (
    execute_mesh_adaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._lineage import EntityLineageKind
from phydrax.meshing._organization import MeshAttribute, MeshAttributeRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._polyhedral_adaptation import (
    adapt_polyhedral_mesh,
    PolyhedralAdaptationOperation,
    PolyhedralAdaptationOutcome,
    PolyhedralAdaptationRequest,
)
from phydrax.meshing._polyhedral_adaptation_shared import polyhedral_face_loops
from phydrax.meshing._polyhedral_generation import (
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing.providers import (
    NativeMeshingOptions,
    NativeMeshingProvider,
    NativePlcSource,
    NativePolyhedralSource,
)


pytestmark = pytest.mark.meshcore


_CUBE_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 7, 6, 2),
    (0, 4, 7, 3),
    (1, 2, 6, 5),
)
_CUBE_VERTICES = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)


def _material_l_mesh() -> tuple[CellMesh, np.ndarray]:
    points: list[tuple[int, int, int]] = []
    point_rows: dict[tuple[int, int, int], int] = {}

    def vertex(point: tuple[int, int, int], /) -> int:
        row = point_rows.get(point)
        if row is None:
            row = len(points)
            point_rows[point] = row
            points.append(point)
        return row

    cells: list[tuple[np.ndarray, ...]] = []
    for x_offset, y_offset in ((0, 0), (1, 0), (0, 1)):
        local = tuple(
            vertex((x + x_offset, y + y_offset, z)) for x, y, z in _CUBE_VERTICES
        )
        cells.append(
            tuple(
                np.asarray([local[index] for index in face], dtype=np.int32)
                for face in _CUBE_FACES
            )
        )
    mesh = CellMesh.from_polyhedra(
        np.asarray(points, dtype=np.float64),
        cells,
        vertex_global_ids=np.arange(100, 100 + len(points), dtype=np.int64),
        cell_global_ids=np.asarray((10, 20, 30), dtype=np.int64),
    )
    classes = np.asarray((0, 0, 1), dtype=np.int64)
    bounding_box_volume = float(np.prod(np.ptp(np.asarray(mesh.coordinates), axis=0)))
    assert bounding_box_volume == 4.0
    return mesh, classes


def _l_volume_oracle() -> float:
    # The L is the disjoint union of three unit boxes; the missing fourth box is
    # excluded even though it lies inside their axis-aligned bounding box.
    return 3.0


def _cell_box_measure(mesh: CellMesh, cell_id: int, /) -> float:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Test mesh must retain polyhedral connectivity.")
    ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    row = int(np.flatnonzero(ids == cell_id)[0])
    offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
    values = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
    vertices = values[offsets[row] : offsets[row + 1]]
    extents = np.ptp(np.asarray(mesh.coordinates)[vertices], axis=0)
    return float(np.prod(extents))


def _cell_classes_by_id(outcome: PolyhedralAdaptationOutcome, /) -> dict[int, int]:
    hierarchy = outcome.hierarchy
    return {
        int(identifier): int(region)
        for identifier, region in zip(
            np.asarray(hierarchy.target_cell_ids),
            np.asarray(hierarchy.region_classes),
            strict=True,
        )
    }


def _shared_face_signatures(mesh: CellMesh, /) -> tuple[tuple[int, ...], ...]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Test mesh must retain polyhedral connectivity.")
    loops = polyhedral_face_loops(mesh)
    owners = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbors = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    owner_local = np.asarray(connectivity.face_owner_local, dtype=np.int64)
    neighbor_local = np.asarray(connectivity.face_neighbor_local, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    signatures: list[tuple[int, ...]] = []
    for face in np.flatnonzero(neighbors >= 0):
        first = vertex_ids[loops[int(owners[face])][int(owner_local[face])]]
        second = vertex_ids[loops[int(neighbors[face])][int(neighbor_local[face])]]
        first_signature = tuple(sorted(int(value) for value in first))
        second_signature = tuple(sorted(int(value) for value in second))
        assert first_signature == second_signature
        signatures.append(first_signature)
    return tuple(signatures)


def _material_interface_area(outcome: PolyhedralAdaptationOutcome, /) -> float:
    mesh = outcome.target_mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Test mesh must retain polyhedral connectivity.")
    classes = _cell_classes_by_id(outcome)
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    owner = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    owner_local = np.asarray(connectivity.face_owner_local, dtype=np.int64)
    loops = polyhedral_face_loops(mesh)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    areas: list[float] = []
    for face in np.flatnonzero(neighbor >= 0):
        first_cell, second_cell = int(owner[face]), int(neighbor[face])
        if classes[int(cell_ids[first_cell])] == classes[int(cell_ids[second_cell])]:
            continue
        loop = np.asarray(loops[first_cell][int(owner_local[face])], dtype=np.int64)
        polygon = points[loop]
        area = 0.0
        for index in range(1, polygon.shape[0] - 1):
            area += 0.5 * float(
                np.linalg.norm(
                    np.cross(polygon[index] - polygon[0], polygon[index + 1] - polygon[0])
                )
            )
        areas.append(area)
    return float(sum(areas))


def _plc_material_l() -> PiecewiseLinearComplex:
    points: list[tuple[int, int, int]] = []
    point_rows: dict[tuple[int, int, int], int] = {}

    def vertex(point: tuple[int, int, int], /) -> int:
        row = point_rows.get(point)
        if row is None:
            row = len(points)
            point_rows[point] = row
            points.append(point)
        return row

    face_owner: dict[tuple[int, ...], tuple[tuple[int, ...], int]] = {}
    interfaces: list[tuple[tuple[int, ...], tuple[int, int]]] = []
    for x_offset, y_offset, region in ((0, 0, 0), (1, 0, 0), (0, 1, 1)):
        local = tuple(
            vertex((x + x_offset, y + y_offset, z)) for x, y, z in _CUBE_VERTICES
        )
        for face in _CUBE_FACES:
            loop = tuple(local[index] for index in face)
            key = tuple(sorted(loop))
            previous = face_owner.pop(key, None)
            if previous is None:
                face_owner[key] = (loop, region)
            elif previous[1] != region:
                previous_loop, previous_region = previous
                if previous_region == 0:
                    oriented = tuple(reversed(previous_loop))
                else:
                    oriented = previous_loop
                interfaces.append((oriented, (0, 1)))
    polygons: list[tuple[int, ...]] = []
    region_sides: list[tuple[int, int]] = []
    for loop, region in face_owner.values():
        polygons.append(loop)
        region_sides.append((-1, region))
    for loop, sides in interfaces:
        polygons.append(loop)
        region_sides.append(sides)
    identifiers = np.arange(len(polygons), dtype=np.int64)
    return PiecewiseLinearComplex(
        np.asarray(points, dtype=np.float64),
        tuple(np.asarray(loop, dtype=np.int64) for loop in polygons),
        identifiers,
        np.asarray(region_sides, dtype=np.int64),
        ("base", "upper"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"material-l-vertex-{row}" for row in range(len(points))),
        vertex_occurrence_paths=((),) * len(points),
        facet_ids=tuple(f"material-l-facet-{row}" for row in range(len(polygons))),
        facet_occurrence_paths=((),) * len(polygons),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
    )


def _regeneration_request() -> PolyhedralAdaptationRequest:
    complex_ = _plc_material_l()
    source = NativePolyhedralSource(
        complex_,
        "polyhedral-adaptation-material-l",
        "successor-sites",
        sites=np.asarray(
            ((0.4, 0.5, 0.5), (1.4, 0.5, 0.5), (0.5, 1.5, 0.5)),
            dtype=np.float64,
        ),
        weights=np.zeros((3,), dtype=np.float64),
        site_ids=("base-west", "base-east", "upper-site"),
    )
    scope = MeshingScope.whole(source, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("polyhedron",)),
        ),
        scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
        region_seeds=(
            RegionSeed(
                np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
                "base",
                "base-material",
                RegionRole.SOLID,
            ),
            RegionSeed(
                np.asarray((0.5, 1.5, 0.5), dtype=np.float64),
                "upper",
                "upper-material",
                RegionRole.SOLID,
            ),
        ),
    )
    return PolyhedralAdaptationRequest(
        PolyhedralAdaptationOperation.REGENERATE,
        regeneration_source=source,
        volume_specification=specification,
        schedule=NativePolyhedralSchedule(require_every_site=True),
    )


def test_material_l_plane_split_closes_faces_and_preserves_regions() -> None:
    source, classes = _material_l_mesh()
    outcome = adapt_polyhedral_mesh(
        source,
        PolyhedralAdaptationRequest(
            PolyhedralAdaptationOperation.PLANE_SPLIT,
            planes=np.asarray(((0.0, 1.0, 0.0, 0.5),), dtype=np.float64),
        ),
        source_geometry=CellGeometrySpec.affine(source),
        cell_classes=classes,
    )

    target = outcome.target_mesh
    connectivity = target.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    assert connectivity.cell_count == 5
    signatures = _shared_face_signatures(target)
    assert len(signatures) == int(
        np.count_nonzero(np.asarray(connectivity.face_neighbor) >= 0)
    )
    assert all(
        value > 0.0 for value in np.asarray(outcome.conservative_remap.target_volumes)
    )
    assert sum(
        _cell_box_measure(target, int(identifier))
        for identifier in outcome.hierarchy.target_cell_ids
    ) == pytest.approx(_l_volume_oracle(), abs=1e-13)
    assert _material_interface_area(outcome) == pytest.approx(1.0, abs=1e-13)
    assert set(_cell_classes_by_id(outcome).values()) == {0, 1}
    assert outcome.edit.shared_faces is not None
    assert outcome.vertex_stencil.preserves_constants


def test_material_l_same_region_agglomeration_keeps_material_interface() -> None:
    source, classes = _material_l_mesh()
    outcome = adapt_polyhedral_mesh(
        source,
        PolyhedralAdaptationRequest(
            PolyhedralAdaptationOperation.AGGLOMERATE,
            agglomerations=((10, 20),),
        ),
        source_geometry=CellGeometrySpec.affine(source),
        cell_classes=classes,
    )

    connectivity = outcome.target_mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    assert connectivity.cell_count == 2
    assert sum(
        _cell_box_measure(outcome.target_mesh, int(identifier))
        for identifier in outcome.hierarchy.target_cell_ids
    ) == pytest.approx(_l_volume_oracle(), abs=1e-13)
    assert _material_interface_area(outcome) == pytest.approx(1.0, abs=1e-13)
    assert bool(np.asarray(outcome.conservative_remap.report.coverage_complete))
    merged = int(np.setdiff1d(outcome.hierarchy.target_cell_ids, (30,))[0])
    hierarchy_row = int(np.flatnonzero(outcome.hierarchy.target_cell_ids == merged)[0])
    start, stop = outcome.hierarchy.parent_offsets[hierarchy_row : hierarchy_row + 2]
    np.testing.assert_array_equal(outcome.hierarchy.parent_cell_ids[start:stop], (10, 20))
    relation = outcome.lineage.entity_lineage(3)
    assert int(EntityLineageKind.MERGED_INTO) in np.asarray(relation.relation_kinds)


def test_material_l_site_regeneration_uses_physical_conservative_overlap() -> None:
    source, classes = _material_l_mesh()
    outcome = adapt_polyhedral_mesh(
        source,
        _regeneration_request(),
        source_geometry=CellGeometrySpec.affine(source),
        cell_classes=classes,
    )

    assert outcome.evidence.construction_id is not None
    assert outcome.evidence.site_ids == ("base-west", "base-east", "upper-site")
    assert outcome.vertex_stencil.preserves_constants is False
    assert not np.any(np.asarray(outcome.vertex_stencil.valid))
    assert outcome.common_refinement.succeeded
    assert np.max(np.diff(np.asarray(outcome.common_refinement.target_offsets))) >= 2
    assert outcome.common_refinement.evidence.source_gap_count == 0
    assert outcome.common_refinement.evidence.target_gap_count == 0
    assert bool(np.asarray(outcome.conservative_remap.report.coverage_complete))
    assert float(
        np.sum(np.asarray(outcome.conservative_remap.target_volumes))
    ) == pytest.approx(_l_volume_oracle(), abs=1e-9)
    assert _material_interface_area(outcome) == pytest.approx(1.0, abs=1e-9)

    source_connectivity = source.connectivity
    assert isinstance(source_connectivity, PolyhedralConnectivity)
    source_cell_ids = np.asarray(source_connectivity.cell_global_ids, dtype=np.int64)
    source_average_by_id = {10: 2.0, 20: 5.0, 30: 11.0}
    source_averages = np.asarray(
        [source_average_by_id[int(identifier)] for identifier in source_cell_ids],
        dtype=np.float64,
    )
    target_averages = np.asarray(outcome.conservative_remap.apply(source_averages))
    transferred_content = float(
        np.dot(
            target_averages,
            np.asarray(outcome.conservative_remap.target_volumes, dtype=np.float64),
        )
    )
    assert transferred_content == pytest.approx(18.0, abs=1e-10)
    assert sum(source_average_by_id.values()) == pytest.approx(18.0)


def test_native_polyhedral_route_publishes_conservative_plane_split() -> None:
    source, _ = _material_l_mesh()
    source_geometry = CellGeometrySpec.affine(source)
    source_result = certify_cell_mesh(
        source,
        SpatialCoordinateContract.si(),
        geometry=source_geometry,
    )
    request = PolyhedralAdaptationRequest(
        PolyhedralAdaptationOperation.PLANE_SPLIT,
        planes=np.asarray(((0.0, 1.0, 0.0, 0.5),), dtype=np.float64),
    )
    prepared = prepare_mesh_adaptation(
        source_result,
        request,
        policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_POLYHEDRAL),
    )

    adapted = execute_mesh_adaptation(prepared)

    remap = adapted.conservative_remap
    assert remap is not None
    assert bool(np.asarray(remap.report.coverage_complete))
    target = adapted.target.mesh
    connectivity = target.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    assert sum(
        _cell_box_measure(target, int(identifier))
        for identifier in np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    ) == pytest.approx(_l_volume_oracle(), abs=1e-13)


def test_native_power_regeneration_republishes_site_evidence_and_refuses_user_data() -> (
    None
):
    request = _regeneration_request()
    successor = request.regeneration_source
    specification = request.volume_specification
    assert successor is not None and specification is not None
    initial = NativePolyhedralSource(
        successor.complex,
        successor.source_id,
        "initial-sites",
        sites=np.asarray(
            ((0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (0.5, 1.5, 0.5)),
            dtype=np.float64,
        ),
        weights=np.zeros(3, dtype=np.float64),
        site_ids=("old-west", "old-east", "old-upper"),
    )
    source_spec = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        MeshingScope.whole(initial, 2),
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
        region_seeds=specification.region_seeds,
    )
    source = (
        NativeMeshingProvider(NativeMeshingOptions("plc_restricted_power"))
        .plan(initial, source_spec, coordinate_contract=SpatialCoordinateContract.si())
        .execute()
    )
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_POLYHEDRAL)
    adapted = execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )
    target = adapted.target
    assert target.certification is not None and target.certification.passed
    assert target.certification.inputs.source_revision == successor.source_revision
    assert {a.name for a in target.attributes} == {
        "power_site_index",
        "power_component_index",
        "power_site_parent_index",
        "power_site_weight",
    }
    assert {f"power-site:{site}" for site in successor.site_ids} <= {
        label.name for label in target.labels
    }
    attributes = {attribute.name: attribute for attribute in target.attributes}
    site_attribute = attributes["power_site_index"]
    site_rows = np.asarray(site_attribute.values, dtype=np.int64)
    ids = np.asarray(site_attribute.scope.entity_ids, dtype=np.int64)
    labels = {label.name: label for label in target.labels}
    for index, site in enumerate(successor.site_ids):
        assert np.array_equal(
            np.sort(ids[site_rows == index]),
            np.asarray(labels[f"power-site:{site}"].scope.entity_ids, dtype=np.int64),
        )
    assert np.array_equal(
        np.asarray(attributes["power_site_weight"].values),
        np.asarray(successor.weights)[site_rows],
    )
    assert all(
        a.source_revision == successor.source_revision for a in target.associations
    )
    assert adapted.transition is not None
    assert adapted.transition.source_mesh_id == source.mesh.mesh_id
    assert adapted.transition.target.mesh.mesh_id == target.mesh.mesh_id
    assert adapted.conservative_remap is not None
    assert bool(np.asarray(adapted.conservative_remap.report.coverage_complete))

    extra = MeshAttribute(
        "user_temperature",
        MeshAttributeRole.USER,
        source.attributes[0].scope,
        np.ones(len(source.attributes[0].scope.entity_ids), dtype=np.float64),
    )
    assert source.certification is not None
    audited = certify_cell_mesh(
        source.mesh,
        source.coordinate_contract,
        geometry=source.geometry,
        certification_inputs=source.certification.inputs,
        patches=source.patches,
        zones=source.zones,
        labels=source.labels,
        attributes=(*source.attributes, extra),
        associations=source.associations,
        region_boundary_evidence=source.region_boundary_evidence,
    )
    with_user_data = CellMeshingResult(
        source.mesh,
        source.geometry,
        source.coordinate_contract,
        audited.audit,
        source.quality,
        source.compliance,
        source.trace,
        source.provider,
        source.runtime,
        source.derivative_mode,
        source.provenance,
        patches=source.patches,
        zones=source.zones,
        labels=source.labels,
        attributes=(*source.attributes, extra),
        associations=source.associations,
        region_boundary_evidence=source.region_boundary_evidence,
        certification=audited.certification,
    )
    with pytest.raises(ValueError, match="explicit attribute remap"):
        prepare_mesh_adaptation(with_user_data, request, policy=policy)


def test_auto_power_site_is_the_first_recovered_carrier_centroid() -> None:
    request = _regeneration_request()
    original = request.regeneration_source
    spec = request.volume_specification
    assert original is not None and spec is not None
    source = NativePlcSource(original.complex, original.source_id, "automatic-sites")
    source_spec = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
        region_seeds=spec.region_seeds,
    )
    construction = generate_polyhedral_volume(
        source, source_spec, NativePolyhedralSchedule(),
        frame_id=SpatialCoordinateContract.si().spatial_id,
    )
    carrier = construction.carrier.mesh
    first = np.asarray(carrier.blocks[0].vertices[0], dtype=np.int64)
    expected = np.mean(np.asarray(carrier.coordinates, dtype=np.float64)[first], axis=0)
    np.testing.assert_array_equal(construction.site_points, expected[None, :])
    np.testing.assert_array_equal(construction.site_weights, np.zeros(1, dtype=np.float64))
    assert construction.site_ids == (
        canonical_fingerprint(
            {
                "kind": "plc-first-carrier-tetrahedron-centroid-site",
                "source_binding": source.binding_id,
                "carrier_mesh": carrier.mesh_id,
                "carrier_cell_id": int(carrier.blocks[0].global_ids[0]),
                "coordinates": array_tree_fingerprint(expected[None, :]),
            }
        ),
    )
