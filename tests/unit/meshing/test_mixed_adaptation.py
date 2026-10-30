from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._cell_geometry import CellGeometryElement, CellGeometrySpec
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.meshing._mixed_adaptation import (
    MixedAdaptationHierarchy,
    MixedAdaptationOutcome,
    MixedLayerColumns,
    adapt_mixed_mesh,
)
from phydrax.meshing._adaptation import (
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    execute_mesh_adaptation,
    prepare_mesh_adaptation,
)
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._topology_edit import assemble_topology_edit

from phydrax.discretization.fem import lagrange_element

pytestmark = pytest.mark.meshcore


@pytest.fixture
def tet_prism_mesh() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, -1.0),
            (1.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    return CellMesh(
        points,
        (
            CellBlock("tetrahedra", "tetrahedron", np.asarray(((0, 1, 2, 3),), dtype=np.int32), global_ids=np.asarray((101,), dtype=np.int64)),
            CellBlock("prisms", "prism", np.asarray(((0, 2, 1, 4, 5, 6),), dtype=np.int32), global_ids=np.asarray((202,), dtype=np.int64)),
        ),
        vertex_global_ids=np.arange(700, 707, dtype=np.int64),
    )


@pytest.fixture
def prism_pyramid_mesh() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.5, -0.5, 0.5),
        ),
        dtype=np.float64,
    )
    return CellMesh(
        points,
        (
            CellBlock("prisms", "prism", np.asarray(((0, 1, 2, 3, 4, 5),), dtype=np.int32), global_ids=np.asarray((303,), dtype=np.int64)),
            CellBlock("pyramids", "pyramid", np.asarray(((0, 1, 4, 3, 6),), dtype=np.int32), global_ids=np.asarray((404,), dtype=np.int64)),
        ),
        vertex_global_ids=np.arange(800, 807, dtype=np.int64),
    )


@pytest.fixture
def hex_pyramid_mesh() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.5, 0.5, 2.0),
        ),
        dtype=np.float64,
    )
    return CellMesh(
        points,
        (
            CellBlock("hexahedra", "hexahedron", np.asarray((tuple(range(8)),), dtype=np.int32), global_ids=np.asarray((505,), dtype=np.int64)),
            CellBlock("pyramids", "pyramid", np.asarray(((4, 5, 6, 7, 8),), dtype=np.int32), global_ids=np.asarray((606,), dtype=np.int64)),
        ),
        vertex_global_ids=np.arange(900, 909, dtype=np.int64),
    )


def _determinant_volume(vertices: np.ndarray) -> float:
    return abs(float(np.linalg.det((vertices[1:] - vertices[0]).T))) / 6.0


def _cell_volume(kind: str, vertices: np.ndarray) -> float:
    if kind == "tetrahedron":
        return _determinant_volume(vertices)
    if kind == "prism":
        return sum(
            _determinant_volume(vertices[np.asarray(local, dtype=np.int64)])
            for local in ((0, 1, 2, 3), (1, 2, 4, 3), (2, 4, 5, 3))
        )
    if kind == "pyramid":
        return sum(
            _determinant_volume(vertices[np.asarray(local, dtype=np.int64)])
            for local in ((0, 1, 2, 4), (0, 2, 3, 4))
        )
    if kind == "hexahedron":
        return sum(
            _determinant_volume(vertices[np.asarray(local, dtype=np.int64)])
            for local in (
                (0, 1, 2, 6),
                (0, 2, 3, 6),
                (0, 3, 7, 6),
                (0, 7, 4, 6),
                (0, 4, 5, 6),
                (0, 5, 1, 6),
            )
        )
    raise ValueError(f"Unexpected family {kind!r}.")


def _cell_records(mesh: CellMesh) -> dict[int, tuple[str, np.ndarray]]:
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    records: dict[int, tuple[str, np.ndarray]] = {}
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int64)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        for identifier, cell in zip(identifiers, cells, strict=True):
            records[int(identifier)] = (block.cell_kind, coordinates[cell])
    return records


def _cell_measures(mesh: CellMesh) -> dict[int, float]:
    return {
        identifier: _cell_volume(kind, vertices)
        for identifier, (kind, vertices) in _cell_records(mesh).items()
    }


def _assert_whole_cell_validity(mesh: CellMesh) -> None:
    validity = certify_cell_geometry_validity(mesh)
    assert validity.invalid_count == 0
    assert validity.unresolved_count == 0
    assert validity.certified_valid_count == mesh.entity_set(3).count


def _interface_signatures(
    mesh: CellMesh, /, *, axis: int, coordinate: float
) -> tuple[dict[str, set[tuple[tuple[float, ...], ...]]], dict[tuple[tuple[float, ...], ...], list[str]]]:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    signatures: dict[str, set[tuple[tuple[float, ...], ...]]] = defaultdict(set)
    occurrences: dict[tuple[tuple[float, ...], ...], list[str]] = defaultdict(list)
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        cells = np.asarray(block.vertices, dtype=np.int64)
        for cell in cells:
            for local_face in topology.entities[2]:
                face_points = points[cell[np.asarray(local_face, dtype=np.int64)]]
                if not np.all(np.isclose(face_points[:, axis], coordinate, rtol=0.0, atol=1.0e-12)):
                    continue
                signature = tuple(sorted(tuple(float(value) for value in point) for point in face_points))
                signatures[block.cell_kind].add(signature)
                occurrences[signature].append(block.cell_kind)
    return signatures, occurrences


def _assert_interface_matches(
    mesh: CellMesh,
    first_kind: str,
    second_kind: str,
    /,
    *,
    axis: int,
    coordinate: float,
) -> None:
    signatures, occurrences = _interface_signatures(mesh, axis=axis, coordinate=coordinate)
    assert signatures[first_kind]
    assert signatures[first_kind] == signatures[second_kind]
    for signature in signatures[first_kind]:
        assert sorted(occurrences[signature]) == sorted((first_kind, second_kind))


def _assert_content_transfer(
    source: CellMesh,
    target: CellMesh,
    outcome: MixedAdaptationOutcome,
    source_classes: dict[int, int],
    /,
) -> None:
    transfer = outcome.cell_content_transfer
    source_measures = _cell_measures(source)
    target_measures = _cell_measures(target)
    densities = {cell_class: 1.75 + 0.8 * cell_class for cell_class in set(source_classes.values())}
    source_ids = transfer.source_cell_ids
    source_content = np.asarray(
        [densities[source_classes[int(identifier)]] * source_measures[int(identifier)] for identifier in source_ids],
        dtype=np.float64,
    )
    target_content = transfer.apply(source_ids, source_content)
    target_classes = {cell.cell_id: cell.cell_class for cell in outcome.hierarchy.active_cells}
    for row, identifier in enumerate(transfer.target_cell_ids):
        assert target_content[row] == pytest.approx(
            densities[target_classes[int(identifier)]] * target_measures[int(identifier)],
            rel=1.0e-12,
            abs=1.0e-13,
        )
    for cell_class, density in densities.items():
        source_total = sum(
            density * measure
            for identifier, measure in source_measures.items()
            if source_classes[identifier] == cell_class
        )
        target_total = sum(
            float(target_content[row])
            for row, identifier in enumerate(transfer.target_cell_ids)
            if target_classes[int(identifier)] == cell_class
        )
        assert target_total == pytest.approx(source_total, rel=1.0e-12, abs=1.0e-13)


def _adapt_and_restore(
    mesh: CellMesh,
    refine_id: int,
    classes: np.ndarray,
    /,
    *,
    layer_columns: MixedLayerColumns | None = None,
) -> tuple[MixedAdaptationOutcome, CellMesh, MixedAdaptationOutcome, CellMesh]:
    source_cells = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    class_by_id = dict(zip(source_cells.tolist(), classes.tolist(), strict=True))
    refined = adapt_mixed_mesh(
        mesh,
        refine_cell_ids=np.asarray((refine_id,), dtype=np.int64),
        cell_classes=classes,
        layer_columns=layer_columns,
    )
    fine, _, _ = assemble_topology_edit(
        mesh, refined.edit, numeric_version="mixed-adaptation-refined"
    )
    _assert_whole_cell_validity(mesh)
    _assert_whole_cell_validity(fine)
    _assert_content_transfer(mesh, fine, refined, class_by_id)
    fine_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in fine.blocks]
    )
    coarsened = adapt_mixed_mesh(
        fine,
        coarsen_cell_ids=fine_ids,
        hierarchy=refined.hierarchy,
    )
    restored, _, _ = assemble_topology_edit(
        fine, coarsened.edit, numeric_version="mixed-adaptation-restored"
    )
    _assert_whole_cell_validity(restored)
    fine_classes = {
        cell.cell_id: cell.cell_class for cell in refined.hierarchy.active_cells
    }
    _assert_content_transfer(fine, restored, coarsened, fine_classes)
    return refined, fine, coarsened, restored


def test_tetrahedron_prism_face_closure_and_conservative_coarsening(
    tet_prism_mesh: CellMesh,
) -> None:
    refined, fine, coarsened, restored = _adapt_and_restore(
        tet_prism_mesh,
        101,
        np.asarray((4, 9), dtype=np.int64),
    )
    assert set(refined.evidence.closure_cell_ids.tolist()) == {202}
    _assert_interface_matches(
        fine, "tetrahedron", "prism", axis=2, coordinate=0.0
    )
    assert sorted(refined.evidence.refined_cell_ids.tolist()) == [101]
    assert sorted(coarsened.evidence.coarsened_cell_ids.tolist()) == sorted(
        np.concatenate([np.asarray(block.global_ids) for block in fine.blocks]).tolist()
    )
    assert {block.cell_kind for block in restored.blocks} == {"tetrahedron", "prism"}
    assert _cell_measures(restored) == pytest.approx(_cell_measures(tet_prism_mesh))
    assert {cell.cell_class for cell in coarsened.hierarchy.active_cells} == {4, 9}
    coarsening_map_children = {
        int(child)
        for record in coarsened.coarsening_maps
        for child in np.asarray(record.child_cell_ids, dtype=np.int64)
    }
    assert coarsening_map_children == set(coarsened.evidence.coarsened_cell_ids.tolist())
    for record in coarsened.coarsening_maps:
        assert len(record.child_reference_vertices) == len(record.child_cell_ids)
        assert all(
            len(point) == 3
            for child in record.child_reference_vertices
            for point in child
        )
        assert all(
            isinstance(value, Fraction)
            for child in record.child_reference_vertices
            for point in child
            for value in point
        )


def test_prism_pyramid_face_closure_and_conservative_coarsening(
    prism_pyramid_mesh: CellMesh,
) -> None:
    refined, fine, coarsened, restored = _adapt_and_restore(
        prism_pyramid_mesh,
        303,
        np.asarray((3, 8), dtype=np.int64),
    )
    assert set(refined.evidence.closure_cell_ids.tolist()) == {404}
    _assert_interface_matches(
        fine, "prism", "pyramid", axis=1, coordinate=0.0
    )
    assert sorted(refined.evidence.refined_cell_ids.tolist()) == [303]
    assert not coarsened.evidence.rejected_coarsening_ids.size
    assert {block.cell_kind for block in restored.blocks} == {"prism", "pyramid"}
    assert _cell_measures(restored) == pytest.approx(_cell_measures(prism_pyramid_mesh))
    assert {cell.cell_class for cell in coarsened.hierarchy.active_cells} == {3, 8}
    assert {record.parent_cell_kind for record in coarsened.coarsening_maps} == {
        "prism",
        "pyramid",
    }


def test_hex_pyramid_layers_keep_first_interval_and_region_interface(
    hex_pyramid_mesh: CellMesh,
) -> None:
    columns = MixedLayerColumns(
        np.asarray((505,), dtype=np.int64),
        np.asarray((77,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        hard_first_thickness=True,
        axial_refinement=False,
    )
    refined, fine, coarsened, restored = _adapt_and_restore(
        hex_pyramid_mesh,
        505,
        np.asarray((5, 12), dtype=np.int64),
        layer_columns=columns,
    )
    assert set(refined.evidence.closure_cell_ids.tolist()) == {606}
    assert not refined.evidence.schedule_changed_cell_ids.size
    _assert_interface_matches(
        fine, "hexahedron", "pyramid", axis=2, coordinate=1.0
    )
    fine_layer_cells = [
        cell for cell in refined.hierarchy.active_cells if cell.column_id == 77
    ]
    assert len(fine_layer_cells) == 4
    fine_records = _cell_records(fine)
    for cell in fine_layer_cells:
        vertices = fine_records[cell.cell_id][1]
        assert np.min(vertices[:, 2]) == pytest.approx(0.0)
        assert np.max(vertices[:, 2]) == pytest.approx(1.0)
        assert cell.interval_index == 0
    assert {block.cell_kind for block in restored.blocks} == {"hexahedron", "pyramid"}
    restored_layer = [
        cell for cell in coarsened.hierarchy.active_cells if cell.column_id == 77
    ]
    assert [(cell.cell_id, cell.interval_index) for cell in restored_layer] == [(505, 0)]
    assert _cell_measures(restored) == pytest.approx(_cell_measures(hex_pyramid_mesh))
    assert {record.parent_cell_kind for record in coarsened.coarsening_maps} == {
        "hexahedron",
        "pyramid",
    }

def _curved_hex_pyramid_geometry(mesh: CellMesh) -> CellGeometrySpec:
    elements: dict[str, CellGeometryElement] = {}
    routes: dict[str, np.ndarray] = {}
    coefficient_rows: list[np.ndarray] = []
    cursor = 0
    for block in mesh.blocks:
        degree = 2 if block.cell_kind == "hexahedron" else 1
        element = lagrange_element(block.cell_kind, degree)
        reference = np.asarray(element.reference_nodes, dtype=np.float64)
        x, y, z = reference.T
        if block.cell_kind == "hexahedron":
            coordinates = np.column_stack((x, y + 0.05 * x * z * (1.0 - z), z))
        else:
            coordinates = np.column_stack((x, y, z + 1.0))
        elements[block.name] = element
        routes[block.name] = np.arange(
            cursor, cursor + reference.shape[0], dtype=np.int32
        )[None, :]
        coefficient_rows.append(coordinates)
        cursor += reference.shape[0]
    return CellGeometrySpec(elements, routes, np.concatenate(coefficient_rows))


def _hex_reference_image(
    corners: tuple[tuple[Fraction, ...], ...],
    point: tuple[Fraction, ...],
    /,
) -> tuple[Fraction, ...]:
    weights = tuple(
        (point[0] if corner[0] else 1 - point[0])
        * (point[1] if corner[1] else 1 - point[1])
        * (point[2] if corner[2] else 1 - point[2])
        for corner in reference_cell_topology("hexahedron").vertices
    )
    return tuple(
        sum(
            (weights[vertex] * corners[vertex][axis] for vertex in range(8)),
            Fraction(0),
        )
        for axis in range(3)
    )


def test_high_order_source_geometry_restricts_and_restores_mixed_parents(
    hex_pyramid_mesh: CellMesh,
) -> None:
    source_geometry = _curved_hex_pyramid_geometry(hex_pyramid_mesh)
    source_validity = certify_cell_geometry_validity(
        source_geometry, mesh=hex_pyramid_mesh
    )
    assert source_validity.invalid_count == 0
    assert source_validity.unresolved_count == 0

    refined = adapt_mixed_mesh(
        hex_pyramid_mesh,
        refine_cell_ids=np.asarray((505,), dtype=np.int64),
        cell_classes=np.asarray((2, 6), dtype=np.int64),
        source_geometry=source_geometry,
    )
    assert refined.target_geometry is not None
    fine, _, _ = assemble_topology_edit(
        hex_pyramid_mesh, refined.edit, numeric_version="mixed-curved-refinement"
    )
    fine_validity = certify_cell_geometry_validity(
        refined.target_geometry, mesh=fine
    )
    assert fine_validity.invalid_count == 0
    assert fine_validity.unresolved_count == 0
    hex_sibling = next(
        record
        for record in refined.hierarchy.siblings
        if record.parent_id == 505
    )
    parent_ids = np.full(
        (len(hex_sibling.child_ids),), hex_sibling.parent_id, dtype=np.int64
    )
    source_corner_images = source_geometry.evaluate_exact(
        hex_pyramid_mesh,
        parent_ids,
        hex_sibling.child_reference_vertices,
    )
    local_corners = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology("hexahedron").vertices
    )
    fine_corner_images = refined.target_geometry.evaluate_exact(
        fine,
        np.asarray(hex_sibling.child_ids, dtype=np.int64),
        tuple((local_corners,) for _ in hex_sibling.child_ids),
    )
    assert fine_corner_images == source_corner_images

    center = (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))
    parent_centers = tuple(
        (_hex_reference_image(child, center),)
        for child in hex_sibling.child_reference_vertices
    )
    source_center_images = source_geometry.evaluate_exact(
        hex_pyramid_mesh, parent_ids, parent_centers
    )
    fine_center_images = refined.target_geometry.evaluate_exact(
        fine,
        np.asarray(hex_sibling.child_ids, dtype=np.int64),
        tuple((center,) for _ in hex_sibling.child_ids),
    )
    assert fine_center_images == source_center_images
    assert any(
        float(image[0][1]) != pytest.approx(float(parent[1]))
        for image, (parent,) in zip(
            source_center_images, parent_centers, strict=True
        )
    )

    fine_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in fine.blocks]
    )
    coarsened = adapt_mixed_mesh(
        fine,
        coarsen_cell_ids=fine_ids,
        hierarchy=refined.hierarchy,
        source_geometry=refined.target_geometry,
    )
    assert coarsened.target_geometry is not None
    restored, _, _ = assemble_topology_edit(
        fine, coarsened.edit, numeric_version="mixed-curved-coarsening"
    )
    restored_validity = certify_cell_geometry_validity(
        coarsened.target_geometry, mesh=restored
    )
    assert restored_validity.invalid_count == 0
    assert restored_validity.unresolved_count == 0
    sample_points = {
        "hexahedron": (center,),
        "pyramid": ((Fraction(1, 2), Fraction(1, 2), Fraction(1, 4)),),
    }
    for cell_id, (kind, _) in _cell_records(hex_pyramid_mesh).items():
        reference = sample_points[kind]
        expected = source_geometry.evaluate_exact(
            hex_pyramid_mesh,
            np.asarray((cell_id,), dtype=np.int64),
            (reference,),
        )
        actual = coarsened.target_geometry.evaluate_exact(
            restored,
            np.asarray((cell_id,), dtype=np.int64),
            (reference,),
        )
        assert actual == expected


def test_native_mixed_route_preserves_high_order_geometry_through_coarsening(
    hex_pyramid_mesh: CellMesh,
) -> None:
    source_geometry = _curved_hex_pyramid_geometry(hex_pyramid_mesh)
    source = certify_cell_mesh(
        hex_pyramid_mesh,
        SpatialCoordinateContract.si(),
        geometry=source_geometry,
    )
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_MIXED)
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(np.asarray((505,), dtype=np.int64)),
            policy=policy,
        )
    )
    fine = refined.target
    hierarchy = refined.hierarchy
    assert hierarchy is not None
    assert isinstance(hierarchy, MixedAdaptationHierarchy)
    sibling = next(record for record in hierarchy.siblings if record.parent_id == 505)
    center = (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))
    parent_ids = np.full(len(sibling.child_ids), 505, dtype=np.int64)
    expected_fine = source_geometry.evaluate_exact(
        hex_pyramid_mesh,
        parent_ids,
        tuple(
            (_hex_reference_image(child, center),)
            for child in sibling.child_reference_vertices
        ),
    )
    actual_fine = fine.geometry.evaluate_exact(
        fine.mesh,
        np.asarray(sibling.child_ids, dtype=np.int64),
        tuple((center,) for _ in sibling.child_ids),
    )
    assert actual_fine == expected_fine

    fine_cell_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64)
            for block in fine.mesh.blocks
        )
    )
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            fine,
            MarkedMeshAdaptation(
                coarsen_cell_ids=fine_cell_ids,
                hierarchy=hierarchy,
            ),
            policy=policy,
        )
    )
    restored_geometry = coarsened.target.geometry
    assert coarsened.cell_content_transfer is not None
    assert coarsened.target.certification is not None
    coarsened.target.certification.require_passed()
    for cell_id, (kind, _) in _cell_records(hex_pyramid_mesh).items():
        reference = (
            center
            if kind == "hexahedron"
            else (Fraction(1, 2), Fraction(1, 2), Fraction(1, 4))
        )
        expected = source_geometry.evaluate_exact(
            hex_pyramid_mesh,
            np.asarray((cell_id,), dtype=np.int64),
            (reference,),
        )
        actual = restored_geometry.evaluate_exact(
            coarsened.target.mesh,
            np.asarray((cell_id,), dtype=np.int64),
            (reference,),
        )
        assert actual == expected
