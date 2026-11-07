from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import ParamSpec, TypeVar

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import CellGeometryElement, CellGeometrySpec
from phydrax.discretization._cell_geometry_transition_types import CellGeometryTransition
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from phydrax.discretization._nested_reference_types import ReferenceCellPartition
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import PeriodicIsometryGroup
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import lagrange_element
from phydrax.meshing import MeshAdaptationResult
from phydrax.meshing._adaptation import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._layer_core_topology import mesh_entity_scope
from phydrax.meshing._marker_transfer import transfer_cell_markers, validate_cell_markers
from phydrax.meshing._mixed_adaptation import (
    _point_in_root_hull,
    adapt_mixed_mesh,
    MixedAdaptationHierarchy,
    MixedAdaptationOutcome,
    MixedLayerColumns,
    MixedSibling,
)
from phydrax.meshing._mixed_hierarchy_reproof import (
    require_rooted_mixed_hierarchy_geometry,
)
from phydrax.meshing._organization import MeshAttribute, MeshAttributeRole
from phydrax.meshing._periodic import (
    periodic_entity_orbit_map,
    require_periodic_geometry_face_maps,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._topology_edit import assemble_topology_edit, entity_keys
from tests._support.validity_reuse import (
    assert_incremental_matches_full,
    cell_map_digests,
)


pytestmark = pytest.mark.meshcore

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _record_replays(
    function: Callable[_P, _T], records: list[tuple[Callable[[], _T], _T]]
) -> Callable[_P, _T]:
    def record(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        result = function(*args, **kwargs)
        records.append((lambda: function(*args, **kwargs), result))
        return result

    return record


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
            CellBlock(
                "tetrahedra",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((101,), dtype=np.int64),
            ),
            CellBlock(
                "prisms",
                "prism",
                np.asarray(((0, 2, 1, 4, 5, 6),), dtype=np.int32),
                global_ids=np.asarray((202,), dtype=np.int64),
            ),
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
            CellBlock(
                "prisms",
                "prism",
                np.asarray(((0, 1, 2, 3, 4, 5),), dtype=np.int32),
                global_ids=np.asarray((303,), dtype=np.int64),
            ),
            CellBlock(
                "pyramids",
                "pyramid",
                np.asarray(((0, 1, 4, 3, 6),), dtype=np.int32),
                global_ids=np.asarray((404,), dtype=np.int64),
            ),
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
            CellBlock(
                "hexahedra",
                "hexahedron",
                np.asarray((tuple(range(8)),), dtype=np.int32),
                global_ids=np.asarray((505,), dtype=np.int64),
            ),
            CellBlock(
                "pyramids",
                "pyramid",
                np.asarray(((4, 5, 6, 7, 8),), dtype=np.int32),
                global_ids=np.asarray((606,), dtype=np.int64),
            ),
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
) -> tuple[
    dict[str, set[tuple[tuple[float, ...], ...]]],
    dict[tuple[tuple[float, ...], ...], list[str]],
]:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    signatures: dict[str, set[tuple[tuple[float, ...], ...]]] = defaultdict(set)
    occurrences: dict[tuple[tuple[float, ...], ...], list[str]] = defaultdict(list)
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        cells = np.asarray(block.vertices, dtype=np.int64)
        for cell in cells:
            for local_face in topology.entities[2]:
                face_points = points[cell[np.asarray(local_face, dtype=np.int64)]]
                if not np.all(
                    np.isclose(face_points[:, axis], coordinate, rtol=0.0, atol=1.0e-12)
                ):
                    continue
                signature = tuple(
                    sorted(
                        tuple(float(value) for value in point) for point in face_points
                    )
                )
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
    signatures, occurrences = _interface_signatures(
        mesh, axis=axis, coordinate=coordinate
    )
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
    densities = {
        cell_class: 1.75 + 0.8 * cell_class for cell_class in set(source_classes.values())
    }
    source_ids = transfer.source_cell_ids
    source_content = np.asarray(
        [
            densities[source_classes[int(identifier)]] * source_measures[int(identifier)]
            for identifier in source_ids
        ],
        dtype=np.float64,
    )
    target_content = transfer.apply(source_ids, source_content)
    target_classes = {
        cell.cell_id: cell.cell_class for cell in outcome.hierarchy.active_cells
    }
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


def test_rooted_mixed_hierarchy_reproof_rejects_stored_mutations(
    tet_prism_mesh: CellMesh,
) -> None:
    root = tet_prism_mesh
    outcome = adapt_mixed_mesh(
        root,
        refine_cell_ids=np.asarray((101,), dtype=np.int64),
        cell_classes=np.asarray((4, 9), dtype=np.int64),
    )
    target, _, _ = assemble_topology_edit(
        root, outcome.edit, numeric_version="hierarchy-proof"
    )
    partition = ReferenceCellPartition(outcome.reference_maps)
    hierarchy = outcome.hierarchy

    def reprove(value: MixedAdaptationHierarchy) -> None:
        require_rooted_mixed_hierarchy_geometry(
            root,
            target,
            partition,
            value,
            root_classes={101: 4, 202: 9},
            root_layers={},
        )

    reprove(hierarchy)
    bad_template = copy.copy(hierarchy)
    sibling = hierarchy.siblings[0]
    altered_chart = list(sibling.child_reference_vertices)
    first = list(altered_chart[0])
    first[0] = (Fraction(17, 19), *first[0][1:])
    altered_chart[0] = tuple(first)
    object.__setattr__(
        bad_template,
        "siblings",
        (
            sibling._replace(child_reference_vertices=tuple(altered_chart)),
            *hierarchy.siblings[1:],
        ),
    )
    with pytest.raises(ValueError):
        reprove(bad_template)
    bad_point = copy.copy(hierarchy)
    points = list(hierarchy.vertex_points)
    points[0] = (points[0][0], ((999999, Fraction(1)),))
    object.__setattr__(bad_point, "vertex_points", tuple(points))
    with pytest.raises(ValueError, match="stencil"):
        reprove(bad_point)
    bad_allocator = copy.copy(hierarchy)
    object.__setattr__(bad_allocator, "next_cell_id", 0)
    with pytest.raises(ValueError, match="allocator"):
        reprove(bad_allocator)


def test_mixed_cell_markers_refine_restore_and_reject_conflicts(
    tet_prism_mesh: CellMesh,
) -> None:
    mesh = tet_prism_mesh
    ids = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    scope = mesh_entity_scope(mesh, 3, ids)
    markers = (
        MeshAttribute(
            "layer_index", MeshAttributeRole.MARKER, scope, np.asarray((-1, 0))
        ),
        MeshAttribute(
            "layer_column", MeshAttributeRole.MARKER, scope, np.asarray((-1, 23))
        ),
    )
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=np.asarray((101,)))
    fine, forward, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="marker-fine"
    )
    fine_markers = transfer_cell_markers(mesh, markers, forward, fine)
    fine_ids = np.asarray(fine.entity_set(3).entity_ids, dtype=np.int64)
    restored_outcome = adapt_mixed_mesh(
        fine, coarsen_cell_ids=fine_ids, hierarchy=outcome.hierarchy
    )
    restored, reverse, _ = assemble_topology_edit(
        fine, restored_outcome.edit, numeric_version="marker-restored"
    )
    restored_markers = transfer_cell_markers(fine, fine_markers, reverse, restored)
    for original, actual in zip(markers, restored_markers, strict=True):
        np.testing.assert_array_equal(actual.scope.entity_ids, original.scope.entity_ids)
        np.testing.assert_array_equal(actual.values, original.values)

    record = reverse.entity_lineage(3)
    targets = np.asarray(record.target_global_ids, dtype=np.int64)
    parents = np.asarray(record.source_global_ids, dtype=np.int64)
    target = next(
        identifier
        for identifier in np.unique(targets)
        if np.count_nonzero(targets == identifier) > 1
    )
    parent = int(parents[np.flatnonzero(targets == target)[0]])
    bad_values = np.asarray(fine_markers[0].values).copy()
    bad_values[
        np.flatnonzero(np.asarray(fine_markers[0].scope.entity_ids) == parent)[0]
    ] = 99
    bad = MeshAttribute(
        "material", MeshAttributeRole.MARKER, fine_markers[0].scope, bad_values
    )
    with pytest.raises(ValueError, match="conflicts"):
        transfer_cell_markers(fine, (bad,), reverse, restored)


def test_mixed_cell_markers_reject_unknown_semantics(tet_prism_mesh: CellMesh) -> None:
    ids = np.sort(np.asarray(tet_prism_mesh.entity_set(3).entity_ids, dtype=np.int64))
    scope = mesh_entity_scope(tet_prism_mesh, 3, ids)
    attribute = MeshAttribute(
        "temperature", MeshAttributeRole.USER, scope, np.asarray((1.0, 2.0))
    )
    with pytest.raises(ValueError, match="MARKER"):
        validate_cell_markers(tet_prism_mesh, (attribute,))


def test_tetrahedron_prism_face_closure_and_conservative_coarsening(
    tet_prism_mesh: CellMesh,
) -> None:
    refined, fine, coarsened, restored = _adapt_and_restore(
        tet_prism_mesh,
        101,
        np.asarray((4, 9), dtype=np.int64),
    )
    assert set(refined.evidence.closure_cell_ids.tolist()) == {202}
    _assert_interface_matches(fine, "tetrahedron", "prism", axis=2, coordinate=0.0)
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


def test_mixed_postcertification_owner_replays_actual_phase(
    tet_prism_mesh: CellMesh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry._mesh_certificates import _same_tree_content
    from phydrax.meshing import _adaptation

    source = certify_cell_mesh(tet_prism_mesh, SpatialCoordinateContract.si())
    prepared = prepare_mesh_adaptation(
        source,
        MarkedMeshAdaptation(np.asarray((101,), dtype=np.int64)),
        policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_MIXED),
    )
    captured: list[
        tuple[Callable[[], _adaptation._NativeTarget], _adaptation._NativeTarget]
    ] = []
    original = _adaptation._finish_certified_native_target
    monkeypatch.setattr(
        _adaptation,
        "_finish_certified_native_target",
        _record_replays(original, captured),
    )
    actual = execute_mesh_adaptation(prepared)
    assert actual.transition is not None and len(captured) == 1
    replay, first = captured[0]
    replayed = replay()
    assert _same_tree_content(first, replayed)
    assert replayed.transition.transition_id == actual.transition.transition_id


def test_local_template_point_cache_preserves_exact_facts(
    tet_prism_mesh: CellMesh,
    hex_pyramid_mesh: CellMesh,
) -> None:
    from phydrax.meshing._mixed_adaptation import _Cell, _template_facts, _templates

    cache = {}
    for mesh in (tet_prism_mesh, hex_pyramid_mesh):
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        for block in mesh.blocks:
            vertices = vertex_ids[np.asarray(block.vertices, dtype=np.int64)[0]]
            for actual in (vertices, vertices[::-1].copy()):
                cell = _Cell(1, block.name, block.cell_kind, actual, 0, None, None)
                for axial in (False, True):
                    for template in _templates(block.cell_kind, axial):
                        assert _template_facts(
                            cell, template, point_cache=cache
                        ) == _template_facts(cell, template)


def test_periodic_witness_rebind_survives_nonidentity_vertex_reordering() -> None:
    from phydrax.discretization._periodic_exact import exact_periodic_element
    from phydrax.meshing._topology_edit import _rebind_periodic_topology
    from tests.unit.meshing.test_periodic_mesh_ownership import _reordered_periodic_mesh

    original = _reordered_periodic_mesh()
    topology = original.periodic_topology
    assert topology is not None
    ids = np.asarray(original.vertex_global_ids, dtype=np.int64)
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    points = np.asarray(original.coordinates, dtype=np.float64)
    exact_rows = []
    for row, root in enumerate(representatives):
        matrix = exact_periodic_element(topology.cell, shifts[row])
        root_point = tuple(Fraction(float(value)) for value in points[int(root)])
        exact_rows.append(
            tuple(
                sum(
                    (
                        matrix[axis][column] * (*root_point, Fraction(1))[column]
                        for column in range(3)
                    ),
                    Fraction(0),
                )
                for axis in range(2)
            )
        )
    assert np.array_equal(np.asarray(exact_rows, dtype=np.float64), points)
    bank = ExactPlcCellGeometrySource(
        "exact-reorder",
        original.mesh_id,
        ids,
        tuple(exact_rows),
        vertex_source_entity_ids=(None,) * ids.size,
        vertex_occurrence_paths=((),) * ids.size,
    )
    lifted = CellMesh(points, original.blocks, vertex_global_ids=ids)
    old_geometry = topology._actual_geometry
    assert old_geometry is not None
    geometry = CellGeometrySpec(
        dict(zip(old_geometry.block_names, old_geometry.elements, strict=True)),
        dict(zip(old_geometry.block_names, old_geometry.geometry_dofs, strict=True)),
        old_geometry.coordinates,
        source_map=bank,
        source_mesh=lifted,
    )
    witnessed = PeriodicMeshTopology(
        lifted,
        topology.cell,
        representatives,
        shifts,
        actual_geometry=geometry,
        exact_coordinate_source=bank,
    )
    permutation = np.arange(ids.size - 1, -1, -1, dtype=np.int64)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(ids.size, dtype=np.int64)
    blocks = tuple(
        CellBlock(
            block.name,
            block.cell_kind,
            inverse[np.asarray(block.vertices, dtype=np.int64)],
            global_ids=block.global_ids,
        )
        for block in lifted.blocks
    )
    target = CellMesh(points[permutation], blocks, vertex_global_ids=ids[permutation])
    rebound = _rebind_periodic_topology(target, witnessed)
    assert rebound.periodic_topology is not None
    rebound_bank = rebound.periodic_topology.exact_coordinate_source
    assert isinstance(rebound_bank, ExactPlcCellGeometrySource)
    assert rebound_bank.exact_coordinates == tuple(
        exact_rows[int(row)] for row in permutation
    )
    assert rebound_bank.source_map_id != bank.source_map_id
    lifted_geometry = rebound.periodic_topology._actual_geometry
    assert lifted_geometry is not None
    assert lifted_geometry.source_mesh_id != rebound.mesh_id
    current_geometry = CellGeometrySpec(
        dict(zip(lifted_geometry.block_names, lifted_geometry.elements, strict=True)),
        dict(
            zip(lifted_geometry.block_names, lifted_geometry.geometry_dofs, strict=True)
        ),
        lifted_geometry.coordinates,
        source_map=rebound_bank,
        source_mesh=rebound,
    )
    restored = CellMesh.from_stored_fields(rebound, actual_geometry=current_geometry)
    assert restored.periodic_topology is not None
    assert restored.periodic_topology.exact_coordinate_source is not None
    assert (
        restored.periodic_topology.exact_coordinate_source.source_map_id
        == rebound_bank.source_map_id
    )


def test_periodic_exact_plc_witness_reorders_with_source_ancestry() -> None:
    from phydrax.meshing._topology_edit import _reorder_periodic_coordinate_source

    ids = np.asarray((11, 42), dtype=np.int64)
    exact = ((Fraction(1, 3), Fraction(1, 5)), (Fraction(4, 3), Fraction(1, 5)))
    source = ExactPlcCellGeometrySource(
        "actual-source",
        "revision",
        ids,
        exact,
        vertex_source_entity_ids=("first", "second"),
        vertex_occurrence_paths=(("a",), ("b",)),
    )
    permutation = np.asarray((1, 0), dtype=np.int64)
    reordered = _reorder_periodic_coordinate_source(
        source, ids, ids[permutation], permutation
    )
    assert isinstance(reordered, ExactPlcCellGeometrySource)
    assert reordered.source_map_id != source.source_map_id
    assert reordered.exact_coordinates == tuple(exact[int(row)] for row in permutation)
    assert reordered.vertex_source_entity_ids == ("second", "first")
    assert reordered.vertex_occurrence_paths == (("b",), ("a",))
    cell = PeriodicCell(np.eye(2, dtype=np.float64))
    reordered.require_periodic_lift(
        np.asarray(reordered.rounded_vertices),
        cell,
        np.asarray((1, 1)),
        np.asarray(((1, 0), (0, 0))),
    )
    assert (
        ExactPlcCellGeometrySource.from_stored_fields(reordered).source_map_id
        == reordered.source_map_id
    )
    with pytest.raises(ValueError, match="stable vertex identities"):
        _reorder_periodic_coordinate_source(source, ids, ids, permutation)
    with pytest.raises(ValueError, match="different source vertex rows"):
        _reorder_periodic_coordinate_source(
            source, ids[::-1], ids[::-1], np.asarray((0, 1))
        )


def test_retained_coordinate_bank_rebuilds_data_without_source_authority() -> None:
    from phydrax.meshing._mixed_adaptation import _RetainedCoordinateBank

    exact = ((Fraction(1, 3), Fraction(2, 5)), (Fraction(3, 7), Fraction(4, 9)))
    bank = _RetainedCoordinateBank("source", "revision", "provenance-map", exact)
    rebuilt = _RetainedCoordinateBank.from_stored_fields(
        bank, coordinates=np.asarray(exact, dtype=np.float64)
    )
    assert rebuilt.source_coordinates() == exact
    assert rebuilt.source_map_id == "provenance-map"
    with pytest.raises(ValueError, match="execution shadows"):
        _RetainedCoordinateBank.from_stored_fields(bank, coordinates=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="exact scalar bank"):
        _RetainedCoordinateBank("source", "revision", "provenance-map", ((0.25,),))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="exact scalar bank"):
        _RetainedCoordinateBank(
            "source", "revision", "provenance-map", ((Fraction(1),), ())
        )


def test_refined_chart_triangle_preserves_exact_split_source_pieces() -> None:
    from phydrax.meshing._adaptation import (
        _require_disjoint_chart_occurrences,
        _split_refined_chart_triangle,
    )

    zero, one, half = Fraction(0), Fraction(1), Fraction(1, 2)
    a, b, c = (zero, zero, zero), (one, zero, zero), (zero, one, zero)
    middle = (half, half, zero)
    source = (a, b, c)
    old = ((a, b, middle), (a, middle, c))
    parameters = tuple(tuple((point[0], point[1]) for point in row) for row in old)
    pieces = _split_refined_chart_triangle(source, source, (0, 1), old, parameters)
    assert len(pieces) == 2
    assert {piece[0] for piece in pieces} == {0, 1}
    for _, target, uv in pieces:
        assert tuple((point[0], point[1]) for point in target) == uv
        assert all(point[2] == zero for point in target)
    keys: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("chart-a", "face-a", ()),
        ("chart-b", "face-b", ()),
    )
    rows: list[
        tuple[
            tuple[str, str, tuple[str, ...]],
            dict[str, int | tuple[tuple[Fraction, ...], ...]],
        ]
    ] = [
        (keys[index], {"cell_id": 1, "local_entity": 0, "reference": target})
        for index, target, _ in pieces
    ]
    _require_disjoint_chart_occurrences(rows)
    with pytest.raises(ValueError, match="occurrences overlap"):
        _require_disjoint_chart_occurrences([rows[0], (keys[1], rows[0][1])])
    with pytest.raises(ValueError, match="one target cell chart"):
        _require_disjoint_chart_occurrences(
            [rows[0], (keys[1], {**rows[1][1], "cell_id": 2})]
        )
    with pytest.raises(ValueError, match="disjoint exact source-piece cover"):
        _split_refined_chart_triangle(source, source, (0,), old, parameters)
    with pytest.raises(ValueError, match="disjoint exact source-piece cover"):
        _split_refined_chart_triangle(source, source, (0, 0, 1), old, parameters)
    off_plane = tuple((x, y, one) for x, y, _ in old[1])
    with pytest.raises(ValueError, match="disjoint exact source-piece cover"):
        _split_refined_chart_triangle(
            source, source, (0, 1), (old[0], off_plane), parameters
        )


def test_identity_entity_relation_shortcut_requires_exact_local_map_and_kind(
    tet_prism_mesh: CellMesh,
) -> None:
    from phydrax.meshing._mixed_adaptation import _entity_relations, _TargetCell

    block = tet_prism_mesh.blocks[0]
    vertices = tuple(
        int(tet_prism_mesh.vertex_global_ids[int(row)])
        for row in np.asarray(block.vertices)[0]
    )
    corners = tuple(
        tuple(Fraction(value) for value in point)
        for point in reference_cell_topology(block.cell_kind).vertices
    )
    target = _TargetCell(
        101,
        block.name,
        block.cell_kind,
        vertices,
        0,
        None,
        None,
        (101,),
        (Fraction(1),),
        corners,
    )
    identity = _entity_relations(tet_prism_mesh, tet_prism_mesh, [target], {}, ())
    assert identity[1].source_keys.shape[0] == identity[2].source_keys.shape[0] == 0

    permuted = target._replace(reference_vertices=(corners[1], corners[0], *corners[2:]))
    nonidentity = _entity_relations(tet_prism_mesh, tet_prism_mesh, [permuted], {}, ())
    assert nonidentity[1].source_keys.shape[0] > 0
    assert nonidentity[2].source_keys.shape[0] > 0

    wrong_kind = target._replace(kind="simplex:3")
    with pytest.raises(ValueError, match="mismatched source family"):
        _entity_relations(tet_prism_mesh, tet_prism_mesh, [wrong_kind], {}, ())


def test_exact_prism_partition_extrusion_distinguishes_contact_and_overlap() -> None:
    from phydrax.discretization._nested_reference_types import (
        _partition_interiors_disjoint,
    )

    zero, one, quarter = Fraction(0), Fraction(1), Fraction(1, 4)
    base = (
        (zero, zero, zero),
        (one, zero, zero),
        (zero, one, zero),
        (zero, zero, one),
        (one, zero, one),
        (zero, one, one),
    )

    def translated(dx: Fraction, dz: Fraction) -> tuple[tuple[Fraction, ...], ...]:
        return tuple((x + dx, y, z + dz) for x, y, z in base)

    assert _partition_interiors_disjoint("prism", base, "prism", translated(one, zero))
    assert _partition_interiors_disjoint("prism", base, "prism", translated(zero, one))
    assert not _partition_interiors_disjoint(
        "prism", base, "prism", translated(quarter, zero)
    )
    assert not _partition_interiors_disjoint(
        "prism", base, "prism", translated(zero, quarter)
    )
    skewed = tuple(
        (x + (quarter if index >= 3 else zero), y, z)
        for index, (x, y, z) in enumerate(base)
    )
    assert not _partition_interiors_disjoint("prism", base, "prism", skewed)


def test_mixed_canonical_simplex_kind_survives_refine_coarsen(
    tet_prism_mesh: CellMesh,
) -> None:
    blocks = tuple(
        CellBlock(
            block.name,
            "simplex:3" if block.cell_kind == "tetrahedron" else block.cell_kind,
            np.asarray(block.vertices),
            global_ids=np.asarray(block.global_ids),
        )
        for block in tet_prism_mesh.blocks
    )
    mesh = CellMesh(
        tet_prism_mesh.coordinates,
        blocks,
        vertex_global_ids=tet_prism_mesh.vertex_global_ids,
    )
    refined = adapt_mixed_mesh(mesh, refine_cell_ids=np.asarray((101,), dtype=np.int64))
    fine, _, _ = assemble_topology_edit(
        mesh, refined.edit, numeric_version="simplex-kind-fine"
    )
    assert {block.cell_kind for block in fine.blocks} == {"simplex:3", "prism"}
    fine_ids = np.concatenate([np.asarray(block.global_ids) for block in fine.blocks])
    coarsened = adapt_mixed_mesh(
        fine, coarsen_cell_ids=fine_ids, hierarchy=refined.hierarchy
    )
    restored, _, _ = assemble_topology_edit(
        fine, coarsened.edit, numeric_version="simplex-kind-restored"
    )
    assert {block.cell_kind for block in restored.blocks} == {"simplex:3", "prism"}
    assert set(np.asarray(restored.entity_set(3).entity_ids)) == set(
        np.asarray(mesh.entity_set(3).entity_ids)
    )


def test_mixed_simplex_and_prism_restore_original_root_maps(
    tet_prism_mesh: CellMesh,
) -> None:
    exact = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(tet_prism_mesh.coordinates)
    )
    source_map = ExactPlcCellGeometrySource(
        "mixed-root-test",
        "1",
        tet_prism_mesh.vertex_global_ids,
        exact,
        vertex_source_entity_ids=(None,) * len(exact),
        vertex_occurrence_paths=((),) * len(exact),
    )
    geometry = CellGeometrySpec.power(tet_prism_mesh, source_map)
    refined = adapt_mixed_mesh(
        tet_prism_mesh,
        refine_cell_ids=np.asarray((101,), dtype=np.int64),
        source_geometry=geometry,
    )
    assert refined.target_geometry is not None
    assert refined.target_geometry.source_map is not None
    assert refined.target_geometry.source_map.source_map_id == source_map.source_map_id
    assert refined.target_geometry.source_coordinates() == exact
    fine, _, _ = assemble_topology_edit(
        tet_prism_mesh, refined.edit, numeric_version="mixed-adaptation-mapped-geometry"
    )
    assert fine.mesh_id == refined.target_geometry.source_mesh_id
    fine_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in fine.blocks]
    )
    restored = adapt_mixed_mesh(
        fine,
        coarsen_cell_ids=fine_ids,
        hierarchy=refined.hierarchy,
        source_geometry=refined.target_geometry,
    )
    assert restored.coarsening_partition is not None
    assert restored.target_geometry is not None
    assert restored.target_geometry.source_map is not None
    assert restored.target_geometry.source_map.source_map_id == source_map.source_map_id
    assert restored.target_geometry.source_coordinates() == exact
    root = restored.target_geometry.restriction_source
    assert root is not None
    assert root.source_geometry_id == cell_geometry_id(geometry)
    assert root.source_topology_id == tet_prism_mesh.topology_id
    assert set(root.block_names) == {block.name for block in tet_prism_mesh.blocks}
    for block, witness in zip(root.block_names, root.parent_reference_maps, strict=True):
        assert witness is not None
        original = next(item for item in tet_prism_mesh.blocks if item.name == block)
        np.testing.assert_array_equal(witness.source_cell_ids, original.global_ids)
        assert witness.source_reference_chart_arguments is not None


def test_prism_pyramid_face_closure_and_conservative_coarsening(
    prism_pyramid_mesh: CellMesh,
) -> None:
    refined, fine, coarsened, restored = _adapt_and_restore(
        prism_pyramid_mesh,
        303,
        np.asarray((3, 8), dtype=np.int64),
    )
    assert set(refined.evidence.closure_cell_ids.tolist()) == {404}
    _assert_interface_matches(fine, "prism", "pyramid", axis=1, coordinate=0.0)
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
    _assert_interface_matches(fine, "hexahedron", "pyramid", axis=2, coordinate=1.0)
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


def test_retained_coordinate_bank_rejects_rehashed_foreign_provenance(
    hex_pyramid_mesh: CellMesh,
    tmp_path: Path,
) -> None:
    from phydrax.meshing._curving import _ExactCoordinateBankMap
    from phydrax.meshing._mixed_adaptation import (
        _require_retained_bank_origin,
        _RetainedCoordinateBank,
    )

    base = _curved_hex_pyramid_geometry(hex_pyramid_mesh)
    exact = base.source_coordinates()
    source_map = _ExactCoordinateBankMap(
        hex_pyramid_mesh,
        "actual-source",
        "r1",
        ("geometry",),
        exact,
        np.asarray(base.coordinates),
    )
    source_geometry = CellGeometrySpec(
        dict(zip(base.block_names, base.elements, strict=True)),
        dict(zip(base.block_names, base.geometry_dofs, strict=True)),
        base.coordinates,
        source_map=source_map,
        source_mesh=hex_pyramid_mesh,
    )
    outcome = adapt_mixed_mesh(
        hex_pyramid_mesh,
        refine_cell_ids=np.asarray((505,), dtype=np.int64),
        source_geometry=source_geometry,
    )
    target, _, _ = assemble_topology_edit(
        hex_pyramid_mesh, outcome.edit, numeric_version="mixed-adaptation-mapped-geometry"
    )
    geometry = outcome.target_geometry
    assert geometry is not None and type(geometry.source_map) is _RetainedCoordinateBank
    _require_retained_bank_origin(source_geometry, geometry)
    retained = geometry.source_map
    forged = _RetainedCoordinateBank(
        retained.source_id,
        retained.source_revision,
        "different-source-map",
        retained.exact_coordinates,
    )
    forged_geometry = CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        geometry.coordinates,
        source_map=forged,
        source_mesh=target,
        restriction_source=geometry.restriction_source,
    )
    assert cell_geometry_id(forged_geometry) != cell_geometry_id(geometry)
    with pytest.raises(ValueError, match="actual source geometry"):
        _require_retained_bank_origin(source_geometry, forged_geometry)

    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax.discretization._cell_geometry_transition_types import (
        CellGeometryTransitionEvidence,
    )
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )

    def archive_transition(target_geometry: CellGeometrySpec, name: str) -> Composition:
        evidence = CellGeometryTransitionEvidence(
            "nested_restriction",
            reference_map_count=sum(
                len(group.target_cell_ids) for group in outcome.reference_maps
            ),
        )
        transition = CellGeometryTransition(
            hex_pyramid_mesh,
            source_geometry,
            target,
            target_geometry,
            evidence=evidence,
            reference_map=outcome.reference_maps,
        )
        entry = CompositionEntry(
            transition,
            entry_id="transition",
            role="model-state",
            owner_id="phydrax.meshing",
            structure_id=target.topology_id,
            revision_id=transition.transition_id,
            semantics_id="actual-exact-transition",
        )
        path = tmp_path / name
        write_model_artifact(
            path,
            Composition((entry,), boundary_id="retained-coordinate-origin"),
            limits=MODEL_ARCHIVE_LIMITS,
        )
        return read_model_artifact(
            path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
            expected_content_id=model_archive_content_id(
                path, limits=MODEL_ARCHIVE_LIMITS
            ),
        )

    restored = archive_transition(geometry, "real-transition.npz")
    assert restored.value("transition").transition_id
    with pytest.raises(
        (ArrayArchiveCorruptionError, ValueError), match="source geometry|archive"
    ):
        archive_transition(forged_geometry, "forged-transition.npz")


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


def test_refined_target_validity_reuses_unchanged_source_cell_proofs(
    hex_pyramid_mesh: CellMesh, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A tetrahedron sharing no face with the refined hexahedron keeps its map.
    far = np.asarray(
        ((3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.0, 1.0, 0.0), (3.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        np.concatenate((np.asarray(hex_pyramid_mesh.coordinates), far)),
        (
            *hex_pyramid_mesh.blocks,
            CellBlock(
                "tetrahedra",
                "tetrahedron",
                np.asarray(((9, 10, 11, 12),), dtype=np.int32),
                global_ids=np.asarray((707,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(900, 913, dtype=np.int64),
    )
    source_geometry = _curved_hex_pyramid_geometry(mesh)
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=mesh)
    refined = adapt_mixed_mesh(
        mesh,
        refine_cell_ids=np.asarray((505,), dtype=np.int64),
        cell_classes=np.asarray((2, 6, 7), dtype=np.int64),
        source_geometry=source_geometry,
    )
    assert refined.target_geometry is not None
    fine, _, _ = assemble_topology_edit(
        mesh, refined.edit, numeric_version="mixed-curved-refinement"
    )

    full, proofs = assert_incremental_matches_full(
        monkeypatch, refined.target_geometry, source_validity, mesh=fine
    )

    assert full.invalid_count == full.unresolved_count == 0
    table = np.frombuffer(full.cell_map_digests, dtype=np.uint8).reshape(-1, 2, 16)
    reused = cell_map_digests(full) & cell_map_digests(source_validity)
    tetrahedron = full.block_offsets[full.block_names.index("tetrahedra")]
    assert {row.tobytes() for row in table[tetrahedron]} <= reused
    assert proofs == len(cell_map_digests(full)) - len(reused)


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
    fine_root = refined.target_geometry.restriction_source
    assert fine_root is not None
    assert fine_root.source_geometry_id == cell_geometry_id(source_geometry)
    assert fine_root.source_topology_id == hex_pyramid_mesh.topology_id
    assert all(witness is not None for witness in fine_root.parent_reference_maps)
    fine, _, _ = assemble_topology_edit(
        hex_pyramid_mesh, refined.edit, numeric_version="mixed-curved-refinement"
    )
    fine_validity = certify_cell_geometry_validity(refined.target_geometry, mesh=fine)
    assert fine_validity.invalid_count == 0
    assert fine_validity.unresolved_count == 0
    hex_sibling = next(
        record for record in refined.hierarchy.siblings if record.parent_id == 505
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
        tuple(local_corners for _ in hex_sibling.child_ids),
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
        for image, (parent,) in zip(source_center_images, parent_centers, strict=True)
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
    restored_root = coarsened.target_geometry.restriction_source
    assert restored_root is not None
    assert restored_root.source_geometry_id == fine_root.source_geometry_id
    assert restored_root.source_topology_id == fine_root.source_topology_id
    assert all(witness is not None for witness in restored_root.parent_reference_maps)
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
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in fine.mesh.blocks)
    )
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            fine,
            MarkedMeshAdaptation(
                (),
                fine_cell_ids,
                hierarchy=hierarchy,
            ),
            policy=policy,
        )
    )
    restored_geometry = coarsened.target.geometry
    assert coarsened.cell_content_transfer is not None
    assert coarsened.transition is not None
    transition = coarsened.transition.geometry_transition
    assert transition is not None and transition.coarsening_partition is not None
    renewed = CellGeometryTransition.from_stored_fields(transition)
    assert renewed.transition_id == transition.transition_id
    # The source carries no acceptance inputs, so the route must not invent a
    # certification; whole-cell validity of the restored curved map is checked.
    assert coarsened.target.certification is None
    restored_validity = certify_cell_geometry_validity(
        restored_geometry, mesh=coarsened.target.mesh
    )
    assert restored_validity.invalid_count == 0
    assert restored_validity.unresolved_count == 0
    for cell_id, (kind, _) in _cell_records(hex_pyramid_mesh).items():
        reference = (
            center
            if kind == "hexahedron"
            else (Fraction(1, 2), Fraction(1, 2), Fraction(1, 4))
        )
        expected = source_geometry.evaluate_exact(
            hex_pyramid_mesh,
            np.asarray((cell_id,), dtype=np.int64),
            ((reference,),),
        )
        actual = restored_geometry.evaluate_exact(
            coarsened.target.mesh,
            np.asarray((cell_id,), dtype=np.int64),
            ((reference,),),
        )
        assert actual == expected


def _periodic_hex_pyramid(mesh: CellMesh) -> CellMesh:
    representatives = np.asarray((0, 0, 3, 3, 4, 4, 7, 7, 8), dtype=np.int32)
    shifts = np.zeros((9, 1), dtype=np.int32)
    shifts[np.asarray((1, 2, 5, 6)), 0] = 1
    periodic = PeriodicMeshTopology(
        mesh, PeriodicCell(np.asarray(((1.0, 0.0, 0.0),))), representatives, shifts
    )
    return CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        periodic_topology=periodic,
    )


def _periodic_curved_hex_pyramid(mesh: CellMesh) -> CellGeometrySpec:
    geometry = _curved_hex_pyramid_geometry(mesh)
    coordinates = np.asarray(geometry.coordinates).copy()
    hex_index = geometry.block_names.index("hexahedra")
    nodes = np.asarray(
        lagrange_element("hexahedron", 2).reference_nodes, dtype=np.float64
    )
    rows = np.asarray(geometry.geometry_dofs[hex_index], dtype=np.int64)[0]
    coordinates[rows, 1] = nodes[:, 1] + 0.05 * nodes[:, 2] * (1.0 - nodes[:, 2])
    return CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        coordinates,
    )


def _periodic_entity_ids(mesh: CellMesh, degree: int) -> dict[tuple[int, ...], int]:
    orbit_map = periodic_entity_orbit_map(mesh, degree)
    quotient = np.asarray(orbit_map.quotient_entity_ids, dtype=np.int64)
    rows = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
    return {
        tuple(int(value) for value in key if value >= 0): int(quotient[row])
        for key, row in zip(entity_keys(mesh, degree), rows, strict=True)
    }


def test_periodic_prepared_exact_maps_match_cellwise_source_charts(
    hex_pyramid_mesh: CellMesh,
) -> None:
    from phydrax.discretization._cell_geometry_transition_types import (
        _geometry_map_expressions,
    )
    from phydrax.discretization._coordinate_enclosure import expression_equal
    from phydrax.meshing._hex_dominant import _source_cell_chart

    mesh = _periodic_hex_pyramid(hex_pyramid_mesh)
    geometry = _periodic_curved_hex_pyramid(mesh)
    elements, routes, _ = geometry.resolve(mesh)
    prepared = _geometry_map_expressions(mesh, geometry, (elements, routes))
    row = 0
    for block in mesh.blocks:
        for identifier in np.asarray(block.global_ids, dtype=np.int64):
            actual = prepared.expressions(int(identifier))
            original = _source_cell_chart(mesh, geometry, row).physical_expressions
            assert all(
                expression_equal(a, b) for a, b in zip(actual, original, strict=True)
            )
            row += 1


def test_periodic_curved_mixed_translation_group_embedding(
    hex_pyramid_mesh: CellMesh,
) -> None:
    lattice_mesh = _periodic_hex_pyramid(hex_pyramid_mesh)
    lattice_topology = lattice_mesh.periodic_topology
    assert lattice_topology is not None
    generator = np.eye(4, dtype=np.float64)
    generator[0, 3] = 1.0
    topology = PeriodicMeshTopology(
        hex_pyramid_mesh,
        PeriodicIsometryGroup(generator[None, :, :]),
        np.asarray(lattice_topology.vertex_representatives),
        np.asarray(lattice_topology.vertex_shifts),
    )
    mesh = CellMesh(
        hex_pyramid_mesh.coordinates,
        hex_pyramid_mesh.blocks,
        vertex_global_ids=hex_pyramid_mesh.vertex_global_ids,
        periodic_topology=topology,
    )
    geometry = _periodic_curved_hex_pyramid(mesh)
    require_periodic_geometry_face_maps(mesh, geometry)
    result = certify_cell_mesh(mesh, SpatialCoordinateContract.si(), geometry=geometry)
    assert result.audit.passed


def test_periodic_mixed_curved_two_rounds_restore_quotient_ids(
    hex_pyramid_mesh: CellMesh,
) -> None:
    mesh = _periodic_hex_pyramid(hex_pyramid_mesh)
    geometry = _periodic_curved_hex_pyramid(mesh)
    require_periodic_geometry_face_maps(mesh, geometry)
    source = certify_cell_mesh(mesh, SpatialCoordinateContract.si(), geometry=geometry)
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_MIXED)

    def run(
        state: CellMeshingResult,
        hierarchy: MixedAdaptationHierarchy | None,
        refine: tuple[int, ...] | np.ndarray,
        coarsen: tuple[int, ...] | np.ndarray,
    ) -> MeshAdaptationResult:
        return execute_mesh_adaptation(
            prepare_mesh_adaptation(
                state,
                MarkedMeshAdaptation(refine, coarsen, hierarchy=hierarchy),
                policy=policy,
            )
        )

    first = run(source, None, (505,), ())
    assert isinstance(first.hierarchy, MixedAdaptationHierarchy)
    replay = run(source, None, (505,), ())
    assert isinstance(replay.hierarchy, MixedAdaptationHierarchy)
    assert first.target.mesh.mesh_id == replay.target.mesh.mesh_id
    assert first.hierarchy.hierarchy_id == replay.hierarchy.hierarchy_id
    child = min(
        cell.cell_id
        for cell in first.hierarchy.active_cells
        if cell.block_name == "hexahedra"
    )
    second = run(first.target, first.hierarchy, (child,), ())
    assert isinstance(second.hierarchy, MixedAdaptationHierarchy)
    require_periodic_geometry_face_maps(second.target.mesh, second.target.geometry)
    second_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in second.target.mesh.blocks]
    )
    one_round = run(second.target, second.hierarchy, (), second_ids)
    assert isinstance(one_round.hierarchy, MixedAdaptationHierarchy)
    first_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in one_round.target.mesh.blocks]
    )
    restored = run(one_round.target, one_round.hierarchy, (), first_ids)
    for state in (first, second, one_round, restored):
        root = state.target.geometry.restriction_source
        assert root is not None
        assert root.source_geometry_id == cell_geometry_id(geometry)
        assert root.source_topology_id == mesh.topology_id
        assert all(witness is not None for witness in root.parent_reference_maps)
    previous_points: dict[int, tuple[tuple[int, Fraction], ...]] = {}
    for state, source_mesh in zip(
        (first, second, one_round, restored),
        (mesh, first.target.mesh, second.target.mesh, one_round.target.mesh),
        strict=True,
    ):
        assert isinstance(state.hierarchy, MixedAdaptationHierarchy)
        history = state.hierarchy.periodic_quotient_history
        assert history is not None
        assert history.source_mesh_id == source_mesh.mesh_id
        for rows in history.entity_keys_by_degree:
            assert len(rows) == len({key for key, _ in rows})
        current_points = dict(state.hierarchy.vertex_points)
        assert current_points.items() >= previous_points.items()
        previous_points = current_points
    for degree in range(4):
        assert _periodic_entity_ids(restored.target.mesh, degree) == _periodic_entity_ids(
            mesh, degree
        )
    require_periodic_geometry_face_maps(restored.target.mesh, restored.target.geometry)


def test_periodic_face_proof_rejects_curved_trace_mismatch(
    hex_pyramid_mesh: CellMesh,
) -> None:
    mesh = _periodic_hex_pyramid(hex_pyramid_mesh)
    with pytest.raises(ValueError, match="face coordinate maps disagree"):
        require_periodic_geometry_face_maps(mesh, _curved_hex_pyramid_geometry(mesh))


def test_exact_lineage_does_not_alias_disjoint_same_support_edges() -> None:
    def point(x: Fraction) -> tuple[tuple[int, Fraction], ...]:
        return tuple(
            (vertex, weight) for vertex, weight in ((10, 1 - x), (20, x)) if weight
        )

    left = (point(Fraction(0)), point(Fraction(1, 2)))
    right = (point(Fraction(1, 2)), point(Fraction(1)))
    assert _point_in_root_hull(point(Fraction(1, 4)), left)
    assert not _point_in_root_hull(point(Fraction(3, 4)), left)
    assert not all(_point_in_root_hull(value, left) for value in right)


def test_periodic_mixed_edit_is_stable_under_vertex_row_permutation(
    hex_pyramid_mesh: CellMesh,
) -> None:
    source = _periodic_hex_pyramid(hex_pyramid_mesh)
    permutation = np.asarray((8, 3, 2, 1, 0, 7, 6, 5, 4), dtype=np.int64)
    inverse = np.argsort(permutation)
    blocks = tuple(
        CellBlock(
            block.name,
            block.cell_kind,
            inverse[np.asarray(block.vertices, dtype=np.int64)],
            global_ids=block.global_ids,
        )
        for block in source.blocks
    )
    reordered = CellMesh(
        np.asarray(source.coordinates)[permutation],
        blocks,
        vertex_global_ids=np.asarray(source.vertex_global_ids)[permutation],
    )
    topology = source.periodic_topology
    assert topology is not None
    periodic = PeriodicMeshTopology(
        reordered,
        topology.cell,
        inverse[np.asarray(topology.vertex_representatives)[permutation]],
        np.asarray(topology.vertex_shifts)[permutation],
    )
    reordered = CellMesh(
        reordered.coordinates,
        reordered.blocks,
        vertex_global_ids=reordered.vertex_global_ids,
        periodic_topology=periodic,
    )

    def refine(mesh: CellMesh) -> CellMesh:
        outcome = adapt_mixed_mesh(
            mesh,
            refine_cell_ids=np.asarray((505,), dtype=np.int64),
            source_geometry=_periodic_curved_hex_pyramid(mesh),
        )
        target, _, _ = assemble_topology_edit(
            mesh, outcome.edit, numeric_version="refine"
        )
        return target

    def cells(mesh: CellMesh) -> dict[int, tuple[str, tuple[int, ...]]]:
        vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        return {
            int(identifier): (block.cell_kind, tuple(vertices[row].tolist()))
            for block in mesh.blocks
            for identifier, row in zip(
                np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
            )
        }

    first, second = refine(source), refine(reordered)
    assert cells(first) == cells(second)
    for degree in range(4):
        assert _periodic_entity_ids(first, degree) == _periodic_entity_ids(second, degree)


def test_retained_mixed_result_reproof_and_tampering(hex_pyramid_mesh: CellMesh) -> None:
    source = certify_cell_mesh(
        hex_pyramid_mesh,
        SpatialCoordinateContract.si(),
        geometry=_curved_hex_pyramid_geometry(hex_pyramid_mesh),
    )
    request = MarkedMeshAdaptation(np.asarray((505,), dtype=np.int64))
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_MIXED)
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )
    assert isinstance(result.hierarchy, MixedAdaptationHierarchy)
    assert result.transition is not None
    geometry = result.transition.geometry_transition
    assert geometry is not None

    def restore(
        raw: MeshAdaptationResult, hierarchy: MixedAdaptationHierarchy
    ) -> MeshAdaptationResult:
        return MeshAdaptationResult.from_stored_fields(
            raw,
            source=source,
            target=result.target,
            policy=policy,
            request=request,
            geometry_transition=geometry,
            hierarchy=hierarchy,
            expected_result_id=result.result_id,
        )

    restored = restore(result, result.hierarchy)
    assert restored.result_id == result.result_id
    assert restored.cell_content_transfer is not None
    assert result.cell_content_transfer is not None
    assert (
        restored.cell_content_transfer.transfer_id
        == result.cell_content_transfer.transfer_id
    )
    children = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in restored.target.mesh.blocks)
    )
    assert isinstance(restored.hierarchy, MixedAdaptationHierarchy)
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            restored.target,
            MarkedMeshAdaptation((), children, hierarchy=restored.hierarchy),
            policy=policy,
        )
    )
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id

    def replacement(
        *,
        siblings: tuple[MixedSibling, ...] | None = None,
        vertex_points: tuple[tuple[int, tuple[tuple[int, Fraction], ...]], ...]
        | None = None,
        retired_entities: tuple[tuple[int, tuple[int, ...], int], ...] | None = None,
    ) -> MixedAdaptationHierarchy:
        hierarchy = result.hierarchy
        assert isinstance(hierarchy, MixedAdaptationHierarchy)
        return MixedAdaptationHierarchy(
            hierarchy.siblings if siblings is None else siblings,
            hierarchy.active_cells,
            hierarchy.forest_roots,
            hierarchy.vertex_points if vertex_points is None else vertex_points,
            hierarchy.retired_entities if retired_entities is None else retired_entities,
            hierarchy.layer_policy,
            hierarchy.next_cell_id,
            hierarchy.next_vertex_id,
            hierarchy.periodic_quotient_history,
            hierarchy.active_lifted_topology_id,
        )

    hierarchy = result.hierarchy
    sibling = hierarchy.siblings[0]
    charts = list(sibling.child_reference_vertices)
    first_child = list(charts[0])
    point = list(first_child[0])
    point[0] += Fraction(1, 8)
    first_child[0] = tuple(point)
    charts[0] = tuple(first_child)
    bad_sibling = sibling._replace(child_reference_vertices=tuple(charts))
    cases = [replacement(siblings=(bad_sibling, *hierarchy.siblings[1:]))]
    points = list(hierarchy.vertex_points)
    row = next(i for i, (_, key) in enumerate(points) if len(key) == 2)
    vertex, key = points[row]
    first_weight = Fraction(1, 3) if key[0][1] == Fraction(1, 4) else Fraction(1, 4)
    points[row] = (vertex, ((key[0][0], first_weight), (key[1][0], 1 - first_weight)))
    cases.append(replacement(vertex_points=tuple(points)))
    assert hierarchy.retired_entities
    retired = list(hierarchy.retired_entities)
    degree, key, _ = retired[0]
    retired[0] = (
        degree,
        key,
        int(np.asarray(result.target.mesh.entity_set(degree).entity_ids)[0]),
    )
    cases.append(replacement(retired_entities=tuple(retired)))
    for bad in cases:
        raw = copy.copy(result)
        object.__setattr__(raw, "hierarchy", bad)
        with pytest.raises(ValueError):
            restore(raw, bad)

    raw = copy.copy(result)
    object.__setattr__(raw, "source", result.target)
    with pytest.raises(ValueError):
        restore(raw, hierarchy)
    assert result.lineage is not None
    changed_record = copy.copy(result.lineage.entities[-1])
    ids = np.asarray(changed_record.source_global_ids, dtype=np.int64).copy()
    ids[0] += 1
    object.__setattr__(changed_record, "source_global_ids", ids)
    changed_lineage = copy.copy(result.lineage)
    object.__setattr__(
        changed_lineage, "entities", (*result.lineage.entities[:-1], changed_record)
    )
    raw = copy.copy(result)
    object.__setattr__(raw, "lineage", changed_lineage)
    with pytest.raises(ValueError):
        restore(raw, hierarchy)


def test_retained_periodic_mixed_result_rejects_changed_action(
    hex_pyramid_mesh: CellMesh,
) -> None:
    from phydrax.meshing._periodic import PeriodicQuotientHistory

    mesh = _periodic_hex_pyramid(hex_pyramid_mesh)
    source = certify_cell_mesh(
        mesh, SpatialCoordinateContract.si(), geometry=_periodic_curved_hex_pyramid(mesh)
    )
    request = MarkedMeshAdaptation(np.asarray((505,), dtype=np.int64))
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_MIXED)
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )
    hierarchy = result.hierarchy
    assert isinstance(hierarchy, MixedAdaptationHierarchy)
    assert (
        result.transition is not None
        and result.transition.geometry_transition is not None
    )
    history = hierarchy.periodic_quotient_history
    assert history is not None
    assert (
        MeshAdaptationResult.from_stored_fields(
            result,
            source=source,
            target=result.target,
            policy=policy,
            request=request,
            geometry_transition=result.transition.geometry_transition,
            expected_result_id=result.result_id,
            hierarchy=hierarchy,
        ).result_id
        == result.result_id
    )
    actions = list(history.vertex_actions_by_id)
    identifier, fixed, winding = actions[0]
    assert winding
    changed_winding = (winding[0] + 1, *winding[1:])
    actions[0] = (identifier, fixed, changed_winding)
    with pytest.raises(ValueError):
        changed_history = PeriodicQuotientHistory(
            history.identification_id,
            history.source_mesh_id,
            history.entity_keys_by_degree,
            history.allocator_next_ids,
            history.occurrences_by_degree,
            tuple(actions),
        )
        changed_hierarchy = MixedAdaptationHierarchy(
            hierarchy.siblings,
            hierarchy.active_cells,
            hierarchy.forest_roots,
            hierarchy.vertex_points,
            hierarchy.retired_entities,
            hierarchy.layer_policy,
            hierarchy.next_cell_id,
            hierarchy.next_vertex_id,
            changed_history,
            hierarchy.active_lifted_topology_id,
        )
        raw = copy.copy(result)
        object.__setattr__(raw, "hierarchy", changed_hierarchy)
        MeshAdaptationResult.from_stored_fields(
            raw,
            source=source,
            target=result.target,
            policy=policy,
            request=request,
            geometry_transition=result.transition.geometry_transition,
            expected_result_id=result.result_id,
            hierarchy=changed_hierarchy,
        )


def test_restricted_mixed_source_uses_exact_nonaffine_root_vertices() -> None:
    from phydrax.discretization._cell_geometry import (
        CellGeometryRestrictionSource,
        PerCellRestrictedCellGeometryElement,
    )
    from phydrax.discretization._coordinate_enclosure import add, axes, multiply, scale
    from phydrax.discretization._nested_reference_types import (
        freeze_reference_chart_arguments,
        ReferenceCellMapWitnesses,
    )
    from phydrax.meshing._lineage import EntityLineageKind

    corners = tuple(
        tuple(Fraction(int(value)) for value in point)
        for point in reference_cell_topology("tetrahedron").vertices
    )
    points = np.asarray(corners, dtype=np.float64)
    vertices = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    vertex_ids = np.asarray((10, 11, 12, 13), dtype=np.int64)
    root = CellMesh(
        points,
        (CellBlock("cells", "tetrahedron", vertices, global_ids=(5,)),),
        vertex_global_ids=vertex_ids,
    )
    source = CellMesh(
        points,
        (CellBlock("cells", "tetrahedron", vertices, global_ids=(101,)),),
        vertex_global_ids=vertex_ids,
    )
    x, y, z = axes(3)
    chart = (add(x, scale(multiply(x, y), Fraction(-1, 2))), y, z)
    witness = ReferenceCellMapWitnesses(
        np.asarray((0,), dtype=np.int64),
        np.asarray((5,), dtype=np.int64),
        "tetrahedron",
        "tetrahedron",
        points[None, :, :],
        (corners,),
        freeze_reference_chart_arguments((chart,)),
    )
    root_geometry = CellGeometrySpec.affine(root)
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(root_geometry),
        root.topology_id,
        {"cells": np.asarray((5,), dtype=np.int64)},
        {"cells": vertex_ids[None, :]},
        block_source_blocks={"cells": "cells"},
        block_parent_reference_maps={"cells": witness},
    )
    element = PerCellRestrictedCellGeometryElement(
        lagrange_element("tetrahedron", 1), "tetrahedron", (101,), (5,), (chart,)
    )
    geometry = CellGeometrySpec(
        {"cells": element}, {"cells": vertices}, points, restriction_source=restriction
    )
    with pytest.raises(ValueError, match="requires its exact root mesh"):
        adapt_mixed_mesh(source, refine_cell_ids=(101,), source_geometry=geometry)
    outcome = adapt_mixed_mesh(
        source, refine_cell_ids=(101,), source_geometry=geometry, root_mesh=root
    )
    assert outcome.target_geometry is not None
    assert outcome.target_geometry.restriction_source is not None
    assert (
        outcome.target_geometry.restriction_source.source_geometry_id
        == cell_geometry_id(root_geometry)
    )
    assert len(outcome.hierarchy.forest_roots) == 1
    assert outcome.hierarchy.forest_roots[0].root_cell_id == 5
    bank = dict(outcome.hierarchy.vertex_points)
    edge_midpoint = None
    assert outcome.edit.stencil_exact_weights is not None
    for row, identifier in enumerate(
        np.asarray(outcome.edit.vertex_global_ids, dtype=np.int64)
    ):
        weights = outcome.edit.stencil_exact_weights[row]
        sources = np.asarray(outcome.edit.stencil_sources[row], dtype=np.int64)
        valid = np.asarray(outcome.edit.stencil_valid[row], dtype=np.bool_)
        point = tuple(
            sum(
                (
                    weight
                    * corners[int(np.flatnonzero(vertex_ids == source_id)[0])][axis]
                    for source_id, weight, active in zip(
                        sources, weights, valid, strict=True
                    )
                    if active
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        rx = point[0] - point[0] * point[1] / 2
        expected = tuple(
            (int(vertex), weight)
            for vertex, weight in zip(
                vertex_ids,
                (1 - rx - point[1] - point[2], rx, point[1], point[2]),
                strict=True,
            )
            if weight
        )
        assert bank[int(identifier)] == expected
        if point == (Fraction(1, 2), Fraction(1, 2), Fraction(0)):
            edge_midpoint = int(identifier)
    assert edge_midpoint is not None
    edges = next(value for value in outcome.edit.relations if value.dimension == 1)
    descendants = {
        tuple(int(value) for value in target if value >= 0)
        for source_key, target, kind in zip(
            edges.source_keys, edges.target_keys, edges.kinds, strict=True
        )
        if tuple(int(value) for value in source_key if value >= 0) == (11, 12)
        and int(kind) == int(EntityLineageKind.REFINED_FROM)
    }
    assert descendants == {
        tuple(sorted((11, edge_midpoint))),
        tuple(sorted((12, edge_midpoint))),
    }
    fine, _, _ = assemble_topology_edit(
        source, outcome.edit, numeric_version="nonaffine-source-fine"
    )
    fine_ids = np.asarray(fine.blocks[0].global_ids, dtype=np.int64)
    restarted = adapt_mixed_mesh(
        fine,
        refine_cell_ids=(int(np.min(fine_ids)),),
        source_geometry=outcome.target_geometry,
        root_mesh=root,
    )
    assert {row.cell_id for row in restarted.hierarchy.forest_roots} == set(fine_ids)
    assert all(row.root_cell_id == 5 for row in restarted.hierarchy.forest_roots)
    restored = adapt_mixed_mesh(
        fine,
        coarsen_cell_ids=fine_ids,
        hierarchy=outcome.hierarchy,
        source_geometry=outcome.target_geometry,
        root_mesh=root,
    )
    assert restored.coarsening_partition is not None
    assert restored.target_geometry is not None
    restored_mesh, _, _ = assemble_topology_edit(
        fine, restored.edit, numeric_version="nonaffine-source-restored"
    )
    sample = (Fraction(1, 3), Fraction(1, 4), Fraction(1, 8))
    assert restored.target_geometry.evaluate_exact(
        restored_mesh, np.asarray((101,), dtype=np.int64), ((sample,),)
    ) == geometry.evaluate_exact(source, np.asarray((101,), dtype=np.int64), ((sample,),))
    reverse_edges = next(
        value for value in restored.edit.relations if value.dimension == 1
    )
    ancestors = {
        tuple(int(value) for value in source_key if value >= 0)
        for source_key, target_key, kind in zip(
            reverse_edges.source_keys,
            reverse_edges.target_keys,
            reverse_edges.kinds,
            strict=True,
        )
        if tuple(int(value) for value in target_key if value >= 0) == (11, 12)
        and int(kind) == int(EntityLineageKind.COARSENED_INTO)
    }
    assert ancestors == descendants


def test_canonical_simplex_family_preserves_exact_child_reference_map() -> None:
    from phydrax.meshing._mixed_adaptation import _exact_child_to_parent_affine_map

    corners = tuple(
        tuple(Fraction(int(value)) for value in point)
        for point in reference_cell_topology("simplex:3").vertices
    )
    child = tuple(
        (point[0] / 2 + Fraction(1, 4), point[1] / 2, point[2] / 2) for point in corners
    )
    matrix, offset = _exact_child_to_parent_affine_map("simplex:3", child)
    assert offset == (Fraction(1, 4), Fraction(0), Fraction(0))
    assert matrix == tuple(
        tuple(Fraction(int(row == column), 2) for column in range(3)) for row in range(3)
    )
