#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import itertools
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._cell_geometry import (
    cell_geometry_preparation_query_upper,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import affine_vertex_positions
from phydrax.lifecycle import read_meshing_source_closure, write_meshing_source_closure
from phydrax.meshing import (
    BisectionCompatibility,
    BisectionHierarchy,
    BisectionUniformRefinement,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MeshingEntityKind,
    MeshingScope,
    prepare_mesh_adaptation,
)
from tests._support.cell_meshes import incompatible_simplex, kuhn_cube, triangle_grid
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.distributed_meshing.cold_lifecycle_identities import record_array
from tests._support.forced_devices import run_on_forced_devices
from tests._support.mesh_certificate_fixtures import (
    affine_tetrahedral_root_restrictions,
    TETRAHEDRON,
)


# Qualification bounds retained from test_bisection.py at d2dfcb941.
_GEOMETRY_ROUNDOFF = 1.0e-12
_AFFINE_TRANSFER_ROUNDOFF = 1.0e-13
_HANGING_VERTEX_SEPARATION = 1.0e-9
_TETRA_SIMILARITY_FLOOR = 0.999
# Empirical agreement threshold retained from the pre-refactor midpoint and P2-node
# fixtures in test_conforming_closure_* and test_bisection_keeps_curved_source_maps_*.
_MIDPOINT_ROUNDOFF = 1.0e-15


def _signed_measures(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    corners = np.asarray(points)[cells]
    edges = corners[:, 1:] - corners[:, :1]
    if cells.shape[1] == 3:
        return 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    return np.linalg.det(edges) / 6.0


def _certified(mesh: CellMesh) -> Any:
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _adapt(
    source: Any,
    refine: Any = (),
    coarsen: Any = (),
    /,
    *,
    hierarchy: Any = None,
    refinement_generations: int = 1,
    refinement_target_cells: int | None = None,
    **options: Any,
) -> Any:
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION, **options)
    request = MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
        refinement_generations=refinement_generations,
        refinement_target_cells=refinement_target_cells,
    )
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )


def _cells(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(block.vertices) for block in mesh.blocks))


def _cell_ids(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(block.global_ids) for block in mesh.blocks))


def _canonical_cells(mesh: CellMesh) -> tuple[np.ndarray, np.ndarray]:
    identifiers = _cell_ids(mesh)
    vertices = np.asarray(mesh.vertex_global_ids)[_cells(mesh)]
    order = np.argsort(identifiers, kind="stable")
    return identifiers[order], vertices[order]


def _edge_keys(mesh: CellMesh) -> np.ndarray:
    match mesh.connectivity:
        case (
            PolygonalConnectivity()
            | PolyhedralConnectivity()
            | TetrahedralConnectivity() as connectivity
        ):
            edges = np.asarray(connectivity.edges)
        case _:
            raise TypeError(
                f"bisection.mesh.connectivity must expose simplex edges; got {type(mesh.connectivity).__name__}"
            )
    return np.sort(np.asarray(mesh.vertex_global_ids)[edges], axis=1)


def _minimum_angles(mesh: CellMesh) -> np.ndarray:
    corners = np.asarray(mesh.coordinates)[_cells(mesh)]
    angles = []
    for vertex in range(3):
        first = corners[:, (vertex + 1) % 3] - corners[:, vertex]
        second = corners[:, (vertex + 2) % 3] - corners[:, vertex]
        cosine = np.sum(first * second, axis=1) / (
            np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        )
        angles.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.min(np.stack(angles, axis=1), axis=1)


def _mean_ratios(mesh: CellMesh) -> np.ndarray:
    points = np.asarray(mesh.coordinates)
    cells = _cells(mesh)
    pairs = np.asarray(tuple(itertools.combinations(range(4), 2)))
    corners = points[cells]
    squared = np.sum((corners[:, pairs[:, 0]] - corners[:, pairs[:, 1]]) ** 2, axis=2)
    volumes = np.abs(_signed_measures(points, cells))
    return 12.0 * (3.0 * volumes) ** (2.0 / 3.0) / np.sum(squared, axis=1)


def _assert_conforming(mesh: CellMesh, measure: float) -> None:
    """Facets bound one or two cells, no vertex hangs on an edge, measure is kept."""

    cells = _cells(mesh)
    points = np.asarray(mesh.coordinates)
    width = cells.shape[1]
    facets = np.sort(
        np.stack([np.delete(cells, i, axis=1) for i in range(width)], axis=1), axis=2
    ).reshape((-1, width - 1))
    _, counts = np.unique(facets, axis=0, return_counts=True)
    assert set(np.unique(counts).tolist()) <= {1, 2}
    pairs = np.asarray(tuple(itertools.combinations(range(width), 2)))
    edges = np.unique(np.sort(cells[:, pairs].reshape((-1, 2)), axis=1), axis=0)
    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    for start in range(0, midpoints.shape[0], 512):
        chunk = midpoints[start : start + 512, None, :] - points[None, :, :]
        assert np.min(np.linalg.norm(chunk, axis=2)) > _HANGING_VERTEX_SEPARATION
    measures = _signed_measures(points, cells)
    assert np.all(measures > 0.0)
    assert np.isclose(np.sum(measures), measure, rtol=0.0, atol=_GEOMETRY_ROUNDOFF)


def _assert_boundary_on_unit_box(mesh: CellMesh) -> None:
    cells = _cells(mesh)
    width = cells.shape[1]
    facets = np.sort(
        np.stack([np.delete(cells, i, axis=1) for i in range(width)], axis=1), axis=2
    ).reshape((-1, width - 1))
    keys, counts = np.unique(facets, axis=0, return_counts=True)
    corners = np.asarray(mesh.coordinates)[keys[counts == 1]]
    on_face = np.all(np.isclose(corners, 0.0), axis=1) | np.all(
        np.isclose(corners, 1.0), axis=1
    )
    assert np.all(np.any(on_face, axis=1))


def _corner_cells(mesh: CellMesh) -> np.ndarray:
    corner = np.argmin(np.linalg.norm(np.asarray(mesh.coordinates), axis=1))
    return np.sort(_cell_ids(mesh)[np.any(_cells(mesh) == corner, axis=1)])


@pytest.mark.parametrize(
    ("mesh_builder", "cycles", "stride"),
    (
        pytest.param(partial(triangle_grid, 4, 4), 6, 7, id="triangles"),
        pytest.param(
            partial(kuhn_cube, 2, orientation_swap=(0, 1)), 4, 11, id="tetrahedra"
        ),
    ),
)
def test_repeated_bisection_preserves_conformity_and_boundary(
    mesh_builder: Callable[[], CellMesh], cycles: int, stride: int
) -> None:
    mesh = mesh_builder()
    result = _adapt(_certified(mesh), (0, 5, 17, 30))
    for _ in range(cycles):
        target = result.target.mesh
        marks = np.union1d(_corner_cells(target), np.sort(_cell_ids(target))[::stride])
        result = _adapt(result.target, marks, hierarchy=result.hierarchy)
        _assert_conforming(result.target.mesh, 1.0)
        _assert_boundary_on_unit_box(result.target.mesh)
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.evidence.maximum_generation >= cycles


def test_triangle_bisection_preserves_uniform_similarity_angle_bound() -> None:
    source = _certified(triangle_grid(2, 2, height=0.6))
    bound = _uniform_class_bound(source, _minimum_angles, 4)
    result = _adapt(source, _corner_cells(source.mesh))
    for _ in range(20):
        assert np.min(_minimum_angles(result.target.mesh)) >= bound - _GEOMETRY_ROUNDOFF
        result = _adapt(
            result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
        )
    assert np.min(_minimum_angles(result.target.mesh)) >= bound - _GEOMETRY_ROUNDOFF
    assert result.evidence.maximum_generation >= 20


def test_tetrahedral_bisection_preserves_uniform_similarity_quality_bound() -> None:
    source = _certified(kuhn_cube(1, orientation_swap=(0, 1)))
    bound = _uniform_class_bound(source, _mean_ratios, 3)
    result = _adapt(source, _corner_cells(source.mesh))
    for _ in range(9):
        assert np.min(_mean_ratios(result.target.mesh)) >= _TETRA_SIMILARITY_FLOOR * bound
        result = _adapt(
            result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
        )
    assert np.min(_mean_ratios(result.target.mesh)) >= _TETRA_SIMILARITY_FLOOR * bound
    assert result.evidence.maximum_generation >= 9


def _uniform_class_bound(source: Any, quality: Any, rounds: int) -> float:
    """Minimum quality over the first uniform bisection generations."""

    bound = float(np.min(quality(source.mesh)))
    current, hierarchy = source, None
    for _ in range(rounds):
        result = _adapt(current, np.sort(_cell_ids(current.mesh)), hierarchy=hierarchy)
        current, hierarchy = result.target, result.hierarchy
        bound = min(bound, float(np.min(quality(current.mesh))))
    return bound


@pytest.mark.parametrize(
    "mesh_builder",
    (
        pytest.param(partial(triangle_grid, 3, 3), id="triangles"),
        pytest.param(partial(kuhn_cube, 2, orientation_swap=(0, 1)), id="tetrahedra"),
    ),
)
def test_coarsening_restores_source_topology_and_coordinates(
    mesh_builder: Callable[[], CellMesh],
) -> None:
    mesh = mesh_builder()
    source = _certified(mesh)
    refined = _adapt(source, np.sort(_cell_ids(mesh))[::3])
    refined = _adapt(
        refined.target,
        _corner_cells(refined.target.mesh),
        hierarchy=refined.hierarchy,
    )
    assert refined.target.mesh.topology_id != source.mesh.topology_id
    restored = _adapt(
        refined.target,
        (),
        np.sort(_cell_ids(refined.target.mesh)),
        hierarchy=refined.hierarchy,
    )
    target = restored.target.mesh
    # Marked cells that were never refined have no family to merge into.
    assert np.all(np.isin(restored.evidence.rejected_coarsening_ids, _cell_ids(mesh)))
    assert restored.evidence.coarsened_vertices == (
        refined.target.mesh.coordinates.shape[0] - mesh.coordinates.shape[0]
    )
    assert target.topology_id == source.mesh.topology_id
    np.testing.assert_array_equal(target.vertex_global_ids, mesh.vertex_global_ids)
    target_ids, target_cells = _canonical_cells(target)
    source_ids, source_cells = _canonical_cells(mesh)
    np.testing.assert_array_equal(target_ids, source_ids)
    np.testing.assert_array_equal(target_cells, source_cells)
    np.testing.assert_array_equal(target.coordinates, mesh.coordinates)


@pytest.mark.parametrize(
    "mesh_builder",
    (
        pytest.param(partial(triangle_grid, 3, 3), id="triangles"),
        pytest.param(partial(kuhn_cube, 2, orientation_swap=(0, 1)), id="tetrahedra"),
    ),
)
def test_bisection_transfer_reproduces_affine_fields(
    mesh_builder: Callable[[], CellMesh],
) -> None:
    mesh = mesh_builder()
    result = _adapt(_certified(mesh), np.sort(_cell_ids(mesh))[::2])
    result = _adapt(
        result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
    )
    slope = np.linspace(0.5, 1.5, mesh.ambient_dimension)
    before = np.asarray(result.source.mesh.coordinates) @ slope + 0.25
    expected = np.asarray(result.target.mesh.coordinates) @ slope + 0.25
    np.testing.assert_allclose(
        np.asarray(result.transfer.apply(before)),
        expected,
        rtol=0.0,
        atol=_AFFINE_TRANSFER_ROUNDOFF,
    )


def test_protected_edge_retains_identity_and_reports_rejected_marks() -> None:
    source = _certified(triangle_grid(4, 4))
    mesh = source.mesh
    edges = mesh.entity_set(1)
    keys = _edge_keys(mesh)
    points = np.asarray(mesh.coordinates)
    offsets = points[keys[:, 1]] - points[keys[:, 0]]
    diagonal = np.flatnonzero(np.all(np.isclose(offsets, 0.25), axis=1))[4]
    guarded = keys[diagonal]
    identifier = np.asarray(edges.entity_ids)[diagonal]
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        edges.entity_set_id,
        np.asarray([identifier]),
    )
    cell_vertices = np.asarray(mesh.vertex_global_ids)[_cells(mesh)]
    through = np.sort(
        _cell_ids(mesh)[np.sum(np.isin(cell_vertices, guarded), axis=1) == 2]
    )
    elsewhere = np.setdiff1d(_cell_ids(mesh), through)[[0, -1]]
    result = _adapt(source, np.union1d(through, elsewhere), protected_scopes=(scope,))
    assert result.status is MeshAdaptationStatus.PARTIAL
    np.testing.assert_array_equal(result.evidence.rejected_refinement_ids, through)
    target = result.target.mesh
    row = np.flatnonzero(np.all(_edge_keys(target) == guarded, axis=1))
    assert row.size == 1
    assert np.asarray(target.entity_set(1).entity_ids)[row[0]] == identifier
    assert not np.any(np.isin(_cell_ids(target), elsewhere))


def test_coarsening_preserves_protected_bisection_vertex() -> None:
    source = _certified(triangle_grid(2, 2))
    refined = _adapt(source, np.sort(_cell_ids(source.mesh)))
    refined = _adapt(
        refined.target,
        np.sort(_cell_ids(refined.target.mesh)),
        hierarchy=refined.hierarchy,
    )
    mesh = refined.target.mesh
    identifiers = np.asarray(mesh.vertex_global_ids)
    created = np.setdiff1d(identifiers, np.asarray(source.mesh.vertex_global_ids))
    kept = created[:1]
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        mesh.entity_set(0).entity_set_id,
        kept,
    )
    result = _adapt(
        refined.target,
        (),
        np.sort(_cell_ids(mesh)),
        hierarchy=refined.hierarchy,
        protected_scopes=(scope,),
    )
    remaining = np.asarray(result.target.mesh.vertex_global_ids)
    surviving = np.setdiff1d(remaining, np.asarray(source.mesh.vertex_global_ids))
    np.testing.assert_array_equal(surviving, kept)
    assert result.evidence.coarsened_vertices == created.size - 1
    _assert_conforming(result.target.mesh, 1.0)


def test_uniform_compatibility_preparation_closes_incompatible_source_labels() -> None:
    points = np.asarray([(0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)])
    mesh = CellMesh.from_triangles(points, np.asarray([(0, 1, 2), (0, 3, 1)]))
    source = _certified(mesh)
    with pytest.raises(ValueError, match=r"Bisection\.compatibility"):
        _adapt(source, (0,))
    result = _adapt(source, (0,), compatibility=BisectionCompatibility.UNIFORM_REFINEMENT)
    assert result.evidence.uniform_refinement_applied
    assert result.evidence.initially_compatible is False
    _assert_conforming(result.target.mesh, 3.5)
    follow = _adapt(
        result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
    )
    _assert_conforming(follow.target.mesh, 3.5)


def test_bisection_repeats_with_identical_result_and_forest_ids() -> None:
    first = _adapt(_certified(kuhn_cube(2, orientation_swap=(0, 1))), (0, 7, 13))
    second = _adapt(_certified(kuhn_cube(2, orientation_swap=(0, 1))), (0, 7, 13))
    assert first.result_id == second.result_id
    assert first.hierarchy.hierarchy_id == second.hierarchy.hierarchy_id


def test_changed_target_receipt_prefix_preserves_independent_bisection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing._native_preparation import NativePreparationLedger

    initial = _certified(triangle_grid(2, 2))
    source = _adapt(initial, np.sort(_cell_ids(initial.mesh))).target
    marks = np.sort(_cell_ids(source.mesh))
    with monkeypatch.context() as patch:
        patch.setattr(NativePreparationLedger, "retain_receipt_snapshot", lambda *args, **kwargs: None)
        reference = _adapt(source, marks)
    actual = _adapt(source, marks)
    assert actual.result_id == reference.result_id
    assert actual.target.result_id == reference.target.result_id
    assert actual.hierarchy.hierarchy_id == reference.hierarchy.hierarchy_id
    for name in ("mesh", "geometry", "audit", "certification"):
        assert phx.array_tree_fingerprint(getattr(actual.target, name)) == (
            phx.array_tree_fingerprint(getattr(reference.target, name))
        )
    for name in ("hierarchy", "lineage", "stencil"):
        assert phx.array_tree_fingerprint(getattr(actual, name)) == (
            phx.array_tree_fingerprint(getattr(reference, name))
        )
    _assert_conforming(actual.target.mesh, 1.0)


@pytest.mark.parametrize(
    ("dimension", "siblings"),
    (pytest.param(2, 6, id="triangles"), pytest.param(3, 24, id="tetrahedra")),
)
def test_uniform_source_inverse_restores_scientific_identity_after_cold_archive(
    dimension: int, siblings: int, tmp_path: Path
) -> None:
    source = _certified(incompatible_simplex(dimension))
    refined = _adapt(
        source, (0,), compatibility=BisectionCompatibility.UNIFORM_REFINEMENT
    )
    assert refined.status is MeshAdaptationStatus.COMPLETE
    assert refined.hierarchy.uniform_refinement is not None
    lineage = refined.hierarchy.uniform_refinement
    assert lineage.child_ids.shape[1] == siblings
    assert np.any(np.asarray(lineage.barycentric_weights) == 1.0 / 3.0)
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    cold = write_meshing_source_closure(
        tmp_path / "uniform-cold",
        (source, refined.target, refined.hierarchy, policy),
    )
    cold_record = run_on_forced_devices(
        "tests._support.distributed_meshing.cold_lifecycle",
        "uniform_source_inverse",
        (str(cold.path), cold.content_id),
        devices=1,
        env={},
    )
    assert cold_record["status"] == MeshAdaptationStatus.COMPLETE.value
    cold_original, cold_target = cold_record["original"], cold_record["target"]
    np.testing.assert_array_equal(
        record_array(cold_target["coordinates"]),
        record_array(cold_original["coordinates"]),
    )
    np.testing.assert_array_equal(
        record_array(cold_target["vertex_global_ids"]),
        record_array(cold_original["vertex_global_ids"]),
    )
    for before, after in zip(cold_original["blocks"], cold_target["blocks"], strict=True):
        np.testing.assert_array_equal(
            record_array(after["global_ids"]), record_array(before["global_ids"])
        )
        np.testing.assert_array_equal(
            record_array(after["vertices"]), record_array(before["vertices"])
        )
    for degree in range(source.mesh.topological_dimension + 1):
        np.testing.assert_array_equal(
            record_array(cold_record["target_entity_ids"][degree]),
            record_array(cold_record["original_entity_ids"][degree]),
        )
    np.testing.assert_array_equal(
        record_array(cold_record["tags"]), record_array(cold_record["parent_tags"])
    )
    np.testing.assert_array_equal(
        record_array(cold_record["generations"]),
        record_array(cold_record["parent_levels"]),
    )
    assert cold_record["fine_work"] == cold_record["before_work"]
    assert cold_record["total_work_units"] > 0
    assert cold_record["host_storage_peak_bytes_upper"] > 0
    assert cold_record["uniform_refinement_absent"]


@pytest.mark.parametrize(
    ("dimension", "siblings"),
    (pytest.param(2, 6, id="triangles"), pytest.param(3, 24, id="tetrahedra")),
)
def test_uniform_source_inverse_preserves_protected_rollback_and_can_continue(
    dimension: int, siblings: int, tmp_path: Path
) -> None:
    source = _certified(incompatible_simplex(dimension))
    original_ids = _cell_ids(source.mesh)
    refined = _adapt(
        source, (0,), compatibility=BisectionCompatibility.UNIFORM_REFINEMENT
    )
    assert refined.status is MeshAdaptationStatus.COMPLETE
    assert refined.hierarchy.uniform_refinement is not None
    lineage = refined.hierarchy.uniform_refinement
    assert lineage.child_ids.shape[1] == siblings
    assert np.any(np.asarray(lineage.barycentric_weights) == 1.0 / 3.0)
    receipt = write_meshing_source_closure(
        tmp_path / "uniform-triangle", (refined.target, refined.hierarchy)
    )
    target, hierarchy = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    born = np.setdiff1d(
        np.asarray(target.mesh.vertex_global_ids),
        np.asarray(source.mesh.vertex_global_ids),
    )
    scope = MeshingScope(
        target.mesh.mesh_id,
        target.mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        target.mesh.entity_set(0).entity_set_id,
        born[:1],
    )
    refused = _adapt(
        target, (), _cell_ids(target.mesh), hierarchy=hierarchy, protected_scopes=(scope,)
    )
    assert refused.status is MeshAdaptationStatus.PARTIAL
    np.testing.assert_array_equal(
        target.mesh.vertex_global_ids, refined.target.mesh.vertex_global_ids
    )
    np.testing.assert_array_equal(
        target.geometry.coordinates, refined.target.geometry.coordinates
    )
    protected_row = np.flatnonzero(
        np.asarray(refused.target.mesh.vertex_global_ids) == born[0]
    )
    assert protected_row.size == 1
    np.testing.assert_array_equal(
        np.asarray(refused.target.mesh.coordinates)[protected_row[0]],
        np.asarray(target.mesh.coordinates)[
            np.flatnonzero(np.asarray(target.mesh.vertex_global_ids) == born[0])[0]
        ],
    )
    with coordinate_reference_budget(
        queries_upper=(
            cell_geometry_preparation_query_upper(refused.target.geometry)
            + cell_geometry_preparation_query_upper(target.geometry)
        ),
    ) as budget:
        assert refused.target.geometry.source_coordinates(
            budget=budget
        ) == target.geometry.source_coordinates(budget=budget)
    assert (
        refused.hierarchy.uniform_refinement.lineage_id
        == hierarchy.uniform_refinement.lineage_id
    )
    restored = _adapt(target, (), _cell_ids(target.mesh), hierarchy=hierarchy)
    assert restored.status is MeshAdaptationStatus.COMPLETE
    np.testing.assert_array_equal(_cell_ids(restored.target.mesh), original_ids)
    np.testing.assert_array_equal(
        restored.target.mesh.vertex_global_ids, source.mesh.vertex_global_ids
    )
    np.testing.assert_array_equal(
        restored.target.mesh.coordinates, source.mesh.coordinates
    )
    np.testing.assert_array_equal(_cells(restored.target.mesh), _cells(source.mesh))
    for degree in range(dimension + 1):
        np.testing.assert_array_equal(
            restored.target.mesh.entity_set(degree).entity_ids,
            source.mesh.entity_set(degree).entity_ids,
        )
    np.testing.assert_array_equal(restored.hierarchy.tags, lineage.parent_tags)
    np.testing.assert_array_equal(restored.hierarchy.generations, lineage.parent_levels)
    assert restored.hierarchy.uniform_refinement is None
    continued = _adapt(
        restored.target,
        original_ids[:1],
        hierarchy=restored.hierarchy,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    assert continued.status is MeshAdaptationStatus.COMPLETE
    _assert_conforming(
        continued.target.mesh,
        float(
            np.sum(
                _signed_measures(np.asarray(source.mesh.coordinates), _cells(source.mesh))
            )
        ),
    )


@pytest.mark.parametrize("dimension", (2, 3), ids=("triangles", "tetrahedra"))
def test_partial_uniform_inverse_preserves_original_block_rows_and_coefficient_routes(
    dimension: int,
    tmp_path: Path,
) -> None:

    width = dimension + 1
    simplex = np.concatenate(
        (np.zeros((1, dimension), dtype=np.float64), np.eye(dimension, dtype=np.float64))
    )
    if dimension == 2:
        component = np.asarray(
            ((0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)), dtype=np.float64
        )
        component_cells = np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32)
    else:
        component = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (1.0, 0.5, 0.0),
                (1.0, 0.2, 1.0),
                (1.0, -3.0, -1.0),
            ),
            dtype=np.float64,
        )
        component_cells = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int32)
    points = np.concatenate((component, simplex + 4.0))
    kind = "triangle" if dimension == 2 else "tetrahedron"
    original = CellMesh(
        points,
        (
            CellBlock(
                "z-root",
                kind,
                component_cells,
                global_ids=np.asarray((90, 10), dtype=np.int64),
            ),
            CellBlock(
                "a-root",
                kind,
                np.arange(component.shape[0], component.shape[0] + width, dtype=np.int32)[
                    None
                ],
                global_ids=np.asarray((50,), dtype=np.int64),
            ),
        ),
    )
    original = phx.meshing.canonicalize_cell_mesh(original)
    coefficient_order = np.arange(points.shape[0] - 1, -1, -1, dtype=np.int32)
    coordinate_rows = np.argsort(coefficient_order).astype(np.int32)
    element = coordinate_lagrange_element(kind, 1)
    geometry = CellGeometrySpec(
        {block.name: element for block in original.blocks},
        {
            block.name: coordinate_rows[np.asarray(block.vertices)]
            for block in original.blocks
        },
        points[coefficient_order],
    )
    source = phx.meshing.certify_cell_mesh(
        original, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    fine = _adapt(source, (), compatibility=BisectionCompatibility.UNIFORM_REFINEMENT)
    lineage = fine.hierarchy.uniform_refinement
    assert lineage is not None
    original_weights = np.asarray(lineage.barycentric_weights).copy()
    root = np.flatnonzero(np.asarray(lineage.parent_ids) == 90)[0]

    wrong_sci = eqx.tree_at(
        lambda value: value.uniform_refinement.parent_blocks,
        fine.hierarchy,
        lineage.parent_blocks.at[root].set(
            tuple(block.name for block in original.blocks).index("a-root")
        ),
    )
    with pytest.raises(ValueError, match="BisectionUniformRefinement.parent_blocks"):
        _adapt(fine.target, (), np.asarray(lineage.child_ids)[root], hierarchy=wrong_sci)
    wrong_support = eqx.tree_at(
        lambda value: value.uniform_refinement.barycentric_weights,
        fine.hierarchy,
        lineage.barycentric_weights.at[root, 0, 0, 0].set(0.25),
    )
    with pytest.raises(ValueError, match="barycentric_weights"):
        _adapt(
            fine.target, (), np.asarray(lineage.child_ids)[root], hierarchy=wrong_support
        )
    wrong_bank = eqx.tree_at(
        lambda value: value.geometry.coordinates,
        fine.target,
        fine.target.geometry.coordinates.at[0, 0].add(1.0),
    )
    with pytest.raises(ValueError, match=r"BisectionUniformRefinement\.coefficient_bank"):
        _adapt(
            wrong_bank, (), np.asarray(lineage.child_ids)[root], hierarchy=fine.hierarchy
        )
    selected_roots = np.isin(
        np.asarray(lineage.parent_ids), np.asarray((90, 10), dtype=np.int64)
    )
    partial = _adapt(
        fine.target,
        (),
        np.asarray(lineage.child_ids)[selected_roots].reshape(-1),
        hierarchy=fine.hierarchy,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    restored_block = next(
        block
        for block in partial.target.mesh.blocks
        if 90 in np.asarray(block.global_ids)
    )
    assert restored_block.name == "z-root"
    original_block = next(block for block in original.blocks if block.name == "z-root")
    np.testing.assert_array_equal(restored_block.global_ids, original_block.global_ids)
    elements, routes, bank = partial.target.geometry.resolve(partial.target.mesh)
    slot = tuple(block.name for block in partial.target.mesh.blocks).index("z-root")
    assert elements[slot].element_id == element.element_id
    np.testing.assert_array_equal(
        routes[slot], coordinate_rows[np.asarray(original_block.vertices)]
    )
    np.testing.assert_array_equal(bank, geometry.coordinates)
    np.testing.assert_array_equal(lineage.barycentric_weights, original_weights)
    receipt = write_meshing_source_closure(
        tmp_path / "partial-uniform", (source, partial.target, partial.hierarchy)
    )
    original_source, reopened, hierarchy = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    active_ids = _cell_ids(reopened.mesh)
    remaining_children = active_ids[~np.isin(active_ids, _cell_ids(original_source.mesh))]
    completed = _adapt(
        reopened,
        (),
        remaining_children,
        hierarchy=hierarchy,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    assert completed.status is MeshAdaptationStatus.COMPLETE
    for before, after in zip(
        original_source.mesh.blocks, completed.target.mesh.blocks, strict=True
    ):
        assert after.name == before.name
        np.testing.assert_array_equal(after.global_ids, before.global_ids)
        np.testing.assert_array_equal(after.vertices, before.vertices)
    np.testing.assert_array_equal(
        completed.target.geometry.coordinates, original_source.geometry.coordinates
    )


@pytest.mark.parametrize("degree", (1, 2), ids=("edges", "cells"))
def test_uniform_source_inverse_refuses_changed_scientific_classes(degree: int) -> None:
    points = np.asarray(
        ((0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)), dtype=np.float64
    )
    source = _certified(
        CellMesh.from_triangles(
            points, np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32)
        )
    )
    refined = _adapt(source, (), compatibility=BisectionCompatibility.UNIFORM_REFINEMENT)
    fine = refined.target
    scope = MeshingScope(
        fine.mesh.mesh_id,
        fine.mesh.numeric_version,
        MeshingEntityKind.MESH,
        degree,
        fine.mesh.entity_set(degree).entity_set_id,
        np.asarray(fine.mesh.entity_set(degree).entity_ids, dtype=np.int64)[:1],
    )
    changed = phx.meshing.certify_cell_mesh(
        fine.mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=fine.geometry,
        labels=(phx.meshing.MeshLabel("changed-scientific-class", scope),),
    )
    refused = _adapt(
        changed, (), np.sort(_cell_ids(fine.mesh)), hierarchy=refined.hierarchy
    )
    assert refused.status is MeshAdaptationStatus.PARTIAL
    assert refused.evidence.rejected_coarsening_ids.size > 0
    assert refused.hierarchy.uniform_refinement is not None
    np.testing.assert_array_equal(refused.target.mesh.coordinates, fine.mesh.coordinates)
    np.testing.assert_array_equal(_cell_ids(refused.target.mesh), _cell_ids(fine.mesh))
    np.testing.assert_array_equal(_cells(refused.target.mesh), _cells(fine.mesh))


def _jittered_triangle_grid() -> CellMesh:
    """Unit-square grid whose jittered interior breaks the longest-edge matching."""

    mesh = triangle_grid(4, 4)
    points = np.asarray(mesh.coordinates).copy()
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    offsets = np.random.default_rng(13).uniform(-0.06, 0.06, size=points.shape)
    points[interior] += offsets[interior]
    return CellMesh.from_triangles(points, _cells(mesh).astype(np.int32))


def test_conforming_closure_bisects_incompatible_triangle_labels_locally() -> None:
    mesh = _jittered_triangle_grid()
    source = _certified(mesh)
    marks = np.sort(_cell_ids(mesh))[::5]
    with pytest.raises(ValueError, match=r"Bisection\.compatibility"):
        _adapt(source, marks)
    result = _adapt(
        source, marks, compatibility=BisectionCompatibility.CONFORMING_CLOSURE
    )
    evidence = result.evidence
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert evidence.initially_compatible is False
    assert evidence.incompatible_facets > 0
    assert not evidence.uniform_refinement_applied
    assert evidence.accepted_refinements == marks.size
    assert evidence.bisections > marks.size
    assert evidence.maximum_generation <= 2
    target = result.target.mesh
    assert sum(block.cell_count for block in target.blocks) <= 4 * sum(
        block.cell_count for block in mesh.blocks
    )
    _assert_conforming(target, 1.0)
    _assert_boundary_on_unit_box(target)
    # A closure from the labelled start splits source edges only.
    cells = _cells(mesh)
    pairs = np.asarray(tuple(itertools.combinations(range(3), 2)))
    edges = np.unique(np.sort(cells[:, pairs].reshape((-1, 2)), axis=1), axis=0)
    points = np.asarray(mesh.coordinates)
    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    created = ~np.isin(
        np.asarray(target.vertex_global_ids), np.asarray(mesh.vertex_global_ids)
    )
    assert np.count_nonzero(created) == evidence.created_vertices
    for point in np.asarray(target.coordinates)[created]:
        assert np.min(np.linalg.norm(midpoints - point, axis=1)) < _MIDPOINT_ROUNDOFF
    follow = _adapt(result.target, _corner_cells(target), hierarchy=result.hierarchy)
    assert follow.status is MeshAdaptationStatus.COMPLETE
    _assert_conforming(follow.target.mesh, 1.0)
    restored = _adapt(
        result.target, (), np.sort(_cell_ids(target)), hierarchy=result.hierarchy
    )
    # Restoration rejoins every root to its authored block.
    assert restored.target.mesh.topology_id == source.mesh.topology_id
    target_ids, target_cells = _canonical_cells(restored.target.mesh)
    source_ids, source_cells = _canonical_cells(mesh)
    np.testing.assert_array_equal(target_ids, source_ids)
    np.testing.assert_array_equal(target_cells, source_cells)
    np.testing.assert_array_equal(restored.target.mesh.coordinates, mesh.coordinates)


def test_conforming_closure_keeps_incompatible_tetrahedral_labels_refused() -> None:
    mesh = kuhn_cube(2, orientation_swap=(0, 1))
    points = np.asarray(mesh.coordinates).copy()
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    points[interior] += 0.1
    source = _certified(CellMesh.from_tetrahedra(points, _cells(mesh).astype(np.int32)))
    with pytest.raises(ValueError, match=r"Bisection\.compatibility"):
        _adapt(source, (0,), compatibility=BisectionCompatibility.CONFORMING_CLOSURE)


def test_device_bisection_refuses_label_conforming_closure() -> None:
    with pytest.raises(ValueError, match="NATIVE_BISECTION"):
        MeshAdaptationPolicy(
            MeshAdaptationRoute.DEVICE_BISECTION,
            compatibility=BisectionCompatibility.CONFORMING_CLOSURE,
            device_policy=phx.discretization.AdaptiveSimplexPolicy(
                vertex_capacity=64, cell_capacity=64
            ),
        )


@pytest.mark.parametrize(
    ("limits", "requested", "achieved"),
    [
        pytest.param(
            phx.meshing.MeshingLimits(maximum_cells=40),
            "maximum_cells",
            "active_cells",
            id="cells",
        ),
        pytest.param(
            phx.meshing.MeshingLimits(maximum_vertices=26),
            "maximum_vertices",
            "vertices",
            id="vertices",
        ),
        pytest.param(
            phx.meshing.MeshingLimits(maximum_work_units=11),
            "maximum_work_units",
            "bisection_steps",
            id="work-units",
        ),
    ],
)
def test_bisection_closure_refuses_batches_beyond_declared_capacity(
    limits: Any, requested: str, achieved: str
) -> None:
    source = _certified(triangle_grid(4, 4))
    marks = np.sort(_cell_ids(source.mesh))[::3]
    assert marks.size == 11
    with pytest.raises(phx.meshing.MeshingFailure) as refused:
        _adapt(source, marks, limits=limits)
    failure = refused.value
    assert failure.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert failure.stage == "bisection-closure"
    assert (
        dict(failure.evidence.achieved)[achieved]
        > dict(failure.evidence.requested)[requested]
    )


def _quadratic_geometry(mesh: CellMesh) -> Any:
    """Quadratic coordinate map whose edge nodes bow off the straight corners."""

    element = phx.discretization.coordinate_lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes)
    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    coordinates = (
        corners[:, :1]
        + nodes[None, :, :1] * (corners[:, 1:2] - corners[:, :1])
        + nodes[None, :, 1:] * (corners[:, 2:3] - corners[:, :1])
    )
    edge_dofs = np.asarray(
        [dof for entity in element.entity_dofs[1] for dof in entity], dtype=np.int32
    )
    coordinates[:, edge_dofs, 1] += 0.01
    routes = np.arange(coordinates.shape[0] * coordinates.shape[1]).reshape(
        coordinates.shape[:2]
    )
    return phx.discretization.CellGeometrySpec(
        {mesh.blocks[0].name: element},
        {mesh.blocks[0].name: routes},
        coordinates.reshape((-1, 2)),
    )


def test_bisection_keeps_curved_source_maps_and_affine_successors_affine() -> None:
    mesh = phx.meshing.canonicalize_cell_mesh(triangle_grid(2, 2))
    curved = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=_quadratic_geometry(mesh)
    )

    refined = _adapt(curved, (0,))
    target = refined.target
    assert refined.status is MeshAdaptationStatus.COMPLETE
    assert target.geometry.elements[0].element_id == (
        curved.geometry.elements[0].element_id
    )
    # A P2 map sends an edge's reference midpoint to that edge's node, so every
    # bisection vertex is a source edge node, not the straight corner midpoint.
    old = np.asarray(curved.mesh.vertex_global_ids)
    created = ~np.isin(np.asarray(target.mesh.vertex_global_ids), old)
    nodes = np.asarray(curved.geometry.coordinates)
    element = phx.discretization.coordinate_lagrange_element("triangle", 2)
    edge_dofs = np.asarray(
        [dof for entity in element.entity_dofs[1] for dof in entity], dtype=np.int32
    )
    edge_nodes = nodes[
        np.asarray(curved.geometry.geometry_dofs[0])[:, edge_dofs].reshape(-1)
    ]
    for point in np.asarray(target.mesh.coordinates)[created]:
        assert np.min(np.linalg.norm(edge_nodes - point, axis=1)) < _MIDPOINT_ROUNDOFF
    assert np.any(np.abs(np.asarray(target.mesh.coordinates)[created, 1] % 0.5) > 1e-3)

    result = _adapt(_certified(mesh), (0,))
    affine = phx.discretization.CellGeometrySpec.affine(result.target.mesh)
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.geometry_transition is None
    # P1 coefficient actions retain ancestry in the coordinate layout identity
    # rather than flattening into the incidental vertex-coordinate presentation.
    assert result.target.geometry.geometry_layout_id != affine.geometry_layout_id
    assert all(element.degree == 1 for element in result.target.geometry.elements)
    elements, routes, coefficients = result.target.geometry.resolve(result.target.mesh)
    for block, element, route in zip(
        result.target.mesh.blocks, elements, routes, strict=True
    ):
        corners = phx.discretization.coordinate_lagrange_element(
            block.cell_kind, 1
        ).reference_nodes
        represented = (
            np.asarray(element.tabulate(corners)[0])
            @ np.asarray(coefficients)[np.asarray(route)]
        )
        np.testing.assert_array_equal(
            represented,
            np.asarray(result.target.mesh.coordinates)[np.asarray(block.vertices)],
        )


@pytest.fixture(scope="module")
def uniform_refinement_state() -> MeshAdaptationResult:
    source = _certified(incompatible_simplex(2))
    return _adapt(source, (), compatibility=BisectionCompatibility.UNIFORM_REFINEMENT)


def test_stored_uniform_actions_rebuild_without_changing_scientific_identity(
    uniform_refinement_state: MeshAdaptationResult,
) -> None:
    hierarchy = uniform_refinement_state.hierarchy
    assert isinstance(hierarchy, BisectionHierarchy)
    lineage = hierarchy.uniform_refinement
    assert lineage is not None
    rebuilt = BisectionUniformRefinement.from_stored_fields(lineage)
    assert rebuilt is not lineage
    assert rebuilt.source is lineage.source
    assert rebuilt.lineage_id == lineage.lineage_id
    expected_arrays = lineage.host_arrays()
    rebuilt_arrays = rebuilt.host_arrays()
    for name, expected in expected_arrays.items():
        np.testing.assert_array_equal(rebuilt_arrays[name], expected)


def test_stored_uniform_actions_refuse_corrupted_barycentric_support(
    uniform_refinement_state: MeshAdaptationResult,
) -> None:
    hierarchy = uniform_refinement_state.hierarchy
    assert isinstance(hierarchy, BisectionHierarchy)
    lineage = hierarchy.uniform_refinement
    assert lineage is not None
    corrupt = eqx.tree_at(
        lambda value: value.barycentric_weights,
        lineage,
        lineage.barycentric_weights.at[0, 0, 0, 0].add(0.25),
    )
    with pytest.raises(
        ValueError, match="BisectionUniformRefinement.barycentric_weights"
    ):
        BisectionUniformRefinement.from_stored_fields(corrupt)


def test_stored_hierarchy_rebuild_retains_cell_labels_and_issuers(
    uniform_refinement_state: MeshAdaptationResult,
) -> None:
    hierarchy = uniform_refinement_state.hierarchy
    assert isinstance(hierarchy, BisectionHierarchy)
    rebuilt = BisectionHierarchy.from_stored_fields(hierarchy)
    assert rebuilt is not hierarchy
    assert rebuilt.hierarchy_id == hierarchy.hierarchy_id
    assert rebuilt.uniform_refinement is hierarchy.uniform_refinement
    assert rebuilt.next_vertex_id == hierarchy.next_vertex_id
    assert rebuilt.next_cell_id == hierarchy.next_cell_id
    np.testing.assert_array_equal(rebuilt.cell_global_ids, hierarchy.cell_global_ids)
    np.testing.assert_array_equal(rebuilt.ordered_vertices, hierarchy.ordered_vertices)
    np.testing.assert_array_equal(rebuilt.tags, hierarchy.tags)
    np.testing.assert_array_equal(rebuilt.generations, hierarchy.generations)
    np.testing.assert_array_equal(
        rebuilt.scientific_cell_ids, hierarchy.scientific_cell_ids
    )
    np.testing.assert_array_equal(
        rebuilt.scientific_block_ids, hierarchy.scientific_block_ids
    )
    for restored, retained in zip(
        rebuilt.retired_entity_ids, hierarchy.retired_entity_ids, strict=True
    ):
        np.testing.assert_array_equal(restored, retained)


def test_stored_hierarchy_refuses_corrupted_simplex_incidence(
    uniform_refinement_state: MeshAdaptationResult,
) -> None:
    hierarchy = uniform_refinement_state.hierarchy
    assert isinstance(hierarchy, BisectionHierarchy)
    corrupt = eqx.tree_at(
        lambda value: value.ordered_vertices,
        hierarchy,
        hierarchy.ordered_vertices.at[0, 1].set(hierarchy.ordered_vertices[0, 0]),
    )
    with pytest.raises(ValueError, match="Bisection.ordered_vertices"):
        BisectionHierarchy.from_stored_fields(corrupt)


def test_rectangular_triangle_fixture_retains_storage_and_cell_identity() -> None:
    mesh = triangle_grid(2, 1, height=0.5)
    np.testing.assert_array_equal(
        mesh.coordinates,
        ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.5), (0.5, 0.5), (1.0, 0.5)),
    )
    np.testing.assert_array_equal(
        _cells(mesh), ((0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4))
    )
    np.testing.assert_array_equal(_cell_ids(mesh), (0, 1, 2, 3))


def test_restricted_parent_bisection_publishes_each_child_map_and_conserves() -> None:
    # Children of a restricted parent own distinct restricted charts; each is
    # published in its own block while cell identities, the untouched sibling
    # parent, the rational source images and the conservative transfer survive.
    mesh, geometry = affine_tetrahedral_root_restrictions(TETRAHEDRON + 5.0)
    source = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    marked, untouched = (int(value) for value in _cell_ids(source.mesh))
    refined = _adapt(source, (marked,))
    assert refined.status is MeshAdaptationStatus.COMPLETE
    target = refined.target
    identifiers = _cell_ids(target.mesh)
    assert identifiers.size == 3
    assert untouched in identifiers and marked not in identifiers
    elements = target.geometry.elements
    assert all(isinstance(element, RestrictedCellGeometryElement) for element in elements)
    assert len({element.element_id for element in elements}) == len(target.mesh.blocks)
    with coordinate_reference_budget() as budget:
        positions = affine_vertex_positions(target.mesh, target.geometry, budget=budget)
    for point, carrier in zip(
        positions, np.asarray(target.mesh.coordinates).tolist(), strict=True
    ):
        assert tuple(float(value) for value in point) == tuple(carrier)
    transfer = refined.transfer
    assert transfer is not None and transfer.conservative
    density = 2.0 + np.asarray(source.mesh.coordinates) @ np.asarray((0.2, 0.3, 0.4))
    transferred = np.asarray(transfer.apply(density))

    def inventory(cells_mesh: CellMesh, values: np.ndarray) -> float:
        cells = _cells(cells_mesh)
        measures = np.abs(_signed_measures(np.asarray(cells_mesh.coordinates), cells))
        return float(np.sum(measures * np.mean(values[cells], axis=1)))

    np.testing.assert_allclose(
        inventory(target.mesh, transferred),
        inventory(source.mesh, density),
        rtol=0.0,
        atol=_AFFINE_TRANSFER_ROUNDOFF,
    )


@pytest.mark.parametrize("generations", (1, 2))
def test_whole_edge_star_refines_every_tetrahedron_without_uniform_prefix(
    generations: int,
) -> None:
    mesh = kuhn_cube(2, jitter=0.03)
    source = _certified(mesh)
    marks = np.sort(_cell_ids(mesh))
    refined = _adapt(
        source, marks, compatibility=BisectionCompatibility.WHOLE_EDGE_STAR,
        refinement_generations=generations,
    )
    assert refined.status is MeshAdaptationStatus.COMPLETE
    assert refined.evidence.accepted_refinements == marks.size
    assert not refined.evidence.uniform_refinement_applied
    assert not np.intersect1d(marks, _cell_ids(refined.target.mesh)).size
    assert refined.hierarchy.refinement_rule == "whole_edge_star"
    assert 2**generations * marks.size <= _cell_ids(refined.target.mesh).size <= 8**generations * marks.size
    _assert_conforming(refined.target.mesh, 1.0)
    from tools.meshing_qualification._descent import certify_descent

    descent = certify_descent(mesh, refined.target.mesh, refined.hierarchy)
    assert descent["refinement_rule"] == "whole_edge_star"
    assert descent["shape_class_cap"] is None
    assert descent["quality_floor_rule"] == (
        "finite-forest-announced-edge-tagged-reference-floor"
    )
    rebuilt = BisectionHierarchy.from_stored_fields(refined.hierarchy)
    assert rebuilt.hierarchy_id == refined.hierarchy.hierarchy_id
    assert rebuilt.refinement_rule == "whole_edge_star"
    restored = _adapt(
        refined.target, (), np.sort(_cell_ids(refined.target.mesh)),
        hierarchy=refined.hierarchy,
    )
    assert restored.target.mesh.topology_id == mesh.topology_id
    assert restored.hierarchy.refinement_rule == "whole_edge_star"
    from tools.meshing_qualification._runtime import QualificationFailure

    corrupted = eqx.tree_at(
        lambda value: value.record_vertex_ids,
        refined.hierarchy,
        np.asarray(refined.hierarchy.record_vertex_ids) + 1,
    )
    with pytest.raises(QualificationFailure):
        certify_descent(mesh, refined.target.mesh, corrupted)


def test_whole_edge_star_continuation_keeps_unsplit_inherited_labels() -> None:
    from tools.meshing_qualification._descent import certify_descent

    mesh = kuhn_cube(1)
    first = _adapt(
        _certified(mesh), np.sort(_cell_ids(mesh)),
        compatibility=BisectionCompatibility.WHOLE_EDGE_STAR,
    )
    second = _adapt(
        first.target, np.asarray(first.hierarchy.cell_global_ids)[:1],
        hierarchy=first.hierarchy,
        compatibility=BisectionCompatibility.WHOLE_EDGE_STAR,
    )
    before, after = first.hierarchy, second.hierarchy
    retained, old_rows, new_rows = np.intersect1d(
        before.cell_global_ids, after.cell_global_ids, return_indices=True
    )
    assert retained.size
    np.testing.assert_array_equal(
        np.asarray(before.tags)[old_rows], np.asarray(after.tags)[new_rows]
    )
    np.testing.assert_array_equal(
        np.asarray(before.ordered_vertices)[old_rows],
        np.asarray(after.ordered_vertices)[new_rows],
    )
    assert certify_descent(mesh, second.target.mesh, after)["status"] == "certified"


def test_whole_edge_star_refuses_triangle_input() -> None:
    with pytest.raises(ValueError, match="WHOLE_EDGE_STAR requires tetrahedral"):
        _adapt(
            _certified(triangle_grid(1, 1)), (0,),
            compatibility=BisectionCompatibility.WHOLE_EDGE_STAR,
        )


def test_source_edge_star_subdivision_has_eight_child_hard_bound() -> None:
    # Exhaust all 6! local edge orders. A global source-edge schedule induces
    # one of these orders in each source tetrahedron, or a prefix thereof.
    edges = tuple(itertools.combinations(range(4), 2))
    for order in itertools.permutations(edges):
        leaves = {(0, 1, 2, 3)}
        for midpoint, edge in enumerate(order, start=4):
            children = set()
            for cell in leaves:
                if set(edge) <= set(cell):
                    for endpoint in edge:
                        children.add(tuple(sorted(
                            midpoint if vertex == endpoint else vertex for vertex in cell
                        )))
                else:
                    children.add(cell)
            leaves = children
            assert len(leaves) <= 8
        assert len(leaves) == 8


@pytest.mark.parametrize("dimension", (2, 3))
def test_exact_nested_partition_reuses_barycentric_source_embedding(
    dimension: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from phydrax.geometry import _restricted_embedding as owner

    fallback = owner._nested_vertices
    fallback_calls = 0

    def counted_fallback(*args: Any, **kwargs: Any) -> Any:
        nonlocal fallback_calls
        fallback_calls += 1
        return fallback(*args, **kwargs)

    monkeypatch.setattr(owner, "_nested_vertices", counted_fallback)

    mesh = triangle_grid(1, 1) if dimension == 2 else kuhn_cube()
    source = _certified(mesh)
    refined = _adapt(
        source, np.sort(_cell_ids(mesh)),
        compatibility=(
            BisectionCompatibility.CONFORMING_CLOSURE if dimension == 2
            else BisectionCompatibility.WHOLE_EDGE_STAR
        ),
    )
    limits = phx.geometry.MeshCertificateLimits()
    embedding = phx.geometry.certify_global_embedding(
        mesh, source.geometry, source.audit.validity, limits=limits
    )
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            mesh, source.geometry, source.audit.validity, embedding,
            refined.target.mesh, refined.target.geometry,
            source_limits=limits, budget=budget,
        )
        assert set(predecessor.source_rows) == {-1}
        inherited = phx.geometry.certify_global_embedding(
            refined.target.mesh, refined.target.geometry,
            refined.target.audit.validity, limits=limits,
            predecessor=predecessor, budget=budget,
        )
    assert inherited.status == "certified"
    assert "restriction_parent_partition" in inherited.evaluated_checks
    assert fallback_calls == 0


def test_refinement_generations_are_canonical_and_route_specific() -> None:
    one = MarkedMeshAdaptation((0,), refinement_generations=1)
    two = MarkedMeshAdaptation((0,), refinement_generations=2)
    assert one.request_id != two.request_id
    with pytest.raises(ValueError):
        MarkedMeshAdaptation((0,), refinement_generations=0)
    with pytest.raises(ValueError, match="refinement_generations"):
        _adapt(
            _certified(triangle_grid(1, 1)), (0,),
            compatibility=BisectionCompatibility.CONFORMING_CLOSURE,
            refinement_generations=2,
        )


def test_quadratic_interpolation_proxy_matches_independent_exact_integral() -> None:
    from tools.meshing_qualification._physics import _physical_error
    from tools.meshing_qualification.scenarios.native_lifecycle import (
        _corpus_refinement_plan,
    )

    source = _certified(kuhn_cube())
    _, _, plan = _corpus_refinement_plan(source, 0.05, 160_000, material_x=False)
    points = np.asarray(source.mesh.coordinates)
    nodal = np.sum(points * points, axis=1)
    expected = _physical_error(source.mesh, nodal)
    assert plan["interpolation_l2_proxy"] == pytest.approx(expected, rel=1.0e-13, abs=0.0)
    assert plan["convergence_claim"] is False


def test_whole_edge_star_preserves_every_protected_source_edge() -> None:
    mesh = kuhn_cube()
    source = _certified(mesh)
    edges = mesh.entity_set(1)
    protected = MeshingScope(
        mesh.mesh_id, mesh.numeric_version, MeshingEntityKind.MESH, 1,
        edges.entity_set_id, np.asarray(edges.entity_ids, dtype=np.int64),
    )
    marks = np.sort(_cell_ids(mesh))
    result = _adapt(
        source, marks,
        compatibility=BisectionCompatibility.WHOLE_EDGE_STAR,
        protected_scopes=(protected,),
    )
    assert result.status is MeshAdaptationStatus.PARTIAL
    np.testing.assert_array_equal(result.evidence.rejected_refinement_ids, marks)
    assert result.evidence.accepted_refinements == 0
    np.testing.assert_array_equal(result.target.mesh.coordinates, mesh.coordinates)
    assert result.target.mesh.topology_id == mesh.topology_id


def test_whole_edge_star_recovers_intermediate_parent_of_packed_action() -> None:
    source = _certified(kuhn_cube())
    refined = _adapt(
        source, np.sort(_cell_ids(source.mesh)),
        compatibility=BisectionCompatibility.WHOLE_EDGE_STAR,
        refinement_generations=2,
    )
    vertex = int(np.max(np.asarray(refined.hierarchy.record_vertex_ids)))
    rows = _cells(refined.target.mesh)
    vertices = np.asarray(refined.target.mesh.vertex_global_ids)[rows]
    star = np.sort(_cell_ids(refined.target.mesh)[np.any(vertices == vertex, axis=1)])
    coarsened = _adapt(
        refined.target, (), star, hierarchy=refined.hierarchy
    )
    assert coarsened.evidence.coarsened_vertices > 0
    assert vertex not in np.asarray(coarsened.target.mesh.vertex_global_ids)
    assert _cell_ids(source.mesh).size < _cell_ids(coarsened.target.mesh).size
    assert _cell_ids(coarsened.target.mesh).size < _cell_ids(refined.target.mesh).size
    slope = np.asarray((0.2, 0.3, 0.4))
    values = np.asarray(refined.target.mesh.coordinates) @ slope + 0.5
    np.testing.assert_allclose(
        np.asarray(coarsened.transfer.apply(values)),
        np.asarray(coarsened.target.mesh.coordinates) @ slope + 0.5,
        rtol=0.0, atol=_AFFINE_TRANSFER_ROUNDOFF,
    )


def test_whole_edge_star_stops_only_after_complete_marked_generation() -> None:
    source = _certified(kuhn_cube())
    marks = np.sort(_cell_ids(source.mesh))
    result = _adapt(
        source, marks, compatibility=BisectionCompatibility.WHOLE_EDGE_STAR,
        refinement_generations=3, refinement_target_cells=1,
    )
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.evidence.accepted_refinements == marks.size
    assert result.evidence.closure_iterations == 1
    assert not np.intersect1d(marks, _cell_ids(result.target.mesh)).size
    assert _cell_ids(result.target.mesh).size >= 2 * marks.size
    assert result.request.refinement_target_cells == 1
    assert result.request.refinement_generations == 3
    assert result.request.request_id != MarkedMeshAdaptation(
        marks, refinement_generations=3
    ).request_id
    with pytest.raises(ValueError):
        MarkedMeshAdaptation(marks, refinement_target_cells=0)
    with pytest.raises(ValueError, match="target_cells"):
        _adapt(
            _certified(triangle_grid(1, 1)), (0,),
            compatibility=BisectionCompatibility.CONFORMING_CLOSURE,
            refinement_target_cells=1,
        )


@pytest.mark.parametrize("support_size", (2, 3))
def test_nested_partition_requires_complete_lower_dimensional_source_stars(
    support_size: int,
) -> None:
    from fractions import Fraction

    from phydrax.geometry._restricted_embedding import _nested_complete_source_stars

    corners = np.asarray(
        ((0, 1, 2, 3), (0, 1, 4, 5)) if support_size == 2
        else ((0, 1, 2, 3), (0, 2, 1, 4)),
        dtype=np.int64,
    )
    vertices = np.arange(6, dtype=np.int64)
    key = tuple((vertex, Fraction(1, support_size)) for vertex in range(support_size))
    with coordinate_reference_budget() as budget:
        assert not _nested_complete_source_stars(
            {key: {0}}, corners, vertices, 3, budget=budget
        )
    with coordinate_reference_budget() as budget:
        assert _nested_complete_source_stars(
            {key: {0, 1}}, corners, vertices, 3, budget=budget
        )


def test_mapped_plc_affine_corner_bank_reuses_whole_route_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing import _plc_mapped_support as owner

    source = _certified(kuhn_cube())
    refined = _adapt(
        source, np.sort(_cell_ids(source.mesh)),
        compatibility=BisectionCompatibility.WHOLE_EDGE_STAR,
    )
    mesh, geometry = refined.target.mesh, refined.target.geometry
    with coordinate_reference_budget() as budget:
        reference = owner._cell_maps(mesh, geometry, prepare_expressions=True, budget=budget)
    original = owner.algebra.coordinate_map_images
    per_cell_calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal per_cell_calls
        per_cell_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(owner.algebra, "coordinate_map_images", counted)
    with coordinate_reference_budget() as budget:
        shared = owner._cell_maps(mesh, geometry, prepare_expressions=False, budget=budget)
    assert per_cell_calls == 0
    assert shared.vertex_images == reference.vertex_images
    assert shared.degree_one_exactness_work_units == reference.degree_one_exactness_work_units == 0
    for actual, expected in zip(shared.cells, reference.cells, strict=True):
        assert actual.kind == expected.kind
        assert actual.vertices == expected.vertices
        assert actual.element is expected.element
        assert actual.controls == expected.controls
        assert actual.root_cell_id == expected.root_cell_id
        assert actual.root_vertex_ids == expected.root_vertex_ids
        assert actual.coordinates is None and expected.coordinates is not None


@pytest.mark.parametrize("dimension", (2, 3))
def test_authored_binary_partition_earns_closed_chain_without_native_request_records(
    dimension: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        CellGeometryRestrictionSource,
        require_p1_cardinal_source,
    )
    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from phydrax.discretization._cell_geometry_validity import (
        cell_geometry_id,
        certify_cell_geometry_validity,
    )
    from phydrax.geometry import _restricted_embedding as owner

    width = dimension + 1
    kind = "triangle" if dimension == 2 else "tetrahedron"
    points = np.vstack((np.zeros(dimension), np.eye(dimension)))
    parent = CellMesh(points, (CellBlock("parent", kind, np.arange(width)[None, :]),))
    source_geometry = CellGeometrySpec.affine(parent)
    root = require_p1_cardinal_source(source_geometry.elements[0])
    weights = np.eye(width)
    weights[-1] = 0.0
    weights[-1, (0, dimension)] = 0.5
    second = np.eye(width)
    second[0] = weights[-1]
    midpoint = 0.5 * (points[0] + points[-1])
    fine_points = np.vstack((points, midpoint))
    first_vertices = np.arange(width)
    first_vertices[-1] = width
    second_vertices = np.arange(width)
    second_vertices[0] = width
    target = CellMesh(fine_points, (
        CellBlock("first", kind, first_vertices[None, :], global_ids=np.asarray((1,))),
        CellBlock("second", kind, second_vertices[None, :], global_ids=np.asarray((2,))),
    ))
    geometry = CellGeometrySpec(
        {"first": BarycentricCellGeometryElement(root, weights),
         "second": BarycentricCellGeometryElement(root, second)},
        {name: np.arange(width)[None, :] for name in ("first", "second")},
        source_geometry.coordinates,
        restriction_source=CellGeometryRestrictionSource(
            cell_geometry_id(source_geometry), parent.topology_id,
            {name: np.asarray((0,)) for name in ("first", "second")},
            {name: np.arange(width)[None, :] for name in ("first", "second")},
            block_source_blocks={name: "parent" for name in ("first", "second")},
        ),
    )
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=parent)
    target_validity = certify_cell_geometry_validity(geometry, mesh=target)
    limits = phx.geometry.MeshCertificateLimits()
    source_embedding = phx.geometry.certify_global_embedding(
        parent, source_geometry, source_validity, limits=limits
    )
    original = owner._nested_closed_chain
    chain_results: list[object] = []

    def checked_chain(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        chain_results.append(result)
        return result

    monkeypatch.setattr(owner, "_nested_closed_chain", checked_chain)
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            parent, source_geometry, source_validity, source_embedding,
            target, geometry, source_limits=limits, budget=budget,
        )
        certificate = phx.geometry.certify_global_embedding(
            target, geometry, target_validity, limits=limits,
            predecessor=predecessor, budget=budget,
        )
    assert certificate.status == "certified"
    assert "restriction_parent_partition" in certificate.evaluated_checks
    assert chain_results == [None]


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron"))
@pytest.mark.parametrize("nonunit_source", (False, True))
def test_exact_dyadic_reference_chart_uses_equivalent_barycentric_owner(
    kind: str, nonunit_source: bool,
) -> None:
    from fractions import Fraction

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_corner_images,
        coordinate_polynomials,
    )
    from phydrax.meshing._adaptation import uniform_rational_p1_chart

    root = coordinate_lagrange_element(kind, 1)
    width = root.local_dof_count
    source = root
    if nonunit_source:
        inner = np.eye(width)
        inner[0, 0], inner[0, 1] = 1.5, 0.125
        source = BarycentricCellGeometryElement(root, inner)
    rows = [list(row) for row in np.eye(width)]
    rows[0][0], rows[0][1] = 0.5, 0.5
    rows[1][1], rows[1][2] = 0.75, 0.25
    exact = tuple(tuple(Fraction(float(value)) for value in row) for row in rows)
    ticket = HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=1_000_000)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
    )
    with budget.activate():
        actual = uniform_rational_p1_chart(source, exact, budget=budget)
        assert isinstance(actual, BarycentricCellGeometryElement)
        baseline = PolynomialComposedCellGeometryElement(
            source, root,
            np.asarray([[value.numerator for value in row[1:]] for row in exact], dtype=object),
            np.asarray([[value.denominator for value in row[1:]] for row in exact], dtype=object),
            budget=budget,
        )
        assert actual.element_id != baseline.element_id
        controls = tuple(
            tuple(Fraction(2 + node * 5 + axis) for axis in range(root.topological_dimension))
            for node in range(width)
        )
        assert coordinate_polynomials(actual, controls, budget=budget) == (
            coordinate_polynomials(baseline, controls, budget=budget)
        )
        assert coordinate_corner_images(actual, controls, budget=budget) == (
            coordinate_corner_images(baseline, controls, budget=budget)
        )
    points = np.full((2, root.topological_dimension), 0.125)
    points[1, 0] = 0.25
    actual_values, actual_gradients = actual.tabulate(points)
    expected_values, expected_gradients = baseline.tabulate(points)
    np.testing.assert_array_equal(actual_values, expected_values)
    np.testing.assert_array_equal(actual_gradients, expected_gradients)


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron"))
def test_nondyadic_reference_chart_retains_actual_rational_owner(kind: str) -> None:
    from fractions import Fraction

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._cell_geometry import (
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.discretization._coordinate_enclosure import coordinate_corner_images
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.meshing._adaptation import uniform_rational_p1_chart

    root = coordinate_lagrange_element(kind, 1)
    width = root.local_dof_count
    rows = [list(map(Fraction, row)) for row in np.eye(width)]
    rows[0][0], rows[0][1] = Fraction(2, 3), Fraction(1, 3)
    exact = tuple(tuple(row) for row in rows)
    ticket = HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=1_000_000)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
    )
    with budget.activate():
        actual = uniform_rational_p1_chart(root, exact, budget=budget)
        assert isinstance(actual, PolynomialComposedCellGeometryElement)
        controls = tuple(
            tuple(Fraction(value) for value in point)
            for point in reference_cell_topology(kind).vertices
        )
        corners = coordinate_corner_images(actual, controls, budget=budget)
        assert corners is not None
        assert corners[0][0] == Fraction(1, 3)
        # A rational owner remains rational even when a later action is dyadic.
        identity = tuple(tuple(Fraction(value) for value in row) for row in np.eye(width))
        composed = uniform_rational_p1_chart(actual, identity, budget=budget)
        assert isinstance(composed, PolynomialComposedCellGeometryElement)
        assert coordinate_corner_images(composed, controls, budget=budget) == corners


def test_whole_edge_star_cover_prefers_announced_edge_shared_by_all_marks() -> None:
    # Edge (0,1) is longest in the small third sector and touches all three
    # marks. Each long outer edge is announced by only one of the other marks.
    # Coverage-first selects the genuine complete central star rather than
    # cutting those two marks again through their longer outer announcements.
    points = np.asarray(
        ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (3.0, 0.0, 0.0),
         (-0.25, 0.5, 0.0), (-0.25, -0.5, 0.0))
    )
    mesh = CellMesh.from_tetrahedra(
        points, np.asarray(((0, 1, 2, 3), (0, 1, 3, 4), (0, 1, 4, 2)))
    )
    source = _certified(mesh)
    marks = np.sort(_cell_ids(mesh))
    refined = _adapt(
        source, marks, compatibility=BisectionCompatibility.WHOLE_EDGE_STAR
    )
    assert refined.status is MeshAdaptationStatus.COMPLETE
    assert _cell_ids(refined.target.mesh).size == 2 * marks.size
    assert refined.target.mesh.coordinates.shape[0] == points.shape[0] + 1
    assert not np.any(np.isin(marks, _cell_ids(refined.target.mesh)))
    created = np.asarray(refined.target.mesh.coordinates)[
        ~np.isin(refined.target.mesh.vertex_global_ids, mesh.vertex_global_ids)
    ]
    np.testing.assert_array_equal(created, np.asarray(((0.0, 0.0, 0.0),)))
    assert refined.target.audit.validity.all_certified


@pytest.mark.parametrize("jitter", (0.0, 0.4))
def test_whole_edge_star_uncovered_cover_matches_independent_greedy_oracle(
    jitter: float,
) -> None:
    from fractions import Fraction

    mesh = kuhn_cube(2 if jitter == 0.0 else 3, jitter=jitter)
    points, cells = np.asarray(mesh.coordinates), _cells(mesh)
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    point_by_id = {int(identifier): point for identifier, point in zip(vertex_ids, points, strict=True)}
    cell_edges = [
        {tuple(sorted((int(vertex_ids[row[first]]), int(vertex_ids[row[second]]))))
         for first, second in itertools.combinations(range(4), 2)}
        for row in cells
    ]
    edges = set().union(*cell_edges)
    # Preserve the existing binary64 priority, not an exact-length theorem.
    priority = {
        edge: (-float(np.sum((point_by_id[edge[1]] - point_by_id[edge[0]]) ** 2)), edge)
        for edge in edges
    }
    announced = {min(local, key=priority.__getitem__) for local in cell_edges}
    stars = {
        edge: {owner for owner, local in enumerate(cell_edges) if edge in local}
        for edge in announced
    }
    remaining = set(range(len(cells)))
    selected = []
    changed_coverage = False
    while remaining:
        chosen = min(
            announced,
            key=lambda edge: (-len(stars[edge] & remaining), priority[edge]),
        )
        covered = stars[chosen] & remaining
        assert covered
        changed_coverage |= len(covered) != len(stars[chosen])
        selected.append(chosen)
        remaining -= covered
        announced.remove(chosen)
    if jitter:
        assert changed_coverage
    expected = sorted(
        tuple(
            float((Fraction(float(point_by_id[first][axis]))
                   + Fraction(float(point_by_id[second][axis]))) / 2)
            for axis in range(3)
        )
        for first, second in selected
    )
    source = _certified(mesh)
    marks = np.sort(_cell_ids(mesh))
    result = _adapt(
        source, marks, compatibility=BisectionCompatibility.WHOLE_EDGE_STAR
    )
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert not np.any(np.isin(marks, _cell_ids(result.target.mesh)))
    created = np.asarray(result.target.mesh.coordinates)[
        ~np.isin(result.target.mesh.vertex_global_ids, mesh.vertex_global_ids)
    ]
    assert sorted(map(tuple, created)) == expected
    _assert_conforming(result.target.mesh, 1.0)
    reordered = _adapt(
        source, marks[::-1], compatibility=BisectionCompatibility.WHOLE_EDGE_STAR
    )
    assert reordered.status is MeshAdaptationStatus.COMPLETE
    first_ids, first_rows = _canonical_cells(result.target.mesh)
    second_ids, second_rows = _canonical_cells(reordered.target.mesh)
    np.testing.assert_array_equal(first_ids, second_ids)
    np.testing.assert_array_equal(first_rows, second_rows)
    np.testing.assert_array_equal(
        result.target.mesh.coordinates, reordered.target.mesh.coordinates
    )


def test_whole_edge_star_metered_priorities_keep_prior_tuple_order() -> None:
    import heapq

    from phydrax.meshing._bisection_kernel_host import _StarCoverPriority

    priorities = [(-4, 12, 3), (-1, 1, 9), (-4, 3, 2),
                  (-4, 3, 0), (-4, 3, 1), (-2, 0, 5)]
    with coordinate_reference_budget() as budget:
        budget.reserve(0, 512 + 256 * len(priorities))
        heap = [_StarCoverPriority(*key, budget=budget) for key in priorities]
        heapq.heapify(heap)
        actual = []
        while heap:
            item = heapq.heappop(heap)
            actual.append((item.negative_coverage, item.rank, item.slot))
        assert actual == sorted(priorities)
        assert budget.work_units > 0


def test_whole_edge_star_comparator_refusal_is_terminal() -> None:
    import heapq

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.meshing._bisection_kernel_host import _StarCoverPriority

    ticket = HostBatchTicket(1, bytes_upper=1 << 20, queries_upper=1)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=1, maximum_memory_bytes=1 << 20
    )
    with budget.activate():
        budget.reserve(0, 1024)
        heap = [
            _StarCoverPriority(0, 0, 0, budget=budget),
            _StarCoverPriority(0, 1, 1, budget=budget),
        ]
        # The equal coverage field consumes the first unit. Admission refuses
        # before the rank comparison; heapq's partial mutation is not resumed.
        with pytest.raises(CoordinateEnclosureResourceError):
            heapq.heapify(heap)
        assert budget.work_units == 1
