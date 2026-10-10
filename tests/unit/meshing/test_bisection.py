#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import itertools
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellMesh
from phydrax.meshing import (
    BisectionCompatibility,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MeshingEntityKind,
    MeshingScope,
    prepare_mesh_adaptation,
)


def _signed_measures(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    corners = np.asarray(points)[cells]
    edges = corners[:, 1:] - corners[:, :1]
    if cells.shape[1] == 3:
        return 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    return np.linalg.det(edges) / 6.0


def _triangle_grid(columns: int, rows: int, height: float = 1.0) -> CellMesh:
    xs = np.linspace(0.0, 1.0, columns + 1)
    ys = np.linspace(0.0, height, rows + 1)
    points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2))
    cells = []
    for j, i in itertools.product(range(rows), range(columns)):
        a = j * (columns + 1) + i
        b, c, d = a + 1, a + columns + 2, a + columns + 1
        cells.extend(((a, b, c), (a, c, d)))
    return CellMesh.from_triangles(points, np.asarray(cells, dtype=np.int32))


def _kuhn_grid(size: int) -> CellMesh:
    axis = np.linspace(0.0, 1.0, size + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = points.reshape((-1, 3))
    cells = []
    for origin in itertools.product(range(size), repeat=3):
        for order in itertools.permutations(range(3)):
            corner = list(origin)
            path = [(corner[0] * (size + 1) + corner[1]) * (size + 1) + corner[2]]
            for direction in order:
                corner[direction] += 1
                path.append((corner[0] * (size + 1) + corner[1]) * (size + 1) + corner[2])
            cells.append(path)
    cells = np.asarray(cells, dtype=np.int32)
    negative = _signed_measures(points, cells) < 0.0
    cells[negative] = cells[negative][:, [1, 0, 2, 3]]
    return CellMesh.from_tetrahedra(points, cells)


def _certified(mesh: CellMesh) -> Any:
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _adapt(
    source: Any,
    refine: Any = (),
    coarsen: Any = (),
    /,
    *,
    hierarchy: Any = None,
    **options: Any,
) -> Any:
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION, **options)
    request = MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
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
    # ty: ignore[unresolved-attribute]
    edges = np.asarray(mesh.connectivity.edges)
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
        assert np.min(np.linalg.norm(chunk, axis=2)) > 1.0e-9
    measures = _signed_measures(points, cells)
    assert np.all(measures > 0.0)
    assert np.isclose(np.sum(measures), measure, rtol=0.0, atol=1.0e-12)


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


def test_bisection_scenario_1() -> None:
    for mesh, cycles, stride in [(_triangle_grid(4, 4), 6, 7), (_kuhn_grid(2), 4, 11)]:
        result = _adapt(_certified(mesh), (0, 5, 17, 30))
        for _ in range(cycles):
            target = result.target.mesh
            marks = np.union1d(
                _corner_cells(target), np.sort(_cell_ids(target))[::stride]
            )
            result = _adapt(result.target, marks, hierarchy=result.hierarchy)
            _assert_conforming(result.target.mesh, 1.0)
            _assert_boundary_on_unit_box(result.target.mesh)
        assert result.status is MeshAdaptationStatus.COMPLETE
        assert result.evidence.maximum_generation >= cycles
    source = _certified(_triangle_grid(2, 2, height=0.6))
    bound = _uniform_class_bound(source, _minimum_angles, 4)
    result = _adapt(source, _corner_cells(source.mesh))
    for _ in range(20):
        assert np.min(_minimum_angles(result.target.mesh)) >= bound - 1.0e-12
        result = _adapt(
            result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
        )
    assert np.min(_minimum_angles(result.target.mesh)) >= bound - 1.0e-12
    assert result.evidence.maximum_generation >= 20
    source = _certified(_kuhn_grid(1))
    bound = _uniform_class_bound(source, _mean_ratios, 3)
    result = _adapt(source, _corner_cells(source.mesh))
    for _ in range(9):
        assert np.min(_mean_ratios(result.target.mesh)) >= 0.999 * bound
        result = _adapt(
            result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
        )
    assert np.min(_mean_ratios(result.target.mesh)) >= 0.999 * bound
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


def test_bisection_scenario_2() -> None:
    for mesh in [_triangle_grid(3, 3), _kuhn_grid(2)]:
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
    for mesh in [_triangle_grid(3, 3), _kuhn_grid(2)]:
        result = _adapt(_certified(mesh), np.sort(_cell_ids(mesh))[::2])
        result = _adapt(
            result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
        )
        slope = np.linspace(0.5, 1.5, mesh.ambient_dimension)
        before = np.asarray(result.source.mesh.coordinates) @ slope + 0.25
        expected = np.asarray(result.target.mesh.coordinates) @ slope + 0.25
        np.testing.assert_allclose(
            np.asarray(result.transfer.apply(before)), expected, rtol=0.0, atol=1.0e-13
        )
    source = _certified(_triangle_grid(4, 4))
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


def test_bisection_scenario_3() -> None:
    source = _certified(_triangle_grid(2, 2))
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
    points = np.asarray([(0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)])
    mesh = CellMesh.from_triangles(points, np.asarray([(0, 1, 2), (0, 3, 1)]))
    source = _certified(mesh)
    with pytest.raises(ValueError, match="UNIFORM_REFINEMENT"):
        _adapt(source, (0,))
    result = _adapt(source, (0,), compatibility=BisectionCompatibility.UNIFORM_REFINEMENT)
    assert result.evidence.uniform_refinement_applied
    assert result.evidence.initially_compatible is False
    _assert_conforming(result.target.mesh, 3.5)
    follow = _adapt(
        result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
    )
    _assert_conforming(follow.target.mesh, 3.5)
    first = _adapt(_certified(_kuhn_grid(2)), (0, 7, 13))
    second = _adapt(_certified(_kuhn_grid(2)), (0, 7, 13))
    assert first.result_id == second.result_id
    assert first.hierarchy.hierarchy_id == second.hierarchy.hierarchy_id


@pytest.mark.parametrize("dimension", (2, 3))
def test_uniform_source_inverse_survives_archive_and_protected_rollback(
    dimension: int, tmp_path: Path
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    if dimension == 2:
        points = np.asarray(
            ((0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)), dtype=np.float64
        )
        mesh = CellMesh.from_triangles(
            points, np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32)
        )
    else:
        points = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (1.0, 0.5, 0.0),
                (1.0, 0.2, 1.0),
                (1.0, -3.0, -1.0),
            ),
            dtype=np.float64,
        )
        mesh = CellMesh.from_tetrahedra(
            points, np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int32)
        )
    source = _certified(mesh)
    original_ids = _cell_ids(source.mesh)
    refined = _adapt(
        source, (0,), compatibility=BisectionCompatibility.UNIFORM_REFINEMENT
    )
    assert refined.status is MeshAdaptationStatus.COMPLETE
    assert refined.hierarchy.uniform_refinement is not None
    lineage = refined.hierarchy.uniform_refinement
    assert lineage.child_ids.shape[1] == (6 if dimension == 2 else 24)
    assert np.any(np.asarray(lineage.barycentric_weights) == 1.0 / 3.0)
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    cold = write_meshing_source_closure(
        tmp_path / "uniform-cold",
        (source, refined.target, refined.hierarchy, policy),
    )
    script = textwrap.dedent("""
        import sys
        import numpy as np
        import phydrax as phx
        from phydrax.lifecycle._meshing_sources import read_meshing_source_closure
        original, fine, hierarchy, policy = read_meshing_source_closure(
            sys.argv[1], expected_content_id=sys.argv[2])
        before_work = np.asarray(fine.execution_evidence.work).copy()
        lineage = hierarchy.uniform_refinement
        request = phx.meshing.MarkedMeshAdaptation(
            (), np.sort(np.concatenate([np.asarray(block.global_ids) for block in fine.mesh.blocks])),
            hierarchy=hierarchy)
        result = phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(fine, request, policy=policy))
        assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
        np.testing.assert_array_equal(result.target.mesh.coordinates, original.mesh.coordinates)
        np.testing.assert_array_equal(result.target.mesh.vertex_global_ids, original.mesh.vertex_global_ids)
        for before, after in zip(original.mesh.blocks, result.target.mesh.blocks, strict=True):
            np.testing.assert_array_equal(after.global_ids, before.global_ids)
            np.testing.assert_array_equal(after.vertices, before.vertices)
        for degree in range(original.mesh.topological_dimension + 1):
            np.testing.assert_array_equal(result.target.mesh.entity_set(degree).entity_ids,
                                          original.mesh.entity_set(degree).entity_ids)
        np.testing.assert_array_equal(result.hierarchy.tags, lineage.parent_tags)
        np.testing.assert_array_equal(result.hierarchy.generations, lineage.parent_levels)
        np.testing.assert_array_equal(fine.execution_evidence.work, before_work)
        assert int(result.target.execution_evidence.total_work_units) > 0
        assert int(result.target.execution_evidence.host_storage_peak_bytes_upper) > 0
        assert result.hierarchy.uniform_refinement is None
    """)
    subprocess.run(
        (sys.executable, "-c", script, str(cold.path), cold.content_id),
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )
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
    assert (
        refused.target.geometry.source_coordinates()
        == target.geometry.source_coordinates()
    )
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


@pytest.mark.parametrize("dimension", (2, 3))
def test_partial_uniform_inverse_preserves_original_block_rows_and_coefficient_routes(
    dimension: int,
    tmp_path: Path,
) -> None:
    from phydrax.discretization import CellBlock, CellGeometrySpec
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

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
    import equinox as eqx

    wrong_sci = eqx.tree_at(
        lambda value: value.uniform_refinement.parent_blocks,
        fine.hierarchy,
        lineage.parent_blocks.at[root].set(
            tuple(block.name for block in original.blocks).index("a-root")
        ),
    )
    with pytest.raises(ValueError, match="scientific block"):
        _adapt(fine.target, (), np.asarray(lineage.child_ids)[root], hierarchy=wrong_sci)
    wrong_support = eqx.tree_at(
        lambda value: value.uniform_refinement.barycentric_weights,
        fine.hierarchy,
        lineage.barycentric_weights.at[root, 0, 0, 0].set(0.25),
    )
    with pytest.raises(ValueError, match="canonical barycentric"):
        _adapt(
            fine.target, (), np.asarray(lineage.child_ids)[root], hierarchy=wrong_support
        )
    wrong_bank = eqx.tree_at(
        lambda value: value.geometry.coordinates,
        fine.target,
        fine.target.geometry.coordinates.at[0, 0].add(1.0),
    )
    with pytest.raises(ValueError):
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


@pytest.mark.parametrize("degree", (1, 2))
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

    mesh = _triangle_grid(4, 4)
    points = np.asarray(mesh.coordinates).copy()
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    offsets = np.random.default_rng(13).uniform(-0.06, 0.06, size=points.shape)
    points[interior] += offsets[interior]
    return CellMesh.from_triangles(points, _cells(mesh).astype(np.int32))


def test_conforming_closure_bisects_incompatible_triangle_labels_locally() -> None:
    mesh = _jittered_triangle_grid()
    source = _certified(mesh)
    marks = np.sort(_cell_ids(mesh))[::5]
    with pytest.raises(ValueError, match="CONFORMING_CLOSURE"):
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
        assert np.min(np.linalg.norm(midpoints - point, axis=1)) < 1.0e-15
    follow = _adapt(result.target, _corner_cells(target), hierarchy=result.hierarchy)
    assert follow.status is MeshAdaptationStatus.COMPLETE
    _assert_conforming(follow.target.mesh, 1.0)
    restored = _adapt(
        result.target, (), np.sort(_cell_ids(target)), hierarchy=result.hierarchy
    )
    # Complete restoration rejoins every root to its authored block.
    assert restored.target.mesh.topology_id == source.mesh.topology_id
    target_ids, target_cells = _canonical_cells(restored.target.mesh)
    source_ids, source_cells = _canonical_cells(mesh)
    np.testing.assert_array_equal(target_ids, source_ids)
    np.testing.assert_array_equal(target_cells, source_cells)
    np.testing.assert_array_equal(restored.target.mesh.coordinates, mesh.coordinates)


def test_conforming_closure_keeps_incompatible_tetrahedral_labels_refused() -> None:
    mesh = _kuhn_grid(2)
    points = np.asarray(mesh.coordinates).copy()
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    points[interior] += 0.1
    source = _certified(CellMesh.from_tetrahedra(points, _cells(mesh).astype(np.int32)))
    with pytest.raises(ValueError, match="triangle meshes only"):
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
    source = _certified(_triangle_grid(4, 4))
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
    mesh = phx.meshing.canonicalize_cell_mesh(_triangle_grid(2, 2))
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
        assert np.min(np.linalg.norm(edge_nodes - point, axis=1)) < 1e-15
    assert np.any(np.abs(np.asarray(target.mesh.coordinates)[created, 1] % 0.5) > 1e-3)

    result = _adapt(_certified(mesh), (0,))
    affine = phx.discretization.CellGeometrySpec.affine(result.target.mesh)
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.geometry_transition is None
    # Exact P1 coefficient actions retain ancestry in canonical layout identity
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
