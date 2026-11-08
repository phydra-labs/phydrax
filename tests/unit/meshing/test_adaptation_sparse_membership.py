"""Adaptation organization classes stay sparse and equal the dense ranking."""

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellMesh
from phydrax.meshing._adaptation import (
    _cell_classes,
    _edge_classes,
    _membership,
    _membership_classes,
    _row_classes,
)
from phydrax.meshing._scope import (
    MeshingEntityKind,
    MeshingScope,
    resolve_mesh_scope,
    resolve_mesh_scope_rows,
)
from phydrax.meshing._topology_edit import entity_keys, key_rows


def _dense(
    mesh: CellMesh, scopes: tuple[MeshingScope, ...], dimension: int
) -> np.ndarray:
    """The entities x scopes matrix the sparse membership replaces."""
    columns = [
        np.asarray(resolve_mesh_scope(mesh, scope).mask, dtype=np.int64)
        for scope in scopes
        if scope.entity_dimension == dimension
    ]
    count = mesh.entity_set(dimension).count
    return np.stack(columns, axis=1) if columns else np.zeros((count, 0), np.int64)


def _scope(mesh: CellMesh, dimension: int, ids: np.ndarray) -> MeshingScope:
    entities = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        np.asarray(ids, dtype=np.int64),
    )


def _kuhn_cube() -> CellMesh:
    corners = np.asarray(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    )
    cells = np.asarray(
        (
            (0, 1, 3, 7),
            (0, 1, 5, 7),
            (0, 2, 3, 7),
            (0, 2, 6, 7),
            (0, 4, 5, 7),
            (0, 4, 6, 7),
        ),
        dtype=np.int64,
    )
    tets = corners[cells]
    negative = np.linalg.det(tets[:, 1:] - tets[:, :1]) < 0.0
    cells[negative] = cells[negative][:, (0, 2, 1, 3)]
    return CellMesh(
        corners,
        (CellBlock("tets", "tetrahedron", cells, global_ids=np.arange(10, 16)),),
    )


def _random_scopes(
    mesh: CellMesh, rng: np.random.Generator, dimension: int, count: int
) -> tuple[MeshingScope, ...]:
    ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    scopes = []
    for _ in range(count):
        size = int(rng.integers(1, ids.size + 1))
        scopes.append(
            _scope(mesh, dimension, np.sort(rng.choice(ids, size, replace=False)))
        )
    return tuple(scopes)


@pytest.mark.parametrize("seed", range(12))
def test_membership_classes_equal_dense_row_classes(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = int(rng.integers(0, 60))
    columns = int(rng.integers(0, 40))
    density = float(rng.choice((0.0, 0.02, 0.2, 0.7)))
    dense = (rng.random((rows, columns)) < density).astype(np.int64)
    leading = rng.integers(-2, 3, size=(rows, int(rng.integers(1, 4)))).astype(np.int64)
    member_rows, member_columns = np.nonzero(dense)
    sparse = _membership_classes(
        leading, (member_rows.astype(np.int64), member_columns.astype(np.int64))
    )
    expected = _row_classes(np.column_stack((leading, dense)))
    assert sparse.dtype == expected.dtype
    np.testing.assert_array_equal(sparse, expected)


def test_scope_rows_equal_masks_and_refuse_undeclared_ids() -> None:
    mesh = _kuhn_cube()
    rng = np.random.default_rng(3)
    scopes = tuple(
        scope
        for dimension in range(4)
        for scope in _random_scopes(mesh, rng, dimension, 5)
    )
    for scope, rows in zip(scopes, resolve_mesh_scope_rows(mesh, scopes), strict=True):
        np.testing.assert_array_equal(
            rows, np.flatnonzero(np.asarray(resolve_mesh_scope(mesh, scope).mask))
        )
    facets = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    undeclared = _scope(mesh, 2, np.asarray((int(np.max(facets)) + 1,)))
    for resolve in (
        lambda: resolve_mesh_scope(mesh, undeclared),
        lambda: resolve_mesh_scope_rows(mesh, (undeclared,)),
    ):
        with pytest.raises(ValueError, match="undeclared mesh entity IDs"):
            resolve()


@pytest.mark.parametrize("seed", range(6))
def test_cell_and_facet_classes_equal_the_dense_membership(seed: int) -> None:
    mesh = _kuhn_cube()
    rng = np.random.default_rng(seed)
    facets = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    # One patch per facet, as on a reconstructed PLC, plus overlapping scopes.
    scopes = (
        *(_scope(mesh, 2, facets[[row]]) for row in rng.permutation(facets.size)),
        *_random_scopes(mesh, rng, 2, 4),
        *_random_scopes(mesh, rng, 3, 3),
    )
    sparse_facets = _membership_classes(
        np.zeros((facets.size, 1), dtype=np.int64), _membership(mesh, scopes, 2)
    )
    dense_facets = _row_classes(
        np.column_stack((np.zeros((facets.size,), np.int64), _dense(mesh, scopes, 2)))
    )
    np.testing.assert_array_equal(sparse_facets, dense_facets)
    cells = key_rows(np.arange(10, 16)[:, None], entity_keys(mesh, 3))
    np.testing.assert_array_equal(
        _cell_classes(mesh, scopes),
        _row_classes(
            np.column_stack((np.zeros((6,), np.int64)[cells], _dense(mesh, scopes, 3)))
        ),
    )


@pytest.mark.parametrize("seed", range(6))
def test_planar_edge_classes_equal_the_dense_membership(seed: int) -> None:
    corners = np.asarray([[x, y] for y in (0.0, 1.0, 2.0) for x in (0.0, 1.0, 2.0)])
    triangles = np.asarray(
        (
            (0, 1, 4),
            (0, 4, 3),
            (1, 2, 5),
            (1, 5, 4),
            (3, 4, 7),
            (3, 7, 6),
            (4, 5, 8),
            (4, 8, 7),
        ),
        dtype=np.int64,
    )
    mesh = CellMesh(
        corners,
        (CellBlock("tris", "triangle", triangles, global_ids=np.arange(8)),),
    )
    rng = np.random.default_rng(seed)
    scopes = (*_random_scopes(mesh, rng, 1, 6), *_random_scopes(mesh, rng, 2, 2))
    cell_classes = _cell_classes(mesh, scopes)
    relation = mesh.topology.incidences[1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    edges = np.asarray(relation.source_indices)[valid]
    incident_cells = cell_classes[np.asarray(relation.target_indices)[valid]]
    count = mesh.entity_set(1).count
    incident = np.bincount(edges, minlength=count)
    low = np.full((count,), np.iinfo(np.int64).max, dtype=np.int64)
    high = np.full((count,), -1, dtype=np.int64)
    np.minimum.at(low, edges, incident_cells)
    np.maximum.at(high, edges, incident_cells)
    member = _dense(mesh, scopes, 1)
    feature = (incident != 2) | (low != high) | np.any(member, axis=1)
    expected = np.zeros((count,), dtype=np.int64)
    keys = np.column_stack((incident == 1, low, high, member)).astype(np.int64)
    expected[feature] = 1 + _row_classes(keys[feature])
    np.testing.assert_array_equal(_edge_classes(mesh, scopes, cell_classes), expected)
