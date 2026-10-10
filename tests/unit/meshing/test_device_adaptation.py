#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import itertools
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.discretization import (
    AdaptiveSimplexPolicy,
    AdaptiveSimplexStatus,
    CellMesh,
    coarsen_adaptive_simplex,
    refine_adaptive_simplex,
    refine_adaptive_simplex_parts,
)
from phydrax.meshing import (
    BisectionCompatibility,
    commit_adaptive_simplex,
    commit_partitioned_adaptive_simplex,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MeshingEntityKind,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingScope,
    MeshPart,
    MeshPartitionKind,
    MeshPartitionPolicy,
    partition_adaptive_simplex,
    prepare_adaptive_simplex,
    prepare_mesh_adaptation,
    prepare_mesh_distribution,
)
from phydrax.meshing._bisection import BisectionHierarchy


def _triangle_grid(columns: int, rows: int) -> CellMesh:
    xs = np.linspace(0.0, 1.0, columns + 1)
    ys = np.linspace(0.0, 1.0, rows + 1)
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
    corners = points[cells]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    cells[negative] = cells[negative][:, [1, 0, 2, 3]]
    return CellMesh.from_tetrahedra(points, cells)


def _certified(mesh: CellMesh) -> Any:
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def test_device_adaptation_scenario_1() -> None:
    prepared = prepare_adaptive_simplex(
        _certified(_triangle_grid(1, 1)),
        policy=_policy(MeshAdaptationRoute.DEVICE_BISECTION),
    )
    mesh = prepared.state.mesh
    phx.typing.validate(mesh)

    widened = eqx.tree_at(
        lambda value: value.coordinates,
        mesh,
        jnp.concatenate((mesh.coordinates, mesh.coordinates[:1]), axis=0),
    )
    with pytest.raises(ValueError, match="coordinates"):
        phx.typing.validate(widened)

    wrong_dtype = eqx.tree_at(
        lambda value: value.cells,
        mesh,
        mesh.cells.astype(jnp.float64),
    )
    with pytest.raises(ValueError, match="cells"):
        phx.typing.validate(wrong_dtype)
    for mesh in [_triangle_grid(3, 3), _kuhn_grid(1)]:
        source = _certified(mesh)
        prepared = prepare_adaptive_simplex(
            source, policy=_policy(MeshAdaptationRoute.DEVICE_BISECTION)
        )
        marks = prepared.cell_marks(np.sort(_cell_ids(source.mesh))[::2])
        state = refine_adaptive_simplex(prepared.layout, prepared.state, marks).state
        geometry = phx.discretization.evaluate_masked_fv_geometry(state.mesh)
        flux = jax.random.normal(jax.random.key(3), (geometry.face_capacity,))
        residual = phx.discretization.masked_fv_flux_divergence(geometry, flux)
        volumes = np.asarray(geometry.cell_volumes)
        active = np.asarray(state.mesh.cell_active)
        boundary = np.asarray(geometry.boundary_faces)
        content = np.sum(volumes * np.asarray(residual))
        assert np.isclose(content, -np.sum(np.asarray(flux)[boundary]), atol=1e-12)
        assert np.all(np.asarray(residual)[~active] == 0.0)
        assert np.isclose(np.sum(volumes[active]), 1.0, atol=1e-12)
        ledger = phx.discretization.evaluate_masked_fv_conservation(
            geometry, flux, jnp.zeros((state.mesh.cell_capacity,))
        )
        assert np.all(np.abs(np.asarray(ledger.residual)) < 1e-12)
    for mesh in [_triangle_grid(4, 4), _kuhn_grid(2)]:
        host_route = MeshAdaptationRoute.NATIVE_BISECTION
        device_route = MeshAdaptationRoute.DEVICE_BISECTION
        source = _certified(mesh)
        marks = np.sort(_cell_ids(mesh))[::3]
        host = _adapt(host_route, source, marks)
        device = _adapt(device_route, source, marks)
        _assert_same_target(host, device)
        assert host.lineage.lineage_id == device.lineage.lineage_id

        identifiers = np.sort(_cell_ids(host.target.mesh))
        refine, coarsen = _corner_cells(host.target.mesh), identifiers[-40:]
        coarsen = np.setdiff1d(coarsen, refine)
        host = _adapt(host_route, host.target, refine, coarsen, hierarchy=host.hierarchy)
        device = _adapt(
            device_route, device.target, refine, coarsen, hierarchy=device.hierarchy
        )
        _assert_same_target(host, device)
        np.testing.assert_array_equal(
            np.asarray(host.evidence.rejected_coarsening_ids),
            np.asarray(device.evidence.rejected_coarsening_ids),
        )

        everything = np.sort(_cell_ids(host.target.mesh))
        host = _adapt(host_route, host.target, (), everything, hierarchy=host.hierarchy)
        device = _adapt(
            device_route, device.target, (), everything, hierarchy=device.hierarchy
        )
        _assert_same_target(host, device)
        assert device.evidence.coarsened_vertices == host.evidence.coarsened_vertices > 0
    source = _certified(_triangle_grid(4, 4))
    scope, through = _guarded_diagonal(source)
    marks = np.union1d(through, np.setdiff1d(_cell_ids(source.mesh), through)[[0, -1]])
    host = _adapt(
        MeshAdaptationRoute.NATIVE_BISECTION, source, marks, protected_scopes=(scope,)
    )
    device = _adapt(
        MeshAdaptationRoute.DEVICE_BISECTION, source, marks, protected_scopes=(scope,)
    )
    assert device.status is MeshAdaptationStatus.PARTIAL
    np.testing.assert_array_equal(
        np.asarray(device.evidence.rejected_refinement_ids), through
    )
    _assert_same_target(host, device)
    for route in (
        MeshAdaptationRoute.NATIVE_BISECTION,
        MeshAdaptationRoute.DEVICE_BISECTION,
    ):
        rejected = _adapt(route, source, through, protected_scopes=(scope,))
        assert rejected.status is MeshAdaptationStatus.PARTIAL
        assert not rejected.status.converged
        assert rejected.target.result_id == source.result_id
        assert rejected.transition is rejected.lineage is rejected.transfer is None
        np.testing.assert_array_equal(
            np.asarray(rejected.evidence.rejected_refinement_ids), through
        )
    source = _certified(_triangle_grid(4, 4))
    scope, _ = _guarded_diagonal(source)
    partition = MeshPartitionPolicy(MeshPartitionKind.MORTON, 1)
    distribution = prepare_mesh_distribution(MeshPart("domain", source), policy=partition)
    policy = _policy(
        MeshAdaptationRoute.DEVICE_BISECTION,
        protected_scopes=(scope,),
        distribution=distribution,
        partition_policy=partition,
    )
    partitioned = partition_adaptive_simplex(
        prepare_adaptive_simplex(source, policy=policy)
    )
    # Parts have no per-mark admissibility: the union closure splits the edge.
    update = refine_adaptive_simplex_parts(
        partitioned.layout,
        partitioned.parts,
        partitioned.states,
        partitioned.cell_marks(_cell_ids(source.mesh)),
    )
    conflict = int(AdaptiveSimplexStatus.PROTECTED_CONFLICT)
    assert np.all(np.asarray(update.report.status) & conflict)
    assert np.all(np.asarray(update.state.status_flags) & conflict)
    np.testing.assert_array_equal(
        np.asarray(update.state.cursors), np.asarray(partitioned.states.cursors)
    )
    with pytest.raises(MeshingFailure) as rejection:
        commit_partitioned_adaptive_simplex(partitioned, update.state)
    assert rejection.value.category is MeshingFailureCategory.INVALID_SPECIFICATION


@pytest.mark.skipif(
    not meshcore_available(), reason="native execution ledger unavailable"
)
@pytest.mark.parametrize("dimension", (2, 3))
def test_device_uniform_inverse_keeps_compiled_counters_and_source_authority(
    dimension: int,
    tmp_path: Path,
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
    policy = _policy(
        MeshAdaptationRoute.DEVICE_BISECTION,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
        device_policy=AdaptiveSimplexPolicy(vertex_capacity=256, cell_capacity=1024),
    )
    fine = execute_mesh_adaptation(
        prepare_mesh_adaptation(source, MarkedMeshAdaptation(), policy=policy)
    )
    assert isinstance(fine.hierarchy, BisectionHierarchy)
    lineage = fine.hierarchy.uniform_refinement
    assert lineage is not None
    assert lineage.child_ids.shape[1] == (6 if dimension == 2 else 24)
    receipt = write_meshing_source_closure(
        tmp_path / "device-uniform",
        (fine.target, fine.hierarchy, policy),
    )
    target, hierarchy, reopened_policy = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert reopened_policy.policy_id == policy.policy_id
    prepared = prepare_adaptive_simplex(
        target, policy=reopened_policy, hierarchy=hierarchy
    )
    assert prepared.execution_evidence is not None
    assert prepared.execution_evidence.owner_id == prepared.adaptation.prepared_id
    marks = prepared.cell_marks(np.sort(_cell_ids(target.mesh)))
    refined = refine_adaptive_simplex(prepared.layout, prepared.state, marks).state
    state = coarsen_adaptive_simplex(
        prepared.layout, refined, refined.mesh.cell_active
    ).state
    counters = np.array(state.counters, copy=True)
    flags = np.array(state.status_flags, copy=True)
    raw_cells = np.array(state.mesh.cells, copy=True)
    raw_ids = np.array(state.mesh.cell_ids, copy=True)
    raw_active = np.array(state.mesh.cell_active, copy=True)
    raw_cursors = np.array(state.cursors, copy=True)
    restored = commit_adaptive_simplex(prepared, state)
    assert restored.status is MeshAdaptationStatus.COMPLETE
    np.testing.assert_array_equal(state.counters, counters)
    np.testing.assert_array_equal(state.status_flags, flags)
    np.testing.assert_array_equal(state.mesh.cells, raw_cells)
    np.testing.assert_array_equal(state.mesh.cell_ids, raw_ids)
    np.testing.assert_array_equal(state.mesh.cell_active, raw_active)
    np.testing.assert_array_equal(state.cursors, raw_cursors)
    np.testing.assert_array_equal(
        restored.target.mesh.coordinates, source.mesh.coordinates
    )
    np.testing.assert_array_equal(
        restored.target.mesh.vertex_global_ids, source.mesh.vertex_global_ids
    )
    np.testing.assert_array_equal(_cells(restored.target.mesh), _cells(source.mesh))
    np.testing.assert_array_equal(_cell_ids(restored.target.mesh), _cell_ids(source.mesh))
    for degree in range(dimension + 1):
        np.testing.assert_array_equal(
            restored.target.mesh.entity_set(degree).entity_ids,
            source.mesh.entity_set(degree).entity_ids,
        )
    assert isinstance(restored.hierarchy, BisectionHierarchy)
    np.testing.assert_array_equal(restored.hierarchy.tags, lineage.parent_tags)
    np.testing.assert_array_equal(restored.hierarchy.generations, lineage.parent_levels)
    assert restored.hierarchy.uniform_refinement is None
    assert restored.target.execution_evidence is not None
    assert int(restored.target.execution_evidence.total_work_units) > 0
    np.testing.assert_array_equal(
        restored.target.execution_evidence.source_preparation_work_units,
        prepared.execution_evidence.total_work_units,
    )
    assert int(restored.target.execution_evidence.host_storage_peak_bytes_upper) > 0
    continued = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            restored.target,
            MarkedMeshAdaptation(
                np.sort(_cell_ids(source.mesh))[:1], hierarchy=restored.hierarchy
            ),
            policy=policy,
        )
    )
    assert continued.status is MeshAdaptationStatus.COMPLETE
    invalid = eqx.tree_at(
        lambda value: value.state.counters,
        prepared,
        -jnp.ones_like(prepared.state.counters),
    )
    with pytest.raises(ValueError, match="baseline"):
        commit_adaptive_simplex(invalid, state)


@pytest.mark.skipif(
    not meshcore_available(), reason="native execution ledger unavailable"
)
def test_device_commit_refuses_foreign_and_changed_preparation_receipts() -> None:
    source = _certified(_triangle_grid(1, 1))
    policy = _policy(MeshAdaptationRoute.DEVICE_BISECTION)
    prepared = prepare_adaptive_simplex(source, policy=policy)
    foreign_source = _certified(
        CellMesh.from_triangles(
            np.asarray(source.mesh.coordinates) + 2.0,
            _cells(source.mesh).astype(np.int32),
        )
    )
    foreign = prepare_adaptive_simplex(foreign_source, policy=policy)
    substituted = eqx.tree_at(
        lambda value: value.execution_evidence, prepared, foreign.execution_evidence
    )
    with pytest.raises(ValueError, match="scientific operation"):
        commit_adaptive_simplex(substituted, prepared.state)
    changed = eqx.tree_at(
        lambda value: value.execution_evidence.source_preparation_work_units,
        prepared,
        jnp.asarray(1, dtype=jnp.uint64),
    )
    # A root receipt has no predecessor. Inventing scalar source work must not
    # silently recharge a different phase, even when its operation tag matches.
    with pytest.raises(ValueError, match="preparation"):
        commit_adaptive_simplex(changed, prepared.state)


def _policy(route: Any, **options: Any) -> Any:
    if route is MeshAdaptationRoute.DEVICE_BISECTION:
        options.setdefault("device_policy", AdaptiveSimplexPolicy())
    return MeshAdaptationPolicy(route, **options)


def _adapt(
    route: Any,
    source: Any,
    refine: Any = (),
    coarsen: Any = (),
    /,
    *,
    hierarchy: Any = None,
    **options: Any,
) -> Any:
    request = MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
    )
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=_policy(route, **options))
    )


def _cells(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(block.vertices) for block in mesh.blocks))


def _cell_ids(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(block.global_ids) for block in mesh.blocks))


def _assert_same_target(host: Any, device: Any) -> None:
    first, second = host.target.mesh, device.target.mesh
    assert host.status is device.status
    assert first.topology_id == second.topology_id
    np.testing.assert_array_equal(
        np.asarray(first.coordinates), np.asarray(second.coordinates)
    )
    np.testing.assert_array_equal(
        np.asarray(first.vertex_global_ids), np.asarray(second.vertex_global_ids)
    )
    np.testing.assert_array_equal(_cell_ids(first), _cell_ids(second))
    assert host.hierarchy.hierarchy_id == device.hierarchy.hierarchy_id


def _signed_measures(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    corners = np.asarray(points)[cells]
    edges = corners[:, 1:] - corners[:, :1]
    if cells.shape[1] == 3:
        return 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    return np.linalg.det(edges) / 6.0


def _assert_conforming(mesh: CellMesh, measure: float) -> None:
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
    distances = np.linalg.norm(midpoints[:, None, :] - points[None, :, :], axis=2)
    assert np.min(distances) > 1.0e-9
    measures = _signed_measures(points, cells)
    assert np.all(measures > 0.0)
    assert np.isclose(np.sum(measures), measure, rtol=0.0, atol=1.0e-12)


def _corner_cells(mesh: CellMesh) -> np.ndarray:
    corner = np.argmin(np.linalg.norm(np.asarray(mesh.coordinates), axis=1))
    return np.sort(_cell_ids(mesh)[np.any(_cells(mesh) == corner, axis=1)])


def _guarded_diagonal(source: Any) -> Any:
    """Protected-scope of one interior diagonal and the cells holding it."""

    mesh = source.mesh
    edges = mesh.entity_set(1)
    keys = np.sort(
        np.asarray(mesh.vertex_global_ids)[np.asarray(mesh.connectivity.edges)], axis=1
    )
    points = np.asarray(mesh.coordinates)
    offsets = points[keys[:, 1]] - points[keys[:, 0]]
    diagonal = np.flatnonzero(np.all(np.isclose(offsets, 0.25), axis=1))[4]
    guarded = keys[diagonal]
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        edges.entity_set_id,
        np.asarray([np.asarray(edges.entity_ids)[diagonal]]),
    )
    cell_vertices = np.asarray(mesh.vertex_global_ids)[_cells(mesh)]
    through = np.sort(
        _cell_ids(mesh)[np.sum(np.isin(cell_vertices, guarded), axis=1) == 2]
    )
    return scope, through


def _moved_vertex(state: Any, old: Any, new: Any) -> Any:
    coordinates = np.asarray(state.mesh.coordinates).copy()
    coordinates[np.all(coordinates == np.asarray(old), axis=1)] = new
    return eqx.tree_at(
        lambda value: value.mesh.coordinates, state, jnp.asarray(coordinates)
    )


def _assert_same_arrays(first: Any, second: Any) -> None:
    before, after = jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)
    assert all(
        np.array_equal(left, right) for left, right in zip(before, after, strict=True)
    )


def test_device_commit_owns_uniform_inverse_without_mutating_compiled_summary(
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    points = np.asarray(
        ((0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)), dtype=np.float64
    )
    source = _certified(
        CellMesh.from_triangles(
            points, np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32)
        )
    )
    refined = _adapt(
        MeshAdaptationRoute.NATIVE_BISECTION,
        source,
        (0,),
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    assert refined.status is MeshAdaptationStatus.COMPLETE
    prepared = prepare_adaptive_simplex(
        refined.target,
        policy=_policy(MeshAdaptationRoute.DEVICE_BISECTION),
        hierarchy=refined.hierarchy,
    )
    update = coarsen_adaptive_simplex(
        prepared.layout, prepared.state, prepared.state.mesh.cell_active
    )
    before = [
        np.array(leaf, copy=True) for leaf in jax.tree_util.tree_leaves(update.state)
    ]
    counters = np.array(update.state.counters, dtype=np.int64, copy=True)
    result = commit_adaptive_simplex(prepared, update.state)
    assert result.status is MeshAdaptationStatus.COMPLETE
    for original, leaf in zip(
        before, jax.tree_util.tree_leaves(update.state), strict=True
    ):
        np.testing.assert_array_equal(leaf, original)
    np.testing.assert_array_equal(update.state.counters, counters)
    np.testing.assert_array_equal(_cells(result.target.mesh), _cells(source.mesh))
    np.testing.assert_array_equal(_cell_ids(result.target.mesh), _cell_ids(source.mesh))
    np.testing.assert_array_equal(
        result.target.mesh.vertex_global_ids, source.mesh.vertex_global_ids
    )
    np.testing.assert_array_equal(result.target.mesh.coordinates, source.mesh.coordinates)
    assert isinstance(result.hierarchy, BisectionHierarchy)
    assert result.hierarchy.uniform_refinement is None
    assert result.target.execution_evidence is not None
    assert int(result.target.execution_evidence.externally_charged_work) > 0
    receipt = write_meshing_source_closure(
        tmp_path / "device-uniform", (result.target, result.hierarchy)
    )
    target, hierarchy = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    continued = _adapt(
        MeshAdaptationRoute.NATIVE_BISECTION,
        target,
        _cell_ids(target.mesh)[:1],
        hierarchy=hierarchy,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    assert continued.status is MeshAdaptationStatus.COMPLETE
    _assert_conforming(continued.target.mesh, 3.5)
    negative = eqx.tree_at(
        lambda state: state.counters,
        update.state,
        update.state.counters.at[0].set(-1),
    )
    with pytest.raises(ValueError, match="baseline"):
        commit_adaptive_simplex(prepared, negative)


def test_device_adaptation_scenario_2() -> None:
    points = np.asarray([(0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)])
    source = _certified(
        CellMesh.from_triangles(points, np.asarray([(0, 1, 2), (0, 3, 1)]))
    )
    options = {"compatibility": BisectionCompatibility.UNIFORM_REFINEMENT}
    host = _adapt(MeshAdaptationRoute.NATIVE_BISECTION, source, (0,), **options)
    device = _adapt(MeshAdaptationRoute.DEVICE_BISECTION, source, (0,), **options)
    assert device.evidence.uniform_refinement_applied
    _assert_same_target(host, device)
    _assert_conforming(device.target.mesh, 3.5)
    """Refine, coarsen back, and refine again before one commit."""

    source = _certified(_triangle_grid(3, 3))
    marks = np.sort(_cell_ids(source.mesh))[::2]
    policy = _policy(MeshAdaptationRoute.DEVICE_BISECTION)
    prepared = prepare_adaptive_simplex(source, policy=policy)
    layout = prepared.layout
    refined = refine_adaptive_simplex(layout, prepared.state, prepared.cell_marks(marks))
    everything = refined.state.mesh.cell_active
    restored = coarsen_adaptive_simplex(layout, refined.state, everything)
    again = refine_adaptive_simplex(layout, restored.state, prepared.cell_marks(marks))
    device = commit_adaptive_simplex(prepared, again.state)

    first = _adapt(MeshAdaptationRoute.NATIVE_BISECTION, source, marks)
    back = _adapt(
        MeshAdaptationRoute.NATIVE_BISECTION,
        first.target,
        (),
        _cell_ids(first.target.mesh),
        hierarchy=first.hierarchy,
    )
    host = _adapt(
        MeshAdaptationRoute.NATIVE_BISECTION, back.target, marks, hierarchy=back.hierarchy
    )
    assert back.target.mesh.topology_id == source.mesh.topology_id
    assert device.target.mesh.topology_id == host.target.mesh.topology_id
    # Pin the numerical content as diagnostics for canonical identity equality.
    source_order = np.argsort(_cell_ids(source.mesh), kind="stable")
    back_order = np.argsort(_cell_ids(back.target.mesh), kind="stable")
    np.testing.assert_array_equal(
        _cell_ids(back.target.mesh)[back_order], _cell_ids(source.mesh)[source_order]
    )
    np.testing.assert_array_equal(
        _cells(back.target.mesh)[back_order], _cells(source.mesh)[source_order]
    )
    device_order = np.argsort(_cell_ids(device.target.mesh), kind="stable")
    host_order = np.argsort(_cell_ids(host.target.mesh), kind="stable")
    np.testing.assert_array_equal(
        _cell_ids(device.target.mesh)[device_order],
        _cell_ids(host.target.mesh)[host_order],
    )
    np.testing.assert_array_equal(
        _cells(device.target.mesh)[device_order], _cells(host.target.mesh)[host_order]
    )
    np.testing.assert_array_equal(
        np.asarray(device.target.mesh.coordinates),
        np.asarray(host.target.mesh.coordinates),
    )
    assert isinstance(device.hierarchy, BisectionHierarchy)
    assert isinstance(host.hierarchy, BisectionHierarchy)
    assert device.hierarchy.hierarchy_id == host.hierarchy.hierarchy_id
    np.testing.assert_array_equal(
        device.hierarchy.cell_global_ids, host.hierarchy.cell_global_ids
    )
    np.testing.assert_array_equal(
        device.hierarchy.ordered_vertices, host.hierarchy.ordered_vertices
    )
    np.testing.assert_array_equal(device.hierarchy.tags, host.hierarchy.tags)
    np.testing.assert_array_equal(
        device.hierarchy.generations, host.hierarchy.generations
    )
    np.testing.assert_array_equal(
        device.hierarchy.record_parent_ids, host.hierarchy.record_parent_ids
    )
    np.testing.assert_array_equal(
        device.hierarchy.record_child_ids, host.hierarchy.record_child_ids
    )
    np.testing.assert_array_equal(
        device.hierarchy.record_vertex_ids, host.hierarchy.record_vertex_ids
    )
    slope = np.asarray((0.75, -1.25))
    before = np.asarray(source.mesh.coordinates) @ slope + 0.5
    after = np.asarray(device.target.mesh.coordinates) @ slope + 0.5
    np.testing.assert_allclose(
        # ty: ignore[unresolved-attribute]
        np.asarray(device.transfer.apply(before)),
        after,
        rtol=0.0,
        atol=1e-13,
    )
    source = _certified(_kuhn_grid(1))
    prepared = prepare_adaptive_simplex(
        source, policy=_policy(MeshAdaptationRoute.DEVICE_BISECTION)
    )
    layout, state = prepared.layout, prepared.state
    allowed = int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION)
    flags = 0
    for _ in range(4):
        points = state.mesh.coordinates[state.mesh.cells]
        near = jnp.min(jnp.linalg.norm(points, axis=2), axis=1) < 1.0e-12
        update = refine_adaptive_simplex(layout, state, near & state.mesh.cell_active)
        assert (int(update.report.status) & ~allowed) == 0
        assert int(update.report.operations) > 0
        flags |= int(update.report.status)
        state = update.state
    assert int(state.status_flags) == flags
    width = layout.dimension + 1
    neighbors = np.asarray(state.mesh.facet_neighbors)
    active = np.asarray(state.mesh.cell_active)
    owner, local = np.nonzero(active[:, None] & (neighbors >= 0))
    partner = neighbors[owner, local]
    assert np.all(active[partner // width])
    np.testing.assert_array_equal(
        neighbors[partner // width, partner % width], owner * width + local
    )
    result = commit_adaptive_simplex(prepared, state)
    # ty: ignore[unresolved-attribute]
    assert result.evidence.maximum_generation >= 4
    _assert_conforming(result.target.mesh, 1.0)
    cells = _cells(result.target.mesh)
    facets = np.sort(
        np.stack([np.delete(cells, i, axis=1) for i in range(width)], axis=1), axis=2
    ).reshape((-1, width - 1))
    _, counts = np.unique(facets, axis=0, return_counts=True)
    boundary = np.count_nonzero(np.asarray(state.mesh.boundary_facets))
    assert boundary == np.count_nonzero(counts == 1)


def _capacity_failure(source: Any) -> Any:
    policy = _policy(
        MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=AdaptiveSimplexPolicy(vertex_capacity=10, cell_capacity=12),
    )
    prepared = prepare_adaptive_simplex(source, policy=policy)
    return prepared, prepared.state, prepared.state.mesh.cell_active


def _closure_failure(source: Any) -> Any:
    policy = _policy(MeshAdaptationRoute.DEVICE_BISECTION, maximum_closure_iterations=1)
    prepared = prepare_adaptive_simplex(source, policy=policy)
    marks = prepared.cell_marks(np.sort(_cell_ids(source.mesh))[:1])
    return prepared, prepared.state, marks


def _geometry_failure(source: Any) -> Any:
    prepared = prepare_adaptive_simplex(
        source, policy=_policy(MeshAdaptationRoute.DEVICE_BISECTION)
    )
    # Vertex (0.5, 0) moved above the diagonal (0, 0)-(0.5, 0.5) inverts one cell.
    state = _moved_vertex(prepared.state, (0.5, 0.0), (0.25, 0.3))
    return prepared, state, jnp.zeros_like(state.mesh.cell_active)


def test_device_adaptation_scenario_3() -> None:
    for failure, flag, category in [
        (
            _capacity_failure,
            AdaptiveSimplexStatus.CAPACITY_EXCEEDED,
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
        ),
        (
            _closure_failure,
            AdaptiveSimplexStatus.CLOSURE_LIMIT,
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
        ),
        (
            _geometry_failure,
            AdaptiveSimplexStatus.INVALID_GEOMETRY,
            MeshingFailureCategory.QUALITY_REJECTED,
        ),
    ]:
        source = _certified(_triangle_grid(2, 2))
        prepared, state, marks = failure(source)
        layout = prepared.layout
        failed = refine_adaptive_simplex(layout, state, marks)
        assert AdaptiveSimplexStatus(int(failed.report.status)) & flag
        assert bool(failed.report.failed)
        assert AdaptiveSimplexStatus(int(failed.state.status_flags)) & flag
        # Every array but the recorded status is rolled back to the input.
        _assert_same_arrays(
            state, eqx.tree_at(lambda value: value.clocks, failed.state, state.clocks)
        )
        for call in (refine_adaptive_simplex, coarsen_adaptive_simplex):
            refused = call(layout, failed.state, failed.state.mesh.cell_active)
            report = refused.report
            assert AdaptiveSimplexStatus(int(report.status)) & flag
            assert bool(report.failed)
            assert int(report.accepted) == int(report.operations) == 0
            assert int(report.iterations) == int(report.vertices) == 0
            _assert_same_arrays(failed.state, refused.state)
        with pytest.raises(MeshingFailure) as rejection:
            commit_adaptive_simplex(prepared, failed.state)
        assert rejection.value.category is category
    source = _certified(_triangle_grid(2, 2))
    policy = _policy(
        MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=AdaptiveSimplexPolicy(maximum_coarsening_passes=1),
    )
    prepared = prepare_adaptive_simplex(source, policy=policy)
    layout, state = prepared.layout, prepared.state
    for _ in range(2):
        state = refine_adaptive_simplex(layout, state, state.mesh.cell_active).state
    coarsened = coarsen_adaptive_simplex(layout, state, state.mesh.cell_active)
    assert AdaptiveSimplexStatus(int(coarsened.report.status)) & (
        AdaptiveSimplexStatus.PASS_LIMIT
    )
    assert int(coarsened.report.operations) > 0
    assert AdaptiveSimplexStatus(int(coarsened.state.status_flags)) & (
        AdaptiveSimplexStatus.PASS_LIMIT
    )
    result = commit_adaptive_simplex(prepared, coarsened.state)
    assert result.status is MeshAdaptationStatus.PASS_LIMIT
    assert result.status.converged is False
    # ty: ignore[unresolved-attribute]
    assert 0 < result.evidence.coarsened_vertices < result.evidence.created_vertices
    _assert_conforming(result.target.mesh, 1.0)


@pytest.mark.parametrize(
    ("offset", "certified"),
    [
        pytest.param(
            2.0**-55,
            True,
            marks=pytest.mark.skipif(
                not meshcore_available(), reason="exact orientation requires meshcore"
            ),
        ),
        (0.0, False),
    ],
    ids=["positive", "collinear"],
)
def test_uncertain_device_orientation_is_resolved_exactly_at_commit(
    offset: Any, certified: Any
) -> None:
    source = _certified(_triangle_grid(2, 2))
    prepared = prepare_adaptive_simplex(
        source, policy=_policy(MeshAdaptationRoute.DEVICE_BISECTION)
    )
    # Vertex (0.5, 0) on (or one rounding below) the diagonal (0, 0)-(0.5, 0.5):
    # FILTERED_DEVICE cannot decide that cell's orientation.
    state = _moved_vertex(prepared.state, (0.5, 0.0), (0.25, 0.25 - offset))
    update = refine_adaptive_simplex(
        prepared.layout, state, jnp.zeros_like(state.mesh.cell_active)
    )
    assert int(update.report.uncertain_cells) == 1
    assert int(update.report.invalid_cells) == 0
    assert not bool(update.report.failed)
    assert (
        AdaptiveSimplexStatus(int(update.state.status_flags))
        is AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION
    )
    if certified:
        result = commit_adaptive_simplex(prepared, update.state)
        assert result.status is MeshAdaptationStatus.UNCHANGED
        return
    with pytest.raises(MeshingFailure) as rejection:
        commit_adaptive_simplex(prepared, update.state)
    assert rejection.value.category is MeshingFailureCategory.QUALITY_REJECTED


def _compact_poisson(mesh: CellMesh, source_term: Any) -> Any:
    kind = mesh.blocks[0].cell_kind
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element(kind, 1)
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.FiniteElementForm(
        "poisson",
        "u",
        (
            phx.equations.DiffusionAction("u"),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    lambda x, _: source_term(x), coefficient_id="affine-source"
                ),
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=phx.discretization.dirichlet_constraint(discretization, "u"),
        dirichlet_values=0.0,
    )
    system, rhs = compiled.linear_system()
    return compiled.expand(phx.linalg.solve(system, rhs).value)


def test_masked_poisson_on_the_device_state_equals_the_committed_solve() -> None:
    def source_term(points: Any) -> Any:
        return 1.0 + points[..., 0] + 2.0 * points[..., 1]

    source = _certified(_triangle_grid(3, 3))
    prepared = prepare_adaptive_simplex(
        source, policy=_policy(MeshAdaptationRoute.DEVICE_BISECTION)
    )
    marks = prepared.cell_marks(np.sort(_cell_ids(source.mesh))[:4])
    update = refine_adaptive_simplex(prepared.layout, prepared.state, marks)
    second = update.state.mesh.cell_active & (update.state.mesh.cell_ids % 3 == 0)
    update = refine_adaptive_simplex(prepared.layout, update.state, second)
    mesh = update.state.mesh
    plan = phx.discretization.MaskedFiniteElementPlan(mesh)
    system = phx.discretization.assemble_masked_finite_element(plan, mesh)
    load = jnp.where(mesh.vertex_active, source_term(mesh.coordinates), 0.0)
    rhs = jnp.where(system.boundary_dofs, 0.0, system.mass.mv(load))
    operator = phx.discretization.constrain_masked_dofs(
        system.stiffness, system.boundary_dofs
    )
    masked = phx.linalg.solve(phx.linalg.LinearSystem(operator), rhs)
    assert bool(jnp.all(masked.successful))

    committed = commit_adaptive_simplex(prepared, update.state).target.mesh
    expected = _compact_poisson(committed, source_term)
    identifiers = np.asarray(mesh.vertex_ids)
    active = np.asarray(mesh.vertex_active)
    slots = np.searchsorted(
        np.where(identifiers >= 0, identifiers, np.iinfo(np.int64).max),
        np.asarray(committed.vertex_global_ids),
    )
    assert np.count_nonzero(active) == committed.coordinates.shape[0]
    np.testing.assert_allclose(
        np.asarray(masked.value)[slots], np.asarray(expected), rtol=1e-9, atol=1e-11
    )
    assert np.all(np.asarray(masked.value)[~active] == 0.0)


@pytest.mark.parametrize("scenario", ("chain", "owners", "protected", "capacity"))
def test_part_sharded_neighbor_closure_semantics(scenario: str) -> None:
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    subprocess.run(
        [sys.executable, "-c", _NEIGHBOR_PARTS_SCRIPT, scenario],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )


_NEIGHBOR_PARTS_SCRIPT = textwrap.dedent(
    """
    import sys

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from phydrax.discretization._adaptive_simplex import (
        AdaptiveSimplexLayout,
        AdaptiveSimplexParts,
        AdaptiveSimplexState,
        AdaptiveSimplexStatus,
        adaptive_simplex_state,
        refine_adaptive_simplex,
        refine_adaptive_simplex_parts,
    )

    scenario = sys.argv[1]
    angles = np.linspace(0.0, np.pi, 5)
    points = np.vstack((np.zeros((1, 2)), np.stack((np.cos(angles), np.sin(angles)), axis=1)))
    rows = np.asarray([(0, index, index + 1) for index in range(1, 5)], dtype=np.int32)
    owners = np.asarray([3, 0, 2, 1] if scenario == "owners" else [0, 1, 2, 3], dtype=np.int32)
    layout = AdaptiveSimplexLayout(
        2, 2, vertex_capacity=64,
        cell_capacity=3 if scenario == "capacity" else 64,
        protected_edge_capacity=4, maximum_closure_iterations=64,
        maximum_coarsening_passes=1,
    )

    def state_for(indices: np.ndarray, bucket: AdaptiveSimplexLayout) -> AdaptiveSimplexState:
        global_rows = rows[indices]
        vertex_ids = np.unique(global_rows).astype(np.int64)
        local_rows = np.searchsorted(vertex_ids, global_rows).astype(np.int32)
        count = indices.size
        protected = np.empty((0, 2), dtype=np.int32)
        if scenario == "protected" and 3 in indices:
            protected = np.searchsorted(vertex_ids, np.asarray([[0, 5]])).astype(np.int32)
        return adaptive_simplex_state(
            bucket,
            coordinates=points[vertex_ids],
            vertex_ids=vertex_ids,
            vertex_active=np.ones(vertex_ids.size, dtype=np.bool_),
            vertex_parents=np.full((vertex_ids.size, 2), -1, dtype=np.int32),
            vertex_protected=np.zeros(vertex_ids.size, dtype=np.bool_),
            cells=local_rows, tuples=local_rows,
            tags=np.full(count, 2, dtype=np.int32),
            blocks=np.zeros(count, dtype=np.int32),
            generations=np.zeros(count, dtype=np.int32),
            parents=np.full(count, -1, dtype=np.int32),
            children=np.full((count, 2), -1, dtype=np.int32),
            bisection_vertices=np.full(count, -1, dtype=np.int32),
            cell_ids=indices.astype(np.int64) + 10,
            cell_active=np.ones(count, dtype=np.bool_),
            cell_classes=np.zeros(count, dtype=np.int32),
            facet_classes=np.zeros((count, 3), dtype=np.int32),
            protected_edges=protected, next_vertex_id=6, next_cell_id=14,
        )

    pieces = [state_for(np.flatnonzero(owners == part), layout) for part in range(4)]
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *pieces)
    # Every shared edge is routed, but center-only vertex overlap is not a
    # topology route. A forced cavity must advance through the whole chain.
    neighbors = tuple(
        pair for index in range(3)
        for pair in ((int(owners[index]), int(owners[index + 1])),
                     (int(owners[index + 1]), int(owners[index])))
    )
    parts = AdaptiveSimplexParts(jax.devices("cpu"), neighbor_pairs=neighbors)
    marks = np.zeros((4, layout.cell_capacity), dtype=np.bool_)
    marks[owners[0], 0] = True
    update = refine_adaptive_simplex_parts(layout, parts, stacked, marks)

    def same_tree(first: AdaptiveSimplexState, second: AdaptiveSimplexState) -> None:
        for left, right in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second), strict=True):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    if scenario in ("protected", "capacity"):
        expected = AdaptiveSimplexStatus.PROTECTED_CONFLICT if scenario == "protected" else AdaptiveSimplexStatus.CAPACITY_EXCEEDED
        assert np.all(np.asarray(update.report.status) & int(expected))
        assert np.all(np.asarray(update.report.accepted) == 0)
        assert np.all(np.asarray(update.report.operations) == 0)
        cleared = eqx.tree_at(
            lambda state: state.clocks, update.state,
            update.state.clocks.at[:, 2].set(0),
        )
        same_tree(cleared, stacked)
        refused = refine_adaptive_simplex_parts(layout, parts, update.state, marks)
        same_tree(refused.state, update.state)
        assert np.all(np.asarray(refused.report.failed))
    else:
        assert np.all(np.asarray(update.report.status) == 0)
        grown = np.asarray(update.state.cursors[:, 1] - stacked.cursors[:, 1])
        assert np.all(grown > 0), grown
        assert np.all(np.asarray(update.report.iterations) >= 3)
        serial_state = state_for(np.arange(4, dtype=np.int32), layout)
        serial_marks = np.zeros(layout.cell_capacity, dtype=np.bool_)
        serial_marks[0] = True
        serial = refine_adaptive_simplex(layout, serial_state, serial_marks)
        assert int(serial.report.status) == 0
        mesh = update.state.mesh
        active = np.asarray(mesh.cell_active)
        cell_ids = np.asarray(mesh.cell_ids)[active]
        connectivity = np.asarray(mesh.vertex_ids)[np.arange(4)[:, None, None], np.asarray(mesh.cells)][active]
        order = np.argsort(cell_ids)
        reference = serial.state.mesh
        reference_active = np.asarray(reference.cell_active)
        reference_order = np.argsort(np.asarray(reference.cell_ids)[reference_active])
        np.testing.assert_array_equal(
            cell_ids[order], np.asarray(reference.cell_ids)[reference_active][reference_order],
        )
        np.testing.assert_array_equal(
            connectivity[order],
            np.asarray(reference.vertex_ids)[np.asarray(reference.cells)[reference_active]][reference_order],
        )
        held = np.asarray(mesh.vertex_active)
        ids = np.asarray(mesh.vertex_ids)[held]
        coordinates = np.asarray(mesh.coordinates)[held]
        unique, first = np.unique(ids, return_index=True)
        for identifier, coordinate in zip(ids, coordinates, strict=True):
            np.testing.assert_array_equal(coordinate, coordinates[first[np.searchsorted(unique, identifier)]])
        reference_held = np.asarray(reference.vertex_active)
        np.testing.assert_array_equal(unique, np.asarray(reference.vertex_ids)[reference_held])
        np.testing.assert_array_equal(coordinates[first], np.asarray(reference.coordinates)[reference_held])
    """
)
