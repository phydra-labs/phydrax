#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import itertools
from collections.abc import Callable
from functools import partial
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
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    CellMesh,
    coarsen_adaptive_simplex,
    refine_adaptive_simplex,
    refine_adaptive_simplex_parts,
)
from phydrax.meshing import (
    BisectionCompatibility,
    BisectionHierarchy,
    CellMeshingResult,
    commit_adaptive_simplex,
    commit_partitioned_adaptive_simplex,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
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
    PreparedAdaptiveSimplex,
)
from phydrax.meshing._measurements import NativeExecutionRecord
from tests._support.cell_meshes import kuhn_cube, triangle_grid
from tests._support.distributed_meshing.device_adaptation import (
    _adapt,
    _cell_ids,
    _cells,
    _certified,
    _policy,
)
from tests._support.forced_devices import run_on_forced_devices


# Reference envelopes retained from d2dfcb941's fixed float64 volume, affine
# transfer, and masked-versus-compact Poisson fixtures; none are mesh policies.
_MEASURE_AND_BALANCE_ATOL = 1.0e-12
_AFFINE_TRANSFER_ATOL = 1.0e-13
_POISSON_COMPARISON_RTOL = 1.0e-9
_POISSON_COMPARISON_ATOL = 1.0e-11
# Four unit-cube bisections keep nonzero corner distances above 1/16; this
# threshold separates those corners from the exactly represented origin.
_ORIGIN_SELECTION_RADIUS = 1.0e-12


def test_device_mesh_contract_refuses_wrong_extent_and_dtype() -> None:
    prepared = prepare_adaptive_simplex(
        _certified(triangle_grid(1, 1)),
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
    with pytest.raises(TypeError, match="cells"):
        phx.typing.validate(wrong_dtype)


@pytest.mark.parametrize(
    "mesh_factory",
    (partial(triangle_grid, 3, 3), partial(kuhn_cube, 1, orientation_swap=(0, 1))),
    ids=("triangle", "tetrahedron"),
)
def test_masked_fv_conservation_after_device_refinement(
    mesh_factory: Callable[[], CellMesh],
) -> None:
    mesh = mesh_factory()
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
    assert np.isclose(
        content, -np.sum(np.asarray(flux)[boundary]), atol=_MEASURE_AND_BALANCE_ATOL
    )
    assert np.all(np.asarray(residual)[~active] == 0.0)
    assert np.isclose(np.sum(volumes[active]), 1.0, atol=_MEASURE_AND_BALANCE_ATOL)
    ledger = phx.discretization.evaluate_masked_fv_conservation(
        geometry, flux, jnp.zeros((state.mesh.cell_capacity,))
    )
    assert np.all(np.abs(np.asarray(ledger.residual)) < _MEASURE_AND_BALANCE_ATOL)


@pytest.mark.parametrize(
    "mesh_factory",
    (partial(triangle_grid, 4, 4), partial(kuhn_cube, 2, orientation_swap=(0, 1))),
    ids=("triangle", "tetrahedron"),
)
def test_device_and_host_marked_lifecycle_have_identical_targets(
    mesh_factory: Callable[[], CellMesh],
) -> None:
    mesh = mesh_factory()
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


def test_device_protected_scope_refuses_only_incident_marks() -> None:
    source = _certified(triangle_grid(4, 4))
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


@pytest.mark.parametrize(
    "route",
    (MeshAdaptationRoute.NATIVE_BISECTION, MeshAdaptationRoute.DEVICE_BISECTION),
    ids=("native", "device"),
)
def test_protected_only_request_preserves_the_source_epoch(
    route: MeshAdaptationRoute,
) -> None:
    source = _certified(triangle_grid(4, 4))
    scope, through = _guarded_diagonal(source)
    rejected = _adapt(route, source, through, protected_scopes=(scope,))
    assert rejected.status is MeshAdaptationStatus.PARTIAL
    assert not rejected.status.converged
    assert rejected.target.result_id == source.result_id
    assert rejected.transition is rejected.lineage is rejected.transfer is None
    np.testing.assert_array_equal(
        np.asarray(rejected.evidence.rejected_refinement_ids), through
    )


def test_partitioned_protected_conflict_retains_clocks_and_refuses_commit() -> None:
    source = _certified(triangle_grid(4, 4))
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
    update = refine_adaptive_simplex_parts(partitioned.layout, type(partitioned).execution_parts(partitioned), partitioned.states,
    partitioned.cell_marks(_cell_ids(source.mesh)),)
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
    source = _certified(triangle_grid(1, 1))
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
    forged = eqx.tree_at(
        lambda value: value.execution_evidence.source_preparation_work_units,
        prepared,
        jnp.asarray(1, dtype=jnp.uint64),
    )
    # The chained host preparation makes the forged aggregate fail first.
    with pytest.raises(
        ValueError,
        match=r"NativeExecutionRecord\.(receipt_id must bind|preparation_aggregates must match)",
    ):
        commit_adaptive_simplex(forged, prepared.state)
    original = prepared.execution_evidence
    assert original is not None
    # The host preparation is this request's chained phase, not an alias.
    assert original.preparation_evidence is prepared.adaptation.preparation_evidence
    assert original.consumer_evidence is None
    assert int(np.asarray(original.imported_preparation_count)) == 0
    changed = NativeExecutionRecord(
        original.native_evidence(),
        source_preparation_work_units=1,
        source_preparation_geometry_queries=int(
            np.asarray(original.source_preparation_geometry_queries)
        ),
        preparation_seconds=float(np.asarray(original.preparation_seconds)),
        owner_id=original.owner_id,
    )
    assert changed.receipt_hex() != original.receipt_hex()
    changed.require_valid()
    # Preparation is checked at construction, so commit cannot receive this record.
    with pytest.raises(ValueError, match="preparation"):
        PreparedAdaptiveSimplex(
            prepared.adaptation,
            prepared.layout,
            prepared.state,
            prepared.anchor,
            execution_evidence=changed,
        )


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
    assert np.min(distances) > _POISSON_COMPARISON_RTOL
    measures = _signed_measures(points, cells)
    assert np.all(measures > 0.0)
    assert np.isclose(np.sum(measures), measure, rtol=0.0, atol=_MEASURE_AND_BALANCE_ATOL)


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


def test_device_uniform_compatibility_matches_the_host_target() -> None:
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


@pytest.mark.parametrize("uniform", (False, True), ids=("root-labels", "uniform"))
def test_source_solver_owners_inherit_canonical_preclosure_lineage(
    uniform: bool,
) -> None:
    from phydrax.meshing._device_adaptation import source_solver_cell_owners

    mesh = (
        CellMesh.from_triangles(
            np.asarray(
                ((0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)),
                dtype=np.float64,
            ),
            np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32),
        )
        if uniform
        else triangle_grid(1, 1)
    )
    source = _certified(mesh)
    partition_policy = MeshPartitionPolicy(MeshPartitionKind.MORTON, 2)
    distribution = prepare_mesh_distribution(
        MeshPart("domain", source),
        policy=partition_policy,
    )
    prepared = prepare_adaptive_simplex(
        source,
        policy=_policy(
            MeshAdaptationRoute.DEVICE_BISECTION,
            compatibility=(
                BisectionCompatibility.UNIFORM_REFINEMENT
                if uniform
                else BisectionCompatibility.REJECT
            ),
            distribution=distribution,
            partition_policy=partition_policy,
        ),
    )
    expected = {
        int(identifier): int(owner)
        for identifier, owner in zip(
            np.asarray(distribution.cell_global_ids),
            np.asarray(distribution.partition.cell_owner),
            strict=True,
        )
    }
    assert set(expected.values()) == {0, 1}
    lineage = prepared.anchor.start.uniform_refinement
    if uniform:
        assert lineage is not None
        expected = {
            int(child): expected[int(parent)]
            for parent, children in zip(
                lineage.parent_ids, lineage.child_ids, strict=True
            )
            for child in children
        }
    else:
        assert lineage is None
    state = prepared.state
    active = np.asarray(state.mesh.cell_active)
    identifiers = np.asarray(state.mesh.cell_ids)
    assert set(identifiers[active]) == set(expected)
    owners = source_solver_cell_owners(prepared, state)
    assert owners.dtype == jnp.int32
    np.testing.assert_array_equal(
        np.asarray(owners)[active],
        np.asarray([expected[int(key)] for key in identifiers[active]], dtype=np.int32),
    )
    assert np.all(np.asarray(owners)[~active] == -1)
    if not uniform:
        np.testing.assert_array_equal(
            owners, source_solver_cell_owners(prepared.adaptation, state)
        )
        local = eqx.tree_at(
            lambda value: (value.anchor.cell_ids, value.anchor.slot_origins),
            prepared,
            (prepared.anchor.cell_ids[1:], prepared.anchor.slot_origins[1:]),
        )
        np.testing.assert_array_equal(owners, source_solver_cell_owners(local, state))

    slot = int(np.flatnonzero(active)[0])
    absent = int(np.max(identifiers)) + 1
    missing = eqx.tree_at(
        lambda value: value.mesh.cell_ids,
        state,
        state.mesh.cell_ids.at[slot].set(absent),
    )
    for preparation in (prepared, prepared.adaptation):
        with pytest.raises(ValueError, match="solver ownership bank"):
            source_solver_cell_owners(preparation, missing)
    if uniform:
        origins = prepared.anchor.slot_origins.copy()
        origins[prepared.anchor.cell_ids == identifiers[slot]] = absent
        missing_owner = eqx.tree_at(
            lambda value: value.anchor.slot_origins, prepared, origins
        )
        with pytest.raises(ValueError, match="solver ownership bank"):
            source_solver_cell_owners(missing_owner, state)


def test_device_refine_coarsen_refine_matches_host_ancestry_and_affine_transfer() -> None:
    source = _certified(triangle_grid(3, 3))
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
    # Pin the numerical content as diagnostics for identity equality.
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
        atol=_AFFINE_TRANSFER_ATOL,
    )


def test_tetrahedral_device_closure_retains_reciprocal_facets_after_repeated_edits() -> (
    None
):
    source = _certified(kuhn_cube(1, orientation_swap=(0, 1)))
    prepared = prepare_adaptive_simplex(
        source, policy=_policy(MeshAdaptationRoute.DEVICE_BISECTION)
    )
    layout, state = prepared.layout, prepared.state
    allowed = int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION)
    flags = 0
    for _ in range(4):
        points = state.mesh.coordinates[state.mesh.cells]
        near = jnp.min(jnp.linalg.norm(points, axis=2), axis=1) < _ORIGIN_SELECTION_RADIUS
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


@pytest.mark.parametrize(
    "failure, flag, category",
    (
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
    ),
    ids=("capacity", "closure", "geometry"),
)
def test_failed_device_epoch_rolls_back_and_remains_terminal(
    failure: Callable[
        [CellMeshingResult],
        tuple[PreparedAdaptiveSimplex, AdaptiveSimplexState, jax.Array],
    ],
    flag: AdaptiveSimplexStatus,
    category: MeshingFailureCategory,
) -> None:
    source = _certified(triangle_grid(2, 2))
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


def test_device_coarsening_pass_limit_publishes_partial_progress() -> None:
    source = _certified(triangle_grid(2, 2))
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
    source = _certified(triangle_grid(2, 2))
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

    source = _certified(triangle_grid(3, 3))
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
        np.asarray(masked.value)[slots],
        np.asarray(expected),
        rtol=_POISSON_COMPARISON_RTOL,
        atol=_POISSON_COMPARISON_ATOL,
    )
    assert np.all(np.asarray(masked.value)[~active] == 0.0)


@pytest.mark.parametrize("scenario", ("chain", "owners", "protected", "capacity"))
def test_part_sharded_neighbor_closure_semantics(scenario: str) -> None:
    report = run_on_forced_devices(
        "tests._support.distributed_meshing.device_adaptation",
        f"neighbor_{scenario}",
        (),
        devices=4,
        env={},
    )
    status = np.asarray(report["status"])
    if scenario in ("protected", "capacity"):
        expected = (
            AdaptiveSimplexStatus.PROTECTED_CONFLICT
            if scenario == "protected"
            else AdaptiveSimplexStatus.CAPACITY_EXCEEDED
        )
        assert np.all(status & int(expected))
        assert np.all(np.asarray(report["accepted"]) == 0)
        assert np.all(np.asarray(report["operations"]) == 0)
        for left, right in zip(
            report["rollback_leaves"], report["source_leaves"], strict=True
        ):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(
            report["refused_leaves"], report["failed_leaves"], strict=True
        ):
            np.testing.assert_array_equal(left, right)
        assert np.all(report["refused_failed"])
    else:
        assert np.all(status == 0)
        grown = np.asarray(report["grown"])
        assert np.all(grown > 0), grown
        assert np.all(np.asarray(report["iterations"]) >= 3)
        assert report["serial_status"] == 0
        np.testing.assert_array_equal(report["cell_ids"], report["reference_cell_ids"])
        np.testing.assert_array_equal(
            report["connectivity"], report["reference_connectivity"]
        )
        np.testing.assert_array_equal(report["coordinates"], report["shared_coordinates"])
        np.testing.assert_array_equal(
            report["vertex_ids"], report["reference_vertex_ids"]
        )
        np.testing.assert_array_equal(
            report["unique_coordinates"], report["reference_coordinates"]
        )


@pytest.mark.parametrize("scenario", ("identity", "capacity"))
def test_partitioned_commit_preserves_public_contracts(scenario: str) -> None:
    report = run_on_forced_devices(
        "tests._support.distributed_meshing.device_adaptation",
        f"public_{scenario}",
        (),
        devices=4,
        env={},
    )
    if scenario == "capacity":
        flag = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
        status = np.asarray(report["status"])
        assert status.size == 4
        assert np.all(status & flag)
        assert np.all(np.asarray(report["flags"]) & flag)
        np.testing.assert_array_equal(report["cursors"], report["source_cursors"])
        assert np.all(report["refused_failed"])
        assert report["refused_operations"] == 0
        np.testing.assert_array_equal(report["refused_cursors"], report["cursors"])
        assert report["category"] == MeshingFailureCategory.RESOURCE_EXHAUSTED.value
    else:
        assert np.all(np.asarray(report["first_status"]) == 0)
        assert np.all(np.asarray(report["second_status"]) == 0)
        grown = np.asarray(report["grown"])
        assert grown[0] > 0 and np.count_nonzero(grown[1:]) > 0
        assert report["topology_id"] == report["reference_topology_id"]
        assert report["lineage_id"] == report["reference_lineage_id"]
        assert report["hierarchy_id"] == report["reference_hierarchy_id"]
        assert report["source_certification_absent"]
        assert len(report["locals"]) == 4
        for local in report["locals"]:
            assert local["storage_present"]
            assert local["topology_id"] == report["topology_id"]
            assert local["embedding_status"] == "certified"
            assert local["source_audit_id"] == report["source_audit_id"]
            assert local["source_premise_id"] == report["source_premise_id"]
            assert local["distribution_present"]
            assert local["source_result_id"] == report["source_result_id"]
            assert local["rebalanced"]
            assert local["imbalance"] <= report["maximum_imbalance"]
            assert np.all(local["migration_accepted"])
            assert local["sent_cells"] > 0
            np.testing.assert_array_equal(
                local["part_weights"], local["proposed_weights"]
            )
        np.testing.assert_array_equal(report["cell_ids"], report["host_cell_ids"])
        np.testing.assert_array_equal(report["connectivity"], report["host_connectivity"])
        np.testing.assert_array_equal(report["vertex_ids"], report["host_vertex_ids"])
        np.testing.assert_array_equal(report["coordinates"], report["host_coordinates"])
