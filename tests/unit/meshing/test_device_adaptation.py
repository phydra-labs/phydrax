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
    with pytest.raises(TypeError, match="cells"):
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
    np.testing.assert_array_equal(
        np.asarray(device.target.mesh.coordinates),
        np.asarray(host.target.mesh.coordinates),
    )
    # ty: ignore[unresolved-attribute]
    assert device.hierarchy.hierarchy_id == host.hierarchy.hierarchy_id
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
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", _PARTS_SCRIPT],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    lines = dict(line.split(" ", 1) for line in completed.stdout.strip().splitlines())
    assert lines["status"] == "0 0"
    grown = np.asarray(lines["grown"].split(), dtype=np.int64)
    assert grown[0] > 0 and np.count_nonzero(grown[1:]) > 0
    for key in ("single", "lineage", "hierarchy", "host", "coordinates", "distribution"):
        assert lines[key] == "True", key
    capacity = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
    for key in ("failure", "flags"):
        values = np.asarray(lines[key].split(), dtype=np.int64)
        assert values.size == 4 and np.all(values & capacity), key
    assert lines["rolled"] == lines["refused"] == "True"
    assert lines["rejected"] == MeshingFailureCategory.RESOURCE_EXHAUSTED.value


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


_PARTS_SCRIPT = textwrap.dedent(
    """
    import itertools

    import numpy as np

    import phydrax as phx
    from phydrax.discretization import (
        AdaptiveSimplexPolicy,
        CellMesh,
        refine_adaptive_simplex,
        refine_adaptive_simplex_parts,
    )
    from phydrax.meshing import (
        commit_adaptive_simplex,
        commit_partitioned_adaptive_simplex,
        execute_mesh_adaptation,
        MarkedMeshAdaptation,
        MeshAdaptationPolicy,
        MeshAdaptationRoute,
        MeshingFailure,
        MeshPart,
        MeshPartitionKind,
        MeshPartitionPolicy,
        partition_adaptive_simplex,
        prepare_adaptive_simplex,
        prepare_mesh_adaptation,
        prepare_mesh_distribution,
    )

    xs = np.linspace(0.0, 1.0, 9)
    points = np.stack(np.meshgrid(xs, xs), axis=-1).reshape((-1, 2))
    cells = []
    for j, i in itertools.product(range(8), range(8)):
        a = j * 9 + i
        cells.extend(((a, a + 1, a + 10), (a, a + 10, a + 9)))
    mesh = CellMesh.from_triangles(points, np.asarray(cells, dtype=np.int32))
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    partition = MeshPartitionPolicy(MeshPartitionKind.MORTON, 4)
    distribution = prepare_mesh_distribution(MeshPart("domain", source), policy=partition)
    owners = np.asarray(distribution.partition.cell_owner)
    marks = np.asarray(distribution.cell_global_ids)[owners == 0]
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=AdaptiveSimplexPolicy(vertex_capacity=1024, cell_capacity=1024),
        distribution=distribution,
        partition_policy=partition,
    )
    prepared = prepare_adaptive_simplex(source, policy=policy)
    partitioned = partition_adaptive_simplex(prepared)
    layout, parts = partitioned.layout, partitioned.parts
    first = refine_adaptive_simplex_parts(
        layout, parts, partitioned.states, partitioned.cell_marks(marks)
    )
    # Refine every cell part 0 holds again: children split the shared sides.
    again = np.asarray(first.state.mesh.cell_active).copy()
    again[1:] = False
    second = refine_adaptive_simplex_parts(layout, parts, first.state, again)
    grown = np.asarray(second.state.cursors[:, 1] - first.state.cursors[:, 1])
    result = commit_partitioned_adaptive_simplex(partitioned, second.state)

    held = np.asarray(first.state.mesh.cell_ids[0])[again[0]]
    single = refine_adaptive_simplex(
        prepared.layout, prepared.state, prepared.cell_marks(marks)
    )
    active = np.asarray(single.state.mesh.cell_active)
    single = refine_adaptive_simplex(
        prepared.layout,
        single.state,
        active & np.isin(np.asarray(single.state.mesh.cell_ids), held),
    )
    reference = commit_adaptive_simplex(prepared, single.state)

    host_policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION)
    host = execute_mesh_adaptation(
        prepare_mesh_adaptation(source, MarkedMeshAdaptation(np.sort(marks)), policy=host_policy)
    )
    host = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            host.target,
            MarkedMeshAdaptation(np.sort(held), hierarchy=host.hierarchy),
            policy=host_policy,
        )
    )
    target = result.target.mesh

    def cells_by_id(mesh):
        rows = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
        ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
        return ids, np.asarray(mesh.vertex_global_ids)[rows]

    # Two host commits number intermediate edges per cycle; vertices and cells
    # are issued identically.
    device_cells, host_cells = cells_by_id(target), cells_by_id(host.target.mesh)
    print("status", int(first.report.status[0]), int(second.report.status[0]))
    print("grown", " ".join(str(int(value)) for value in grown))
    print("single", target.topology_id == reference.target.mesh.topology_id)
    print("lineage", result.lineage.lineage_id == reference.lineage.lineage_id)
    print("hierarchy", result.hierarchy.hierarchy_id == reference.hierarchy.hierarchy_id)
    print(
        "host",
        np.array_equal(device_cells[0], host_cells[0])
        and np.array_equal(device_cells[1], host_cells[1])
        and np.array_equal(
            np.asarray(target.vertex_global_ids),
            np.asarray(host.target.mesh.vertex_global_ids),
        ),
    )
    print(
        "coordinates",
        np.array_equal(np.asarray(target.coordinates), np.asarray(host.target.mesh.coordinates)),
    )
    print("distribution", result.distribution is not None)

    # Tight buckets: refining part 0's cells overflows part 0 alone, and the
    # collective status fails, records, and refuses on every part.
    tight = partition_adaptive_simplex(
        prepare_adaptive_simplex(
            source,
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.DEVICE_BISECTION,
                device_policy=AdaptiveSimplexPolicy(growth_factor=1.0),
                distribution=distribution,
                partition_policy=partition,
            ),
        )
    )
    overflow = refine_adaptive_simplex_parts(
        tight.layout, tight.parts, tight.states, tight.cell_marks(marks)
    )
    refused = refine_adaptive_simplex_parts(
        tight.layout, tight.parts, overflow.state, tight.cell_marks(marks)
    )
    print("failure", " ".join(str(int(value)) for value in overflow.report.status))
    print("flags", " ".join(str(int(value)) for value in overflow.state.status_flags))
    print(
        "rolled",
        np.array_equal(np.asarray(overflow.state.cursors), np.asarray(tight.states.cursors)),
    )
    print(
        "refused",
        bool(np.all(np.asarray(refused.report.failed)))
        and int(np.sum(refused.report.operations)) == 0
        and np.array_equal(
            np.asarray(refused.state.cursors), np.asarray(overflow.state.cursors)
        ),
    )
    try:
        commit_partitioned_adaptive_simplex(tight, overflow.state)
    except MeshingFailure as error:
        print("rejected", error.category.value)
    else:
        print("rejected none")
    """
)
