import ctypes.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import _config_state
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax._fingerprint import canonical_fingerprint
from phydrax.graph import (
    DistributedWeightedCSRGraph,
    GraphPartitionPlan,
    partition_graph_distributed,
)
from phydrax.lifecycle import CompositionEntry
from phydrax.meshing._assembly import MeshPart
from phydrax.meshing._distribution import (
    _OWNER_LOCAL_IDENTITY_BUCKET_COUNT,
    _owner_local_multiset_identity,
    MeshDistribution,
    OwnerLocalGraphMigrationEvidence,
    ShardedOwnerDirectoryWitness,
)


@pytest.fixture(autouse=True)
def restore_metis_setting() -> Iterator[None]:
    """Own METIS test overrides without changing other provider settings."""
    previous = _config_state._STATE.libraries.copy()
    _config_state._STATE.libraries.pop("metis", None)
    try:
        yield
    finally:
        _config_state._STATE.libraries.clear()
        _config_state._STATE.libraries.update(previous)


def _cell_part(name: Any = "cells", scale: Any = 1.0, single: Any = False) -> Any:
    points = scale * np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    cells = np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int32)
    if single:
        points = points[[0, 1, 3]]
        cells = np.asarray(((0, 1, 2),), dtype=np.int32)
    mesh = phx.discretization.CellMesh.from_triangles(points, cells)
    return MeshPart(
        name, phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    )


def _fe(part: Any) -> Any:
    return phx.discretization.fem.FiniteElementPlan(
        part.carrier.mesh,
        phx.discretization.fem.FiniteElementFieldSpec(
            "u", phx.discretization.fem.discontinuous_element("triangle", 1)
        ),
        coordinate_spec=part.carrier.geometry,
    ).prepare()


def _grid_part(name: Any = "grid", *, periodic: Any = True, scale: Any = 1.0) -> Any:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=periodic),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [scale]]))
    return MeshPart(name, grid, coordinate_contract=phx.SpatialCoordinateContract.si())


def _compiled_fv(part: Any) -> Any:
    discretization = phx.discretization.FiniteVolumePlan(part.carrier).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="meshing-distribution-advection",
    )
    problem = phx.equations.ConservationProblemIR(
        "meshing-distribution-advection",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    return phx.equations.compile_conservation_problem(problem, discretization, method)


def test_distribution_scenario_1() -> None:
    part = _cell_part()
    native_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in part.carrier.mesh.blocks]
    )
    distribution = MeshDistribution(
        part,
        phx.discretization.CellPartition(np.asarray([1, 0]), 2),
        cell_global_ids=native_ids[::-1],
    )
    field = jnp.asarray([3.0, 7.0])
    np.testing.assert_allclose(distribution.gather(0, field), [3.0, 7.0])
    np.testing.assert_allclose(distribution.gather(1, field), [7.0, 3.0])
    phases = distribution.lower_finite_element(part, _fe(part))
    np.testing.assert_allclose(
        sum(phases.local_contribution(rank, field) for rank in range(2)), 10.0
    )
    flux = jnp.asarray([2.0])
    routed = sum(phases.facet_ownership.route_partition(rank, flux) for rank in range(2))
    np.testing.assert_allclose(routed, [2.0, -2.0])
    with pytest.raises(ValueError, match="stale"):
        distribution.lower_finite_element(_cell_part(scale=2.0), _fe(part))
    with pytest.raises(ValueError, match="exact distribution mesh"):
        distribution.lower_finite_element(part, _fe(_cell_part(scale=2.0)))
    part = _cell_part(single=True)
    distribution = MeshDistribution(
        part, phx.discretization.CellPartition(np.asarray([0]), 1)
    )
    phases = distribution.lower_finite_element(part, _fe(part))
    np.testing.assert_allclose(phases.local_contribution(0, jnp.asarray([4.0])), 4.0)
    np.testing.assert_allclose(
        phases.facet_ownership.route_equal_opposite(jnp.empty((0,))), [0.0]
    )
    part, _ = _quad_mesh(8, 6)
    distribution = phx.meshing.prepare_mesh_distribution(
        part,
        policy=phx.meshing.MeshPartitionPolicy(
            phx.meshing.MeshPartitionKind.HILBERT, 3, halo_width=2
        ),
    )
    phases = distribution.lower_finite_element(part, _fe(part))
    ownership = phases.facet_ownership
    np.testing.assert_array_equal(
        np.sum(np.asarray(ownership.evaluation_mask), axis=0), 1
    )
    owner = np.asarray(distribution.partition.cell_owner)
    facets = np.asarray(ownership.facet_cells)
    evaluator = np.argmax(np.asarray(ownership.evaluation_mask), axis=0)
    interface = owner[facets[:, 0]] != owner[facets[:, 1]]
    assert np.any(interface)
    assert np.all((evaluator == owner[facets[:, 0]]) | (evaluator == owner[facets[:, 1]]))
    flux = jnp.arange(1.0, facets.shape[0] + 1.0)
    np.testing.assert_allclose(
        sum(ownership.route_partition(rank, flux) for rank in range(3)),
        ownership.route_equal_opposite(flux),
    )
    field = jnp.arange(96.0)
    np.testing.assert_allclose(
        sum(phases.local_contribution(rank, field) for rank in range(3)), jnp.sum(field)
    )
    assert phases.worksets.owned_cells.shape[1] == np.max(np.bincount(owner))
    assert phases.worksets.halo_cells.shape[1] < owner.size
    part = _cell_part()
    with pytest.raises(TypeError, match="integer vector"):
        phx.discretization.CellPartition(np.asarray([0.2, 1.0]), 2)
    with pytest.raises(ValueError, match="every partition|Every partition"):
        phx.discretization.CellPartition(np.asarray([0, 0]), 2)
    with pytest.raises(ValueError, match="adjacency reach"):
        MeshDistribution(
            part,
            phx.discretization.CellPartition(np.asarray([0, 1]), 2),
            # ty: ignore[invalid-argument-type]
            halo_global_ids=([], []),
        )
    with pytest.raises(ValueError, match="locally owned"):
        MeshDistribution(
            part,
            phx.discretization.CellPartition(np.asarray([0, 1]), 2),
            # ty: ignore[invalid-argument-type]
            halo_global_ids=([0, 1], [0]),
        )
    grid = _grid_part()
    with pytest.raises(ValueError, match="reproduce"):
        MeshDistribution(
            grid,
            phx.discretization.CellPartition(np.asarray([0, 1] * 4), 2),
            split_factors=(2,),
        )


def test_distribution_scenario_2() -> None:
    part = _grid_part()
    distribution = MeshDistribution.cartesian(part, (2,))
    np.testing.assert_array_equal(distribution.halo_global_ids[0], [4, 7])
    np.testing.assert_array_equal(distribution.halo_global_ids[1], [0, 3])
    bounded = _grid_part(periodic=False)
    bounded_distribution = MeshDistribution.cartesian(bounded, (2,))
    np.testing.assert_array_equal(bounded_distribution.halo_global_ids[0], [4])
    bounded_plan = bounded_distribution.lower_finite_volume(
        bounded, phx.discretization.FiniteVolumePlan(bounded.carrier).prepare()
    )
    assert not bounded_plan.periodic[0]
    compiled = _compiled_fv(part)
    serial_distribution = MeshDistribution.cartesian(part, (1,))
    plan = serial_distribution.lower_finite_volume(part, compiled.discretization)
    runtime = plan.prepare((jax.devices()[0],))
    values = jnp.sin(2 * jnp.pi * part.carrier.structured_axes[0].interval_centers)[
        :, None
    ]
    np.testing.assert_allclose(
        runtime.residual(compiled.dynamics, 0.0, runtime.shard_state(values)),
        compiled(0.0, values),
        rtol=1e-12,
        atol=1e-12,
    )
    changed = _grid_part(scale=2.0)
    with pytest.raises(ValueError, match="stale"):
        serial_distribution.lower_finite_volume(changed, compiled.discretization)
    with pytest.raises(ValueError, match="grid revision"):
        runtime.compile_residual(_compiled_fv(changed).dynamics, 0.0)
    for kind in [
        phx.meshing.MeshPartitionKind.MORTON,
        phx.meshing.MeshPartitionKind.HILBERT,
    ]:
        part, _ = _quad_mesh(12, 10)
        weight_by_id = np.random.default_rng(7).uniform(0.5, 3.0, size=240)
        native = np.asarray(part.carrier.mesh.blocks[0].global_ids)
        weights = weight_by_id[native]
        policy = phx.meshing.MeshPartitionPolicy(kind, 5)
        distribution = phx.meshing.prepare_mesh_distribution(
            part, policy=policy, cell_weights=weights
        )
        part_weights = np.bincount(
            np.asarray(distribution.partition.cell_owner), weights=weights, minlength=5
        )
        np.testing.assert_allclose(distribution.evidence.part_weights, part_weights)
        assert np.all(part_weights > 0)
        assert (
            float(distribution.evidence.imbalance)
            <= 1 + 5 * weights.max() / weights.sum()
        )
        again = phx.meshing.prepare_mesh_distribution(
            part, policy=policy, cell_weights=weights
        )
        assert again.distribution_id == distribution.distribution_id
        block = part.carrier.mesh.blocks[0]
        order = np.random.default_rng(3).permutation(240)
        shuffled = MeshPart(
            "shuffled",
            phx.meshing.certify_cell_mesh(
                phx.discretization.CellMesh.from_triangles(
                    np.asarray(part.carrier.mesh.coordinates),
                    np.asarray(block.vertices)[order],
                    cell_global_ids=native[order],
                ),
                phx.SpatialCoordinateContract.si(),
            ),
        )
        # ty: ignore[unresolved-attribute]
        shuffled_native = np.asarray(shuffled.carrier.mesh.blocks[0].global_ids)
        permuted = phx.meshing.prepare_mesh_distribution(
            shuffled, policy=policy, cell_weights=weight_by_id[shuffled_native]
        )
        owner_by_id = np.empty((240,), dtype=np.int32)
        owner_by_id[native] = np.asarray(distribution.partition.cell_owner)
        np.testing.assert_array_equal(
            permuted.partition.cell_owner, owner_by_id[shuffled_native]
        )
    for width in [0, 1, 2, 3]:
        part, _ = _quad_mesh(9, 7)
        distribution = phx.meshing.prepare_mesh_distribution(
            part,
            policy=phx.meshing.MeshPartitionPolicy(
                phx.meshing.MeshPartitionKind.MORTON, 3, halo_width=width
            ),
        )
        owner = np.asarray(distribution.partition.cell_owner)
        expected = _brute_force_halos(part, owner, 3, width)
        native = np.asarray(distribution.cell_global_ids)
        for rank in range(3):
            np.testing.assert_array_equal(
                distribution.halo_global_ids[rank], expected[rank]
            )
            np.testing.assert_array_equal(
                native[np.asarray(distribution.halo_rows[rank])], expected[rank]
            )
            owned = native[np.asarray(distribution.owned_rows[rank])]
            np.testing.assert_array_equal(owned, np.sort(native[owner == rank]))
            np.testing.assert_array_equal(
                np.asarray(distribution.dependencies[rank]),
                np.isin(np.arange(3), owner[np.isin(native, expected[rank])]),
            )
        assert int(distribution.evidence.halo_replicas) == sum(map(len, expected))


def _quad_mesh(nx: Any, ny: Any, *, refine: Any = (), extend: Any = 0) -> Any:
    """Two triangles per unit quad; ``refine`` quads split into four about their
    center (children of the two original triangles) and ``extend`` columns are
    appended as created cells. Returns the part and each cell's source parent ID."""
    width = nx + extend
    xs, ys = np.meshgrid(np.arange(width + 1.0), np.arange(ny + 1.0), indexing="ij")
    points = [np.stack((xs.ravel(), ys.ravel()), axis=1)]
    vertex = np.arange((width + 1) * (ny + 1)).reshape(width + 1, ny + 1)
    next_vertex, next_id = vertex.size, 2 * nx * ny
    cells, ids, parents = [], [], []
    for i in range(width):
        for j in range(ny):
            a, b, c, d = (
                vertex[i, j],
                vertex[i + 1, j],
                vertex[i + 1, j + 1],
                vertex[i, j + 1],
            )
            quad = i * ny + j
            if i >= nx:
                cells += [(a, b, c), (a, c, d)]
                ids += [next_id, next_id + 1]
                parents += [-1, -1]
                next_id += 2
            elif quad in refine:
                center = next_vertex
                next_vertex += 1
                points.append(np.asarray([[i + 0.5, j + 0.5]]))
                cells += [(a, b, center), (b, c, center), (c, d, center), (d, a, center)]
                ids += [next_id + offset for offset in range(4)]
                parents += [2 * quad, 2 * quad, 2 * quad + 1, 2 * quad + 1]
                next_id += 4
            else:
                cells += [(a, b, c), (a, c, d)]
                ids += [2 * quad, 2 * quad + 1]
                parents += [2 * quad, 2 * quad + 1]
    mesh = phx.discretization.CellMesh.from_triangles(
        np.concatenate(points),
        np.asarray(cells, dtype=np.int32),
        cell_global_ids=np.asarray(ids, dtype=np.int64),
    )
    part = MeshPart(
        "adaptive",
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si()),
    )
    by_id = dict(zip(ids, parents, strict=True))
    # ty: ignore[unresolved-attribute]
    native = np.asarray(part.carrier.mesh.blocks[0].global_ids)
    return part, np.asarray([by_id[int(value)] for value in native])


def _cell_lineage(source: Any, target: Any, parents: Any) -> Any:
    target_ids = np.asarray(target.carrier.mesh.blocks[0].global_ids)
    routed = parents >= 0
    kinds = np.where(
        parents[routed] == target_ids[routed],
        int(phx.meshing.EntityLineageKind.PRESERVED),
        int(phx.meshing.EntityLineageKind.REFINED_FROM),
    )
    cells = phx.meshing.EntityLineage(
        2,
        source.carrier.mesh.entity_set(2).entity_set_id,
        target.carrier.mesh.entity_set(2).entity_set_id,
        parents[routed],
        target_ids[routed],
        kinds,
        created_target_ids=target_ids[~routed],
    )
    return phx.meshing.MeshLineage(
        source.carrier.mesh.topology_id, target.carrier.mesh.topology_id, (cells,)
    )


def _brute_force_halos(part: Any, owner: Any, part_count: Any, width: Any) -> Any:
    block = part.carrier.mesh.blocks[0]
    triangles = np.asarray(block.vertices)
    global_ids = np.asarray(block.global_ids)
    by_edge = {}
    for cell, triangle in enumerate(triangles.tolist()):
        for corner in range(3):
            edge = tuple(sorted((triangle[corner], triangle[(corner + 1) % 3])))
            by_edge.setdefault(edge, []).append(cell)
    neighbors = [set() for _ in triangles]
    for sharing in by_edge.values():
        for first in sharing:
            neighbors[first].update(cell for cell in sharing if cell != first)
    halos = []
    for rank in range(part_count):
        visited = {cell for cell in range(len(triangles)) if owner[cell] == rank}
        frontier = set(visited)
        for _ in range(width):
            frontier = set().union(*(neighbors[cell] for cell in frontier)) - visited
            visited |= frontier
        halos.append(
            sorted(int(global_ids[cell]) for cell in visited if owner[cell] != rank)
        )
    return halos


def test_provider_and_native_graph_routes_take_explicit_ownership_or_fail_closed(
    monkeypatch: Any,
) -> None:
    part, _ = _quad_mesh(4, 4)
    provider = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.PROVIDER, 2)
    owners = np.arange(32) % 2
    distribution = phx.meshing.prepare_mesh_distribution(
        part, policy=provider, ownership=owners
    )
    np.testing.assert_array_equal(distribution.partition.cell_owner, owners)
    assert distribution.evidence.kind is phx.meshing.MeshPartitionKind.PROVIDER
    with pytest.raises(ValueError, match="requires ownership"):
        phx.meshing.prepare_mesh_distribution(part, policy=provider)
    graph = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.GRAPH, 2)
    with pytest.raises(ValueError, match="PROVIDER"):
        phx.meshing.prepare_mesh_distribution(part, policy=graph, ownership=owners)
    with monkeypatch.context() as patched:
        patched.setenv("PHYDRAX_METIS_LIBRARY", "/nonexistent/libmetis.dylib")
        native = phx.meshing.prepare_mesh_distribution(part, policy=graph)
    assert native.evidence.provenance.startswith("phydrax-meshcore ")


@pytest.mark.parametrize("path", (None, "", " \t "))
def test_absent_or_blank_metis_path_uses_platform_fallback(
    monkeypatch: pytest.MonkeyPatch, path: str | None
) -> None:
    if path is None:
        monkeypatch.delenv("PHYDRAX_METIS_LIBRARY", raising=False)
    else:
        monkeypatch.setenv("PHYDRAX_METIS_LIBRARY", path)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    part, _ = _quad_mesh(4, 4)
    policy = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.METIS, 2)
    with pytest.raises(
        phx.meshing.MetisUnavailableError,
        match="explicit METIS comparison requires a METIS library",
    ):
        phx.meshing.prepare_mesh_distribution(part, policy=policy)


def test_configured_missing_metis_path_refuses_platform_fallback(tmp_path: Path) -> None:
    phx.config.configure(
        phx.config.ProviderConfig(
            library_paths={"metis": str(tmp_path / "missing-metis")}
        )
    )
    part, _ = _quad_mesh(4, 4)
    policy = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.METIS, 2)
    with pytest.raises(
        phx.meshing.MetisUnavailableError, match="PHYDRAX_METIS_LIBRARY names a missing"
    ):
        phx.meshing.prepare_mesh_distribution(part, policy=policy)


def test_graph_route_partitions_with_native_meshcore_deterministically() -> None:
    part, _ = _quad_mesh(10, 10)
    policy = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.GRAPH, 4, maximum_imbalance=1.1
    )
    first = phx.meshing.prepare_mesh_distribution(part, policy=policy)
    second = phx.meshing.prepare_mesh_distribution(part, policy=policy)
    np.testing.assert_array_equal(first.partition.cell_owner, second.partition.cell_owner)
    assert np.unique(np.asarray(first.partition.cell_owner)).size == 4
    assert first.evidence.provenance.startswith("phydrax-meshcore ")
    assert first.evidence.feasible


def test_owner_local_geometry_selector_compacts_cells_and_coordinate_routes() -> None:
    part, _ = _quad_mesh(12, 12)
    policy = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.MORTON, 4, halo_width=1
    )
    distribution = phx.meshing.prepare_mesh_distribution(part, policy=policy)
    local = distribution.local_cell_geometry(0)
    carrier = part.carrier
    assert np.all(np.diff(np.asarray(local["cell_ids"])) > 0)
    assert local["coordinates"].shape[0] < carrier.geometry.coordinates.shape[0]
    for block in local["blocks"]:
        routes = np.asarray(block["coordinate_routes"])
        assert np.all((routes < 0) | (routes < local["coordinates"].shape[0]))


def test_metis_route_remains_an_explicit_optional_comparison() -> None:
    part, _ = _quad_mesh(10, 10)
    policy = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.METIS, 4, maximum_imbalance=1.1
    )
    try:
        first = phx.meshing.prepare_mesh_distribution(part, policy=policy)
    except phx.meshing.MetisUnavailableError:
        pytest.skip("METIS is not installed.")
    second = phx.meshing.prepare_mesh_distribution(part, policy=policy)
    np.testing.assert_array_equal(first.partition.cell_owner, second.partition.cell_owner)
    assert np.unique(np.asarray(first.partition.cell_owner)).size == 4
    assert first.evidence.provenance.startswith("metis:")


def _check_migration(transition: Any, parents: Any, source_values: Any) -> None:
    source_native = np.asarray(transition.source.cell_global_ids)
    target_native = np.asarray(transition.target.cell_global_ids)
    moved = transition.transfer(source_values)
    defined = parents >= 0
    np.testing.assert_array_equal(transition.target_defined, defined)
    lookup = {int(gid): row for row, gid in enumerate(source_native)}
    expected = np.asarray(source_values)[
        [lookup[int(value)] for value in parents[defined]]
    ]
    np.testing.assert_allclose(np.asarray(moved)[defined], expected)
    np.testing.assert_allclose(np.asarray(moved)[~defined], 0.0)
    parts = transition.source.partition.part_count
    sent = np.concatenate([np.asarray(transition.sent_by(rank)) for rank in range(parts)])
    received = np.concatenate(
        [np.asarray(transition.received_by(rank)) for rank in range(parts)]
    )
    slots = np.arange(transition.send.capacity)
    np.testing.assert_array_equal(np.sort(sent), slots)
    np.testing.assert_array_equal(np.sort(received), slots)
    np.testing.assert_array_equal(
        np.sort(np.asarray(transition.receive.target_indices)),
        np.sort(np.flatnonzero(defined)),
    )
    target_owner = np.asarray(transition.target.partition.cell_owner)
    for rank in range(parts):
        rows = np.asarray(transition.receive.target_indices)[
            np.asarray(transition.received_by(rank))
        ]
        assert np.all(target_owner[rows] == rank)
    assert np.array_equal(
        target_native,
        np.asarray(transition.target.part.carrier.mesh.blocks[0].global_ids),
    )


def test_transition_contracts() -> None:
    source, _ = _quad_mesh(8, 8)
    policy = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.HILBERT, 4, maximum_imbalance=1.3
    )
    distribution = phx.meshing.prepare_mesh_distribution(source, policy=policy)
    target, parents = _quad_mesh(8, 8, refine=(0, 1, 63), extend=1)
    transition = phx.meshing.prepare_distribution_transition(
        distribution, target, _cell_lineage(source, target, parents), policy=policy
    )
    assert not transition.rebalanced
    source_owner = dict(
        zip(
            np.asarray(distribution.cell_global_ids).tolist(),
            np.asarray(distribution.partition.cell_owner).tolist(),
            strict=True,
        )
    )
    target_owner = np.asarray(transition.target.partition.cell_owner)
    defined = parents >= 0
    np.testing.assert_array_equal(
        target_owner[defined], [source_owner[int(value)] for value in parents[defined]]
    )
    assert np.all(target_owner[~defined] >= 0)
    assert int(transition.migrated_cells) == 0
    np.testing.assert_array_equal(
        np.asarray(transition.migration_counts),
        np.diag(np.diag(np.asarray(transition.migration_counts))),
    )
    assert float(transition.target.evidence.imbalance) <= 1.3
    _check_migration(transition, parents, jnp.asarray(distribution.cell_global_ids * 1.5))
    expected_halos = _brute_force_halos(target, target_owner, 4, 1)
    for rank in range(4):
        np.testing.assert_array_equal(
            transition.target.halo_global_ids[rank], expected_halos[rank]
        )
    phases = transition.target.lower_finite_element(target, _fe(target))
    field = jnp.linspace(0.0, 1.0, parents.size)
    np.testing.assert_allclose(
        sum(phases.local_contribution(rank, field) for rank in range(4)), jnp.sum(field)
    )
    with pytest.raises(ValueError, match="stale"):
        transition.source.lower_finite_element(target, _fe(target))
    source, _ = _quad_mesh(8, 8)
    policy = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.HILBERT, 4)
    distribution = phx.meshing.prepare_mesh_distribution(source, policy=policy)
    heavy = tuple(
        quad
        for quad in range(64)
        if np.asarray(distribution.partition.cell_owner)[
            np.flatnonzero(np.asarray(distribution.cell_global_ids) == 2 * quad)[0]
        ]
        == 0
    )
    target, parents = _quad_mesh(8, 8, refine=heavy)
    lineage = _cell_lineage(source, target, parents)
    transition = phx.meshing.prepare_distribution_transition(
        distribution, target, lineage, policy=policy
    )
    assert transition.rebalanced
    assert float(transition.target.evidence.imbalance) <= 1 + 4 / parents.size
    counts = np.asarray(transition.migration_counts)
    assert np.sum(counts) == parents.size
    assert np.sum(counts) - np.trace(counts) > 0
    assert int(transition.migrated_cells) < parents.size // 2
    np.testing.assert_allclose(
        transition.migration_volume, float(transition.migrated_cells)
    )
    values = jnp.stack(
        (distribution.cell_global_ids * 2.0, -distribution.cell_global_ids * 1.0), axis=1
    )
    _check_migration(transition, parents, values)
    with pytest.raises(ValueError, match="endpoints"):
        phx.meshing.prepare_distribution_transition(
            distribution, source, lineage, policy=policy
        )
    with pytest.raises(ValueError, match="PROVIDER"):
        phx.meshing.prepare_distribution_transition(
            distribution, target, lineage, policy=policy, ownership=np.zeros(parents.size)
        )


def _cell_state(distribution: Any, values: Any) -> Any:
    return phx.lifecycle.CompositionEntry(
        jnp.asarray(values),
        entry_id="solid/mass",
        role="physical-state",
        owner_id="solid",
        structure_id=distribution.distribution_id,
        revision_id=f"mass-{distribution.distribution_id}",
        semantics_id="solid.cell-mass",
    )


def _repartition() -> Any:
    part, parents = _quad_mesh(4, 4)
    policy = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.PROVIDER, 2)
    cells = np.arange(parents.size)
    distribution = phx.meshing.prepare_mesh_distribution(
        part, policy=policy, ownership=cells % 2
    )
    return phx.meshing.prepare_distribution_transition(
        distribution,
        part,
        _cell_lineage(part, part, parents),
        policy=policy,
        ownership=cells // (parents.size // 2),
    )


def test_migration_transport_publishes_a_conserving_repartition() -> None:
    transition = _repartition()
    assert int(transition.migrated_cells) > 0
    ids = np.asarray(transition.source.cell_global_ids, dtype=np.float64)
    mass = np.stack((1.5 * ids + 1.0, np.sin(ids)), axis=1)
    source = _cell_state(transition.source, mass)
    composition = phx.lifecycle.Composition((source,), boundary_id="window-3")
    # A repartition renumbers nothing, so every row keeps its native position.
    target = _cell_state(transition.target, mass)
    transport = transition.composition_transport(source, target)
    assert transport.kind == "ownership-migration"
    np.testing.assert_allclose(transport.source_content, mass.sum(axis=0), rtol=1e-6)
    np.testing.assert_allclose(transport.target_content, mass.sum(axis=0), rtol=1e-6)
    receipt = phx.lifecycle.commit_composition_rebind(
        phx.lifecycle.CompositionRebind(composition, transports=(transport,)),
        accepted_boundary=True,
    )
    assert receipt.published
    assert receipt.migrated == ("solid/mass",)
    assert receipt.composition.entry("solid/mass").structure_id == (
        transition.target.distribution_id
    )
    # Same content, but not the migration image: the route's evidence is refused.
    shuffled = _cell_state(transition.target, np.roll(mass, 1, axis=0))
    refused = phx.lifecycle.commit_composition_rebind(
        phx.lifecycle.CompositionRebind(
            composition,
            transports=(transition.composition_transport(source, shuffled),),
        ),
        accepted_boundary=True,
    )
    assert not refused.published
    assert refused.composition is composition


def test_migration_transport_refuses_created_and_refined_rows() -> None:
    source_part, _ = _quad_mesh(4, 4)
    policy = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.HILBERT, 2)
    distribution = phx.meshing.prepare_mesh_distribution(source_part, policy=policy)
    mass = jnp.ones((distribution.cell_global_ids.size,))
    for options, message in (
        ({"extend": 1}, "Created target rows"),
        ({"refine": (0,)}, "deleted or refined rows"),
    ):
        target_part, parents = _quad_mesh(4, 4, **options)
        transition = phx.meshing.prepare_distribution_transition(
            distribution,
            target_part,
            _cell_lineage(source_part, target_part, parents),
            policy=policy,
        )
        staged = _cell_state(
            transition.target, transition.transfer(mass, reduction="sum")
        )
        with pytest.raises(ValueError, match=message):
            transition.composition_transport(_cell_state(distribution, mass), staged)


def test_migration_transport_refuses_entries_of_other_distributions() -> None:
    transition = _repartition()
    mass = jnp.ones((transition.source.cell_global_ids.size,))
    with pytest.raises(ValueError, match="do not live on this transition"):
        transition.composition_transport(
            _cell_state(transition.target, mass), _cell_state(transition.target, mass)
        )


def test_sharded_owner_directory_witness_resolves_owned_ids_and_refuses_unknown() -> None:
    if jax.process_count() != 1:
        pytest.skip("This ownership-rebind contract uses one local process group.")
    process = jax.process_index()
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "single-owner-directory-contract",
            (process,),
            ((process, device.id),),
            mesh_axes=(("owners", 1),),
        ),
        (device,),
    )
    source_authority = CompositionEntry(
        None,
        entry_id="periodic-source",
        role="topology",
        owner_id="phydrax.test.periodic",
        structure_id="source-topology",
        revision_id="source-revision",
        semantics_id="source-meaning",
    )
    part_authority = CompositionEntry(
        None,
        entry_id="periodic-part",
        role="topology",
        owner_id="phydrax.test.periodic",
        structure_id="periodic-topology",
        revision_id="periodic-part-revision",
        semantics_id="periodic-geometry",
        dependencies=(source_authority.binding("structure"),),
    )
    witness = ShardedOwnerDirectoryWitness(
        group,
        0,
        "single-owner-source-manifest",
        "single-owner-directory",
        np.asarray((1, 2), dtype=np.int64),
        np.asarray((0, 0), dtype=np.int32),
        np.asarray((True, True), dtype=np.bool_),
        np.asarray((10,), dtype=np.int64),
        np.asarray((0,), dtype=np.int32),
        np.asarray((20, 21, 22), dtype=np.int64),
        np.asarray((0, 0, 0), dtype=np.int32),
    )
    bound = witness.bind_source_authority(
        part_authority.record_id,
        group,
        axis_name="single-owner-directory-binding",
    )
    assert bound.source_authority_id == part_authority.record_id
    assert witness.source_authority_id is None
    assert bound.directory_id == witness.directory_id
    assert bound.authority_id != witness.authority_id
    assert witness.rank == 0 and witness.part_count == 1
    np.testing.assert_array_equal(
        bound.owner_ranks_for_ids(
            np.asarray((10,), dtype=np.int64), directory_kind="mesh-cells"
        ),
        np.asarray((0,), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="Owner lookup ID is outside"):
        bound.owner_ranks_for_ids(
            np.asarray((999,), dtype=np.int64), directory_kind="mesh-cells"
        )
    with pytest.raises(
        ValueError,
        match="Authority rebinding changed the accepted group, shard, or source binding",
    ):
        bound.bind_source_authority("different-periodic-part-authority", group)


def test_owner_local_identity_is_order_independent_and_content_sensitive() -> None:
    if jax.process_count() != 1:
        pytest.skip("This identity contract uses one local process group.")
    process = jax.process_index()
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "single-owner-logical-identity",
            (process,),
            ((process, device.id),),
            mesh_axes=(("identity", 1),),
        ),
        (device,),
    )
    header = {"periodic_cell": "unit-cell", "dimension": 2}
    records = (
        (91, {"exact_point": ((0, 1), (1, 2)), "weight": (3, 5)}),
        (4, {"exact_point": ((1, 1), (0, 1)), "weight": (0, 1)}),
    )
    identity = _owner_local_multiset_identity(
        "periodic-source-content",
        header,
        records,
        "source",
        "revision",
        group,
        "identity_ordered",
    )
    assert (
        _owner_local_multiset_identity(
            "periodic-source-content",
            header,
            tuple(reversed(records)),
            "source",
            "revision",
            group,
            "identity_reordered",
        )
        == identity
    )
    assert (
        _owner_local_multiset_identity(
            "periodic-source-content",
            header,
            (
                records[0],
                (
                    records[1][0],
                    {"exact_point": ((1, 1), (1, 1)), "weight": (0, 1)},
                ),
            ),
            "source",
            "revision",
            group,
            "identity_changed_coordinate",
        )
        != identity
    )
    assert (
        _owner_local_multiset_identity(
            "periodic-source-content",
            header,
            ((92, records[0][1]), records[1]),
            "source",
            "revision",
            group,
            "identity_changed_key",
        )
        != identity
    )
    assert (
        _owner_local_multiset_identity(
            "periodic-source-content",
            header,
            (records[0],),
            "source",
            "revision",
            group,
            "identity_dropped_record",
        )
        != identity
    )
    with pytest.raises(ValueError, match="duplicate entity keys"):
        _owner_local_multiset_identity(
            "periodic-source-content",
            header,
            (records[0], records[0]),
            "source",
            "revision",
            group,
            "identity_duplicate_key",
        )


def test_owner_local_periodic_source_entry_is_stable_under_coordinate_motion() -> None:
    if jax.process_count() != 1:
        pytest.skip("This source identity contract uses one local process group.")
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.meshing._periodic import PeriodicPointOrbits
    from phydrax.meshing.providers._native_periodic import (
        _owner_local_periodic_source_authority,
        NativePeriodicSource,
    )

    process = jax.process_index()
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "single-owner-periodic-source-motion",
            (process,),
            ((process, device.id),),
            mesh_axes=(("source_identity", 1),),
        ),
        (device,),
    )
    cell = PeriodicCell(np.eye(2, dtype=np.float64))
    point_ids = np.asarray((11, 29), dtype=np.int64)
    weights = np.asarray((0.125, -0.25), dtype=np.float64)
    owners = np.zeros(point_ids.shape, dtype=np.int32)
    source_values = (
        np.asarray(((0.0, 0.25), (0.5, 0.75)), dtype=np.float64),
        np.asarray(((0.125, 0.25), (0.5, 0.75)), dtype=np.float64),
        np.asarray(((-0.0, 0.25), (0.5, 0.75)), dtype=np.float64),
    )
    authorities = tuple(
        _owner_local_periodic_source_authority(
            NativePeriodicSource(
                PeriodicPointOrbits(cell, points, point_ids),
                "periodic-motion-source",
                "periodic-motion-revision",
                site_weights=weights,
                point_owner_ranks=owners,
            ),
            group,
            f"source_identity_{index}",
        )
        for index, points in enumerate(source_values)
    )
    entries = tuple(authority[0].record_id for authority in authorities)
    content = tuple(authority[2] for authority in authorities)
    bindings = tuple(authority[0].value.binding_id for authority in authorities)
    assert entries[0] == entries[1] == entries[2]
    assert authorities[0][1].composition_id == authorities[1][1].composition_id
    assert authorities[0][1].composition_id == authorities[2][1].composition_id
    assert content[0] != content[1]
    assert content[0] != content[2]
    assert bindings[0] != bindings[1]
    assert bindings[0] != bindings[2]


def test_owner_local_graph_migration_evidence_restore_refuses_forged_identity(
    tmp_path: Path,
) -> None:
    import copy

    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax.meshing._model_codec import (
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        write_model_artifact,
    )

    if jax.process_count() != 1:
        pytest.skip("This migration evidence contract uses one local process group.")

    process = jax.process_index()
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "single-owner-graph-evidence-restore",
            (process,),
            ((process, device.id),),
            mesh_axes=(("graph_evidence", 1),),
        ),
        (device,),
    )
    cell_ids = np.asarray((10, 20), dtype=np.int64)
    graph = DistributedWeightedCSRGraph(
        np.asarray((0, 0, 0), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        entity_ids=cell_ids,
        owner_ranks=np.zeros((2,), dtype=np.int32),
        neighbor_owner_ranks=np.empty((0,), dtype=np.int32),
        vertex_weights=np.ones((2,), dtype=np.int64),
    )
    result = partition_graph_distributed(
        graph,
        GraphPartitionPlan(1, maximum_imbalance=1.0),
        group,
        axis_name="graph_evidence_partition",
    )
    evidence = OwnerLocalGraphMigrationEvidence(
        result,
        "graph-evidence-candidate",
        "source-part",
        "periodic-face-coverage",
        "source-directory",
        "target-part",
        "target-directory",
        group.spec.group_id,
        0,
        1,
        cell_ids,
    )
    path = tmp_path / "graph-migration-evidence.phx"
    write_model_artifact(path, evidence, limits=MODEL_ARCHIVE_LIMITS)
    restored = read_model_artifact(
        path,
        expected_type=OwnerLocalGraphMigrationEvidence,
        restore=OwnerLocalGraphMigrationEvidence.from_stored_fields,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert restored.evidence_id == evidence.evidence_id

    forged = copy.copy(evidence)
    object.__setattr__(forged, "evidence_id", "0" * 64)
    write_model_artifact(path, forged, limits=MODEL_ARCHIVE_LIMITS)
    with pytest.raises(ArrayArchiveCorruptionError) as refusal:
        read_model_artifact(
            path,
            expected_type=OwnerLocalGraphMigrationEvidence,
            restore=OwnerLocalGraphMigrationEvidence.from_stored_fields,
            limits=MODEL_ARCHIVE_LIMITS,
        )
    assert "differs from its recomputed identity" in str(refusal.value.__cause__)


def test_owner_local_graph_keyed_state_transfer_preserves_exact_rows() -> None:
    if jax.process_count() != 1:
        pytest.skip("This keyed-row transfer contract uses one local process group.")

    process = jax.process_index()
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "single-owner-graph-state-transfer",
            (process,),
            ((process, device.id),),
            mesh_axes=(("graph_state", 1),),
        ),
        (device,),
    )
    cell_ids = np.asarray((10, 20), dtype=np.int64)
    graph = DistributedWeightedCSRGraph(
        np.asarray((0, 0, 0), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        entity_ids=cell_ids,
        owner_ranks=np.zeros((2,), dtype=np.int32),
        neighbor_owner_ranks=np.empty((0,), dtype=np.int32),
        vertex_weights=np.ones((2,), dtype=np.int64),
    )
    result = partition_graph_distributed(
        graph,
        GraphPartitionPlan(1, maximum_imbalance=1.0),
        group,
        axis_name="graph_state_partition",
    )
    evidence = OwnerLocalGraphMigrationEvidence(
        result,
        "graph-state-candidate",
        "source-part",
        "periodic-face-coverage",
        "source-directory",
        "target-part",
        "target-directory",
        group.spec.group_id,
        0,
        1,
        cell_ids,
    )
    source_keys = np.asarray(((3, 10, 0), (3, 20, 0)), dtype=np.int64)
    target_keys = source_keys[::-1].copy()
    owners = np.zeros((2,), dtype=np.int32)
    values = np.asarray(((1.25, -0.0), (3.5, 4.0)), dtype=np.float64)
    source = CompositionEntry(
        values,
        entry_id="cell-state",
        role="physical-state",
        owner_id="test-solver",
        structure_id="source-part",
        revision_id="state-revision-1",
        semantics_id="velocity-field",
    )

    transfer = evidence.transfer_keyed_values(
        source,
        source_keys,
        owners,
        values,
        target_keys,
        owners,
        group,
        target_structure_id="target-part",
        key_identity_id=canonical_fingerprint(
            {"kind": "periodic-cell-state-keys", "entity_dimension": 3}
        ),
        axis_name="graph_state_transfer",
    )
    np.testing.assert_array_equal(np.asarray(transfer.target_entity_keys), target_keys)
    np.testing.assert_array_equal(
        np.asarray(transfer.target_values).view(np.uint8),
        values[::-1].copy().view(np.uint8),
    )
    assert transfer.source_content_root == transfer.target_content_root
    target = CompositionEntry(
        transfer.target_values,
        entry_id=source.entry_id,
        role=source.role,
        owner_id=source.owner_id,
        structure_id="target-part",
        revision_id=source.revision_id,
        semantics_id=source.semantics_id,
    )
    transport = transfer.composition_transport(source, target)
    assert bool(np.asarray(transport.successful))
    assert transport.source_content is None
    assert transport.target_content is None
    assert transport.content_tolerance is None
    assert transport.source_content_root == transfer.source_content_root
    assert transport.target_content_root == transfer.target_content_root

    local_keys = target_keys[[0, 1, 0]]
    local_values = transfer.gather_target_local_values(
        local_keys,
        group,
        axis_name="graph_state_local_rows",
    )
    np.testing.assert_array_equal(
        np.asarray(local_values).view(np.uint8),
        values[[1, 0, 1]].copy().view(np.uint8),
    )
    empty_local_values = transfer.gather_target_local_values(
        np.empty((0, 3), dtype=np.int64),
        group,
        axis_name="graph_state_empty_local_rows",
    )
    assert empty_local_values.shape == (0, 2)

    key_data = np.asarray(((1, 2), (3, 4)), dtype=np.uint32)
    rng_values = jax.random.wrap_key_data(jnp.asarray(key_data), impl="threefry2x32")
    rng_source = CompositionEntry(
        rng_values,
        entry_id="cell-rng-state",
        role="physical-state",
        owner_id="test-solver",
        structure_id="source-part",
        revision_id="rng-revision-1",
        semantics_id="cell-rng-keys",
    )
    rng_transfer = evidence.transfer_keyed_values(
        rng_source,
        source_keys,
        owners,
        rng_values,
        target_keys,
        owners,
        group,
        target_structure_id="target-part",
        key_identity_id=canonical_fingerprint(
            {"kind": "periodic-cell-rng-keys", "entity_dimension": 3}
        ),
        axis_name="graph_state_rng_transfer",
    )
    assert jax.dtypes.issubdtype(rng_transfer.target_values.dtype, jax.dtypes.prng_key)
    assert str(jax.random.key_impl(rng_transfer.target_values)) == "threefry2x32"
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(rng_transfer.target_values)),
        key_data[::-1],
    )
    rng_target = CompositionEntry(
        rng_transfer.target_values,
        entry_id=rng_source.entry_id,
        role=rng_source.role,
        owner_id=rng_source.owner_id,
        structure_id="target-part",
        revision_id=rng_source.revision_id,
        semantics_id=rng_source.semantics_id,
    )
    rng_transport = rng_transfer.composition_transport(rng_source, rng_target)
    assert bool(np.asarray(rng_transport.successful))
    assert rng_transport.source_content_root == rng_transport.target_content_root
    local_rng_values = rng_transfer.gather_target_local_values(
        local_keys,
        group,
        axis_name="graph_state_rng_local_rows",
    )
    assert jax.dtypes.issubdtype(local_rng_values.dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(local_rng_values)),
        key_data[[1, 0, 1]],
    )
    empty_local_rng = rng_transfer.gather_target_local_values(
        np.empty((0, 3), dtype=np.int64),
        group,
        axis_name="graph_state_empty_rng_rows",
    )
    assert empty_local_rng.shape == (0,)
    assert jax.dtypes.issubdtype(empty_local_rng.dtype, jax.dtypes.prng_key)
    empty_keys = np.empty((0, 3), dtype=np.int64)
    empty_owners = np.empty((0,), dtype=np.int32)
    empty_values = np.empty((0, 2), dtype=np.float64)
    empty_source = CompositionEntry(
        empty_values,
        entry_id="empty-cell-state",
        role="physical-state",
        owner_id="test-solver",
        structure_id="source-part",
        revision_id="state-revision-1",
        semantics_id="empty-cell-field",
    )
    empty_transfer = evidence.transfer_keyed_values(
        empty_source,
        empty_keys,
        empty_owners,
        empty_values,
        empty_keys,
        empty_owners,
        group,
        target_structure_id="target-part",
        key_identity_id=canonical_fingerprint(
            {"kind": "periodic-cell-state-keys", "entity_dimension": 3}
        ),
        axis_name="graph_state_empty_transfer",
    )
    assert empty_transfer.target_values.shape == (0, 2)
    empty_target = CompositionEntry(
        empty_transfer.target_values,
        entry_id=empty_source.entry_id,
        role=empty_source.role,
        owner_id=empty_source.owner_id,
        structure_id="target-part",
        revision_id=empty_source.revision_id,
        semantics_id=empty_source.semantics_id,
    )
    empty_transport = empty_transfer.composition_transport(empty_source, empty_target)
    assert bool(np.asarray(empty_transport.successful))
    assert empty_transport.source_content_root == empty_transport.target_content_root

    tampered = np.asarray(transfer.target_values).copy()
    tampered[0, 0] = 2.0
    bad_target = CompositionEntry(
        tampered,
        entry_id=source.entry_id,
        role=source.role,
        owner_id=source.owner_id,
        structure_id="target-part",
        revision_id=source.revision_id,
        semantics_id=source.semantics_id,
    )
    with pytest.raises(ValueError, match="exact rows"):
        transfer.composition_transport(source, bad_target)


def _logical_bucket_identity_reference(
    kind: str,
    header: object,
    records: tuple[tuple[object, object], ...],
    source_id: str,
    source_revision: str,
    /,
) -> str:
    header_digest = canonical_fingerprint(
        {
            "kind": kind,
            "source_id": source_id,
            "source_revision": source_revision,
            "header": header,
        }
    )
    buckets: list[list[tuple[str, str]]] = [
        [] for _ in range(_OWNER_LOCAL_IDENTITY_BUCKET_COUNT)
    ]
    for key, record in records:
        key_digest = canonical_fingerprint(key)
        record_digest = canonical_fingerprint(
            {
                "kind": kind,
                "header": header_digest,
                "key": key,
                "record": record,
            }
        )
        buckets[bytes.fromhex(key_digest)[0]].append((key_digest, record_digest))
    logical_buckets = tuple(
        (
            bucket,
            canonical_fingerprint(
                {
                    "kind": f"{kind}-logical-bucket",
                    "bucket": bucket,
                    "records": tuple(sorted(buckets[bucket])),
                }
            ),
        )
        for bucket in range(_OWNER_LOCAL_IDENTITY_BUCKET_COUNT)
    )
    return canonical_fingerprint(
        {
            "kind": f"owner-local-{kind}",
            "source_id": source_id,
            "source_revision": source_revision,
            "header": header,
            "logical_buckets": logical_buckets,
        }
    )


def test_owner_message_channels_preserve_typed_sparse_banks() -> None:
    from phydrax.meshing._distribution import _owner_exchange_message_banks

    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    processes = tuple(device.process_index for device in devices)
    if processes != (0, 1, 2) or len(jax.local_devices()) != 1:
        pytest.skip("This exchange test requires three native process owners.")
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "owner-message-byte-channels",
            processes,
            tuple((process, device.id) for process, device in zip(processes, devices)),
            mesh_axes=(("message_test", 3),),
        ),
        devices,
    )
    rank = jax.process_index()
    for iteration, dtype in enumerate((np.int16, np.float64, np.uint8)):
        counts = ((0, 3, 17), (5, 0, 0), (0, 0, 0))

        def bank(sender: int, receiver: int) -> np.ndarray:
            count = counts[sender][receiver]
            # A strided input also exercises the exact contiguous byte packing.
            values = np.arange(count * 6, dtype=dtype).reshape(count, 6)
            if dtype is np.float64 and count:
                values[0, 0] = -0.0
                values[0, 2] = np.asarray((0x7FF8000000000042,), dtype=np.uint64).view(
                    np.float64
                )[0]
            return values[:, ::2]

        received = _owner_exchange_message_banks(
            tuple(bank(rank, receiver) for receiver in processes),
            rank,
            group,
            f"message_test_distinct_callsite_{iteration}",
        )
        assert set(received) == {sender for sender in processes if counts[sender][rank]}
        for sender, values in received.items():
            assert values.dtype == np.dtype(dtype)
            assert (
                np.ascontiguousarray(values).tobytes()
                == np.ascontiguousarray(bank(sender, rank)).tobytes()
            )


def test_owner_message_channels_in_fresh_three_process_group() -> None:
    import os
    import socket
    import sys

    from phydrax import config as pxconfig
    from phydrax._execution_launch import launch_local_processes, LocalProcessLaunchPlan

    if (
        jax.process_count() != 1
        or os.name != "posix"
        or os.environ.get("PYTEST_XDIST_WORKER") is not None
    ):
        pytest.skip(
            "The three-owner coordinator requires POSIX supervision outside a "
            "process group and xdist."
        )
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    for name in (
        "PHYDRAX_COORDINATOR_ADDRESS",
        "PHYDRAX_NUM_PROCESSES",
        "PHYDRAX_PROCESS_ID",
        "PHYDRAX_LOCAL_DEVICE_IDS",
        "PHYDRAX_COORDINATOR_BIND_ADDRESS",
    ):
        environment.pop(name, None)
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(
            enable_x64=True,
            platforms=("cpu",),
            cpu_device_count=1,
            cpu_collectives_implementation="gloo",
        ),
        base=environment,
    )
    environment.update(
        PYTHONPATH=str(repository),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    worker = (
        "import runpy, sys; "
        "runpy.run_path(sys.argv[1])"
        "['test_owner_message_channels_preserve_typed_sparse_banks']()"
    )
    results = launch_local_processes(
        (sys.executable, "-c", worker, str(Path(__file__).resolve())),
        LocalProcessLaunchPlan(3, f"127.0.0.1:{port}", devices_per_process=1),
        environment=environment,
        cwd=str(repository),
        timeout_seconds=180.0,
    )
    assert all(result.returncode == 0 for result in results), results


def test_owner_local_identity_is_process_count_invariant() -> None:
    processes = tuple(sorted({device.process_index for device in jax.devices()}))
    if len(processes) not in (1, 3):
        pytest.skip("The identity contract is qualified on one or three processes.")
    devices_by_process = {
        process: next(
            device for device in jax.devices() if device.process_index == process
        )
        for process in processes
    }
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "logical-bucket-process-invariance",
            processes,
            tuple((process, devices_by_process[process].id) for process in processes),
            mesh_axes=(("identity", len(processes)),),
        ),
        tuple(devices_by_process[process] for process in processes),
    )
    rank = processes.index(jax.process_index())
    records = (
        (91, {"exact_point": ((0, 1), (1, 2)), "weight": (3, 5)}),
        (4, {"exact_point": ((1, 1), (0, 1)), "weight": (0, 1)}),
    )
    local_records = tuple(
        record for index, record in enumerate(records) if index % len(processes) == rank
    )
    header = {"periodic_cell": "unit-cell", "dimension": 2}
    identity = _owner_local_multiset_identity(
        "periodic-source-content",
        header,
        local_records,
        "source",
        "revision",
        group,
        "logical_bucket_process_invariance",
    )
    assert identity == _logical_bucket_identity_reference(
        "periodic-source-content",
        header,
        records,
        "source",
        "revision",
    )
