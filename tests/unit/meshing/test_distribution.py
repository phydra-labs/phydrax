from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.meshing._assembly import MeshPart
from phydrax.meshing._distribution import MeshDistribution


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
    with pytest.raises(ValueError, match="must own at least one cell"):
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


def test_provider_and_metis_routes_take_explicit_ownership_or_fail_closed(
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
    metis = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.METIS, 2)
    with monkeypatch.context() as patched:
        patched.setenv("PHYDRAX_METIS_LIBRARY", "/nonexistent/libmetis.dylib")
        with pytest.raises(phx.meshing.MetisUnavailableError, match="missing"):
            phx.meshing.prepare_mesh_distribution(part, policy=metis)


@pytest.mark.skipif(not meshcore_available(), reason="phydrax-meshcore unavailable")
def test_graph_route_partitions_natively_without_metis(monkeypatch: Any) -> None:
    monkeypatch.setenv("PHYDRAX_METIS_LIBRARY", "/nonexistent/libmetis.dylib")
    part, _ = _quad_mesh(12, 12)
    policy = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.GRAPH, 4, maximum_imbalance=1.1
    )
    first = phx.meshing.prepare_mesh_distribution(part, policy=policy)
    second = phx.meshing.prepare_mesh_distribution(part, policy=policy)
    np.testing.assert_array_equal(first.partition.cell_owner, second.partition.cell_owner)
    assert np.unique(np.asarray(first.partition.cell_owner)).size == 4
    assert first.evidence.kind is phx.meshing.MeshPartitionKind.GRAPH
    assert first.evidence.provenance.startswith("graph-multilevel:balanced:")
    assert float(first.evidence.imbalance) <= 1.1
    curve = phx.meshing.prepare_mesh_distribution(
        part,
        policy=phx.meshing.MeshPartitionPolicy(
            phx.meshing.MeshPartitionKind.HILBERT, 4, maximum_imbalance=1.1
        ),
    )
    assert int(first.evidence.edge_cut) <= int(curve.evidence.edge_cut)
    with pytest.raises(phx.meshing.MetisUnavailableError):
        phx.meshing.prepare_mesh_distribution(
            part,
            policy=phx.meshing.MeshPartitionPolicy(
                phx.meshing.MeshPartitionKind.METIS, 4
            ),
        )


@pytest.mark.skipif(not meshcore_available(), reason="phydrax-meshcore unavailable")
def test_graph_route_records_quantized_real_cell_weights() -> None:
    part, _ = _quad_mesh(8, 8)
    weights = np.linspace(0.5, 2.5, 128)
    distribution = phx.meshing.prepare_mesh_distribution(
        part,
        policy=phx.meshing.MeshPartitionPolicy(
            phx.meshing.MeshPartitionKind.GRAPH, 3, maximum_imbalance=1.05
        ),
        cell_weights=weights,
    )
    assert ":quantized-weights:" in distribution.evidence.provenance
    assert float(distribution.evidence.imbalance) <= 1.05 + 1e-9


def test_metis_comparison_route_partitions_deterministically() -> None:
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
    assert float(first.evidence.imbalance) <= 1.1 + 1e-12


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
