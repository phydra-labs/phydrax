#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _topology() -> Any:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    plan = phx.discretization.ForestPlan(
        grid, maximum_level=3, maximum_leaf_capacity=1024
    )
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    topology = compiler.initialize(1).topology
    marks = np.zeros((topology.signature.leaf_capacity,), dtype=np.int8)
    marks[[0, 1, 6]] = 1
    return compiler, compiler.adapt(topology, marks).topology


def _values(topology: Any, seed: Any, *, components: Any = 2) -> Any:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(topology.signature.leaf_capacity, components)) + 10.0
    return jnp.where(topology.workset.leaf_valid[:, None], jnp.asarray(values), 0.0)


def test_forest_amr_distributed_scenario_1() -> None:
    for stencil in ["face", "corner"]:
        _, topology = _topology()
        weights = np.linspace(1.0, 3.0, topology.leaf_count)
        partition = phx.discretization.ForestPartitionPlan(
            4, ghost_stencil=phx.discretization.AMRBalanceStencil(stencil)
        ).prepare(topology, weights=weights)
        owners = np.asarray(partition.owners)[: topology.leaf_count]
        assert np.all(np.diff(owners) >= 0)
        assert partition.evidence.maximum_imbalance < 0.25
        assert min(partition.evidence.ghost_counts) > 0
        values = _values(topology, 1)
        np.testing.assert_array_equal(partition.unpack(partition.pack(values)), values)
        workset = topology.workset
        face_valid = np.asarray(workset.face_valid)
        face_minus = np.asarray(workset.face_minus)[face_valid]
        face_plus = np.asarray(workset.face_plus)[face_valid]
        for part in range(4):
            touching = np.flatnonzero(
                (owners[face_minus] == part) | (owners[face_plus] == part)
            )
            local = np.asarray(partition.local_face_ids[part])[
                np.asarray(partition.local_face_valid[part])
            ]
            np.testing.assert_array_equal(np.sort(local), touching)
    compiler, source = _topology()
    marks = np.zeros((source.signature.leaf_capacity,), dtype=np.int8)
    marks[[8, 9]] = 1
    marks[:4] = -1
    result = compiler.adapt(source, marks)
    assert result.evidence.coarsened_families == 1
    target = result.topology
    transition = phx.discretization.ForestFieldTransition(source, target)
    plan = phx.discretization.ForestPartitionPlan(3)
    source_partition = plan.prepare(source)
    target_partition = plan.prepare(target)
    migration = source_partition.migration_to(target_partition, transition=transition)
    values = _values(source, 4)
    migrated = target_partition.unpack(migration.migrate(source_partition.pack(values)))
    expected = transition.routes.apply(values)
    np.testing.assert_allclose(migrated, expected.values, rtol=1e-15)
    assert bool(expected.successful)
    assert migration.moved_leaves > 0
    repartitioned = plan.prepare(target, weights=np.arange(1.0, target.leaf_count + 1.0))
    permutation = target_partition.migration_to(repartitioned)
    np.testing.assert_array_equal(
        repartitioned.unpack(permutation.migrate(target_partition.pack(migrated))),
        migrated,
    )
    with pytest.raises(ValueError, match="requires a transition"):
        source_partition.migration_to(target_partition)


def test_part_local_faces_reproduce_global_face_divergence() -> None:
    _, topology = _topology()
    partition = phx.discretization.ForestPartitionPlan(3).prepare(topology)
    workset = topology.workset
    values = _values(topology, 2, components=1)[:, 0]
    minus = jnp.where(workset.face_valid, workset.face_minus, 0)
    plus = jnp.where(workset.face_valid, workset.face_plus, 0)
    jump = jnp.where(workset.face_valid, values[plus] - values[minus], 0.0)
    expected = jnp.zeros_like(values).at[minus].add(jump).at[plus].add(-jump)

    def local_divergence(
        local: Any, face_minus: Any, face_plus: Any, face_valid: Any, owned: Any
    ) -> Any:
        local_jump = jnp.where(face_valid, local[face_plus] - local[face_minus], 0.0)
        result = jnp.zeros_like(local).at[face_minus].add(local_jump)
        return jnp.where(owned, result.at[face_plus].add(-local_jump), 0.0)

    local = jax.vmap(local_divergence)(
        partition.pack_with_ghosts(values),
        partition.local_face_minus,
        partition.local_face_plus,
        partition.local_face_valid,
        partition.halo.local_owned,
    )
    np.testing.assert_allclose(partition.unpack(local), expected, atol=1e-12)


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four JAX devices")
def test_sharded_ghost_exchange_delivers_owner_values() -> None:
    for stencil in ["face", "edge", "corner"]:
        _, topology = _topology()
        partition = phx.discretization.ForestPartitionPlan(
            4, ghost_stencil=phx.discretization.AMRBalanceStencil(stencil)
        ).prepare(topology)
        values = _values(topology, 3)
        exchanged = partition.exchange_ghosts(partition.pack(values))
        np.testing.assert_array_equal(exchanged, partition.pack_with_ghosts(values))
