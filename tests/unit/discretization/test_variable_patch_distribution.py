#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

import phydrax as phx


def _state():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    signature = phx.discretization.PatchShapeSignature((4,), halo_width=1)
    hierarchy = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(signature, 2),)
            ),
        ),
        (
            phx.discretization.LogicalPatchBox(0, (0,), (4,)),
            phx.discretization.LogicalPatchBox(0, (4,), (8,)),
        ),
    )
    topology = phx.discretization.VariablePatchTopologyCompiler(
        hierarchy
    ).initial_topology()
    field = phx.discretization.VariablePatchFieldState(
        topology.levels[0],
        (jnp.arange(8.0).reshape((2, 4)),),
    )
    return topology, phx.discretization.VariablePatchHierarchyState(topology, (field,))


def test_variable_patch_partition_roundtrip_is_canonical_and_explicit():
    topology, state = _state()
    partition = phx.discretization.VariablePatchPartitionPlan(2, ((1,),)).prepare(
        topology
    )

    packed = partition.pack(state)
    restored = partition.unpack(packed)
    successor = partition.repartition_epoch(topology.epoch)

    assert jnp.array_equal(restored.levels[0].values[0], state.levels[0].values[0])
    assert successor.index == topology.epoch.index + 1
    assert successor.topology_id == topology.epoch.topology_id
    assert successor.partition_id == partition.partition_id


def test_variable_patch_partition_is_deterministic_under_positive_costs():
    topology, _ = _state()
    plan = phx.discretization.VariablePatchPartitionPlan(2, ((1,),))
    first = plan.prepare(topology, costs=((jnp.asarray([2.0, 1.0]),),))
    second = plan.prepare(topology, costs=((jnp.asarray([2.0, 1.0]),),))

    assert first.partition_id == second.partition_id
    assert first.evidence.evidence_id == second.evidence.evidence_id
