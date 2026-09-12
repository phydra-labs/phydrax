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


def test_variable_patch_checkpoint_roundtrip_is_canonical(tmp_path):
    topology, state = _state()
    partition = phx.discretization.VariablePatchPartitionPlan(2, ((1,),)).prepare(
        topology
    )
    plan = phx.solver.VariablePatchCheckpointPlan(
        topology,
        ("density",),
        partition=partition,
    )
    path = tmp_path / "variable-patch.vpckpt"

    written = phx.solver.write_variable_patch_checkpoint(path, plan, state)
    loaded = phx.solver.read_variable_patch_checkpoint(path, plan)

    assert written.payload_id == loaded.payload_id
    assert loaded.state.topology.epoch.epoch_id == topology.epoch.epoch_id
    assert jnp.array_equal(loaded.state.levels[0].values[0], state.levels[0].values[0])
