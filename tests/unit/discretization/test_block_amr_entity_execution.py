#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import pytest

import phydrax as phx


def _one_dimensional_plan(*, periodic: Any = False) -> Any:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    signature = phx.discretization.PatchShapeSignature((4,), halo_width=1)
    return phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(signature, 2),)
            ),
            phx.discretization.VariablePatchLevelPlan(
                1, (phx.discretization.PatchBucketPlan(signature, 4),)
            ),
        ),
        (
            phx.discretization.LogicalPatchBox(0, (0,), (4,)),
            phx.discretization.LogicalPatchBox(0, (4,), (8,)),
        ),
    )


def test_variable_patch_contracts() -> None:
    topology = phx.discretization.VariablePatchTopologyCompiler(
        _one_dimensional_plan()
    ).initial_topology()
    complex_ = phx.discretization.VariablePatchEntityComplexPlan(
        phx.discretization.BlockHierarchyCapacityPlan(
            ((16, 16), (32, 32)),
            (16, 32),
        )
    ).prepare(topology)[0]
    nodes = complex_.view(0, (), 0)

    assert nodes.global_indices[0, 4] == nodes.global_indices[1, 0]
    assert int(nodes.owned[0, 4]) + int(nodes.owned[1, 0]) == 1
    assert complex_.complex.dimension == 1
    plan = _one_dimensional_plan()
    compiler = phx.discretization.VariablePatchTopologyCompiler(plan)
    initial = compiler.initial_topology()
    refined = compiler.compile(
        initial,
        ((jnp.asarray([[False, True, False, False], [False, False, False, False]]),),),
    ).topology
    entity_plan = phx.discretization.VariablePatchEntityComplexPlan(
        phx.discretization.BlockHierarchyCapacityPlan(
            ((16, 16), (32, 32)),
            (16, 32),
        )
    )
    old = entity_plan.prepare(initial)[0]
    new = entity_plan.prepare(refined)[0]
    shared_key = ((), (4,))
    old_index = old.entity_keys[0].index(shared_key)
    new_index = new.entity_keys[0].index(shared_key)

    assert (
        old.complex.entities(0).entity_ids[old_index]
        == new.complex.entities(0).entity_ids[new_index]
    )
    assert (
        old.complex.incidences[0].relation.route_shape
        == new.complex.incidences[0].relation.route_shape
    )
    topology = phx.discretization.VariablePatchTopologyCompiler(
        _one_dimensional_plan(periodic=True)
    ).initial_topology()
    complex_ = phx.discretization.VariablePatchEntityComplexPlan(
        phx.discretization.BlockHierarchyCapacityPlan(
            ((16, 16), (32, 32)),
            (16, 32),
        )
    ).prepare(topology)[0]

    assert complex_.complex.entities(0).num_active == 8
    assert complex_.complex.entities(1).num_active == 8
    nodes = complex_.view(0, (), 0)
    assert nodes.global_indices[0, 0] == nodes.global_indices[1, 4]
    topology = phx.discretization.VariablePatchTopologyCompiler(
        _one_dimensional_plan()
    ).initial_topology()
    plan = phx.discretization.VariablePatchEntityComplexPlan(
        phx.discretization.BlockHierarchyCapacityPlan(
            ((16, 16), (32, 32)),
            (1, 32),
        )
    )

    with pytest.raises(ValueError, match="incidence route capacity"):
        plan.prepare(topology)
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((4, 4), halo_width=1)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(signature, 1),)
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0), (4, 4)),),
    )
    topology = phx.discretization.VariablePatchTopologyCompiler(plan).initial_topology()

    complex_ = phx.discretization.VariablePatchEntityComplexPlan(
        phx.discretization.BlockHierarchyCapacityPlan(
            ((32, 64, 32),),
            (80, 64),
        )
    ).prepare(topology)[0]

    assert complex_.complex.dimension == 2
    assert complex_.complex.entities(0).num_active == 25
    assert complex_.complex.entities(1).num_active == 40
    assert complex_.complex.entities(2).num_active == 16
