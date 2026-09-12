#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

import phydrax as phx


def _node_complex():
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
    return phx.discretization.VariablePatchEntityComplexPlan(
        phx.discretization.BlockHierarchyCapacityPlan(((16, 16),), (16,))
    ).prepare(topology)[0]


def test_entity_gather_scatter_is_exact_signed_transpose_at_shared_node():
    complex_ = _node_complex()
    plan = phx.discretization.VariablePatchEntityExecutionPlan(complex_, 0)
    state = phx.discretization.VariablePatchEntityFieldState(
        complex_,
        0,
        jnp.arange(16.0),
    )
    local = plan.gather(state)
    cotangents = tuple(
        jnp.full_like(value, index + 1.0) for index, value in enumerate(local)
    )
    scattered = plan.scatter(cotangents)

    left = sum(
        jnp.vdot(value, cotangent)
        for value, cotangent in zip(local, cotangents, strict=True)
    )
    right = jnp.vdot(state.values, scattered)
    assert jnp.allclose(left, right)
    assert local[0][0, 4] == local[0][1, 0]
    assert scattered[4] == 2.0


def test_entity_execution_supports_oriented_edges_without_patch_local_duplication():
    complex_ = _node_complex()
    plan = phx.discretization.VariablePatchEntityExecutionPlan(complex_, 1)
    state = phx.discretization.VariablePatchEntityFieldState(
        complex_,
        1,
        jnp.ones((16,)),
    )

    local = plan.gather(state)
    restored = plan.scatter(local)

    assert jnp.all(restored[:8] == 1.0)
    assert jnp.all(restored[8:] == 0.0)
