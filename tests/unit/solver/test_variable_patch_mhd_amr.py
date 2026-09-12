#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

import phydrax as phx


def test_variable_patch_reflux_curl_preserves_discrete_divergence():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2),
            phx.discretization.UniformCellAxisSpec(2),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((2, 2), halo_width=1)
    hierarchy = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(signature, 1),)
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0), (2, 2)),),
    )
    topology = phx.discretization.VariablePatchTopologyCompiler(
        hierarchy
    ).initial_topology()
    complex_ = phx.discretization.VariablePatchEntityComplexPlan(
        phx.discretization.BlockHierarchyCapacityPlan(
            ((16, 24, 8),),
            (32, 32),
        )
    ).prepare(topology)[0]
    edge_capacity = complex_.capacity[0]
    face_capacity = complex_.capacity[1]
    register = phx.solver.advanced.ElectromotiveForceRegister(
        jnp.zeros((edge_capacity,)),
        jnp.arange(edge_capacity, dtype=jnp.float64) * 0.01,
        register_id="variable-patch-emf",
    )
    plan = phx.solver.advanced.VariablePatchCochainSynchronizationPlan(complex_)

    updated, diagnostics = plan.reflux_curl(
        jnp.zeros((face_capacity,)),
        register,
    )

    assert updated.shape == (face_capacity,)
    assert jnp.allclose(diagnostics.divergence_after, diagnostics.divergence_before)
    assert diagnostics.divergence_change <= 32.0 * jnp.finfo(updated.dtype).eps
