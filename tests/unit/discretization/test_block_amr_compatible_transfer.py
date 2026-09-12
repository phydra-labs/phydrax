#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

import phydrax as phx


def _complexes():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    signature = phx.discretization.PatchShapeSignature((4,), halo_width=1)
    hierarchy = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(signature, 1),)
            ),
            phx.discretization.VariablePatchLevelPlan(
                1, (phx.discretization.PatchBucketPlan(signature, 2),)
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0,), (4,)),),
    )
    compiler = phx.discretization.VariablePatchTopologyCompiler(hierarchy)
    initial = compiler.initial_topology()
    tags = ((jnp.asarray([[True, True, False, False]]),),)
    refined = compiler.compile(initial, tags).topology
    complexes = phx.discretization.VariablePatchEntityComplexPlan(
        phx.discretization.BlockHierarchyCapacityPlan(
            ((8, 8), (16, 16)),
            (8, 16),
        )
    ).prepare(refined)
    return complexes


def test_compatible_entity_transfer_uses_bounded_routes_and_distinct_reverse_maps():
    coarse, fine = _complexes()
    family = phx.discretization.CompatibleEntityTransferFamily(
        coarse,
        fine,
        2,
        (16, 16),
    )
    node = family.transfer(0)
    edge = family.transfer(1)

    assert node.evidence.prolongation_routes <= node.evidence.route_capacity
    assert edge.evidence.restriction_routes <= edge.evidence.route_capacity
    assert node.dual_pullback.operator_id != node.hilbert_adjoint.operator_id
    assert node.prolongation.target.size == fine.capacity[0]
    assert edge.restriction.target.size == coarse.capacity[1]


def test_compatible_entity_transfer_primal_and_algebraic_transpose_pair_exactly():
    coarse, fine = _complexes()
    transfer = phx.discretization.CompatibleEntityTransferFamily(
        coarse,
        fine,
        2,
        (16, 16),
    ).transfer(0)
    source = jnp.arange(coarse.capacity[0], dtype=jnp.float64)
    target = jnp.linspace(0.0, 1.0, fine.capacity[0])

    left = jnp.vdot(transfer.prolongation.mv(source), target)
    right = jnp.vdot(source, transfer.dual_pullback.mv(target))

    assert jnp.allclose(left, right)
