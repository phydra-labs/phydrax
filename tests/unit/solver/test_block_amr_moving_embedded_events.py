#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

import phydrax as phx


def _geometry_plan():
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
        ),
        (phx.discretization.LogicalPatchBox(0, (0,), (4,)),),
    )
    topology = phx.discretization.VariablePatchTopologyCompiler(
        hierarchy
    ).initial_topology()
    return phx.discretization.VariablePatchGeometryPlan(
        topology,
        lambda point, time, args: point,
        "identity-1d",
    )


def _journal(epoch):
    artifacts = phx.solver.FiniteVolumeTopologyArtifacts(epoch, "initial-prepared")
    return phx.solver.FiniteVolumeTopologyEventJournal(epoch, artifacts, capacity=4)


def test_moving_embedded_boundary_commits_one_conservative_successor_epoch():
    geometry = _geometry_plan()
    source = geometry.state(0.0)
    target = geometry.state(1.0, revision=1)
    plan = phx.solver.MovingEmbeddedBoundaryEventPlan(
        lambda points, time, args: points[:, 0] - (0.2 + 0.2 * time),
        "moving-wall",
    )

    result = plan.transact(
        _journal(source.plan.topology.epoch),
        source,
        target,
        1,
        0.2,
        0.2,
    )

    assert result.committed
    assert result.epoch.index == source.plan.topology.epoch.index + 1
    assert result.journal.current_epoch_id == result.epoch.epoch_id
    assert result.evidence.budget_defect == 0.0


def test_moving_embedded_boundary_rejects_unclosed_swept_volume_budget_atomically():
    geometry = _geometry_plan()
    source = geometry.state(0.0)
    target = geometry.state(1.0, revision=1)
    journal = _journal(source.plan.topology.epoch)
    plan = phx.solver.MovingEmbeddedBoundaryEventPlan(
        lambda points, time, args: points[:, 0] - (0.2 + 0.2 * time),
        "moving-wall",
    )

    result = plan.transact(journal, source, target, 1, 0.2, 0.1)

    assert not result.committed
    assert result.epoch.epoch_id == source.plan.topology.epoch.epoch_id
    assert result.journal.current_epoch_id == source.plan.topology.epoch.epoch_id
    assert result.evidence.budget_defect > 0.0
