#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic public fixed-block AMR finite-volume workflow."""

from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp

import phydrax as phx


def hierarchy_state(topology, value):
    levels = []
    for level_plan, metadata in zip(topology.plan.levels, topology.levels, strict=True):
        values = jnp.zeros(
            (level_plan.maximum_blocks, *level_plan.block_shape, 1),
            dtype=jnp.float64,
        )
        active = metadata.active.reshape(
            (level_plan.maximum_blocks,) + (1,) * (values.ndim - 1)
        )
        values = jnp.where(active, value, values)
        levels.append(phx.discretization.BlockLevelState(level_plan, metadata, values))
    return phx.discretization.BlockHierarchyState(topology, tuple(levels))


def main() -> None:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(0, (4,), 2, halo_width=1),
            phx.discretization.BlockLevelPlan(1, (2,), 8, halo_width=1),
        ),
    )
    prepared = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()

    initial = prepared.initial_topology()
    tags = jnp.zeros((2, 4), dtype=bool).at[0, 1].set(True)
    compiled = prepared.compile_topology(initial, (tags,))
    if not compiled.status.successful:
        raise RuntimeError(compiled.status.message)

    initial_state = hierarchy_state(initial, jnp.asarray(1.0))
    transition = prepared.field_transition(
        initial,
        compiled.topology,
        "conserved_scalar",
        component_shape=(1,),
    )
    transitioned = transition.apply(initial_state)
    if not bool(transitioned.successful):
        raise RuntimeError("The conservative topology transition failed.")

    unresolved = prepared.fill_patch(transitioned.state)
    boundary_values = tuple(
        jnp.zeros_like(workspace.values) for workspace in unresolved.workspaces
    )
    fill_patch = prepared.fill_patch(
        transitioned.state,
        physical_boundary_values=boundary_values,
    )
    fill_patch.require_complete()

    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: args["speed"] * state,
        lambda left, right, axis, args: jnp.full(left.shape[:-1], jnp.abs(args["speed"])),
        system_id="example:block-amr-linear-advection",
    )
    boundary = phx.discretization.ExtrapolationBoundary()
    boundaries = phx.discretization.FiniteVolumeBoundarySet(
        ("x",),
        (phx.discretization.FiniteVolumeBoundaryPair(boundary, boundary),),
    )
    block_fv = phx.discretization.BlockAMRFiniteVolumePlan(
        prepared,
        system,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
        boundaries,
    )
    schedule = phx.solver.AMRTimeSchedulePlan(prepared)
    runtime = phx.solver.BlockAMRRuntimePlan(block_fv, schedule).prepare(
        compiled.topology
    )
    runtime_state = runtime.initial_state(transitioned.state)
    advanced = runtime.advance(
        runtime_state,
        jnp.asarray(0.01),
        {"speed": jnp.asarray(0.2)},
    )
    if not bool(advanced.accepted):
        raise RuntimeError(f"Block advance failed in phase {int(advanced.failed_phase)}.")

    layout = phx.discretization.CompositeAMRCellLayout(
        compiled.topology, dtype=jnp.float64
    )
    diffusion = phx.discretization.CompositeAMRDiffusionPlan(
        layout,
        boundaries={"x": ("dirichlet", "dirichlet")},
    ).prepare(1.0)
    rhs = diffusion.prepare_rhs(1.0, boundary_data={"x": (0.0, 1.0)})
    elliptic = phx.linalg.solve(
        diffusion.linear_system(),
        rhs,
        policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.ConjugateGradient(),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1.0e-10,
                absolute=1.0e-12,
                max_steps=200,
            ),
        ),
    )
    if not bool(elliptic.successful):
        raise RuntimeError("The composite diffusion solve did not converge.")

    # Numeric kernels are differentiable only while this compiled epoch is fixed.
    tangent = jax.tree.map(jnp.ones_like, rhs)
    _, operator_tangent = jax.jvp(diffusion.mv, (rhs,), (tangent,))
    if transition.transition.transfer.properties.differentiable_geometry:
        raise RuntimeError("Topology selection must not acquire a geometry gradient.")

    partition = phx.discretization.BlockAMRPartitionPlan(hierarchy, 1).prepare(
        compiled,
        prepared,
    )
    with TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "block-amr.fvckpt"
        checkpoint_plan = phx.solver.FiniteVolumeCheckpointPlan(
            runtime,
            partition=partition,
        )
        phx.solver.write_finite_volume_checkpoint(
            checkpoint_path,
            checkpoint_plan,
            advanced.runtime_state,
        )
        restored = phx.solver.read_finite_volume_checkpoint(
            checkpoint_path,
            checkpoint_plan,
        ).runtime_state

    print(
        {
            "topology_status": compiled.status.code,
            "active_blocks": compiled.evidence.realized_blocks,
            "transition_defect": float(
                jnp.max(jnp.abs(transitioned.conservation_residual))
            ),
            "fill_patch_complete": bool(fill_patch.complete),
            "accepted_step": int(restored.accepted_step),
            "composite_solve": bool(elliptic.successful),
            "fixed_epoch_tangent_leaves": len(jax.tree.leaves(operator_tangent)),
            "partition_resources": partition.resources.resource_counts,
            "qualification_limit": (
                "This serial example does not qualify multi-device or multi-host execution."
            ),
        }
    )


if __name__ == "__main__":
    main()
