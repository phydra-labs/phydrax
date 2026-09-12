#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Variable-patch, entity, mapped/ALE, and embedded-boundary workflow."""

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((4, 4), halo_width=1)
    hierarchy = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(signature, 1),)
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0), (4, 4)),),
    )
    topology = phx.discretization.VariablePatchTopologyCompiler(
        hierarchy
    ).initial_topology()
    entities = phx.discretization.VariablePatchEntityComplexPlan(
        phx.discretization.BlockHierarchyCapacityPlan(
            ((32, 64, 32),),
            (80, 64),
        )
    ).prepare(topology)[0]
    geometry_plan = phx.discretization.VariablePatchGeometryPlan(
        topology,
        lambda point, time, args: (1.0 + 0.05 * time) * point,
        "dilating-map",
        geometry_family_id="mapped-ale",
    )
    geometry = geometry_plan.state(0.25, revision=1)
    ale = phx.discretization.VariablePatchALEPlan(geometry_plan).prepare_step(
        geometry_plan.state(0.0),
        0.1,
    )
    embedded = phx.discretization.VariablePatchEmbeddedBoundaryPlan(
        geometry_plan.plan_id,
        lambda points, time, args: points[:, 0] - 0.45,
        "wall",
    ).prepare(geometry)

    print(
        {
            "epoch_id": topology.epoch.epoch_id,
            "entity_complex_id": entities.complex_id,
            "mapped_geometry_valid": bool(geometry.valid),
            "ale_step_valid": bool(ale.successful),
            "cut_cells": embedded.evidence.cut_cell_count,
            "volume_closure": embedded.evidence.maximum_volume_closure_defect,
            "face_closure": embedded.evidence.maximum_face_closure_defect,
        }
    )


if __name__ == "__main__":
    main()
