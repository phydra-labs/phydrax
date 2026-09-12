#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import pytest

import phydrax as phx


def _geometry():
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
    plan = phx.discretization.VariablePatchGeometryPlan(
        topology,
        lambda point, time, args: point,
        "identity-2d",
    )
    return plan, plan.state(0.0)


def test_stationary_patch_embedded_boundary_clips_conservatively_and_marks_small_cells():
    geometry_plan, geometry = _geometry()
    embedded = phx.discretization.VariablePatchEmbeddedBoundaryPlan(
        geometry_plan.plan_id,
        lambda points, time, args: points[:, 0] - 0.45,
        "body-a",
        body_tag=7,
        minimum_volume_fraction=0.3,
    ).prepare(geometry)
    fractions = embedded.volume_fraction[0][0][0]
    cut = embedded.cut_face_active[0][0][0]
    normals = embedded.cut_face_normals[0][0][0]
    small = embedded.small_cells[0][0][0]

    assert embedded.evidence.cut_cell_count == 4
    assert embedded.evidence.small_cell_count == 4
    assert jnp.allclose(jnp.sum(embedded.fluid_cell_volumes[0][0][0]), 0.55)
    assert jnp.all(fractions[0] == 0.0)
    assert jnp.all(fractions[3] == 1.0)
    assert jnp.all(cut[1])
    assert jnp.all(normals[1, :, 0] < 0.0)
    assert jnp.all(small[1])
    open_y = embedded.open_face_fractions[0][0][1][0]
    assert jnp.allclose(open_y[1, 1:], 0.2)
    assert embedded.evidence.maximum_face_closure_defect <= 1.0e-12


def test_stationary_patch_embedded_boundary_rejects_ambiguous_vertex_crossing():
    geometry_plan, geometry = _geometry()
    plan = phx.discretization.VariablePatchEmbeddedBoundaryPlan(
        geometry_plan.plan_id,
        lambda points, time, args: points[:, 0] - 0.5,
        "body-a",
    )

    with pytest.raises(ValueError, match="ambiguous"):
        plan.prepare(geometry)
