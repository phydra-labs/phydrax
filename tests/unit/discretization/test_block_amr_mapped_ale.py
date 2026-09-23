#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _topology(shape):
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(size) for size in shape),
        axis_names=tuple(f"axis{index}" for index in range(len(shape))),
    ).prepare(jnp.asarray([[0.0] * len(shape), [1.0] * len(shape)]))
    signature = phx.discretization.PatchShapeSignature(shape, halo_width=1)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(signature, 1),)
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0,) * len(shape), shape),),
    )
    return phx.discretization.VariablePatchTopologyCompiler(plan).initial_topology()


def test_identity_patch_geometry_matches_reference_cell_volumes_in_all_dimensions():
    for shape in ((4,), (4, 4), (2, 2, 2)):
        topology = _topology(shape)
        geometry = phx.discretization.VariablePatchGeometryPlan(
            topology,
            lambda point, time, args: point,
            f"identity-{len(shape)}d",
        ).state(0.0)

        assert bool(geometry.valid)
        expected = 1.0 / float(jnp.prod(jnp.asarray(shape)))
        active = geometry.active_cell_masks[0][0]
        assert jnp.allclose(geometry.cell_volumes[0][0][active], expected)
        assert jnp.allclose(geometry.gcl_defects[0][0][active], 0.0)


def test_affine_mapped_patch_geometry_has_exact_area_and_stationary_gcl():
    topology = _topology((4, 4))
    matrix = jnp.asarray([[1.5, 0.25], [0.0, 2.0]])
    geometry = phx.discretization.VariablePatchGeometryPlan(
        topology,
        lambda point, time, args: matrix @ point,
        "affine-2d",
        geometry_family_id="mapped",
    ).state(0.0)

    active = geometry.active_cell_masks[0][0]
    assert bool(geometry.valid)
    assert jnp.allclose(geometry.cell_volumes[0][0][active], 3.0 / 16.0)
    assert jnp.allclose(geometry.mesh_volume_rates[0][0][active], 0.0)


def test_ale_patch_geometry_satisfies_independent_gcl_face_sweep():
    topology = _topology((4,))
    geometry_plan = phx.discretization.VariablePatchGeometryPlan(
        topology,
        lambda point, time, args: (1.0 + 0.2 * time) * point,
        "uniform-dilation-1d",
        geometry_family_id="ale",
    )
    geometry = geometry_plan.state(0.5, revision=3)

    active = geometry.active_cell_masks[0][0]
    assert bool(geometry.valid)
    assert jnp.allclose(geometry.cell_volumes[0][0][active], 1.1 / 4.0)
    assert jnp.allclose(geometry.mesh_volume_rates[0][0][active], 0.2 / 4.0)


def test_patch_geometry_rejects_negative_orientation():
    topology = _topology((4,))
    geometry = phx.discretization.VariablePatchGeometryPlan(
        topology,
        lambda point, time, args: -point,
        "reflected-1d",
    ).state(0.0)

    assert not bool(geometry.valid)


def test_patch_geometry_rejects_reflected_hexahedron():
    topology = _topology((2, 2, 2))
    geometry = phx.discretization.VariablePatchGeometryPlan(
        topology,
        lambda point, time, args: jnp.asarray((-point[0], point[1], point[2])),
        "reflected-3d",
    ).state(0.0)

    assert not bool(geometry.valid)
    assert jnp.any(geometry.orientation_minima[0][0] < 0.0)


def test_ale_step_prepares_all_ssprk_geometry_and_commits_atomically():
    topology = _topology((4,))
    geometry_plan = phx.discretization.VariablePatchGeometryPlan(
        topology,
        lambda point, time, args: (1.0 + 0.2 * time) * point,
        "uniform-dilation-1d",
        geometry_family_id="ale",
    )
    source = geometry_plan.state(0.0)
    step = phx.discretization.VariablePatchALEPlan(
        geometry_plan,
        endpoint_tolerance=1.0e-12,
    ).prepare_step(source, 0.1)

    assert bool(step.successful)
    assert jnp.allclose(step.stage_endpoint.time, 0.1)
    assert jnp.allclose(step.stage_midpoint.time, 0.05)
    assert jnp.allclose(
        step.committed_geometry().cell_volumes[0][0],
        step.stage_endpoint.cell_volumes[0][0],
    )


def test_patch_geometry_revision_is_exact_bounded_and_cannot_overflow_ale():
    topology = _topology((4,))
    geometry_plan = phx.discretization.VariablePatchGeometryPlan(
        topology,
        lambda point, time, args: point,
        "bounded-revision",
    )

    for revision in (-1, 1.5, 2**31):
        with pytest.raises(ValueError, match="revision"):
            geometry_plan.state(0.0, revision=revision)

    source = geometry_plan.state(
        0.0,
        revision=jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
    )
    with pytest.raises(Exception, match="cannot advance"):
        phx.discretization.VariablePatchALEPlan(geometry_plan).prepare_step(
            source,
            0.1,
        )
