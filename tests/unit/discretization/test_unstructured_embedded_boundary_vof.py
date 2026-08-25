#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume._unstructured_embedded_boundary import (
    EmbeddedBoundaryStabilizationPolicy,
    EmbeddedBoundaryStatus,
)


def _quadrilateral_grid(nx=3, ny=2, *, x_scale=1.0):
    vertices = np.asarray(
        [(x_scale * i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    )
    cells = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            cells.append((lower_left, lower_right, upper_right, upper_left))
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, quadrilaterals=np.asarray(cells)
    ).prepare()


def test_embedded_boundary_clipping_preserves_complement_and_fluid_closure():
    discretization = _quadrilateral_grid()
    fluid_right = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] - 0.45,
        field_id="vertical-plane-fluid-right",
        body_tag=7,
    ).prepare()
    fluid_left = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: 0.45 - points[:, 0],
        field_id="vertical-plane-fluid-left",
        body_tag=7,
    ).prepare()

    np.testing.assert_allclose(fluid_right.report.total_fluid_volume, 0.55)
    np.testing.assert_allclose(fluid_left.report.total_fluid_volume, 0.45)
    np.testing.assert_allclose(
        fluid_right.volume_fraction + fluid_left.volume_fraction, 1.0, atol=2e-12
    )
    np.testing.assert_allclose(jnp.sum(fluid_right.cut_face_measures), 1.0, atol=2e-12)
    assert fluid_right.report.cut_cell_count == 2
    assert fluid_right.report.maximum_fluid_closure_residual < 2e-12
    cut_normals = fluid_right.cut_face_normals[fluid_right.cut_face_active]
    np.testing.assert_allclose(
        cut_normals,
        jnp.broadcast_to(jnp.asarray((-1.0, 0.0)), cut_normals.shape),
        atol=2e-12,
    )
    assert jnp.all(fluid_right.body_tags == 7)
    assert jnp.all(
        fluid_right.safe_inverse_fluid_volume[fluid_right.active_fluid_cells] > 0.0
    )
    np.testing.assert_allclose(
        fluid_right.open_face_measures,
        fluid_right.face_open_fraction * discretization.face_measures,
    )
    assert fluid_right.evidence.passed
    assert fluid_right.evidence.status == int(EmbeddedBoundaryStatus.SUCCESS)


def test_embedded_boundary_has_exact_full_fluid_and_solid_limits():
    discretization = _quadrilateral_grid()
    full = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: jnp.ones((points.shape[0],)),
        field_id="full-fluid",
    ).prepare()
    solid = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: -jnp.ones((points.shape[0],)),
        field_id="full-solid",
    ).prepare()

    np.testing.assert_array_equal(
        full.volume_fraction, jnp.ones_like(full.volume_fraction)
    )
    np.testing.assert_array_equal(full.fluid_cell_volumes, discretization.cell_volumes)
    np.testing.assert_array_equal(
        full.face_open_fraction, jnp.ones_like(full.face_open_fraction)
    )
    np.testing.assert_array_equal(full.open_face_measures, discretization.face_measures)
    np.testing.assert_array_equal(
        solid.volume_fraction, jnp.zeros_like(solid.volume_fraction)
    )
    np.testing.assert_array_equal(
        solid.fluid_cell_volumes, jnp.zeros_like(solid.fluid_cell_volumes)
    )
    np.testing.assert_array_equal(
        solid.safe_inverse_fluid_volume,
        jnp.zeros_like(solid.safe_inverse_fluid_volume),
    )
    np.testing.assert_array_equal(
        solid.face_open_fraction, jnp.zeros_like(solid.face_open_fraction)
    )
    np.testing.assert_array_equal(
        solid.open_face_measures, jnp.zeros_like(solid.open_face_measures)
    )
    assert jnp.all(full.active_fluid_cells)
    assert not jnp.any(solid.active_fluid_cells)
    assert full.evidence.passed
    assert solid.evidence.passed
    assert full.evidence.minimum_nonzero_volume_fraction == 1.0
    assert solid.evidence.minimum_nonzero_volume_fraction == 0.0


@pytest.mark.parametrize("sliver_width", [1.0e-40, 1.0e-50])
def test_embedded_boundary_rejects_float32_extreme_slivers(sliver_width):
    with jax.enable_x64(True):
        discretization = _quadrilateral_grid(nx=1, ny=1)
    with jax.enable_x64(False):
        assert discretization.vertices.dtype == jnp.float64
        vertex_values = np.asarray(
            (sliver_width, -1.0, sliver_width, -1.0), dtype=np.float64
        )
        plan = phx.discretization.EmbeddedBoundaryPlan(
            discretization,
            lambda points, values: values,
            field_id=f"float32-extreme-sliver-{sliver_width}",
        )

        with pytest.raises(ValueError, match="target dtype"):
            plan.prepare(vertex_values)


def test_embedded_boundary_identity_fingerprints_realized_metric_dtype():
    discretization = _quadrilateral_grid(nx=1, ny=1)
    plan = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] - 0.25,
        field_id="realized-metric-dtype",
    )

    with jax.enable_x64(False):
        float32_metrics = plan.prepare()
    with jax.enable_x64(True):
        float64_metrics = plan.prepare()

    assert float32_metrics.volume_fraction.dtype == jnp.float32
    assert float64_metrics.volume_fraction.dtype == jnp.float64
    assert float32_metrics.metrics_id != float64_metrics.metrics_id
    assert jnp.all(
        jnp.isfinite(
            float32_metrics.safe_inverse_fluid_volume[float32_metrics.active_fluid_cells]
        )
    )
    assert jnp.all(
        float32_metrics.volume_fraction[float32_metrics.active_fluid_cells] > 0.0
    )


def test_embedded_boundary_identity_binds_samples_base_and_field_body_provenance():
    discretization = _quadrilateral_grid()
    plan = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, offset: points[:, 0] - offset,
        field_id="argument-driven-plane",
        body_tag=3,
    )
    first = plan.prepare(0.45)
    stale = plan.prepare(0.46)
    renamed = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] - 0.45,
        field_id="renamed-plane",
        body_tag=3,
    ).prepare()
    retagged = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] - 0.45,
        field_id="argument-driven-plane",
        body_tag=4,
    ).prepare()
    geometry_changed_discretization = _quadrilateral_grid(x_scale=2.0)
    geometry_changed = phx.discretization.EmbeddedBoundaryPlan(
        geometry_changed_discretization,
        lambda points, args: points[:, 0] - 0.45,
        field_id="argument-driven-plane",
        body_tag=3,
    ).prepare()
    topology_changed_discretization = _quadrilateral_grid(nx=4)
    topology_changed = phx.discretization.EmbeddedBoundaryPlan(
        topology_changed_discretization,
        lambda points, args: points[:, 0] - 0.45,
        field_id="argument-driven-plane",
        body_tag=3,
    ).prepare()

    assert first.prepared_id == discretization.prepared_id
    assert first.topology_id == discretization.topology_id
    assert first.geometry_id == discretization.geometry_id
    assert first.field_id == "argument-driven-plane"
    assert first.body_tag == 3
    assert first.metrics_id != stale.metrics_id
    assert first.metrics_id != renamed.metrics_id
    assert first.metrics_id != retagged.metrics_id
    assert first.metrics_id != geometry_changed.metrics_id
    assert first.metrics_id != topology_changed.metrics_id
    assert first.prepared_id != geometry_changed.prepared_id
    assert first.geometry_id != geometry_changed.geometry_id
    assert first.topology_id == geometry_changed.topology_id
    assert first.topology_id != topology_changed.topology_id
    assert not jnp.array_equal(first.vertex_values, stale.vertex_values)
    np.testing.assert_array_equal(first.vertex_values, renamed.vertex_values)
    np.testing.assert_array_equal(first.body_tags, jnp.full_like(first.body_tags, 3))
    np.testing.assert_array_equal(
        retagged.body_tags, jnp.full_like(retagged.body_tags, 4)
    )


def test_embedded_boundary_closure_thresholds_produce_typed_failed_evidence():
    base_discretization = _quadrilateral_grid()
    discretization = eqx.tree_at(
        lambda value: value.cell_volumes,
        base_discretization,
        base_discretization.cell_volumes.at[1].add(1.0e-3),
    )
    relaxed_policy = EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.05,
        maximum_recipients=3,
        absolute_tolerance=2.0e-3,
        relative_tolerance=0.0,
    )
    strict_policy = EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.05,
        maximum_recipients=3,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    level_set = lambda points, args: points[:, 0] + 0.31 * points[:, 1] - 0.51
    relaxed = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        level_set,
        field_id="oblique-closure",
        stabilization_policy=relaxed_policy,
    ).prepare()
    failed = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        level_set,
        field_id="oblique-closure",
        stabilization_policy=strict_policy,
    ).prepare()

    assert relaxed.evidence.passed
    assert relaxed.evidence.status == int(EmbeddedBoundaryStatus.SUCCESS)
    assert not failed.evidence.passed
    assert failed.evidence.status == int(EmbeddedBoundaryStatus.FAILED)
    assert failed.metrics_id != relaxed.metrics_id
    threshold_exceeded = (
        jnp.any(
            failed.evidence.volume_closure_defect
            > failed.evidence.volume_closure_tolerance
        )
        | jnp.any(
            failed.evidence.aperture_closure_defect
            > failed.evidence.aperture_closure_tolerance
        )
        | jnp.any(
            failed.evidence.cut_face_closure_defect
            > failed.evidence.cut_face_closure_tolerance
        )
    )
    assert threshold_exceeded


def test_embedded_boundary_float32_oblique_closure_uses_realized_epsilon_floor():
    with jax.enable_x64(False):
        discretization = _quadrilateral_grid()
        policy = EmbeddedBoundaryStabilizationPolicy(
            minimum_volume_fraction=0.05,
            maximum_recipients=3,
            absolute_tolerance=1.0e-12,
            relative_tolerance=1.0e-12,
        )
        metrics = phx.discretization.EmbeddedBoundaryPlan(
            discretization,
            lambda points, args: points[:, 0] + 0.31 * points[:, 1] - 0.51,
            field_id="float32-oblique-closure",
            stabilization_policy=policy,
        ).prepare()

    assert metrics.volume_fraction.dtype == jnp.float32
    assert metrics.evidence.passed
    assert metrics.evidence.status == int(EmbeddedBoundaryStatus.SUCCESS)
    assert jnp.all(metrics.evidence.volume_closure_tolerance > 1.0e-12)
    assert jnp.all(metrics.evidence.aperture_closure_tolerance > 1.0e-12)
    assert jnp.all(
        metrics.evidence.cut_face_closure_tolerance[metrics.cut_face_active] > 1.0e-12
    )
    assert jnp.all(
        metrics.evidence.volume_closure_defect
        <= metrics.evidence.volume_closure_tolerance
    )
    assert jnp.all(
        metrics.evidence.aperture_closure_defect
        <= metrics.evidence.aperture_closure_tolerance
    )
    assert jnp.all(
        metrics.evidence.cut_face_closure_defect
        <= metrics.evidence.cut_face_closure_tolerance
    )


def test_embedded_boundary_cut_normals_are_unit_and_outward_and_solids_own_zero():
    discretization = _quadrilateral_grid()
    gradient = jnp.asarray((1.0, 0.25))
    embedded = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points @ gradient - 0.55,
        field_id="oblique-normal",
        body_tag=11,
    ).prepare()

    cut_normals = embedded.cut_face_normals[embedded.cut_face_active]
    expected_outward = -gradient / jnp.linalg.norm(gradient)
    np.testing.assert_allclose(jnp.linalg.norm(cut_normals, axis=-1), 1.0, atol=2e-12)
    assert jnp.all(cut_normals @ gradient < 0.0)
    np.testing.assert_allclose(
        cut_normals,
        jnp.broadcast_to(expected_outward, cut_normals.shape),
        atol=2e-12,
    )
    inactive = ~embedded.active_fluid_cells
    assert jnp.any(inactive)
    np.testing.assert_array_equal(
        embedded.fluid_cell_volumes[inactive],
        jnp.zeros_like(embedded.fluid_cell_volumes[inactive]),
    )
    np.testing.assert_array_equal(
        embedded.safe_inverse_fluid_volume[inactive],
        jnp.zeros_like(embedded.safe_inverse_fluid_volume[inactive]),
    )
    assert not jnp.any(embedded.cut_face_active[inactive])
    np.testing.assert_array_equal(
        embedded.body_tags, jnp.full_like(embedded.body_tags, 11)
    )


def test_embedded_boundary_stabilization_policy_classifies_slivers_by_identity():
    discretization = _quadrilateral_grid()
    classify = EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.05,
        maximum_recipients=2,
        absolute_tolerance=1.0e-12,
        relative_tolerance=2.0e-12,
    )
    ignore = EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.01,
        maximum_recipients=2,
        absolute_tolerance=1.0e-12,
        relative_tolerance=2.0e-12,
    )
    recipient_changed = EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.05,
        maximum_recipients=3,
        absolute_tolerance=1.0e-12,
        relative_tolerance=2.0e-12,
    )
    absolute_changed = EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.05,
        maximum_recipients=2,
        absolute_tolerance=2.0e-12,
        relative_tolerance=2.0e-12,
    )
    relative_changed = EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.05,
        maximum_recipients=2,
        absolute_tolerance=1.0e-12,
        relative_tolerance=3.0e-12,
    )
    level_set = lambda points, args: points[:, 0] - 0.32
    slivers = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        level_set,
        field_id="sliver-plane",
        stabilization_policy=classify,
    ).prepare()
    unclassified = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        level_set,
        field_id="sliver-plane",
        stabilization_policy=ignore,
    ).prepare()

    assert classify.policy_id != ignore.policy_id
    assert (
        len(
            {
                classify.policy_id,
                ignore.policy_id,
                recipient_changed.policy_id,
                absolute_changed.policy_id,
                relative_changed.policy_id,
            }
        )
        == 5
    )
    assert slivers.stabilization_policy_id == classify.policy_id
    assert unclassified.stabilization_policy_id == ignore.policy_id
    assert slivers.evidence.small_cell_count == 2
    assert unclassified.evidence.small_cell_count == 0
    np.testing.assert_allclose(
        slivers.evidence.minimum_nonzero_volume_fraction, 0.04, atol=2e-12
    )
    assert slivers.metrics_id != unclassified.metrics_id


def test_embedded_boundary_rejects_ambiguous_multi_crossings():
    discretization = _quadrilateral_grid(nx=1, ny=1)
    with pytest.raises(ValueError, match="ambiguous edge crossings"):
        phx.discretization.EmbeddedBoundaryPlan(
            discretization,
            lambda points, args: jnp.asarray((-1.0, 1.0, 1.0, -1.0)),
            field_id="alternating-crossings",
        ).prepare()


def test_plic_reconstructs_planar_interface_and_vof_transport_is_bounded():
    discretization = _quadrilateral_grid()
    embedded = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] - 0.45,
        field_id="plic-reference-plane",
    ).prepare()
    gradient = phx.discretization.CellPolynomialReconstructionPlan(
        1, oversampling=4
    ).prepare(discretization)
    vof = phx.discretization.UnstructuredVOFPlan(discretization, gradient)
    reconstruction = vof.reconstruct(embedded.volume_fraction)

    assert jnp.sum(reconstruction.interface_active) == 2
    active = reconstruction.interface_active
    interface_normals = reconstruction.normals[active]
    np.testing.assert_allclose(
        jnp.linalg.norm(interface_normals, axis=-1), 1.0, atol=2e-12
    )
    assert jnp.all(interface_normals[:, 0] < -0.99)
    mean_normal = jnp.mean(interface_normals, axis=0)
    assert mean_normal[0] < -0.99
    np.testing.assert_allclose(mean_normal[1], 0.0, atol=2e-12)
    endpoint_projection = jnp.sum(
        reconstruction.interface_endpoints[active] * interface_normals[:, None, :],
        axis=-1,
    )
    np.testing.assert_allclose(
        endpoint_projection,
        jnp.broadcast_to(reconstruction.offsets[active, None], endpoint_projection.shape),
        atol=2e-12,
    )
    np.testing.assert_allclose(jnp.sum(reconstruction.interface_measures), 1.0, rtol=1e-2)

    alpha = jnp.asarray((1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    volume_flux = jnp.zeros((discretization.face_measures.size,))
    owner = np.asarray(discretization.owner_cells)
    neighbour = np.asarray(discretization.neighbour_cells)
    first_faces = np.where((owner == 0) & (neighbour == 1))[0]
    second_faces = np.where((owner == 3) & (neighbour == 4))[0]
    assert first_faces.size == second_faces.size == 1
    volume_flux = volume_flux.at[first_faces[0]].set(0.2)
    volume_flux = volume_flux.at[second_faces[0]].set(0.2)
    old_volume = vof.phase_volume(alpha)
    updated = eqx.filter_jit(vof.advance)(alpha, volume_flux, jnp.asarray(0.5))

    assert jnp.all((updated >= 0.0) & (updated <= 1.0))
    np.testing.assert_allclose(vof.phase_volume(updated), old_volume, atol=2e-12)
    np.testing.assert_allclose(updated, (0.7, 0.3, 0.0, 0.7, 0.3, 0.0))
    assert vof.stable_step(alpha, volume_flux) > 0.5
