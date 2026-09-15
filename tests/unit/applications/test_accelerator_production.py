#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _bunch():
    accelerator = phx.applications.accelerator
    convention = accelerator.AcceleratorConvention()
    return accelerator.AcceleratorBunch(
        jnp.asarray(
            [[1.0e-3, 1.0e-4, 0.0, 0.0, 0.0, 0.0], [-1.0e-3, -1.0e-4, 0.0, 0.0, 0.0, 0.0]]
        ),
        jnp.ones(2),
        jnp.asarray([10, 11]),
        reference_rest_energy=1.0,
        reference_momentum=2.0,
        reference_charge=1.0,
        convention=convention,
        bunch_id="test-bunch",
    )


def test_drift_updates_coordinates_and_preserves_particles():
    accelerator = phx.applications.accelerator
    bunch = _bunch()
    plan = accelerator.BeamlinePlan(
        jnp.asarray([int(accelerator.BeamlineElementKind.DRIFT)]),
        jnp.asarray([2.0]),
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        element_ids=("D1",),
        convention=bunch.convention,
    )
    result = accelerator.track_beamline(plan, bunch)
    assert bool(result.accepted)
    assert jnp.all(result.bunch.active)
    assert jnp.allclose(
        result.bunch.coordinates[:, 0],
        bunch.coordinates[:, 0] + 2.0 * bunch.coordinates[:, 1],
    )


def test_aperture_records_first_loss_without_dropping_identity():
    accelerator = phx.applications.accelerator
    bunch = _bunch()
    plan = accelerator.BeamlinePlan(
        jnp.asarray([int(accelerator.BeamlineElementKind.CIRCULAR_APERTURE)]),
        jnp.asarray([0.0]),
        jnp.asarray([5.0e-4]),
        jnp.asarray([0.0]),
        element_ids=("A1",),
        convention=bunch.convention,
    )
    result = accelerator.track_beamline(plan, bunch)
    assert not jnp.any(result.bunch.active)
    assert jnp.array_equal(result.loss_element_indices, jnp.asarray([0, 0]))
    assert jnp.array_equal(result.bunch.particle_ids, bunch.particle_ids)


def test_space_charge_residual_failure_rolls_back_bunch():
    accelerator = phx.applications.accelerator
    bunch = _bunch()
    plan = accelerator.SpaceChargeKickPlan(
        source_plan_id="pic-source",
        frame_transform_id="lab-to-beam",
        boundary_condition_id="open",
        maximum_residual=1.0e-6,
    )
    rejected = accelerator.apply_space_charge_kick(
        plan,
        bunch,
        jnp.ones((2, 2)),
        jnp.asarray(1.0e-3),
    )
    assert not bool(rejected.accepted)
    assert jnp.array_equal(rejected.bunch.coordinates, bunch.coordinates)
