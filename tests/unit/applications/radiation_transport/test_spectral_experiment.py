#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_correlated_k_quadrature_and_sensor_response_reconstruct_scalar_transfer():
    transfer = phx.applications.astrophysics.RayTransferPlan(
        jnp.asarray(((1.0, 2.0),)), ray_id="one-ray"
    )
    spectral = phx.applications.radiation_transport.CorrelatedKDistributionPlan(
        jnp.asarray(((0.25, 0.75), (0.5, 0.5))), ("band-a", "band-b")
    )
    sensor = phx.applications.radiation_transport.RadiativeSensorPlan(
        jnp.asarray((0.2, 0.8)), sensor_id="two-band"
    )
    plan = phx.applications.radiation_transport.ScalarRadiativeExperimentPlan(
        transfer, spectral, sensor
    )
    emission = jnp.zeros((2, 2, 1, 2))
    emission = emission.at[0, 0].set(1.0)
    emission = emission.at[0, 1].set(3.0)
    emission = emission.at[1, 0].set(2.0)
    emission = emission.at[1, 1].set(4.0)
    result = plan.evaluate(emission, jnp.zeros_like(emission))

    assert bool(result.successful)
    np.testing.assert_allclose(result.band_radiance[:, 0], (7.5, 9.0))
    np.testing.assert_allclose(result.measured_radiance, (8.7,))


def test_polarized_spectral_experiment_preserves_stokes_cone():
    sensor = phx.applications.radiation_transport.RadiativeSensorPlan(
        jnp.asarray((0.25, 0.75)), sensor_id="polarized-two-frequency"
    )
    plan = phx.applications.radiation_transport.PolarizedRadiativeExperimentPlan(
        jnp.asarray(((1.0, 1.0), (1.0, 1.0))), sensor
    )
    emission = jnp.zeros((2, 2, 4))
    emission = emission.at[0, :, 0].set(1.0)
    emission = emission.at[0, :, 1].set(0.2)
    emission = emission.at[1, :, 0].set(2.0)
    emission = emission.at[1, :, 1].set(0.1)
    result = plan.evaluate(
        emission,
        jnp.zeros((2, 2, 4, 4)),
        jnp.zeros((2, 4)),
    )

    assert bool(result.successful)
    assert bool(result.physically_valid)
    np.testing.assert_allclose(result.measured_stokes, (3.5, 0.25, 0.0, 0.0))
    assert result.stokes_cone_margin > 0.0
