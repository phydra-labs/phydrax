#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.semiconductor._optical_response import (
    ActiveRegionOpticalProjection,
    evaluate_semiconductor_optical_response,
    LinearizedCarrierOpticalResponsePlan,
)
from phydrax.applications.semiconductor._traveling_wave import (
    simulate_stochastic_traveling_wave_laser,
    solve_traveling_wave_laser_threshold,
    TravelingWaveLaserInput,
    TravelingWaveLaserNoisePlan,
    TravelingWaveSemiconductorLaserPlan,
    TravelingWaveSemiconductorLaserState,
)


@pytest.fixture(autouse=True)
def _double_precision():
    with jax.enable_x64(True):
        yield


def test_projected_semiconductor_state_drives_replayable_reduced_laser_workflow():
    response = LinearizedCarrierOpticalResponsePlan(
        1.0e24,
        1.0e-20,
        1.0e15,
        reference_temperature=300.0,
        background_internal_loss=400.0,
        density_range=(0.5e24, 4.0e24),
        temperature_range=(290.0, 310.0),
        angular_frequency_range=(0.9e15, 1.1e15),
        active_volume=2.0e-16,
        confinement_factor=0.5,
        provenance="synthetic projected active-region laser workflow",
        model_id="workflow-optical-response",
    )
    projection = ActiveRegionOpticalProjection(
        jnp.full((4,), 0.5e-16),
        jnp.asarray((0.0, 1.0, 1.0, 0.0)),
        jnp.asarray((0.2, 1.0, 1.0, 0.2)),
        support_id="workflow-device-control-volumes",
        mode_id="workflow-fundamental-mode",
        provenance="synthetic device-to-mode projection",
    )
    frozen = evaluate_semiconductor_optical_response(
        response,
        jnp.asarray((0.8, 1.7, 2.1, 0.9)) * 1.0e24,
        jnp.asarray((300.0, 301.0, 299.0, 300.0)),
        1.0e15,
        projection=projection,
    )
    np.testing.assert_allclose(frozen.carrier_pair_density, 1.9e24)
    assert frozen.successful

    laser = TravelingWaveSemiconductorLaserPlan(
        jnp.linspace(0.0, 2.0e-4, 5),
        response,
        1.0e15,
        group_velocity=5.0e7,
        active_area=1.0e-12,
        recombination_a=5.0e8,
        recombination_b=0.0,
        recombination_c=0.0,
        left_facet_amplitude_reflection=np.sqrt(0.5) + 0.0j,
        right_facet_amplitude_reflection=-np.sqrt(0.4) + 0.0j,
        carrier_density_bounds=(0.5e24, 3.5e24),
        step_count=8,
        gain_compression=0.02,
        linewidth_enhancement_factor=3.0,
        ledger_tolerance=3.0e-12,
    ).prepare()
    threshold = solve_traveling_wave_laser_threshold(laser, 300.0)
    assert threshold.successful

    initial = TravelingWaveSemiconductorLaserState(
        jnp.full((4,), threshold.threshold_carrier_pair_density),
        jnp.zeros((4,), dtype="complex128"),
        jnp.zeros((4,), dtype="complex128"),
    )
    forcing = TravelingWaveLaserNoisePlan(
        1.0e-10,
        0.0,
        provenance="explicit workflow seed forcing",
    ).realize(laser, jax.random.key(90210))
    inputs = TravelingWaveLaserInput(threshold.threshold_injection_current, 300.0)
    first = simulate_stochastic_traveling_wave_laser(laser, initial, inputs, forcing)
    replay = simulate_stochastic_traveling_wave_laser(laser, initial, inputs, forcing)

    assert first.successful
    assert np.count_nonzero(first.forward_field) > 0
    np.testing.assert_array_equal(first.forward_field, replay.forward_field)
    np.testing.assert_array_equal(first.backward_field, replay.backward_field)
    assert first.ledger.carrier_relative_residual < 3.0e-12
    assert first.ledger.photon_relative_residual < 3.0e-12
    assert first.evidence.prepared_id == laser.prepared_id
