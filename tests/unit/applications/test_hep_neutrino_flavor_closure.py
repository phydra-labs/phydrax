#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _pot_exposure():
    interval = phx.measurement.OperationalInterval(
        phx.measurement.OperationalCoordinate("neutrino", {"run": 1, "spill": 0}),
        phx.measurement.OperationalCoordinate("neutrino", {"run": 2, "spill": 0}),
    )
    return phx.measurement.ExposureRecord(
        phx.measurement.ExposureKind.PROTONS_ON_TARGET,
        1.0e20,
        1.0e18,
        "POT",
        interval,
        authority="beam-test",
        correlation_id="pot-test",
    )


def test_neutrino_oscillation_unitarity_and_rate_chain():
    neutrino = phx.applications.neutrino
    parameters = neutrino.NeutrinoOscillationParameters(
        theta12=0.59,
        theta13=0.15,
        theta23=0.78,
        delta_cp=-1.2,
        delta_m21_squared=7.5e-5,
        delta_m31_squared=2.5e-3,
        ordering=neutrino.NeutrinoMassOrdering.NORMAL,
    )
    identity = neutrino.oscillation_probabilities(
        parameters,
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([0.0, 0.0]),
    )
    assert jnp.all(identity.valid)
    assert jnp.allclose(
        identity.probabilities, jnp.broadcast_to(jnp.eye(3), (2, 3, 3)), atol=1.0e-12
    )

    oscillated = neutrino.oscillation_probabilities(
        parameters,
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([295.0, 295.0]),
        matter_density_g_cm3=2.6,
    )
    flux = neutrino.NeutrinoFlux(
        jnp.asarray([0.5, 1.5, 2.5]),
        jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]]),
        jnp.eye(6),
        _pot_exposure(),
        source_id="flux-test",
    )
    rate_plan = neutrino.NeutrinoRatePlan(
        flux,
        jnp.ones((2, 3)) * 1.0e-40,
        jnp.ones((2, 3)) * 0.8,
        jnp.asarray([[0.8, 0.1], [0.2, 0.9]]),
        target_count=1.0e30,
        provider_id="interaction-test",
    )
    rates = neutrino.predict_neutrino_rates(rate_plan, oscillated)
    assert bool(rates.valid)
    assert jnp.all(rates.reconstructed_rates >= 0.0)
    transfer = neutrino.apply_near_far_transfer(
        jnp.asarray([10.0, 20.0]), jnp.eye(2) * 0.5, transfer_id="near-far"
    )
    assert jnp.allclose(transfer.far_prediction, jnp.asarray([5.0, 10.0]))


def test_coherent_amplitude_interference_and_time_dependent_mixing():
    flavor = phx.applications.flavor_physics
    amplitude_plan = flavor.CoherentAmplitudePlan(
        ("resonance-a", "resonance-b"),
        phase_convention_id="helicity-test",
        normalization_evidence_id="normalization-test",
    )
    components = jnp.asarray([[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, -1.0 + 0.0j]])
    coefficients = jnp.asarray([1.0 + 0.0j, 0.0 + 1.0j])
    evaluated = flavor.evaluate_coherent_amplitude(
        amplitude_plan, components, coefficients, jnp.asarray([0.5, 0.5])
    )
    assert bool(evaluated.valid)
    assert jnp.allclose(evaluated.intensity, jnp.asarray([2.0, 2.0]))
    fractions = flavor.amplitude_interference_fractions(
        amplitude_plan, components, coefficients, jnp.asarray([0.5, 0.5])
    )
    assert bool(fractions.valid)
    assert jnp.isclose(fractions.sum_fraction, 1.0)

    mixing = flavor.NeutralMesonMixingParameters(
        decay_width=1.0,
        mass_difference=0.5,
        width_difference=0.1,
        q_over_p=1.0 + 0.0j,
        convention_id="mixing-test",
    )
    calibration = flavor.TaggingCalibration(0.0, 1.0, 0.0, source_id="tag-test")
    rates = flavor.time_dependent_decay_rate(
        mixing,
        calibration,
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1, -1]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0 + 0.0j, 1.0 + 0.0j]),
        jnp.asarray([0.0 + 0.0j, 0.0 + 0.0j]),
    )
    assert jnp.all(rates.valid)
    assert jnp.allclose(rates.rate, jnp.asarray([1.0, 0.0]))
