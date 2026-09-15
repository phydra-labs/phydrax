#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.semiconductor._optical_response import (
    LinearizedCarrierOpticalResponsePlan,
)
from phydrax.applications.semiconductor._traveling_wave import (
    realize_traveling_wave_laser_noise,
    simulate_stochastic_traveling_wave_laser,
    simulate_traveling_wave_laser,
    solve_traveling_wave_laser_threshold,
    TravelingWaveLaserInput,
    TravelingWaveLaserNoisePlan,
    TravelingWaveLaserStatus,
    TravelingWaveSemiconductorLaserPlan,
    TravelingWaveSemiconductorLaserState,
)


jax.config.update("jax_enable_x64", True)


def _response(*, internal_loss=1.0e3):
    return LinearizedCarrierOpticalResponsePlan(
        1.0e24,
        5.0e-21,
        1.0e15,
        reference_temperature=300.0,
        background_internal_loss=internal_loss,
        density_range=(0.5e24, 4.0e24),
        temperature_range=(290.0, 310.0),
        angular_frequency_range=(0.9e15, 1.1e15),
        active_volume=3.0e-16,
        confinement_factor=1.0,
        provenance="synthetic reduced traveling-wave optical law",
        model_id=f"laser-response-{internal_loss}",
    )


def _plan(
    *,
    steps=12,
    reflectivity=0.3,
    internal_loss=1.0e3,
    recombination_a=0.0,
    recombination_b=0.0,
    recombination_c=0.0,
    gain_compression=0.0,
    linewidth=0.0,
    detuning=None,
    coupling=None,
):
    amplitude_reflection = np.sqrt(reflectivity) * np.exp(0.2j)
    return TravelingWaveSemiconductorLaserPlan(
        jnp.linspace(0.0, 3.0e-4, 4),
        _response(internal_loss=internal_loss),
        1.0e15,
        group_velocity=1.0e8,
        active_area=1.0e-12,
        recombination_a=recombination_a,
        recombination_b=recombination_b,
        recombination_c=recombination_c,
        left_facet_amplitude_reflection=amplitude_reflection,
        right_facet_amplitude_reflection=amplitude_reflection.conjugate(),
        carrier_density_bounds=(0.5e24, 3.0e24),
        step_count=steps,
        gain_compression=gain_compression,
        linewidth_enhancement_factor=linewidth,
        detuning=detuning,
        coupling=coupling,
        grating_provenance="synthetic reduced-laser grating fixture",
        ledger_tolerance=2.0e-12,
    )


def _state(density, amplitude=0.0):
    return TravelingWaveSemiconductorLaserState(
        jnp.full((3,), density),
        jnp.full((3,), amplitude + 0.0j),
        jnp.full((3,), 0.7j * amplitude),
    )


def test_threshold_matches_analytic_uniform_fabry_perot_condition():
    prepared = _plan(recombination_a=1.0e9).prepare()
    result = solve_traveling_wave_laser_threshold(prepared, 300.0)
    length = 3.0e-4
    mirror_loss = -np.log(0.3 * 0.3) / (2.0 * length)
    expected_density = 1.0e24 + (1.0e3 + mirror_loss) / 5.0e-21
    expected_current = 1.602176634e-19 * (3.0e-16) * 1.0e9 * expected_density

    np.testing.assert_allclose(result.mirror_loss, mirror_loss, rtol=1.0e-13)
    np.testing.assert_allclose(
        result.threshold_carrier_pair_density, expected_density, rtol=2.0e-13
    )
    np.testing.assert_allclose(result.threshold_injection_current, expected_current)
    assert abs(result.round_trip_log_power_residual) < 1.0e-12
    assert result.successful


def test_zero_field_is_an_exact_invariant_of_deterministic_transient():
    prepared = _plan(steps=9).prepare()
    result = simulate_traveling_wave_laser(
        prepared,
        _state(1.4e24),
        TravelingWaveLaserInput(0.0, 300.0),
    )

    assert np.count_nonzero(result.forward_field) == 0
    assert np.count_nonzero(result.backward_field) == 0
    np.testing.assert_array_equal(result.carrier_pair_density, 1.4e24)
    assert result.ledger.stimulated_carrier_pairs == 0.0
    assert result.ledger.photon_balance_residual == 0.0
    assert result.successful


def test_abc_recombination_and_current_injection_balance_exactly_at_fixed_state():
    density = 1.5e24
    coefficient_a = 1.0e8
    coefficient_b = 1.0e-16
    coefficient_c = 1.0e-41
    prepared = _plan(
        steps=5,
        recombination_a=coefficient_a,
        recombination_b=coefficient_b,
        recombination_c=coefficient_c,
    ).prepare()
    recombination_rate = (
        coefficient_a * density + coefficient_b * density**2 + coefficient_c * density**3
    )
    current = 1.602176634e-19 * 3.0e-16 * recombination_rate
    result = simulate_traveling_wave_laser(
        prepared,
        _state(density),
        TravelingWaveLaserInput(current, 300.0),
    )

    np.testing.assert_allclose(result.carrier_pair_density, density, rtol=2.0e-16)
    np.testing.assert_allclose(
        result.ledger.injected_carrier_pairs,
        result.ledger.recombined_carrier_pairs,
        rtol=2.0e-15,
    )
    assert result.ledger.carrier_relative_residual < 2.0e-12
    assert result.successful


def test_below_and_above_threshold_fields_decay_and_grow_over_round_trips():
    prepared = _plan(steps=12).prepare()
    below = simulate_traveling_wave_laser(
        prepared,
        _state(1.5e24, 1.0e-8),
        TravelingWaveLaserInput(0.0, 300.0),
    )
    above = simulate_traveling_wave_laser(
        prepared,
        _state(2.5e24, 1.0e-8),
        TravelingWaveLaserInput(0.0, 300.0),
    )

    below_initial = np.sum(np.abs(below.forward_field[0]) ** 2) + np.sum(
        np.abs(below.backward_field[0]) ** 2
    )
    below_final = np.sum(np.abs(below.forward_field[-1]) ** 2) + np.sum(
        np.abs(below.backward_field[-1]) ** 2
    )
    above_initial = np.sum(np.abs(above.forward_field[0]) ** 2) + np.sum(
        np.abs(above.backward_field[0]) ** 2
    )
    above_final = np.sum(np.abs(above.forward_field[-1]) ** 2) + np.sum(
        np.abs(above.backward_field[-1]) ** 2
    )
    assert below_final < below_initial
    assert above_final > above_initial
    assert below.successful
    assert above.successful


def test_gain_compression_limits_power_while_linewidth_factor_changes_only_phase():
    initial = _state(2.0e24, 0.2)
    inputs = TravelingWaveLaserInput(0.0, 300.0)
    uncompressed = simulate_traveling_wave_laser(
        _plan(steps=1, reflectivity=1.0, internal_loss=0.0).prepare(),
        initial,
        inputs,
    )
    compressed = simulate_traveling_wave_laser(
        _plan(
            steps=1,
            reflectivity=1.0,
            internal_loss=0.0,
            gain_compression=2.0,
        ).prepare(),
        initial,
        inputs,
    )
    linewidth = simulate_traveling_wave_laser(
        _plan(
            steps=1,
            reflectivity=1.0,
            internal_loss=0.0,
            gain_compression=2.0,
            linewidth=4.0,
        ).prepare(),
        initial,
        inputs,
    )

    assert jnp.sum(jnp.abs(compressed.forward_field[-1]) ** 2) < jnp.sum(
        jnp.abs(uncompressed.forward_field[-1]) ** 2
    )
    np.testing.assert_allclose(
        jnp.abs(linewidth.forward_field[-1]),
        jnp.abs(compressed.forward_field[-1]),
        rtol=2.0e-14,
    )
    assert not np.allclose(linewidth.forward_field[-1], compressed.forward_field[-1])


def test_distributed_grating_transfers_waves_without_changing_photon_inventory():
    coupling = jnp.full((3,), np.pi / (4.0e-4), dtype=complex)
    detuning = jnp.asarray((100.0, 0.0, 100.0))
    initial = TravelingWaveSemiconductorLaserState(
        jnp.full((3,), 1.0e24),
        jnp.full((3,), 0.2 + 0.0j),
        jnp.zeros((3,), dtype=complex),
    )
    inputs = TravelingWaveLaserInput(0.0, 300.0)
    no_grating = simulate_traveling_wave_laser(
        _plan(steps=1, reflectivity=1.0, internal_loss=0.0).prepare(),
        initial,
        inputs,
    )
    prepared = _plan(
        steps=1,
        reflectivity=1.0,
        internal_loss=0.0,
        detuning=detuning,
        coupling=coupling,
    ).prepare()
    grating = simulate_traveling_wave_laser(prepared, initial, inputs)

    np.testing.assert_array_equal(no_grating.backward_field[-1, :2], 0.0)
    assert np.all(np.abs(grating.backward_field[-1, :2]) > 0.0)
    initial_power = np.sum(np.abs(initial.forward_field) ** 2)
    final_power = np.sum(np.abs(grating.forward_field[-1]) ** 2) + np.sum(
        np.abs(grating.backward_field[-1]) ** 2
    )
    np.testing.assert_allclose(final_power, initial_power, rtol=2.0e-14)
    assert grating.ledger.photon_relative_residual < 2.0e-12
    assert grating.evidence.maximum_grating_unitarity_error < 1.0e-12
    assert grating.evidence.grating_id == prepared.grating_id
    assert grating.successful

    ambiguous_threshold = solve_traveling_wave_laser_threshold(
        prepared, 300.0, mode_iteration_count=16
    )
    assert ambiguous_threshold.status & int(
        TravelingWaveLaserStatus.THRESHOLD_MODE_NOT_ISOLATED
    )
    assert not ambiguous_threshold.successful


def test_distributed_grating_threshold_matches_exact_single_cell_eigenvalue():
    length = 1.0e-4
    coupling_strength = 4.0e3
    left_reflection = np.sqrt(0.2)
    right_reflection = np.sqrt(0.7)
    prepared = TravelingWaveSemiconductorLaserPlan(
        jnp.asarray((0.0, length)),
        _response(),
        1.0e15,
        group_velocity=1.0e8,
        active_area=1.0e-12,
        recombination_a=0.0,
        recombination_b=0.0,
        recombination_c=0.0,
        left_facet_amplitude_reflection=left_reflection,
        right_facet_amplitude_reflection=right_reflection,
        carrier_density_bounds=(0.5e24, 3.0e24),
        step_count=1,
        coupling=jnp.asarray((1j * coupling_strength,)),
        grating_provenance="analytic single-cell reciprocal grating",
    ).prepare()
    angle = coupling_strength * length
    cosine, sine = np.cos(angle), np.sin(angle)
    passive_map = np.asarray(
        (
            (left_reflection * sine, left_reflection * cosine),
            (right_reflection * cosine, -right_reflection * sine),
        )
    )
    passive_modulus = np.max(np.abs(np.linalg.eigvals(passive_map)))
    equivalent_loss = -2.0 * np.log(passive_modulus) / length
    expected_density = 1.0e24 + (1.0e3 + equivalent_loss) / 5.0e-21

    threshold = solve_traveling_wave_laser_threshold(
        prepared,
        300.0,
        mode_iteration_count=8,
    )

    np.testing.assert_allclose(
        threshold.threshold_carrier_pair_density,
        expected_density,
        rtol=5.0e-12,
    )
    np.testing.assert_allclose(abs(threshold.modal_eigenvalue), 1.0, atol=1.0e-10)
    np.testing.assert_allclose(
        np.sum(np.abs(threshold.threshold_mode_forward) ** 2)
        + np.sum(np.abs(threshold.threshold_mode_backward) ** 2),
        1.0,
        rtol=2.0e-14,
    )
    assert threshold.modal_residual < 1.0e-10
    assert threshold.modal_gap > 1.0e-2
    assert threshold.evidence.threshold_mode_converged
    assert threshold.evidence.threshold_mode_isolated
    assert threshold.evidence.threshold_map_applications == (96 + 3) * 20
    assert threshold.evidence.threshold_mode_iteration_count == 8
    assert threshold.successful


def test_stimulated_exchange_is_charge_neutral_and_closes_both_ledgers():
    prepared = _plan(steps=3, reflectivity=1.0, internal_loss=0.0).prepare()
    result = simulate_traveling_wave_laser(
        prepared,
        _state(2.0e24, 0.1),
        TravelingWaveLaserInput(0.0, 300.0),
    )

    assert np.all(result.carrier_pair_density[-1] < result.carrier_pair_density[0])
    assert result.ledger.stimulated_carrier_pairs > 0.0
    np.testing.assert_allclose(
        result.ledger.stimulated_carrier_pairs,
        result.ledger.stimulated_photons,
        rtol=0.0,
        atol=0.0,
    )
    assert result.evidence.charge_neutrality_error == 0.0
    assert result.ledger.carrier_relative_residual < 2.0e-12
    assert result.ledger.photon_relative_residual < 2.0e-12
    assert result.successful


def test_deterministic_and_explicit_stochastic_replay_are_separate():
    prepared = _plan(steps=5).prepare()
    initial = _state(1.4e24)
    inputs = TravelingWaveLaserInput(0.0, 300.0)
    deterministic = simulate_traveling_wave_laser(prepared, initial, inputs)
    noise_plan = TravelingWaveLaserNoisePlan(
        1.0e-9,
        0.0,
        provenance="explicit synthetic stochastic replay fixture",
    )
    realization = realize_traveling_wave_laser_noise(
        noise_plan, prepared, jax.random.key(17)
    )
    first = simulate_stochastic_traveling_wave_laser(
        prepared, initial, inputs, realization
    )
    replay = simulate_stochastic_traveling_wave_laser(
        prepared, initial, inputs, realization
    )

    assert not deterministic.evidence.stochastic
    assert np.count_nonzero(deterministic.forward_field) == 0
    assert first.evidence.stochastic
    assert np.count_nonzero(first.forward_field) > 0
    np.testing.assert_array_equal(first.forward_field, replay.forward_field)
    np.testing.assert_array_equal(first.backward_field, replay.backward_field)
    np.testing.assert_array_equal(first.carrier_pair_density, replay.carrier_pair_density)
    assert first.noise_realization_id == realization.realization_id
    assert first.ledger.photon_relative_residual < 2.0e-12


def test_failure_statuses_and_resource_shapes_are_observable():
    prepared = _plan(steps=4).prepare()
    rejected = simulate_traveling_wave_laser(
        prepared,
        _state(3.5e24),
        TravelingWaveLaserInput(-0.01, 300.0),
    )

    assert rejected.status & int(TravelingWaveLaserStatus.INPUT_REJECTED)
    assert rejected.status & int(TravelingWaveLaserStatus.CARRIER_OUTSIDE_BOUNDS)
    assert not rejected.successful
    assert rejected.carrier_pair_density.shape == (prepared.step_count + 1, 3)
    assert rejected.forward_field.shape == (prepared.step_count + 1, 3)
    assert rejected.backward_field.shape == (prepared.step_count + 1, 3)
    assert rejected.left_output_power.shape == (prepared.step_count,)
    assert rejected.right_output_power.shape == (prepared.step_count,)
    assert prepared.workspace_bytes > 0
    assert prepared.retained_result_bytes > 0
