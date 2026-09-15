#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.discretization import FourierAxisSpec, TensorGridPlan
from phydrax.optics.wave._material_response import (
    DelayedRamanResponsePlan,
    DrudePlasmaResponsePlan,
    IonizingDrudeResponsePlan,
    MultiphotonIonizationRatePlan,
)
from phydrax.optics.wave._pulse_time import PulseTimeSpace


jax.config.update("jax_enable_x64", True)


def _time_space(count: int = 128, duration: float = 2.0) -> PulseTimeSpace:
    grid = TensorGridPlan((FourierAxisSpec(count),), axis_names=("time",)).prepare(
        jnp.asarray([[0.0], [duration]])
    )
    return PulseTimeSpace(grid, topology="periodic-cell")


def _positive_mask(count: int) -> jax.Array:
    return jnp.fft.fftfreq(count) > 0.0


def test_zero_strength_raman_is_exactly_inactive():
    time_space = _time_space()
    field = jnp.full(time_space.shape, 1.7 + 0.0j)
    plan = DelayedRamanResponsePlan(
        0.0,
        3.0,
        0.2,
        provenance_id="declared-zero-strength-raman",
    )
    evaluation = plan.prepare(
        time_space,
        _positive_mask(time_space.size),
        field.shape,
        temporal_axis=0,
    ).evaluate(field)

    assert jnp.all(evaluation.analytic_nonlinear_polarization == 0.0)
    assert jnp.all(evaluation.analytic_free_current == 0.0)
    assert evaluation.physical_state.shape == field.shape + (2,)
    assert evaluation.evidence.initial_state_norm == 0.0
    assert evaluation.evidence.endpoint_state_norm == 0.0
    assert jnp.all(evaluation.physical_state == 0.0)
    assert evaluation.evidence.initialized_at_window_entrance
    assert evaluation.ledger.material_work_density == 0.0
    assert evaluation.provenance_id == "declared-zero-strength-raman"
    assert evaluation.successful


def test_undamped_raman_constant_drive_matches_exact_oscillator_limit():
    time_space = _time_space(256, 1.5)
    amplitude = 1.3
    omega = 4.0
    field = jnp.full(time_space.shape, amplitude + 0.0j)
    evaluation = (
        DelayedRamanResponsePlan(
            2.0e-20,
            omega,
            0.0,
            provenance_id="declared-undamped-raman",
        )
        .prepare(
            time_space,
            _positive_mask(time_space.size),
            field.shape,
            temporal_axis=0,
        )
        .evaluate(field)
    )
    time = time_space.coordinates - time_space.coordinates[0]
    expected_displacement = amplitude**2 * (1.0 - jnp.cos(omega * time))
    expected_velocity = amplitude**2 * omega * jnp.sin(omega * time)

    assert jnp.allclose(
        evaluation.physical_state[..., 0],
        expected_displacement,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert jnp.allclose(
        evaluation.physical_state[..., 1],
        expected_velocity,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert evaluation.ledger.raman_dissipated_energy_density == 0.0
    assert evaluation.ledger.raman_oscillator_energy_density > 0.0


def test_constant_multiphoton_rate_has_exact_bounded_neutral_depletion():
    time_space = _time_space(160, 2.0)
    amplitude = 2.0
    coefficient = 0.3
    neutral_density = 5.0
    field = jnp.full(time_space.shape, amplitude + 0.0j)
    ionization = MultiphotonIonizationRatePlan(
        coefficient,
        2,
        provenance_id="declared-two-photon-power-rate",
    )
    drude = DrudePlasmaResponsePlan(
        0.0,
        provenance_id="declared-collisionless-drude",
        electron_charge_magnitude=1.0,
        electron_mass=1.0,
    )
    evaluation = (
        IonizingDrudeResponsePlan(
            ionization,
            drude,
            neutral_density,
            0.0,
        )
        .prepare(
            time_space,
            _positive_mask(time_space.size),
            field.shape,
            temporal_axis=0,
        )
        .evaluate(field)
    )
    rate = coefficient * amplitude**4
    elapsed = time_space.coordinates - time_space.coordinates[0]
    expected = neutral_density * (1.0 - jnp.exp(-rate * elapsed))
    electron = evaluation.physical_state[..., 0]

    assert jnp.allclose(electron, expected, rtol=2.0e-13, atol=2.0e-13)
    assert jnp.all(electron >= 0.0)
    assert jnp.all(electron <= neutral_density)
    assert evaluation.evidence.electron_bound_violation == 0.0
    assert evaluation.evidence.minimum_electron_density == 0.0
    assert evaluation.evidence.maximum_electron_density < neutral_density
    assert evaluation.successful


def test_drude_zero_density_and_collisionless_limits_are_exact():
    time_space = _time_space(128, 1.0)
    field = jnp.full(time_space.shape, 2.0)
    collisionless = DrudePlasmaResponsePlan(
        0.0,
        provenance_id="declared-collisionless-limit",
        electron_charge_magnitude=1.0,
        electron_mass=1.0,
    ).prepare(time_space, field.shape, temporal_axis=0)

    vacuum = collisionless.evaluate(field, jnp.zeros_like(field))
    assert jnp.all(vacuum.physical_free_current == 0.0)
    assert vacuum.collisional_energy_density == 0.0
    assert vacuum.terminal_kinetic_energy_density == 0.0

    density = jnp.full_like(field, 3.0)
    plasma = collisionless.evaluate(field, density)
    elapsed = time_space.coordinates - time_space.coordinates[0]
    assert jnp.allclose(
        plasma.physical_free_current,
        6.0 * elapsed,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert plasma.collisional_energy_density == 0.0
    assert plasma.terminal_kinetic_energy_density > 0.0
    assert plasma.finite


def test_ionizing_drude_reports_endpoint_and_complete_work_channels():
    time_space = _time_space(192, 1.2)
    field = jnp.full(time_space.shape, 1.4 + 0.0j)
    ionization = MultiphotonIonizationRatePlan(
        0.08,
        2,
        provenance_id="declared-four-photon-power-rate",
    )
    drude = DrudePlasmaResponsePlan(
        0.7,
        provenance_id="declared-collisional-drude",
        electron_charge_magnitude=1.0,
        electron_mass=2.0,
    )
    potential = 0.6
    evaluation = (
        IonizingDrudeResponsePlan(
            ionization,
            drude,
            4.0,
            potential,
        )
        .prepare(
            time_space,
            _positive_mask(time_space.size),
            field.shape,
            temporal_axis=0,
        )
        .evaluate(field)
    )
    ledger = evaluation.ledger
    endpoint_electron = evaluation.physical_state[-1, 0]

    assert evaluation.evidence.initial_state_norm == 0.0
    assert evaluation.evidence.endpoint_state_norm > 0.0
    assert evaluation.evidence.finite_pulse_endpoint_residual > 0.0
    current_spectrum = jnp.fft.ifft(evaluation.analytic_free_current, norm="ortho")
    assert (
        jnp.max(jnp.abs(current_spectrum[jnp.fft.fftfreq(time_space.size) <= 0.0]))
        < 1.0e-11
    )
    assert jnp.allclose(
        ledger.ionization_potential_energy_density,
        potential * endpoint_electron,
    )
    assert ledger.collisional_energy_density > 0.0
    assert ledger.terminal_kinetic_energy_density > 0.0
    assert jnp.allclose(
        ledger.optical_work_density,
        ledger.polarization_work_density + ledger.free_current_work_density,
    )
    assert jnp.allclose(
        ledger.material_work_density,
        ledger.ionization_potential_energy_density
        + ledger.collisional_energy_density
        + ledger.raman_oscillator_energy_density
        + ledger.raman_dissipated_energy_density
        + ledger.terminal_polarization_energy_density
        + ledger.terminal_kinetic_energy_density,
    )
    assert jnp.allclose(
        ledger.energy_closure_defect,
        ledger.optical_work_density - ledger.material_work_density,
    )
    assert evaluation.successful
