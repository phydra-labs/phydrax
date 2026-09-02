#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.qualification._biophysics import (
    antiporter_electrochemical_balance,
    BOLTZMANN_CONSTANT_J_PER_K,
    ELEMENTARY_CHARGE_C,
    eyring_rate,
    GAS_CONSTANT_J_PER_MOL_K,
    nernst_equilibrium_potential,
    PLANCK_CONSTANT_J_S,
    qualify_censored_dwell_times,
    recover_brownian_transport,
    spherical_membrane_capacitance,
    spherical_membrane_ion_count,
)


def test_spherical_membrane_capacitance_and_ion_count_use_si_geometry():
    radius_m = 5.0e-6
    specific_capacitance_f_per_m2 = 1.0e-2
    potential_v = -70.0e-3
    expected_capacitance_f = 4.0 * np.pi * radius_m**2 * specific_capacitance_f_per_m2

    capacitance_f = spherical_membrane_capacitance(
        radius_m, specific_capacitance_f_per_m2
    )
    ion_count = spherical_membrane_ion_count(
        radius_m,
        specific_capacitance_f_per_m2,
        potential_v,
        ion_valence=1,
    )

    np.testing.assert_allclose(capacitance_f, expected_capacitance_f, rtol=2.0e-6)
    np.testing.assert_allclose(
        ion_count,
        abs(expected_capacitance_f * potential_v / ELEMENTARY_CHARGE_C),
        rtol=2.0e-6,
    )


def test_nernst_sign_valence_and_temperature_match_analytic_reference():
    inside_mol_per_m3 = 10.0
    outside_mol_per_m3 = 100.0
    temperature_k = 310.0
    cation = nernst_equilibrium_potential(
        inside_mol_per_m3, outside_mol_per_m3, 1, temperature_k
    )
    anion = nernst_equilibrium_potential(
        inside_mol_per_m3, outside_mol_per_m3, -1, temperature_k
    )
    colder = nernst_equilibrium_potential(inside_mol_per_m3, outside_mol_per_m3, 1, 155.0)

    assert float(cation) > 0.0
    assert float(anion) < 0.0
    np.testing.assert_allclose(anion, -cation)
    np.testing.assert_allclose(colder, 0.5 * cation)

    extreme = nernst_equilibrium_potential(1.0e-30, 1.0e30, 1.0, 310.0)
    expected_extreme = (
        GAS_CONSTANT_J_PER_MOL_K
        * 310.0
        / 96485.33212331001
        * (np.log(1.0e30) - np.log(1.0e-30))
    )
    assert np.isfinite(float(extreme))
    np.testing.assert_allclose(extreme, expected_extreme, rtol=2.0e-6)


def test_eyring_zero_barrier_and_finite_barrier_match_transition_state_theory():
    temperature_k = 300.0
    zero_barrier = eyring_rate(0.0, temperature_k)
    expected_prefactor = BOLTZMANN_CONSTANT_J_PER_K * temperature_k / PLANCK_CONSTANT_J_S
    np.testing.assert_allclose(zero_barrier, expected_prefactor, rtol=2.0e-6)

    barrier_j_per_mol = 50_000.0
    expected = expected_prefactor * np.exp(
        -barrier_j_per_mol / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
    )
    np.testing.assert_allclose(
        eyring_rate(barrier_j_per_mol, temperature_k), expected, rtol=2.0e-6
    )


def test_antiporter_balance_recovers_voltage_and_equal_stoichiometry_is_singular():
    inferred = antiporter_electrochemical_balance(
        10.0,
        100.0,
        20.0,
        5.0,
        1.0,
        1.0,
        2.0,
        1.0,
        300.0,
    )
    assert bool(inferred.finite)
    assert bool(inferred.identifiable)
    assert np.isfinite(float(inferred.equilibrium_potential_v))

    balanced = antiporter_electrochemical_balance(
        10.0,
        100.0,
        20.0,
        5.0,
        1.0,
        1.0,
        2.0,
        1.0,
        300.0,
        inferred.equilibrium_potential_v,
    )
    assert bool(balanced.balanced)

    singular = antiporter_electrochemical_balance(
        10.0,
        100.0,
        20.0,
        5.0,
        1.0,
        1.0,
        1.0,
        1.0,
        300.0,
    )
    assert bool(singular.finite)
    assert not bool(singular.identifiable)
    assert not bool(singular.successful)
    assert np.isnan(float(singular.equilibrium_potential_v))

    extreme_ratio = antiporter_electrochemical_balance(
        1.0e-30,
        1.0e30,
        1.0e30,
        1.0e-30,
        1.0,
        1.0,
        2.0,
        1.0,
        300.0,
    )
    assert bool(extreme_ratio.finite)
    assert bool(extreme_ratio.identifiable)
    assert np.isfinite(float(extreme_ratio.chemical_driving_energy_j_per_mol))
    assert np.isfinite(float(extreme_ratio.equilibrium_potential_v))

    overflow = antiporter_electrochemical_balance(
        1.0e-30,
        1.0e30,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0e308,
        1.0,
        300.0,
    )
    assert not bool(overflow.finite)
    assert not bool(overflow.successful)


def test_brownian_transport_recovery_matches_exact_increment_statistics():
    diffusion_m2_per_s = 3.0
    drift_m_per_s = 2.0
    time_step_s = 0.5
    increment_count = 4
    fluctuation = np.sqrt(
        2.0 * diffusion_m2_per_s * time_step_s * (increment_count - 1) / increment_count
    )
    increments_m = jnp.asarray(
        [
            drift_m_per_s * time_step_s + fluctuation,
            drift_m_per_s * time_step_s - fluctuation,
            drift_m_per_s * time_step_s + fluctuation,
            drift_m_per_s * time_step_s - fluctuation,
        ]
    )
    positions_m = jnp.concatenate((jnp.zeros(1), jnp.cumsum(increments_m)))[None, :, None]

    result = jax.jit(recover_brownian_transport)(positions_m, time_step_s)

    np.testing.assert_allclose(result.drift_velocity_m_per_s, [drift_m_per_s])
    np.testing.assert_allclose(result.diffusion_coefficient_m2_per_s, diffusion_m2_per_s)
    assert int(result.increment_count) == 4
    assert bool(result.successful)


def test_independent_censored_dwell_qualification_uses_survival_terms():
    result = jax.jit(qualify_censored_dwell_times)(
        jnp.asarray([1.0, 2.0, 3.0]),
        jnp.asarray([True, False, True]),
        0.5,
    )
    np.testing.assert_allclose(result.log_likelihood, 2.0 * np.log(0.5) - 3.0)
    np.testing.assert_allclose(result.maximum_likelihood_rate_per_s, 2.0 / 6.0)
    assert bool(result.successful)

    all_censored = qualify_censored_dwell_times(
        jnp.asarray([1.0, 2.0]), jnp.asarray([False, False]), 0.5
    )
    assert bool(all_censored.finite)
    assert not bool(all_censored.identifiable)
    assert np.isnan(float(all_censored.maximum_likelihood_rate_per_s))
