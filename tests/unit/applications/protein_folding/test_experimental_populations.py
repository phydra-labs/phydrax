# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.protein_folding.experiments import (
    celsius_to_kelvin,
    dimer_log_populations,
    DimerThreeStateUnfolding,
    ParallelPathKinetics,
    repeat_transfer_statistics,
    thermal_unfolding_free_energy,
    ThermodynamicConvention,
    ThreeStateUnfolding,
    two_state_log_populations,
)
from phydrax.units import JOULE, KILOJOULE_PER_MOLE


def test_unfolding_sign_thermal_derivative_and_energy_basis():
    convention = ThermodynamicConvention()
    kt = convention.thermal_constant * 298.15
    populations = jax.jit(lambda g: jnp.exp(two_state_log_populations(g, kt)))(
        jnp.array([-1e5, 0.0, 1e5])
    )
    np.testing.assert_allclose(populations, [[0, 1], [0.5, 0.5], [1, 0]], atol=1e-12)
    parameters = jnp.array([12.0, 170.0, 1.2, 0.004, 1e-5])
    free_energy = lambda t: thermal_unfolding_free_energy(parameters, t, 0.0, 298.15)
    np.testing.assert_allclose(free_energy(298.15), 12.0)
    np.testing.assert_allclose(-jax.grad(free_energy)(298.15), (170 - 12) / 298.15)
    np.testing.assert_allclose(-298.15 * jax.grad(jax.grad(free_energy))(298.15), 1.2)
    single = ThermodynamicConvention(energy_unit=JOULE, basis="single-system")
    np.testing.assert_allclose(
        two_state_log_populations(12.0, kt),
        two_state_log_populations(
            12000 / 6.02214076e23, single.thermal_constant * 298.15
        ),
    )
    np.testing.assert_allclose(
        celsius_to_kelvin(jnp.array([0.0, 100.0])), [273.15, 373.15]
    )
    with pytest.raises(ValueError):
        ThermodynamicConvention(energy_unit=JOULE, basis="molar")
    with pytest.raises(ValueError):
        ThermodynamicConvention(energy_unit=KILOJOULE_PER_MOLE, basis="single-system")


def test_dimer_mass_action_infinite_dilution_and_extreme_gradients():
    kt, c0, dg = 2.5, 1000.0, 18.0
    concentration = jnp.logspace(-12, 6, 40)
    fractions = jnp.exp(dimer_log_populations(dg, kt, concentration, c0))
    np.testing.assert_allclose(fractions.sum(axis=-1), 1.0, atol=1e-12)
    monomer = fractions[:, 1] * concentration
    dimer = fractions[:, 0] * concentration / 2
    np.testing.assert_allclose(monomer**2 / dimer, c0 * np.exp(-dg / kt), rtol=1e-11)
    np.testing.assert_allclose(jnp.exp(dimer_log_populations(dg, kt, 0.0, c0)), [0, 1])
    assert np.isnan(np.asarray(dimer_log_populations(dg, kt, -1.0, c0))).all()
    assert fractions[0, 1] > fractions[-1, 1]
    gradient = jax.jit(
        jax.jacrev(lambda g: jnp.exp(dimer_log_populations(g, kt, 1e-9, c0)))
    )
    assert np.isfinite(np.asarray(gradient(-2000.0))).all()
    assert np.isfinite(np.asarray(gradient(2000.0))).all()


def test_consecutive_three_state_and_dimer_monomer_partition():
    convention = ThermodynamicConvention()
    t, d, c = jnp.array([298.15]), jnp.array([0.0]), jnp.array([0.25])
    model = ThreeStateUnfolding(convention)
    p = jnp.zeros(10)
    np.testing.assert_allclose(model.populations(p, t, d, c), [[1 / 3, 1 / 3, 1 / 3]])
    p = p.at[0].set(12.0).at[5].set(-4.0)
    actual = model.populations(p, t, d, c)[0]
    reference = np.exp(-np.array([0, 12, 8]) / (convention.thermal_constant * t[0]))
    np.testing.assert_allclose(actual, reference / reference.sum())
    dimer = DimerThreeStateUnfolding(convention)
    actual = dimer.populations(p, t, d, c)[0]
    np.testing.assert_allclose(actual.sum(), 1.0)
    np.testing.assert_allclose(
        actual[2] / actual[1], np.exp(4 / (convention.thermal_constant * t[0]))
    )
    concentration_i = actual[1] * c[0]
    concentration_n2 = actual[0] * c[0] / 2
    np.testing.assert_allclose(
        concentration_i**2 / concentration_n2,
        convention.standard_concentration
        * np.exp(-12 / (convention.thermal_constant * t[0])),
    )


def test_repeat_transfer_matches_enumeration_and_marginal_derivative():
    g, bonds, kt = jnp.array([-1.0, 2.0, -0.3]), jnp.array([-1.2, 0.5]), 2.5
    states = np.array(tuple(product((0.0, 1.0), repeat=3)))
    energy = states @ np.asarray(g) + (states[:, :-1] * states[:, 1:]) @ np.asarray(bonds)
    weights = np.exp(-energy / kt)
    log_z, marginal = jax.jit(repeat_transfer_statistics)(g, bonds, kt)
    np.testing.assert_allclose(log_z, np.log(weights.sum()), rtol=1e-12)
    np.testing.assert_allclose(
        marginal, (weights[:, None] * states).sum(axis=0) / weights.sum(), rtol=1e-12
    )
    derivative = jax.grad(lambda nodes: repeat_transfer_statistics(nodes, bonds, kt)[0])(
        g
    )
    np.testing.assert_allclose(derivative, -marginal / kt, rtol=1e-12)
    singleton = repeat_transfer_statistics(jnp.array([-1.0]), jnp.empty(0), kt)
    np.testing.assert_allclose(singleton[1], [1 / (1 + np.exp(-1 / kt))])


def test_parallel_paths_share_equilibrium_and_resolve_total_relaxation():
    model = ParallelPathKinetics()
    t, d = jnp.full(17, 298.15), jnp.linspace(0, 6000, 17)
    p = jnp.array([12.0, 0.004, 3.0, 0.001, 1.0, 0.002])
    rates = model.log_rates(p, t, d)
    np.testing.assert_allclose(
        rates[:, 0] - rates[:, 1],
        (12 - 0.004 * d) / (model.convention.thermal_constant * t),
    )
    np.testing.assert_allclose(
        jnp.exp(model.predict_log_rate(p, t, d)), jnp.exp(rates).sum(axis=-1)
    )
    swapped = p[jnp.array([0, 1, 4, 5, 2, 3])]
    np.testing.assert_allclose(
        model.predict_log_rate(p, t, d), model.predict_log_rate(swapped, t, d)
    )
