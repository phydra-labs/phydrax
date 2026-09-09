# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.semiconductor._interfaces import (
    interface_electrostatics,
    ThermionicInterface,
)
from phydrax.applications.semiconductor._quantities import BOLTZMANN_CONSTANT_SI as K


jax.config.update("jax_enable_x64", True)


def test_sheet_charge_and_dipole_have_material_sided_electrostatics():
    area, eps_l, eps_r, dl, dr = 1e-12, 2e-10, 1e-10, 2e-9, 3e-9
    charge, jump = 1e-18, 0.03
    state = interface_electrostatics(
        0.1,
        -0.1,
        eps_l,
        eps_r,
        dl,
        dr,
        area,
        sheet_charge=charge,
        potential_jump=jump,
    )
    assert bool(state.successful)
    np.testing.assert_allclose(
        state.potential_right - state.potential_left, jump, atol=1e-16
    )
    np.testing.assert_allclose(
        state.displacement_right - state.displacement_left, charge, atol=1e-32
    )
    # Each side obeys its own permittivity and field, not an averaged sheet load.
    np.testing.assert_allclose(
        state.displacement_left,
        -eps_l * area * (state.potential_left - 0.1) / dl,
        rtol=1e-14,
    )
    shifted = interface_electrostatics(
        2.1,
        1.9,
        eps_l,
        eps_r,
        dl,
        dr,
        area,
        sheet_charge=charge,
        potential_jump=jump,
    )
    np.testing.assert_allclose(shifted.field_energy, state.field_energy, rtol=2e-14)
    np.testing.assert_allclose(
        shifted.displacement_right, state.displacement_right, rtol=2e-14
    )


def _law():
    return ThermionicInterface(
        2e23,
        temperature_range=(200.0, 600.0),
        energy_reference="test intrinsic datum",
        provenance="Specified analytic Maxwell-Boltzmann transmitting-mode spectrum",
    )


def test_thermionic_equilibrium_has_finite_correct_linear_response():
    law = _law()
    temperature, area = 300.0, 1e-12
    barrier = 5 * K * temperature
    flux = lambda affinity: (
        law.evaluate(
            K * temperature * affinity, 0.0, temperature, temperature, barrier, area
        ).number_flux
    )
    expected = area * law.prefactor * temperature**2 * jnp.exp(-5.0)
    np.testing.assert_array_equal(flux(0.0), 0.0)
    np.testing.assert_allclose(jax.grad(flux)(0.0), expected, rtol=2e-14)
    np.testing.assert_allclose(jax.grad(jax.grad(flux))(0.0), expected, rtol=2e-14)
    tiny_affinity = 1e-12
    np.testing.assert_allclose(
        flux(tiny_affinity), expected * jnp.expm1(tiny_affinity), rtol=2e-14
    )


def test_thermionic_temperature_drive_closes_particle_energy_and_entropy():
    law = _law()
    tl, tr, area = 250.0, 450.0, 1e-12
    mu_l, mu_r, barrier = K * 300.0, -K * 300.0, 8 * K * 300.0
    result = law.evaluate(mu_l, mu_r, tl, tr, barrier, area)
    assert bool(result.successful)
    left = area * law.prefactor * tl**2 * jnp.exp((mu_l - barrier) / (K * tl))
    right = area * law.prefactor * tr**2 * jnp.exp((mu_r - barrier) / (K * tr))
    expected_energy = left * (barrier + 2 * K * tl) - right * (barrier + 2 * K * tr)
    np.testing.assert_allclose(result.number_flux, left - right, rtol=2e-14)
    np.testing.assert_allclose(result.energy_flux, expected_energy, rtol=2e-14)
    np.testing.assert_allclose(
        result.heat_left + result.heat_right,
        (mu_l - mu_r) * result.number_flux,
        rtol=2e-14,
    )
    assert float(result.entropy_production) > 0
    shifted = law.evaluate(mu_l + 1e-18, mu_r + 1e-18, tl, tr, barrier + 1e-18, area)
    np.testing.assert_allclose(shifted.number_flux, result.number_flux, rtol=5e-14)
    np.testing.assert_allclose(shifted.heat_left, result.heat_left, rtol=5e-14)
    np.testing.assert_allclose(
        shifted.energy_flux - result.energy_flux, 1e-18 * result.number_flux, rtol=5e-14
    )


def test_interface_domain_failures_remain_explicit():
    result = _law().evaluate(0.0, 0.0, 300.0, 300.0, 0.0, 1e-12)
    assert not bool(
        result.successful
    )  # A degenerate reservoir is not admitted MB emission.
    geometric = interface_electrostatics(0.0, 1.0, 1e-10, 1e-10, 0.0, 1e-9, 1e-12)
    assert not bool(geometric.successful)
