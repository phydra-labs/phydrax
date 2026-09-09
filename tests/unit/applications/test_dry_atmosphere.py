#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.atmosphere._dry import (
    DryAir,
    DryAtmospherePlan,
    DryHydrostaticReference,
)


@pytest.mark.parametrize("family,shape", [("isothermal", (6,)), ("isentropic", (4, 6))])
def test_hydrostatic_reference_remains_at_rest_to_roundoff(family, shape):
    bounds = ((0.0,), (6000.0,)) if len(shape) == 1 else ((0.0, 0.0), (4000.0, 6000.0))
    prepared = DryAtmospherePlan(
        shape, bounds, reference=DryHydrostaticReference(family)
    ).prepare()
    initial = prepared.initial_state()
    residual = eqx.filter_jit(prepared.balance.evaluate)(initial.conserved)
    np.testing.assert_allclose(residual.residual, 0.0, atol=3e-10)
    dt = 0.4 * eqx.filter_jit(prepared.stable_step)(initial)
    result = eqx.filter_jit(prepared.rollout)(initial, jnp.full((3,), dt))
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.state.conserved, initial.conserved, rtol=2e-13, atol=2e-10
    )
    scale = jnp.maximum(
        jnp.maximum(
            jnp.abs(initial.initial_integral), jnp.abs(result.budget.source_integral)
        ),
        1.0,
    )
    np.testing.assert_allclose(result.budget.closure / scale, 0.0, atol=2e-10)


def test_shared_mass_flux_closes_gravitational_energy_and_species_budgets():
    prepared = DryAtmospherePlan((4, 6), ((0.0, 0.0), (4000.0, 6000.0))).prepare()
    coordinates = prepared.balance.discretization.cell_centers
    warm = (
        0.5
        * jnp.sin(jnp.pi * coordinates[..., 1] / 6000.0)
        * jnp.cos(2.0 * jnp.pi * coordinates[..., 0] / 4000.0)
    )
    initial = prepared.thermal_state(warm)
    residual = eqx.filter_jit(prepared.balance.evaluate)(initial.conserved)
    # The local gas-work term must exactly cancel potential-energy transport
    # after summation, not merely approximate rho*u*g at this perturbed state.
    np.testing.assert_allclose(
        np.asarray(residual.conservation_defect)[[0, 1, 2, 5]], 0.0, atol=2e-7
    )
    dt = 0.3 * eqx.filter_jit(prepared.stable_step)(initial)
    result = eqx.filter_jit(prepared.rollout)(initial, jnp.full((2,), dt))
    assert bool(result.successful)
    closure = result.budget.closure
    np.testing.assert_allclose(closure[:3] / result.budget.species_mass, 0.0, atol=5e-14)
    np.testing.assert_allclose(closure[-1] / result.budget.total_energy, 0.0, atol=5e-14)
    assert float(jnp.max(jnp.abs(result.state.conserved[..., -2]))) > 0.0


def test_prescribed_outflow_energy_uses_boundary_potential_and_restart_is_exact():
    prepared = DryAtmospherePlan(
        (6,), ((0.0,), (3000.0,)), boundaries=(("prescribed", "prescribed"),)
    ).prepare()
    initial = prepared.thermal_state(jnp.zeros((6,)), velocity=jnp.asarray((0.2,)))
    dt = 0.2 * eqx.filter_jit(prepared.stable_step)(initial)
    advance = eqx.filter_jit(prepared.advance)
    first = advance(initial, dt)
    assert bool(first.accepted)
    assert float(jnp.abs(first.budget.boundary_integral[-1])) > 0.0
    np.testing.assert_allclose(
        first.budget.closure[-1] / first.budget.total_energy, 0.0, atol=5e-14
    )
    continued = advance(first.state, dt)
    restored = advance(prepared.restore(prepared.checkpoint(first.state)), dt)
    np.testing.assert_array_equal(restored.state.conserved, continued.state.conserved)
    np.testing.assert_array_equal(
        restored.state.boundary_integral, continued.state.boundary_integral
    )
    np.testing.assert_array_equal(
        restored.state.source_integral, continued.state.source_integral
    )
    assert float(restored.state.time) == float(continued.state.time)
    other = DryAtmospherePlan(
        (6,), ((0.0,), (3000.0,)), reference=DryHydrostaticReference(gravity=9.7)
    ).prepare()
    with pytest.raises(ValueError, match="identity"):
        other.restore(prepared.checkpoint(first.state))


def test_invalid_and_unstable_states_are_not_clipped_or_partially_accepted():
    prepared = DryAtmospherePlan((4,), ((0.0,), (2000.0,))).prepare()
    initial = prepared.initial_state()
    with pytest.raises(Exception, match="inadmissible"):
        prepared.initial_state(initial.conserved * 1e-12)
    dt = eqx.filter_jit(prepared.stable_step)(initial)
    result = eqx.filter_jit(prepared.advance)(initial, 2.0 * dt)
    assert not bool(result.accepted)
    np.testing.assert_array_equal(result.state.conserved, initial.conserved)
    np.testing.assert_array_equal(
        result.state.boundary_integral, initial.boundary_integral
    )
    assert int(result.state.accepted_steps) == 0
    assert float(result.state.time) == 0.0


def test_reference_pressure_drop_agrees_with_integrated_column_weight():
    air = DryAir()
    prepared = DryAtmospherePlan(
        (8,),
        ((0.0,), (8000.0,)),
        air=air,
        reference=DryHydrostaticReference("isentropic"),
    ).prepare()
    pressure = prepared.system.pressure(prepared.balance.face_reference[0])
    mass = jnp.sum(
        prepared.system.density(prepared.balance.reference)
        * prepared.balance.discretization.cell_volumes
    )
    np.testing.assert_allclose(
        pressure[0] - pressure[-1], prepared.plan.reference.gravity * mass, rtol=2e-12
    )
    recovered = prepared.system.recover_thermodynamics(prepared.balance.reference).state
    np.testing.assert_allclose(
        recovered.molar_heat_capacity_pressure / recovered.molar_mass,
        air.heat_capacity_pressure,
        rtol=2e-12,
    )


def test_vertical_periodicity_is_rejected_in_nonperiodic_gravity():
    with pytest.raises(ValueError, match="vertical gravity"):
        DryAtmospherePlan(
            (4,), ((0.0,), (2000.0,)), boundaries=(("periodic", "periodic"),)
        )
