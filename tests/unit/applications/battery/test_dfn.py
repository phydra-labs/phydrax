#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _parameters():
    return phx.applications.battery.DFNParameters(
        negative_length_m=1.0e-4,
        separator_length_m=2.5e-5,
        positive_length_m=1.0e-4,
        negative_porosity=0.3,
        separator_porosity=0.5,
        positive_porosity=0.3,
        negative_surface_area_m2_m3=1.0e5,
        positive_surface_area_m2_m3=1.0e5,
        negative_solid_conductivity_s_m=100.0,
        positive_solid_conductivity_s_m=20.0,
        electrolyte_conductivity_s_m=1.0,
        electrolyte_diffusivity_m2_s=2.0e-10,
        negative_solid_diffusivity_m2_s=1.0e-14,
        positive_solid_diffusivity_m2_s=1.0e-14,
        negative_particle_radius_m=1.0e-5,
        positive_particle_radius_m=1.0e-5,
        negative_max_concentration_mol_m3=3.0e4,
        positive_max_concentration_mol_m3=5.0e4,
        negative_exchange_current_a_m2=1.0e-3,
        positive_exchange_current_a_m2=1.0e-3,
        transference_number=0.4,
        temperature_k=298.15,
        negative_ocp=lambda theta: 0.1 + 0.8 * theta,
        positive_ocp=lambda theta: 4.2 - 0.8 * theta,
    )


def _state(plan, parameters):
    return plan.initial_state(
        parameters,
        electrolyte_concentration_mol_m3=1000.0,
        negative_stoichiometry=0.8,
        positive_stoichiometry=0.4,
    )


def test_zero_current_dfn_recovers_open_circuit_equilibrium() -> None:
    plan = phx.applications.battery.IsothermalDFNPlan(3, 2, 3, 4)
    parameters = _parameters()
    state = _state(plan, parameters)

    evaluation = plan.evaluate(state, parameters, 0.0)
    step = plan.step(state, parameters, 0.0, 1.0)

    expected = parameters.positive_ocp(jnp.asarray(0.4)) - parameters.negative_ocp(
        jnp.asarray(0.8)
    )
    assert bool(evaluation.converged)
    assert evaluation.residual_norm < 1.0e-8
    assert jnp.allclose(evaluation.voltage_v, expected, atol=1.0e-9)
    assert bool(step.accepted)
    assert jnp.allclose(
        step.accepted_state.electrolyte_concentration_mol_m3,
        state.electrolyte_concentration_mol_m3,
    )


def test_dfn_current_step_is_finite_and_mass_directions_are_physical() -> None:
    plan = phx.applications.battery.IsothermalDFNPlan(3, 2, 3, 4)
    parameters = _parameters()
    state = _state(plan, parameters)

    result = plan.step(state, parameters, 0.1, 1.0e-3)

    assert bool(result.accepted)
    assert jnp.isfinite(result.evaluation.voltage_v)
    assert jnp.mean(
        result.accepted_state.negative_particle_concentration_mol_m3
    ) < jnp.mean(state.negative_particle_concentration_mol_m3)
    assert jnp.mean(
        result.accepted_state.positive_particle_concentration_mol_m3
    ) > jnp.mean(state.positive_particle_concentration_mol_m3)


def test_series_pack_commits_cells_atomically_and_sums_voltage() -> None:
    cell = phx.applications.battery.IsothermalDFNPlan(2, 2, 2, 3)
    parameters = _parameters()
    state = _state(cell, parameters)
    pack = phx.applications.battery.SeriesBatteryPackPlan(
        (cell, cell), (parameters, parameters)
    )

    result = pack.step(
        phx.applications.battery.SeriesBatteryPackState((state, state)),
        0.0,
        1.0,
    )

    assert bool(result.accepted)
    assert jnp.allclose(
        result.pack_voltage_v,
        2.0 * result.cell_results[0].evaluation.voltage_v,
    )
    assert all(cell_state.time_s == 1.0 for cell_state in result.state.cell_states)


def test_spatial_dfn_lanes_commit_atomically_and_close_current_distribution():
    local = phx.applications.battery.IsothermalDFNPlan(2, 2, 2, 3)
    spatial = phx.applications.battery.SpatialBatteryCellPlan(
        local,
        _parameters(),
        jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            )
        ),
        current_weights=jnp.asarray((0.25, 0.75)),
    )
    state = spatial.initialize(
        electrolyte_concentration_mol_m3=1000.0,
        negative_stoichiometry=0.8,
        positive_stoichiometry=0.4,
    )
    result = spatial.step(state, 0.0, 1.0e-3)

    assert bool(result.evidence.successful)
    assert jnp.allclose(result.evidence.current_balance_defect_a_m2, 0.0)
    assert jnp.allclose(result.evidence.voltage_spread_v, 0.0)
    assert jnp.all(result.accepted_state.time_s == 1.0e-3)
