#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_macro_ded_step_commits_energy_and_activation() -> None:
    plan = phx.applications.additive_manufacturing.DEDProcessPlan.create(
        jnp.asarray(((0.0, 0.0), (0.01, 0.0))),
        jnp.asarray((1.0e-6, 1.0e-6)),
        jnp.asarray(((0.0, 2.0), (2.0, 0.0))),
        density_kg_m3=7800.0,
        heat_capacity_j_kg_k=500.0,
        convection_w_m2_k=10.0,
        exposed_area_m2=jnp.asarray((1.0e-4, 1.0e-4)),
        ambient_temperature_k=300.0,
        thermal_expansion_k_inv=1.0e-5,
        elastic_modulus_pa=200.0e9,
    )
    state = phx.applications.additive_manufacturing.DEDState(
        jnp.asarray((300.0, 300.0)),
        phx.manufacturing.MaterialActivationState(
            jnp.asarray((True, False)), jnp.asarray((0.0, -1.0))
        ),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    result = plan.step(
        state,
        phx.manufacturing.GaussianMovingSource(100.0, 0.5, 0.01),
        jnp.asarray((0.0, 0.0)),
        1.0e-3,
        activation_selection=jnp.asarray((False, True)),
        added_mass_kg_s=0.01,
    )

    assert bool(result.accepted_step)
    assert jnp.all(result.accepted.activation.active)
    assert result.accepted.supplied_energy_j > 0.0
    assert result.accepted.deposited_mass_kg > 0.0


def test_two_phase_electroviscoelastic_limit_is_finite() -> None:
    law = phx.rheology.ViscoelasticLaw("oldroyd-b", 1.0, 2.0)
    identity = jnp.eye(2)
    result = phx.applications.electroviscoelastic.evaluate_two_phase_electroviscoelastic(
        identity,
        identity,
        law,
        None,
        jnp.asarray((1.0, 0.0)),
        jnp.asarray((1.0, 0.0)),
        2.0,
        2.0,
        jnp.asarray((1.0, 0.0)),
        0.0,
        0.0,
        0.0,
        0.0,
    )

    assert jnp.allclose(result.polymer_stress_minus, 0.0)
    assert jnp.allclose(result.electric_traction_jump, 0.0)
    assert jnp.isclose(result.surface_charge_rate, 0.0)


def test_industrial_and_engineering_profiles_are_executable() -> None:
    heat = phx.applications.industrial_processes.goldak_double_ellipsoid_source(
        jnp.asarray(((0.0, 0.0, 0.0),)),
        jnp.asarray((0.0, 0.0, 0.0)),
        1000.0,
        0.8,
        0.01,
        0.02,
        0.005,
        0.004,
    )
    damage = phx.applications.engineering_systems.miner_damage(
        jnp.asarray((10.0, 20.0)), jnp.asarray((100.0, 200.0))
    )
    pressure = phx.applications.engineering_systems.black_oil_tank_pressure(
        20.0e6, 1.0e6, 0.1e6, 1.0e-9
    )

    assert heat[0] > 0.0
    assert jnp.isclose(damage, 0.2)
    assert pressure < 20.0e6
    assert jnp.isclose(
        phx.applications.engineering_systems.fire_heat_release_w(1.0), 13.1e6
    )
    assert jnp.isclose(
        phx.applications.engineering_systems.composite_longitudinal_modulus(
            200.0, 100.0, 0.5
        ),
        150.0,
    )
    assert jnp.isclose(
        phx.applications.engineering_systems.mineral_recovery(8.0, 10.0),
        0.8,
    )
