#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


R = phx.equations.UNIVERSAL_GAS_CONSTANT


def _calorically_perfect_model():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.028, 0.028)),
        ("X",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((2.5 * R,), (2.5 * R,))),
        jnp.zeros((2,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=4000.0,
    )
    return phx.equations.HomogeneousHelmholtzPlan(
        phx.equations.IdealGasReferenceHelmholtzTerm(schema, species),
        phx.equations.ZeroResidualHelmholtzTerm(schema),
    )


def test_equilibrium_shock_closes_rankine_hugoniot_and_perfect_gas_limit():
    equilibrium = phx.solver.ChemicalEquilibriumPlan(
        _calorically_perfect_model(), phx.solver.ChemicalEquilibriumEnsemble.TP
    )
    plan = phx.applications.reacting_flow.EquilibriumShockPlan(
        equilibrium, tolerance=5.0e-7, maximum_steps=8
    )
    gamma = 1.4
    temperature = 300.0
    pressure = 1.0e5
    molar_mass = 0.028
    sound_speed = np.sqrt(gamma * R * temperature / molar_mass)
    velocity = 2.0 * sound_speed
    pressure_ratio = 4.5
    density_ratio = 8.0 / 3.0
    temperature_ratio = pressure_ratio / density_ratio

    result = plan.solve(
        temperature,
        pressure,
        jnp.asarray((0.5, 0.5)),
        velocity,
        downstream_temperature_guess=temperature * temperature_ratio,
        downstream_pressure_guess=pressure * pressure_ratio,
    )

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(
        result.downstream_pressure / pressure, pressure_ratio, rtol=2.0e-5
    )
    np.testing.assert_allclose(
        result.downstream_temperature / temperature,
        temperature_ratio,
        rtol=2.0e-5,
    )
    np.testing.assert_allclose(result.evidence.mass_residual, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(result.evidence.momentum_residual, 0.0, atol=0.1)
    np.testing.assert_allclose(result.evidence.energy_residual, 0.0, atol=0.1)
