#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _gas_schema():
    return phx.equations.ChemicalSpeciesSchema(
        ("A",),
        (phx.equations.ChemicalPhaseKind.GAS,),
        jnp.asarray((0.01,)),
        ("X",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
    )


def test_nasa7_constant_heat_capacity_identities_and_derivative():
    schema = _gas_schema()
    coefficients = jnp.asarray([[[3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    plan = phx.equations.NASASpeciesThermodynamicsPlan(
        schema,
        phx.equations.NASAPolynomialKind.NASA7,
        coefficients,
        jnp.asarray(((200.0,),)),
        jnp.asarray(((6000.0,),)),
    )
    temperature = jnp.asarray(1000.0)
    fields = plan.evaluate(temperature)
    gas_constant = 8.31446261815324

    np.testing.assert_allclose(fields.molar_heat_capacity_pressure, 3.5 * gas_constant)
    np.testing.assert_allclose(fields.molar_heat_capacity_volume, 2.5 * gas_constant)
    np.testing.assert_allclose(
        fields.molar_internal_energy,
        2.5 * gas_constant * temperature,
    )
    derivative = jax.grad(lambda value: plan.evaluate(value).molar_enthalpy[0])(
        temperature
    )
    np.testing.assert_allclose(
        derivative,
        fields.molar_heat_capacity_pressure[0],
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        fields.molar_gibbs_energy,
        fields.molar_enthalpy - temperature * fields.molar_entropy,
    )
    assert fields.successful
    assert not plan.evaluate(jnp.asarray(100.0)).successful


def test_polynomial_thermodynamics_and_particle_energy_inversion():
    schema = phx.equations.ChemicalSpeciesSchema(
        ("solid",),
        (phx.equations.ChemicalPhaseKind.SOLID,),
        jnp.asarray((0.1,)),
        ("X",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((10.0, 0.01),)),
        jnp.asarray((5.0,)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=1000.0,
    )
    material = phx.equations.ParticleThermodynamicMaterialPlan(species)
    amount = jnp.asarray([[2.0]])
    target_temperature = jnp.asarray([550.0])
    energy = material.energy_from_temperature(target_temperature, amount)
    state = material.state(
        energy,
        amount,
        jnp.asarray([1.0]),
        jnp.asarray([0.5]),
    )

    np.testing.assert_allclose(state.temperature, target_temperature, rtol=1e-10)
    np.testing.assert_allclose(state.energy_residual, 0.0, atol=1e-8)
    assert state.successful
