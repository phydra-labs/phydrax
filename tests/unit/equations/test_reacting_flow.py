#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _reacting_system():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("H2", "O2", "H2O"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 3,
        jnp.asarray((2.0e-3, 32.0e-3, 18.0e-3)),
        ("H", "O"),
        jnp.asarray(((2, 0, 2), (0, 2, 1)), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((20.0,), (22.0,), (25.0,))),
        jnp.asarray((0.0, 0.0, -2.4e5)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=4000.0,
    )
    ideal = phx.equations.IdealGasReferenceHelmholtzTerm(schema, thermodynamics)
    model = phx.equations.HomogeneousHelmholtzPlan(
        ideal, phx.equations.ZeroResidualHelmholtzTerm(schema)
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "hydrogen-combination",
        schema,
        thermodynamics,
        (
            phx.equations.ChemicalReactionSpec(
                "hydrogen-combination",
                {"H2": 2.0, "O2": 1.0},
                {"H2O": 2.0},
                phx.equations.ArrheniusRatePlan(2.0),
            ),
        ),
    ).prepare()
    return phx.equations.HomogeneousMixtureEulerSystem(model), mechanism


def test_reacting_mixture_uses_full_species_and_full_chemical_energy():
    system, mechanism = _reacting_system()
    primitive = jnp.asarray((0.04, 0.32, 0.0, 0.0, 1200.0))
    state = system.primitive_to_conserved(primitive)
    recovered = system.conserved_to_primitive(state)
    concentration = state[: system.species_count] / mechanism.schema.molar_masses
    rate = mechanism.evaluate(
        concentration,
        recovered[-1],
        system.pressure(state),
    )
    mass_rate = rate.species_amount_rate * mechanism.schema.molar_masses
    candidate = state.at[: system.species_count].add(1.0e-5 * mass_rate)

    np.testing.assert_allclose(recovered, primitive, rtol=2.0e-10)
    np.testing.assert_allclose(jnp.sum(mass_rate), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(candidate[-1], state[-1], atol=0.0)
    assert candidate[2] > state[2]
    assert bool(system.admissible(candidate))
    assert bool(rate.successful)


def test_homogeneous_mixture_flux_and_bounds_are_finite():
    system, _ = _reacting_system()
    primitive = jnp.asarray((0.04, 0.32, 0.18, 2.0, 900.0))
    state = system.primitive_to_conserved(primitive)

    flux = system.physical_flux(state, 0)
    lower, upper = system.signal_bounds(state, state, 0)

    assert flux.shape == state.shape
    assert bool(jnp.all(jnp.isfinite(flux)))
    assert lower < upper
