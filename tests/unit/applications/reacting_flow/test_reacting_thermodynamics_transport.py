#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.reacting_flow._state import ReactiveConservedLayout
from phydrax.applications.reacting_flow._thermodynamics import ReactingGasModel
from phydrax.applications.reacting_flow._transport import (
    MixtureAveragedTransportPlan,
    StefanMaxwellTransportPlan,
)
from phydrax.equations._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from phydrax.equations._chemical_thermodynamics import (
    PolynomialSpeciesThermodynamicsPlan,
)


def _gas_model():
    schema = ChemicalSpeciesSchema(
        ("light", "middle", "heavy"),
        (ChemicalPhaseKind.GAS,) * 3,
        jnp.asarray((0.002, 0.016, 0.032)),
        ("E",),
        jnp.asarray(((1, 1, 1),), dtype=jnp.int32),
        jnp.asarray((0, 0, 0), dtype=jnp.int32),
    )
    thermodynamics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 24.0, 28.0)),
        jnp.asarray((0.0, 2.0e4, -1.0e5)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=3000.0,
    )
    return ReactingGasModel(
        schema,
        thermodynamics,
        formation_molar_enthalpies=jnp.asarray((1.0e3, -2.0e3, 3.0e3)),
    )


def _transport(plan_type):
    model = _gas_model()
    diffusion = jnp.asarray(
        (
            (0.0, 1.0e-5, 1.2e-5),
            (1.0e-5, 0.0, 0.8e-5),
            (1.2e-5, 0.8e-5, 0.0),
        )
    )
    return plan_type(
        model,
        diffusion,
        jnp.asarray((1.0e-5, 1.3e-5, 1.7e-5)),
        jnp.asarray((0.02, 0.025, 0.03)),
    )


def test_ideal_mixture_eos_energy_inversion_and_implicit_jvp():
    model = _gas_model()
    mass = jnp.asarray((0.2, 0.3, 0.5))
    state = model.evaluate_pressure(jnp.asarray(1100.0), jnp.asarray(2.0e5), mass)
    inversion = model.temperature_from_internal_energy(
        state.specific_internal_energy, mass
    )
    _, energy_tangent = jax.jvp(
        lambda energy: model.temperature_from_internal_energy(energy, mass).temperature,
        (state.specific_internal_energy,),
        (jnp.asarray(1.0),),
    )

    assert state.successful
    assert inversion.successful
    np.testing.assert_allclose(inversion.temperature, 1100.0, rtol=1.0e-11)
    np.testing.assert_allclose(
        energy_tangent,
        1.0 / state.specific_heat_capacity_volume,
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        state.pressure,
        state.density * state.gas_constant * state.temperature,
        rtol=1.0e-12,
    )


def test_s_minus_one_species_closure_is_exact_and_failed_cells_reject():
    model = _gas_model()
    layout = ReactiveConservedLayout(model, 2)
    mass = jnp.asarray((0.2, 0.3, 0.5))
    conserved = layout.from_thermodynamic_state(
        jnp.asarray(1.2),
        jnp.asarray((4.0, -2.0)),
        jnp.asarray(900.0),
        mass,
    )
    fields = layout.split(conserved)

    np.testing.assert_allclose(jnp.sum(fields.species_density), fields.density)
    np.testing.assert_allclose(fields.mass_fractions, mass)
    assert layout.evidence(conserved).successful

    invalid = conserved.at[1].set(0.9).at[2].set(0.5)
    evidence = layout.evidence(invalid)
    assert not evidence.species_positive
    assert not evidence.successful


def test_mixture_averaged_transport_has_zero_net_mass_flux_and_inert_limit():
    plan = _transport(MixtureAveragedTransportPlan)
    mass = jnp.asarray((0.2, 0.3, 0.5))
    gradient = jnp.asarray(((0.05, -0.03), (-0.02, 0.01), (-0.03, 0.02)))
    result = plan.evaluate(
        jnp.asarray(1000.0),
        jnp.asarray(101325.0),
        jnp.asarray(0.8),
        mass,
        gradient,
        temperature_gradient=jnp.asarray((10.0, -4.0)),
    )
    inert = plan.evaluate(
        jnp.asarray(1000.0),
        jnp.asarray(101325.0),
        jnp.asarray(0.8),
        mass,
        jnp.zeros_like(gradient),
    )

    assert result.successful
    np.testing.assert_allclose(result.net_mass_flux, 0.0, atol=1.0e-18)
    np.testing.assert_allclose(inert.species_mass_flux, 0.0, atol=0.0)
    np.testing.assert_allclose(inert.total_heat_flux, 0.0, atol=0.0)


def test_stefan_maxwell_matches_its_reference_system_and_mass_constraint():
    plan = _transport(StefanMaxwellTransportPlan)
    mass = jnp.asarray((0.2, 0.3, 0.5))
    gradient = jnp.asarray(((0.05,), (-0.02,), (-0.03,)))
    result = plan.evaluate(
        jnp.asarray(1000.0),
        jnp.asarray(101325.0),
        jnp.asarray(0.8),
        mass,
        gradient,
    )
    recovered_rhs = result.evidence.system_matrix @ result.diffusion_velocities

    assert plan.support_tier == "research"
    assert result.successful
    np.testing.assert_allclose(
        recovered_rhs, result.evidence.right_hand_side, rtol=1.0e-10, atol=1.0e-12
    )
    np.testing.assert_allclose(result.net_mass_flux, 0.0, atol=1.0e-18)
    np.testing.assert_allclose(
        jnp.sum(mass[:, None] * result.diffusion_velocities, axis=0),
        0.0,
        atol=1.0e-15,
    )
