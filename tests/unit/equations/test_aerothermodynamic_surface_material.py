import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _surface_mechanism():
    gas = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A",),
        (phx.equations.ChemicalPhaseKind.GAS,),
        jnp.asarray((0.01,)),
        ("A",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    surface = phx.equations.SurfaceSpeciesSchema(
        ("vacant", "A*"),
        jnp.asarray(((0.0, 1.0),)),
        jnp.asarray((0.0, 0.0)),
        jnp.asarray((1.0, 1.0)),
        site_capacity=1.0,
    )
    reaction = phx.equations.GasSurfaceReactionSpec(
        "adsorption",
        phx.equations.ArrheniusRatePlan(1.0),
        jnp.asarray((-1.0,)),
        jnp.asarray((-1.0, 1.0)),
        reaction_heat=-2.0e4,
    )
    return phx.equations.PreparedGasSurfaceMechanism(gas, surface, (reaction,))


def test_surface_chemistry_preserves_sites_elements_and_charge():
    mechanism = _surface_mechanism()
    state = phx.equations.SurfaceChemicalState(
        jnp.asarray((0.8, 0.2)),
        jnp.asarray(500.0),
        jnp.asarray(0.0),
        jnp.asarray(True),
    )
    result = mechanism.evaluate(jnp.asarray((2.0,)), state, 1.0e5)

    assert bool(result.successful)
    np.testing.assert_allclose(result.site_defect, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.element_defect, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.charge_defect, 0.0, atol=1.0e-12)
    assert result.gas_amount_flux[0] < 0.0
    assert result.surface_amount_rate[1] > 0.0


def _material():
    return phx.equations.PorousAblatingMaterialPlan(
        jnp.asarray((1000.0, 800.0)),
        jnp.asarray((0.2, 0.1)),
        jnp.asarray((0.0, 0.0)),
        jnp.asarray((1.0e-3,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
        jnp.asarray(((0.0, 0.5),)),
        jnp.asarray(((50.0,),)),
        jnp.asarray((0.01,)),
        jnp.asarray((0.0,)),
        virgin_porosity=0.1,
        char_porosity=0.5,
        reference_permeability=1.0e-12,
    )


def test_porous_material_decomposition_conserves_closed_mass():
    material = _material()
    state = phx.equations.AblatingMaterialState(
        jnp.asarray((1.0, 0.0)),
        jnp.asarray((0.0,)),
        jnp.asarray(2.0e5),
        jnp.asarray(0.1),
        jnp.asarray(True),
    )
    result = material.advance(state, 0.1, subcycles=8)
    mass_before = jnp.sum(state.solid_component_densities) + jnp.sum(
        state.pore_gas_molar_densities * material.pore_gas_molar_masses
    )
    mass_after = jnp.sum(result.accepted.solid_component_densities) + jnp.sum(
        result.accepted.pore_gas_molar_densities * material.pore_gas_molar_masses
    )

    assert bool(result.successful)
    assert result.accepted.solid_component_densities[0] < 1.0
    assert result.accepted.solid_component_densities[1] > 0.0
    assert result.accepted.pore_gas_molar_densities[0] > 0.0
    np.testing.assert_allclose(mass_after, mass_before, rtol=2.0e-7)


def test_conjugate_exchange_and_recession_remap_are_conservative():
    interface = phx.solver.ConjugateAerothermalInterfacePlan(
        jnp.eye(2),
        jnp.eye(2),
        jnp.asarray((0.5, 0.5)),
        jnp.eye(2),
        jnp.asarray((1.0, 1.0)),
    )
    exchange = interface.exchange(
        jnp.asarray((10.0, 20.0)),
        jnp.asarray((10.0, 20.0)),
        jnp.asarray(((0.1,), (0.2,))),
    )
    assert bool(exchange.successful)
    np.testing.assert_allclose(exchange.interface_energy_defect, 0.0, atol=0.0)

    remap = phx.solver.ConservativeRecessionRemapPlan(
        jnp.asarray(((0.75, 0.0), (0.25, 1.0))),
        jnp.asarray(((0.5, 0.0), (0.5, 1.0))),
    )
    result = remap.apply(
        jnp.asarray(((1.0, 2.0), (3.0, 4.0))),
        jnp.asarray(((5.0,), (7.0,))),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.gas_conservation_defect, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.material_conservation_defect, 0.0, atol=1.0e-12)
