import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_advanced_foundations_scenario_1() -> None:
    system = phx.frequency.CompiledFrequencySystem.create(
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.2,),)),
        jnp.asarray(((4.0,),)),
    )
    result = system.solve(jnp.asarray((0.0, 1.0)), jnp.asarray((1.0,)))
    assert jnp.all(result.successful)
    assert jnp.allclose(
        result.response[:, 0],
        jnp.asarray((0.25, 1.0 / (3.0 + 0.2j))),
    )
    assert jnp.all(result.residual_norm < 1e-6)
    connector_type = phx.system_modeling.ConnectorType.create(
        "electrical",
        (
            phx.system_modeling.ConnectorVariable("voltage", "across", "V"),
            phx.system_modeling.ConnectorVariable("current", "through", "A"),
        ),
    )
    source = phx.system_modeling.Connector("source", connector_type)
    load = phx.system_modeling.Connector("load", connector_type)
    system = phx.system_modeling.AcausalSystem.create(
        (source, load),
        (phx.system_modeling.ConnectionSet.create(("source", "load")),),
    )
    compiled = phx.system_modeling.compile_linear_acausal_system(
        system,
        jnp.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, -2.0))),
        jnp.asarray((10.0, 0.0)),
    )
    result = compiled.solve()
    values = compiled.connector_values(result.values)
    assert bool(result.successful)
    assert jnp.isclose(values["source"]["voltage"], 10.0)
    assert jnp.isclose(values["load"]["voltage"], 10.0)
    assert jnp.isclose(values["source"]["current"], -5.0)
    assert jnp.isclose(values["load"]["current"], 5.0)
    solver = phx.population_balance.ConservativeSectionalSolver.create(
        jnp.asarray((1.0, 2.0, 4.0)),
        jnp.ones((3, 3)),
    )
    state = phx.population_balance.SectionalPopulationState.create(
        jnp.asarray((1.0, 0.0, 0.25))
    )
    initial = solver.moments(state, jnp.asarray((0, 1)))
    step = solver.advance(state, 0.25)
    final = solver.moments(step.state, jnp.asarray((0, 1)))
    assert step.minimum_cell_number >= 0
    assert final[0] < initial[0]
    assert jnp.isclose(final[1], initial[1], atol=1e-10)
    assert jnp.isclose(step.first_moment_residual, 0, atol=1e-10)


def test_advanced_foundations_scenario_2() -> None:
    transport = phx.population_balance.SpatialPopulationTransport.create(
        jnp.asarray((1.0, 1.0))
    )
    step = transport.advance(
        jnp.asarray(((1.0,), (0.0,))),
        jnp.asarray((0.0, 0.5, 0.0)),
        1.0,
    )
    assert jnp.allclose(step.cell_number, jnp.asarray(((0.5,), (0.5,))))
    assert jnp.allclose(step.boundary_balance_residual, 0)
    law = phx.rheology.ViscoelasticLaw("oldroyd-b", 2.0, 1.0)
    solver = phx.rheology.SpatialConformationSolver.create(
        jnp.asarray((1.0, 1.0)),
        jnp.asarray(((-1.0, 1.0), (1.0, -1.0))),
        law,
    )
    result = solver.advance(
        jnp.asarray(
            (jnp.diag(jnp.asarray((2.0, 1.0))), jnp.diag(jnp.asarray((1.0, 2.0))))
        ),
        jnp.zeros((2, 2, 2)),
        0.1,
    )
    assert bool(result.successful)
    assert result.minimum_eigenvalue > 0
    assert jnp.allclose(result.transport_balance_residual, 0, atol=1e-10)
    assert jnp.allclose(
        result.polymer_stress_pa, result.polymer_stress_pa.swapaxes(-1, -2)
    )
    kinetics = phx.interfacial_transport.AdsorptionKinetics(0.1, 0.0, 1.0)
    solver = phx.interfacial_transport.CoupledBulkSurfaceTransport(
        np.asarray(((0, 1),)),
        np.asarray((0, 0)),
        np.asarray((0, 1)),
        np.asarray((1.0, 1.0)),
        kinetics,
        bulk_size=1,
        surface_size=2,
    )
    with pytest.raises(AttributeError):
        solver.tolerance = 1.0e-6
    result = solver.advance(
        jnp.asarray((1.0,)),
        jnp.asarray((0.0, 0.0)),
        jnp.asarray((1.0,)),
        jnp.asarray((0.5, 0.5)),
        jnp.asarray((1.0,)),
        1.0,
    )
    assert bool(result.successful)
    # Backward Euler with k_d = 0: Gamma = 1 - c and Gamma = 0.1 c (1 - Gamma).
    bulk = (np.sqrt(1.4) - 1.0) / 0.2
    assert jnp.isclose(result.bulk_amount_mol[0], bulk, rtol=1e-9)
    assert jnp.allclose(result.surface_amount_mol, 0.5 * (1.0 - bulk), rtol=1e-9)
    assert jnp.isclose(result.total_amount_residual_mol, 0, atol=1e-15)
    system = phx.structural_dynamics.LinearStructuralSystem.create(
        jnp.eye(2),
        jnp.zeros((2, 2)),
        jnp.diag(jnp.asarray((4.0, 9.0))),
    )
    modes = system.modal_analysis()
    assert bool(modes.successful)
    assert jnp.allclose(modes.angular_frequencies_rad_s, jnp.asarray((2.0, 3.0)))
    assert modes.mass_orthogonality_error < 1e-10
    state = phx.structural_dynamics.StructuralDynamicState(
        jnp.zeros(2), jnp.zeros(2), jnp.zeros(2)
    )
    step = system.newmark_step(state, jnp.asarray((1.0, 0.0)), 0.1)
    assert bool(step.successful)
    assert step.equilibrium_residual_norm < 1e-10
    assert step.mechanical_energy_j > 0


def test_advanced_foundations_scenario_3() -> None:
    law = phx.interfacial_transport.LangmuirSurfactantLaw(0.072, 300.0, 2.0e-6)
    surface_concentration = jnp.asarray(0.5e-6)
    pressure_scale = 8.31446261815324 * 300.0 * 2.0e-6
    expected_tension = 0.072 + pressure_scale * np.log(0.75)
    np.testing.assert_allclose(
        law.surface_tension(surface_concentration), expected_tension
    )
    np.testing.assert_allclose(
        law.tension_derivative(surface_concentration),
        -pressure_scale / 1.5e-6,
    )
    np.testing.assert_allclose(
        law.gibbs_elasticity(surface_concentration),
        surface_concentration * pressure_scale / 1.5e-6,
    )

    kinetics = phx.interfacial_transport.AdsorptionKinetics(0.1, 0.02, 2.0e-6)
    expected_flux = 0.1 * 2.0 * 0.75 - 0.02 * 0.5e-6
    np.testing.assert_allclose(
        kinetics.rate(jnp.asarray(2.0), surface_concentration), expected_flux
    )

    wetting = phx.interfacial_transport.CoxVoinovWettingLaw(0.8, 1.0e-8, 1.0e-3)
    expected_angle = np.cbrt(0.8**3 + 9.0e-3 * np.log(1.0e5))
    np.testing.assert_allclose(wetting.dynamic_angle(jnp.asarray(1.0e-3)), expected_angle)

    components = (law, kinetics, wetting)
    parameters, model_state, fixed = phx.partition_parameters(components)
    assert len(jax.tree_util.tree_leaves(parameters)) == 9
    trained = phx.combine_parameters(
        jax.tree_util.tree_map(lambda parameter: 1.01 * parameter, parameters),
        model_state,
        fixed,
    )
    assert not jnp.isclose(
        trained[0].surface_tension(surface_concentration),
        law.surface_tension(surface_concentration),
    )
    assert not jnp.isclose(
        trained[1].rate(jnp.asarray(2.0), surface_concentration),
        kinetics.rate(jnp.asarray(2.0), surface_concentration),
    )
    assert not jnp.isclose(
        trained[2].dynamic_angle(jnp.asarray(1.0e-3)),
        wetting.dynamic_angle(jnp.asarray(1.0e-3)),
    )
    with pytest.raises(AttributeError):
        law.temperature_k = jnp.asarray(301.0)
    with pytest.raises(AttributeError):
        kinetics.adsorption_rate_m_s = jnp.asarray(0.2)
    with pytest.raises(AttributeError):
        wetting.equilibrium_angle_rad = jnp.asarray(0.9)


def test_modal_and_frf_correlation_pair_permuted_modes() -> None:
    correlation = phx.correlation.correlate_modes(
        jnp.asarray((10.0, 20.0)),
        jnp.eye(2),
        jnp.asarray((20.2, 10.1)),
        jnp.asarray(((0.0, 1.0), (1.0, 0.0))),
        jnp.eye(2),
        minimum_mac=0.99,
        maximum_relative_frequency_error=0.02,
    )
    assert correlation.reference_to_candidate == (1, 0)
    assert jnp.all(correlation.accepted)
    response = jnp.asarray(((1.0 + 1.0j,), (2.0 - 0.5j,), (0.5 + 0.0j,)))
    frf = phx.correlation.correlate_frequency_responses(response, response)
    assert jnp.allclose(frf.assurance, 1.0)
    assert jnp.all(frf.accepted)
