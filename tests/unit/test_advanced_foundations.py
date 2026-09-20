import jax.numpy as jnp

import phydrax as phx


def test_compiled_frequency_system_closes_complex_residual():
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


def test_acausal_compiler_enforces_across_and_through_connections():
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


def test_population_aggregation_conserves_first_moment_with_overflow():
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


def test_spatial_population_transport_conserves_closed_domain_inventory():
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


def test_spatial_conformation_transport_preserves_spd_state():
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


def test_bulk_surface_adsorption_conserves_total_species():
    kinetics = phx.interfacial_transport.AdsorptionKinetics(0.1, 0.0, 1.0)
    solver = phx.interfacial_transport.CoupledBulkSurfaceTransport.create(
        jnp.asarray((1.0,)),
        jnp.asarray((0.5, 0.5)),
        jnp.asarray(((-1.0, 1.0), (1.0, -1.0))),
        jnp.asarray(((1.0, 1.0),)),
        kinetics,
    )
    result = solver.advance(jnp.asarray((1.0,)), jnp.asarray((0.0, 0.0)), 1.0)
    assert bool(result.successful)
    assert jnp.isclose(result.bulk_concentration_mol_m3[0], 0.9)
    assert jnp.allclose(result.surface_concentration_mol_m2, 0.1)
    assert jnp.isclose(result.total_mole_balance_residual, 0, atol=1e-12)


def test_structural_modes_and_newmark_step_close_equilibrium():
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


def test_modal_and_frf_correlation_pair_permuted_modes():
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
