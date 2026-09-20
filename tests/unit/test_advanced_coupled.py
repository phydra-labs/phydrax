import jax.numpy as jnp

import phydrax as phx


def test_coupled_ehd_step_conserves_signed_free_charge():
    solver = phx.electrohydrodynamics.CoupledElectrohydrodynamicSolver.create(
        jnp.asarray((1.0, 1.0)),
        jnp.eye(2),
        jnp.eye(2),
        jnp.asarray(((-1.0, 1.0), (1.0, -1.0))),
        jnp.eye(2),
        jnp.asarray((2.0, 2.0)),
        jnp.asarray((0.5, 0.5)),
        1,
    )
    state = phx.electrohydrodynamics.ElectrohydrodynamicState(
        jnp.asarray((1.0, -1.0)), jnp.zeros(2), jnp.zeros((2, 1))
    )
    step = solver.advance(state, jnp.zeros(2), 0.1)
    assert bool(step.successful)
    assert jnp.isclose(step.charge_balance_residual_c, 0, atol=1e-12)
    assert jnp.isclose(jnp.sum(step.state.free_charge_density_c_m3), 0, atol=1e-12)
    assert step.ledger.electric_energy_j > 0


def test_many_particle_phoresis_includes_hydrodynamic_interactions():
    solver = phx.phoresis.HydrodynamicPhoreticSolver.create(
        1.0, jnp.asarray((0.5, 0.5)), 3
    )
    positions = jnp.asarray(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    force = jnp.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    step = solver.advance(
        positions,
        jnp.zeros_like(positions),
        jnp.zeros_like(positions),
        force,
        0.1,
    )
    assert bool(step.successful)
    assert step.minimum_surface_separation_m == 2.0
    assert step.velocity_m_s[1, 0] > 0
    assert step.force_power_w > 0


def test_spatial_piezoelectric_system_closes_reciprocal_block_residual():
    system = phx.smart_materials.SpatialPiezoelectricSystem.create(
        jnp.asarray(((2.0,),)),
        jnp.asarray(((3.0,),)),
        jnp.asarray(((0.5,),)),
    )
    result = system.solve(jnp.asarray((1.0,)), jnp.asarray((0.2,)))
    assert bool(result.successful)
    assert result.residual_norm < 1e-12
    assert jnp.isfinite(result.electric_enthalpy_j)


def test_spatial_chemo_mechanics_conserves_species_and_dissipates():
    system = phx.chemo_mechanics.SpatialChemoMechanicalSystem.create(
        jnp.asarray(((2.0,),)),
        jnp.eye(2),
        jnp.asarray(((0.1, 0.1),)),
        jnp.asarray(((1.0, -1.0), (-1.0, 1.0))),
        jnp.asarray((1.0, 1.0)),
    )
    result = system.advance(
        jnp.asarray((0.2, 0.8)),
        jnp.asarray((0.0,)),
        jnp.zeros(2),
        0.1,
    )
    assert bool(result.successful)
    assert jnp.isclose(jnp.sum(result.concentration), 1.0)
    assert jnp.isclose(result.mass_balance_residual, 0, atol=1e-12)
    assert result.dissipation_rate >= 0


def test_mass_conserving_ehl_tracks_cavitation_and_elastic_pressure():
    solver = phx.tribology.MassConservingEHLSolver.create(
        jnp.ones(3),
        jnp.ones(3),
        jnp.zeros((3, 3)),
        1.0,
        0.0,
        bulk_modulus_pa=1000.0,
    )
    state = phx.tribology.EHLState(
        jnp.asarray((1.1, 0.5, 1.0)),
        jnp.zeros(3),
    )
    step = solver.advance(state, 1e-4)
    assert bool(step.successful)
    assert jnp.isclose(step.mass_balance_residual_m2, 0, atol=1e-12)
    assert jnp.all((step.saturation >= 0) & (step.saturation <= 1))
    assert step.state.pressure_pa[0] > 0
    assert jnp.isclose(step.state.pressure_pa[1], 0, atol=1e-9)
    assert step.complementarity_residual_pa < 1e-8


def test_enclosure_radiation_is_reciprocal_and_energy_conserving():
    enclosure = phx.thermal_systems.DiffuseGrayEnclosure.create(
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1.0, 1.0)),
        jnp.asarray(((0.0, 1.0), (1.0, 0.0))),
    )
    result = enclosure.solve(jnp.asarray((300.0, 400.0)))
    assert bool(result.successful)
    assert jnp.isclose(result.enclosure_balance_w, 0, atol=1e-10)
    assert result.surface_power_w[0] < 0
    assert result.surface_power_w[1] > 0


def test_boiling_curve_and_wall_step_are_energy_closed():
    curve = phx.thermal_systems.PoolBoilingCurve(
        373.0,
        100.0,
        10.0,
        10000.0,
        2000.0,
        20.0,
        500.0,
        2.0e6,
    )
    transition = curve.evaluate(388.0)
    assert transition.regime == "transition"
    wall = curve.advance_wall(378.0, 5000.0, 10000.0, 0.5)
    assert jnp.isclose(wall.energy_balance_residual_j_m2, 0, atol=1e-12)


def test_cryogenic_boiloff_and_venting_close_mass_energy_ledgers():
    model = phx.thermal_systems.CryogenicTankModel(
        1.0, 1000.0, 100.0, 100.0, 10.0, 5.0, 10.0, 100.0
    )
    state = phx.thermal_systems.CryogenicTankState(
        jnp.asarray(0.5), jnp.asarray(0.01), jnp.asarray(100.0)
    )
    step = model.advance(state, 10.0, 1.0)
    assert bool(step.successful)
    assert step.vaporized_mass_kg > 0
    assert step.vented_mass_kg > 0
    assert jnp.isclose(step.mass_balance_residual_kg, 0, atol=1e-12)
    assert jnp.isclose(step.energy_balance_residual_j, 0, atol=1e-10)


def test_ablation_step_partitions_sensible_and_recession_energy():
    model = phx.thermal_systems.AblationSurfaceModel(
        1000.0, 1000.0, 500.0, 1.0e6, 0.0, 1.0
    )
    state = phx.thermal_systems.AblationSurfaceState(
        jnp.asarray(0.01), jnp.asarray(400.0)
    )
    step = model.advance(state, 200000.0, 0.0, 0.0, 400.0, 1.0)
    assert jnp.isclose(step.state.surface_temperature_k, 500.0)
    assert jnp.isclose(step.recession_m, 1e-4)
    assert jnp.isclose(step.energy_balance_residual_j_m2, 0)


def test_crossflow_membrane_module_conserves_every_species():
    module = phx.membranes.CrossflowMembraneModule.create(
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1e-6, 0.5e-6)),
        jnp.asarray((2e5, 2e5)),
        jnp.asarray((1e5, 1e5)),
    )
    result = module.solve(jnp.asarray((10.0, 10.0)), jnp.asarray((1.0, 1.0)))
    assert bool(result.successful)
    assert jnp.allclose(result.species_balance_residual_mol_s, 0, atol=1e-12)
    assert result.stage_cut > 0


def test_segmented_catalyst_preserves_declared_atomic_inventory():
    reactor = phx.surface_chemistry.SegmentedCatalyticReactor.create(
        jnp.asarray((1.0, 1.0)),
        jnp.asarray(((-1.0, 1.0),)),
        jnp.asarray(((1.0, 0.0),)),
        jnp.asarray((0.1,)),
        jnp.asarray((-1000.0,)),
        jnp.asarray(((1.0, 1.0),)),
        1.0,
        100.0,
    )
    result = reactor.solve(jnp.asarray((1.0, 0.0)), 300.0)
    assert bool(result.successful)
    assert jnp.allclose(result.conserved_quantity_residual, 0, atol=1e-12)
    assert result.species_molar_flow_mol_s[-1, 1] > 0
    assert result.temperature_k[-1] > 300.0


def test_stop_system_couples_thermal_mechanical_and_optical_fields():
    system = phx.optomechanics.SpatialOptomechanicalSystem.create(
        jnp.diag(jnp.asarray((2.0, 3.0))),
        jnp.diag(jnp.asarray((4.0, 5.0))),
        jnp.asarray(((0.1, 0.0), (0.0, 0.2))),
        jnp.eye(2),
        jnp.asarray(((0.01, 0.0), (0.0, 0.02))),
        jnp.asarray((1.0, 1.0)),
    )
    result = system.solve(jnp.asarray((2.0, 3.0)), jnp.asarray((1.0, 0.0)))
    assert bool(result.successful)
    assert result.thermal_residual_norm < 1e-12
    assert result.mechanical_residual_norm < 1e-12
    assert result.rms_wavefront_error_m > 0
