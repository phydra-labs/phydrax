import jax.numpy as jnp

import phydrax as phx


def test_spatial_ded_workflow_closes_process_heat_and_mass_ledgers():
    event = phx.manufacturing.ToolpathEvent(
        "track",
        "deposit",
        0.0,
        1.0,
        "machine",
        (0.0,),
        (1.0,),
        power_w=100.0,
        mass_rate_kg_s=1.0,
    )
    runtime = phx.manufacturing.ManufacturingRuntime.create(
        phx.manufacturing.ProcessSchedule.create((event,)),
        jnp.asarray(((0.0,), (1.0,))),
        jnp.asarray((1.0, 1.0)),
        1.0,
    )
    icme = phx.materials.SpatialICMEModel.create(
        jnp.asarray((1.0, 2.0)), jnp.asarray((1.0, 1.0))
    )
    workflow = phx.applications.additive_manufacturing.SpatialDEDWorkflow.create(
        runtime,
        jnp.asarray((10.0, 10.0)),
        jnp.asarray(((1.0, -1.0), (-1.0, 1.0))),
        jnp.zeros(2),
        icme,
        ambient_temperature_k=300.0,
        thermal_expansion_k_inv=1e-5,
        elastic_modulus_pa=1e9,
        reference_temperature_k=300.0,
    )
    state = phx.applications.additive_manufacturing.SpatialDEDState(
        phx.manufacturing.ManufacturingRuntimeState.initialize(2),
        jnp.asarray((300.0, 300.0)),
        jnp.asarray(((1.0, 0.0), (1.0, 0.0))),
    )
    result = workflow.advance(
        state,
        1.0,
        jnp.asarray(((0.0, 1.0), (0.0, 1.0))),
    )
    assert bool(result.successful)
    assert jnp.isclose(result.energy_balance_residual_j, 0, atol=1e-10)
    assert jnp.isclose(result.mass_balance_residual_kg, 0, atol=1e-12)
    assert jnp.isclose(jnp.sum(result.state.runtime.deposited_mass_kg), 1.0)
    assert jnp.max(result.state.temperature_k) > 300.0


def test_spatial_electroviscoelastic_workflow_closes_surface_charge():
    ehd = phx.electrohydrodynamics.CoupledElectrohydrodynamicSolver.create(
        jnp.asarray((1.0, 1.0)),
        jnp.eye(2),
        jnp.eye(2),
        jnp.zeros((2, 2)),
        jnp.eye(2),
        jnp.asarray((2.0, 2.0)),
        jnp.asarray((0.5, 0.5)),
        1,
    )
    law = phx.rheology.ViscoelasticLaw("oldroyd-b", 1.0, 1.0)
    rheology = phx.rheology.SpatialConformationSolver.create(
        jnp.asarray((1.0, 1.0)), jnp.zeros((2, 2)), law
    )
    workflow = (
        phx.applications.electroviscoelastic.SpatialElectroViscoelasticWorkflow.create(
            ehd,
            rheology,
            rheology,
            jnp.asarray((1.0, 0.0)),
            jnp.zeros((2, 2)),
            jnp.zeros((2, 2)),
            jnp.asarray((0,)),
            jnp.asarray((1,)),
            jnp.asarray(((1.0,),)),
            jnp.asarray((1.0,)),
            jnp.zeros((1, 1)),
        )
    )
    state = phx.applications.electroviscoelastic.SpatialElectroViscoelasticState(
        phx.electrohydrodynamics.ElectrohydrodynamicState(
            jnp.asarray((1.0, -1.0)), jnp.zeros(2), jnp.zeros((2, 1))
        ),
        jnp.ones((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.zeros(1),
    )
    result = workflow.advance(state, jnp.zeros(2), 0.1)
    assert bool(result.successful)
    assert jnp.isclose(result.surface_charge_balance_residual_c, 0, atol=1e-12)
    assert jnp.isclose(result.polymer_free_energy_j, 0, atol=1e-12)


def test_vibroacoustic_monolithic_workflow_closes_both_blocks():
    system = phx.acoustics.VibroacousticSystem.create(
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.1,),)),
        jnp.asarray(((4.0,),)),
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.2,),)),
        jnp.asarray(((9.0,),)),
        jnp.asarray(((0.5,),)),
    )
    result = system.solve(1.0, jnp.asarray((1.0,)), jnp.asarray((0.0,)))
    assert bool(result.successful)
    assert result.structural_residual_norm < 1e-12
    assert result.acoustic_residual_norm < 1e-12
    assert jnp.all(jnp.isfinite(result.acoustic_pressure_pa))


def test_porous_electrode_closes_current_and_species_balances():
    system = phx.electrochemistry.PorousElectrodeSystem.create(
        jnp.asarray((1.0, 1.0)),
        jnp.zeros((1, 2, 2)),
        jnp.eye(2),
        jnp.asarray((-1.0,)),
        jnp.ones(2),
        jnp.ones(2),
        jnp.zeros(2),
        jnp.full(2, 300.0),
        1,
    )
    state = phx.electrochemistry.PorousElectrodeState(jnp.ones((2, 1)), jnp.zeros(2))
    result = system.advance(state, jnp.asarray((0.1, 0.1)), 0.1)
    assert bool(result.successful)
    assert result.current_residual_norm_a < 1e-10
    assert jnp.allclose(result.species_balance_residual_mol, 0, atol=1e-12)
    assert jnp.all(result.state.concentration_mol_m3 < 1.0)


def test_equation_oriented_recycle_flowsheet_converges_balances():
    def residual(value):
        product, recycle = value
        return jnp.asarray((product + recycle - 10.0, recycle - 0.25 * product))

    flowsheet = phx.process_systems.EquationOrientedFlowsheet.create(
        residual,
        jnp.asarray((10.0, 2.0)),
        jnp.asarray((10.0, 2.0)),
        lower_bounds=jnp.zeros(2),
    )
    result = flowsheet.solve(jnp.asarray((5.0, 1.0)))
    assert bool(result.successful)
    assert jnp.allclose(result.variables, jnp.asarray((8.0, 2.0)))
    assert result.scaled_residual_norm < 1e-10


def test_fatigue_and_rotordynamics_workflows_expose_physical_diagnostics():
    fatigue = phx.durability.FatigueAssessment(
        phx.durability.SNCurve(1000.0, -0.1), 2000.0
    ).evaluate(jnp.asarray((0.0, 100.0, 0.0, -100.0, 0.0)))
    assert bool(fatigue.successful)
    assert fatigue.miner_damage > 0
    rotor = phx.applications.rotordynamics.RotorSystem.create(
        jnp.eye(2),
        jnp.eye(2) * 0.1,
        jnp.asarray(((0.0, -0.2), (0.2, 0.0))),
        jnp.diag(jnp.asarray((4.0, 9.0))),
    )
    force = rotor.unbalance_force(
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((0.01, 0.02)),
        jnp.zeros(2),
    )
    response = rotor.frequency_response(jnp.asarray((1.0, 2.0)), force)
    assert jnp.all(response.successful)
    assert jnp.all(response.bearing_dissipation_w >= 0)


def test_maxwell_workflow_closes_field_equations():
    maxwell = phx.electromagnetics.MaxwellFrequencySystem.create(
        jnp.asarray(((4.0,),)),
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.1,),)),
    ).solve(1.0, jnp.asarray((1.0,)))
    assert bool(maxwell.successful)
    assert maxwell.residual_norm < 1e-12
    assert maxwell.electric_energy_j > 0


def test_wind_and_marine_workflows_close_dynamic_residuals():
    turbine = phx.applications.wind_energy.AeroHydroServoElasticSystem.create(
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.1,),)),
        jnp.asarray(((4.0,),)),
        jnp.asarray(((0.1,),)),
        jnp.asarray(((0.2,),)),
        jnp.asarray(((0.2,),)),
        jnp.asarray(((1.0,),)),
        jnp.asarray((1.0,)),
        jnp.asarray((2.0,)),
        0,
        rotor_swept_area_m2=10.0,
    )
    turbine_state = phx.applications.wind_energy.WindTurbineState(
        phx.structural_dynamics.StructuralDynamicState(
            jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)
        ),
        jnp.asarray(0.0),
    )
    turbine_step = turbine.advance(turbine_state, 10.0, 0.5, 1.0, 0.1)
    assert bool(turbine_step.successful)
    assert turbine_step.mechanical_residual_norm < 1e-10
    assert turbine_step.available_wind_power_w > 0

    marine = phx.applications.marine_dynamics.MarineFrequencySystem.create(
        jnp.asarray((1.0, 2.0)),
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.1,),)),
        jnp.asarray(((4.0,),)),
        jnp.zeros((2, 1, 1)),
        jnp.ones((2, 1, 1)) * 0.2,
        jnp.asarray(((0.3,),)),
    ).solve(jnp.ones((2, 1)))
    assert jnp.all(marine.successful)
    assert jnp.all(marine.residual_norm < 1e-10)
    assert jnp.all(marine.absorbed_power_w >= 0)


def test_reservoir_pressure_workflow_conserves_pore_volume():
    system = phx.applications.reservoir.ReservoirPressureSystem.create(
        jnp.asarray((1e-6, 1e-6)),
        jnp.asarray(((1e-8, -1e-8), (-1e-8, 1e-8))),
    )
    state = phx.applications.reservoir.ReservoirPressureState(
        jnp.asarray((1e7, 1e7)),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    result = system.advance(
        state,
        jnp.zeros(2),
        jnp.asarray((1e-8, 0.0)),
        jnp.asarray((9e6, 9e6)),
        100.0,
    )
    assert bool(result.successful)
    assert jnp.isclose(result.volume_balance_residual_m3, 0, atol=1e-12)
    assert result.well_production_rate_m3_s[0] > 0


def test_flight_dynamics_preserves_trimmed_rest_state():
    system = phx.applications.flight_dynamics.RigidBodyFlightSystem.create(
        2.0, jnp.diag(jnp.asarray((1.0, 2.0, 3.0)))
    )
    state = phx.applications.flight_dynamics.FlightDynamicsState(
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.asarray((1.0, 0.0, 0.0, 0.0)),
        jnp.zeros(3),
        jnp.asarray(0.0),
    )
    step = system.advance(
        state,
        jnp.asarray((0.0, 0.0, 2.0 * 9.80665)),
        jnp.zeros(3),
        0.1,
    )
    assert bool(step.successful)
    assert jnp.allclose(step.state.velocity_body_m_s, 0, atol=1e-12)
    assert step.quaternion_norm_error < 1e-12


def test_vascular_device_network_closes_mass_and_momentum():
    network = phx.applications.vascular_devices.VascularDeviceNetwork.create(
        jnp.asarray(((1.0,), (-1.0,))),
        jnp.ones(2),
        jnp.ones(1),
        jnp.ones(1),
        jnp.asarray((0.1,)),
    )
    state = phx.applications.vascular_devices.VascularNetworkState(
        jnp.zeros(2), jnp.zeros(1), jnp.asarray(0.0)
    )
    step = network.advance(state, jnp.asarray((1.0, -1.0)), 0.1)
    assert bool(step.successful)
    assert jnp.allclose(step.node_mass_balance_residual_m3_s, 0, atol=1e-12)
    assert jnp.allclose(step.edge_momentum_residual_pa, 0, atol=1e-12)
    assert step.dissipated_power_w >= 0


def test_mineral_recycle_circuit_preserves_component_mass():
    circuit = phx.applications.mineral_processing.RecycleSeparationCircuit.create(
        jnp.asarray(((0.8, 0.2), (0.5, 0.1))),
        jnp.asarray((0.25, 0.1)),
    )
    result = circuit.solve(jnp.asarray((10.0, 90.0)))
    assert bool(result.successful)
    assert jnp.allclose(result.component_balance_residual_kg_s, 0, atol=1e-12)
    assert jnp.all(result.product_concentrate_kg_s > 0)
