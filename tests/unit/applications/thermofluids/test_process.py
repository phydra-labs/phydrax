#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _model():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("air",),
        (phx.equations.ChemicalPhaseKind.GAS,),
        jnp.asarray((0.02897,)),
        ("air",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((2.5 * phx.equations.UNIVERSAL_GAS_CONSTANT,),)),
        jnp.zeros((1,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=2000.0,
    )
    ideal = phx.equations.IdealGasReferenceHelmholtzTerm(schema, species)
    return phx.equations.HomogeneousHelmholtzPlan(
        ideal, phx.equations.ZeroResidualHelmholtzTerm(schema)
    )


def _initialize_process(process, initial_values=None):
    compilation = phx.dynamics.compile_acausal_dae(
        process.source, phx.dynamics.DAEStructuralPolicy(1, 0, tearing="none")
    )
    values = {}
    tf = phx.applications.thermofluids
    for component in process.components:
        prefix = f"{component.dae_component.name}."
        for variable in component.dae_component.variables:
            if "pressure" in variable.name:
                guess = 101325.0
            elif "enthalpy" in variable.name:
                guess = 1.0e4
            elif "temperature" in variable.name:
                guess = 300.0
            elif variable.name == "molar_density":
                guess = 40.0
            else:
                guess = 0.0
            values[prefix + variable.name] = guess
        for port in component.dae_component.ports:
            typed = component.port(port.name)
            if typed.kind is tf.ThermofluidPortKind.MATERIAL:
                inward_sign = (
                    1 if typed.direction is tf.MaterialFlowDirection.INLET else -1
                )
                values[prefix + port.flows[0]] = inward_sign * typed.mass_flow_orientation
                species = tuple(
                    name for name in port.potentials if "mass_fraction" in name
                )
                for name in species:
                    values[prefix + name] = 1.0 / len(species)
    if initial_values is not None:
        values.update(initial_values)
    initial = jnp.asarray(
        tuple(values[name] for name in compilation.analysis.variable_names)
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        compilation.system,
        initial,
        initialization=phx.solver.DAEInitializationSpec.from_masks(
            compilation.fixed_state_mask,
            compilation.fixed_rate_mask,
        ),
        problem_id=process.process_model_id,
    )
    dimension = compilation.system.state_shape[0]
    initialization_method = phx.nonlinear.NewtonKrylov(
        linear_policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(restart=dimension),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1.0e-12,
                absolute=1.0e-13,
                max_steps=4 * dimension,
            ),
        ),
        forcing_policy=phx.nonlinear.NewtonForcingPolicy("constant"),
    )
    result = phx.solver.initialize_dae(
        problem,
        0.0,
        policy=phx.solver.DAESolvePolicy(initialization_method=initialization_method),
    )
    assert bool(result.valid)
    np.testing.assert_allclose(
        compilation.residual_audit(0.0, result.state, result.state_rate),
        0.0,
        atol=1.0e-7,
    )
    return (
        compilation,
        result,
        compilation.reconstruction(result.state, result.state_rate),
    )


def test_chained_material_valves_solve_pressure_and_conserve_directed_flow():
    model = _model()
    tf = phx.applications.thermofluids
    identity = dict(
        catalog_id=model.schema.catalog.catalog_id,
        thermodynamics_id=model.model_id,
    )
    source = tf.material_boundary_component(
        "source",
        specific_enthalpy=4.0e5,
        mass_flow=1.0,
        direction=tf.MaterialFlowDirection.OUTLET,
        **identity,
    )
    valve = tf.isenthalpic_valve_component("valve", pressure_ratio=0.5, **identity)
    second = tf.isenthalpic_valve_component("second", pressure_ratio=0.5, **identity)
    sink = tf.material_boundary_component(
        "sink",
        pressure=1.0e5,
        direction=tf.MaterialFlowDirection.INLET,
        **identity,
    )
    process = tf.ThermofluidProcessPlan(
        (source, valve, second, sink),
        (
            tf.ThermofluidConnection("source", "material", "valve", "inlet"),
            tf.ThermofluidConnection("valve", "outlet", "second", "inlet"),
            tf.ThermofluidConnection("second", "outlet", "sink", "material"),
        ),
    )
    _, _, jet = _initialize_process(process)
    np.testing.assert_allclose(jet.value("source.pressure"), 4.0e5)
    np.testing.assert_allclose(jet.value("sink.specific_enthalpy"), 4.0e5)
    np.testing.assert_allclose(jet.value("valve.inlet_mass_flow"), 1.0)
    np.testing.assert_allclose(jet.value("valve.outlet_mass_flow"), -1.0)
    np.testing.assert_allclose(jet.value("second.inlet_mass_flow"), 1.0)
    np.testing.assert_allclose(jet.value("second.outlet_mass_flow"), -1.0)
    np.testing.assert_allclose(jet.value("sink.mass_flow"), -1.0)


def test_material_connection_rejects_mismatched_thermodynamics():
    model = _model()
    tf = phx.applications.thermofluids
    source = tf.fixed_material_boundary_component(
        "source",
        pressure=1.0e5,
        specific_enthalpy=1.0,
        mass_flow=1.0,
        catalog_id=model.schema.catalog.catalog_id,
        thermodynamics_id=model.model_id,
        direction=tf.MaterialFlowDirection.OUTLET,
    )
    sink = tf.fixed_material_boundary_component(
        "sink",
        pressure=1.0e5,
        specific_enthalpy=1.0,
        mass_flow=-1.0,
        catalog_id=model.schema.catalog.catalog_id,
        thermodynamics_id="different",
        direction=tf.MaterialFlowDirection.INLET,
    )
    with np.testing.assert_raises(ValueError):
        tf.ThermofluidProcessPlan(
            (source, sink),
            (tf.ThermofluidConnection("source", "material", "sink", "material"),),
        )


def test_compressor_map_design_and_ideal_station_balance():
    model = _model()
    tf = phx.applications.thermofluids
    performance_map = tf.CompressorMapPlan(
        jnp.asarray((0.8, 1.0)),
        jnp.asarray((0.0, 1.0)),
        jnp.asarray(((8.0, 9.0), (10.0, 11.0))),
        jnp.asarray(((1.5, 1.6), (2.0, 2.1))),
        jnp.asarray(((0.75, 0.76), (0.8, 0.81))),
        reference_temperature=288.15,
        reference_pressure=101325.0,
        provenance="synthetic qualification map",
    )
    compressor = tf.CompressorPlan(model, performance_map)
    design = compressor.design(
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        corrected_flow=10.0,
        pressure_ratio=2.0,
        isentropic_efficiency=0.8,
    )
    inlet = compressor.station(
        jnp.asarray(300.0),
        jnp.asarray(1.0e5),
        jnp.asarray(1.0),
        jnp.asarray((1.0,)),
    )
    result = compressor.evaluate(inlet, jnp.asarray(1.0), jnp.asarray(0.0), design)

    assert bool(result.successful)
    np.testing.assert_allclose(result.map_evaluation.pressure_ratio, 2.0)
    np.testing.assert_allclose(result.map_evaluation.isentropic_efficiency, 0.8)
    assert result.outlet.total_pressure > inlet.total_pressure
    assert result.outlet.mass_specific_enthalpy > inlet.mass_specific_enthalpy
    assert result.shaft_power > 0.0


def test_two_body_heat_exchange_conserves_energy_with_explicit_orientation():
    tf = phx.applications.thermofluids
    hot = tf.thermal_capacitance_component("hot", heat_capacity=2.0)
    cold = tf.thermal_capacitance_component(
        "cold",
        heat_capacity=1.0,
        orientation=tf.HeatFlowOrientation.OUT_OF_COMPONENT,
    )
    link = tf.thermal_conductor_component(
        "link",
        conductance=1.0,
        right_orientation=tf.HeatFlowOrientation.OUT_OF_COMPONENT,
    )
    process = tf.ThermofluidProcessPlan(
        (hot, cold, link),
        (
            tf.ThermofluidConnection("hot", "heat", "link", "left"),
            tf.ThermofluidConnection("link", "right", "cold", "heat"),
        ),
    )
    compilation = phx.dynamics.compile_acausal_dae(
        process.source, phx.dynamics.DAEStructuralPolicy(1, 0, tearing="none")
    )
    initial_values = {"hot.temperature": 360.0, "cold.temperature": 280.0}
    problem = phx.solver.DifferentialAlgebraicProblem(
        compilation.system,
        jnp.asarray(
            tuple(
                initial_values.get(name, 0.0)
                for name in compilation.analysis.variable_names
            )
        ),
        initialization=phx.solver.DAEInitializationSpec.from_masks(
            compilation.fixed_state_mask,
            compilation.fixed_rate_mask,
        ),
        problem_id=process.process_model_id,
    )
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 0.5, 6), time_id="two-body-heat")
    solution = phx.solver.solve_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(method=phx.solver.BDFMethod(1)),
    )
    assert bool(solution.successful)
    jets = [
        compilation.reconstruction(state, rate)
        for state, rate in zip(solution.states, solution.state_rates, strict=True)
    ]
    hot_temperature = np.asarray([jet.value("hot.temperature") for jet in jets])
    cold_temperature = np.asarray([jet.value("cold.temperature") for jet in jets])
    np.testing.assert_allclose(
        2.0 * hot_temperature + cold_temperature, 1000.0, atol=1e-8
    )
    np.testing.assert_allclose(
        hot_temperature - cold_temperature,
        80.0 / 1.15 ** np.arange(6),
        rtol=1e-7,
    )
    assert np.all(np.diff(hot_temperature) < 0)
    assert np.all(np.diff(cold_temperature) > 0)
    inward = tf.HeatPortBridge(tf.HeatFlowOrientation.INTO_COMPONENT)
    outward = tf.HeatPortBridge(
        tf.HeatFlowOrientation.OUT_OF_COMPONENT,
        temperature_offset=273.15,
    )
    np.testing.assert_allclose(outward.temperature_kelvin(20.0), 293.15)
    for jet in jets:
        hot_heat = inward.heat_into_component(jet.value("hot.heat_flow_0"))
        cold_heat = outward.heat_into_component(jet.value("cold.heat_flow_0"))
        assert hot_heat < 0
        assert cold_heat > 0
        np.testing.assert_allclose(hot_heat + cold_heat, 0.0, atol=1e-8)


def test_mixer_solves_distinct_advected_enthalpies_and_species():
    tf = phx.applications.thermofluids
    identity = dict(
        catalog_id="two-species", thermodynamics_id="shared-caloric-reference"
    )
    first = tf.material_boundary_component(
        "first",
        specific_enthalpy=10.0,
        mass_flow=2.0,
        direction=tf.MaterialFlowDirection.OUTLET,
        species_count=2,
        mass_fractions=(0.2, 0.8),
        **identity,
    )
    second = tf.material_boundary_component(
        "second",
        specific_enthalpy=40.0,
        mass_flow=1.0,
        direction=tf.MaterialFlowDirection.OUTLET,
        species_count=2,
        mass_fractions=(0.8, 0.2),
        **identity,
    )
    mixer = tf.material_mixer_component(
        "mixer",
        inlet_count=2,
        species_count=2,
        **identity,
    )
    sink = tf.material_boundary_component(
        "sink",
        pressure=1.0e5,
        direction=tf.MaterialFlowDirection.INLET,
        species_count=2,
        **identity,
    )
    process = tf.ThermofluidProcessPlan(
        (first, second, mixer, sink),
        (
            tf.ThermofluidConnection("first", "material", "mixer", "inlet_0"),
            tf.ThermofluidConnection("second", "material", "mixer", "inlet_1"),
            tf.ThermofluidConnection("mixer", "outlet", "sink", "material"),
        ),
    )
    _, _, jet = _initialize_process(process)
    np.testing.assert_allclose(jet.value("sink.mass_flow"), -3.0)
    np.testing.assert_allclose(jet.value("sink.specific_enthalpy"), 20.0)
    np.testing.assert_allclose(
        (jet.value("sink.mass_fraction_0"), jet.value("sink.mass_fraction_1")),
        (0.4, 0.6),
    )
    for field in ("specific_enthalpy", "mass_fraction_0", "mass_fraction_1"):
        flux = sum(
            jet.value(f"mixer.{port}_mass_flow") * jet.value(f"mixer.{port}_{field}")
            for port in ("inlet_0", "inlet_1", "outlet")
        )
        np.testing.assert_allclose(flux, 0.0, atol=1e-8)
    assert jet.value("mixer.inlet_0_specific_enthalpy") != jet.value(
        "mixer.inlet_1_specific_enthalpy"
    )
    with np.testing.assert_raises(ValueError):
        tf.ThermofluidProcessPlan(
            (first, second, mixer, sink),
            (
                tf.ThermofluidConnection("first", "material", "mixer", "inlet_0"),
                tf.ThermofluidConnection("second", "material", "mixer", "inlet_0"),
            ),
        )


def test_heat_conversion_accounts_for_environment_and_resistive_losses():
    tf = phx.applications.thermofluids
    for law, delivered, extracted in (
        (tf.ConstantCOPHeatPumpLaw(3.0), 300.0, 200.0),
        (tf.ResistiveHeatingLaw(0.8), 80.0, -20.0),
    ):
        device = tf.heat_conversion_component("device", law=law, electrical_power=100.0)
        supply = tf.temperature_boundary_component("supply", temperature=320.0)
        environment = tf.temperature_boundary_component("environment", temperature=280.0)
        process = tf.ThermofluidProcessPlan(
            (device, supply, environment),
            (
                tf.ThermofluidConnection("device", "supply", "supply", "heat"),
                tf.ThermofluidConnection("device", "environment", "environment", "heat"),
            ),
        )
        _, _, jet = _initialize_process(process)
        evaluation = law.evaluate(
            100.0,
            jet.value("device.source_temperature"),
            jet.value("device.supply_temperature"),
        )
        assert bool(evaluation.successful)
        np.testing.assert_allclose(jet.value("supply.heat_flow"), delivered)
        np.testing.assert_allclose(jet.value("environment.heat_flow"), -extracted)
        np.testing.assert_allclose(
            jet.value("supply.heat_flow") + jet.value("environment.heat_flow"),
            100.0,
        )
    unsupported = tf.ConstantCOPHeatPumpLaw(20.0).evaluate(100.0, 280.0, 320.0)
    assert not bool(unsupported.successful)
    invalid_power = tf.ResistiveHeatingLaw().evaluate(-1.0, 280.0, 320.0)
    assert not bool(invalid_power.successful)


def test_fluid_heat_exchanger_solves_provider_state_and_advective_energy():
    tf = phx.applications.thermofluids
    model = _model()
    identity = dict(
        catalog_id=model.schema.catalog.catalog_id, thermodynamics_id=model.model_id
    )
    inlet_state = model.evaluate(
        jnp.asarray(300.0),
        jnp.asarray(1.0e5 / (phx.equations.UNIVERSAL_GAS_CONSTANT * 300.0)),
        jnp.asarray((1.0,)),
    )
    inlet_enthalpy = float(inlet_state.molar_enthalpy / inlet_state.molar_mass)
    source = tf.material_boundary_component(
        "source",
        specific_enthalpy=inlet_enthalpy,
        mass_flow=1.0,
        direction=tf.MaterialFlowDirection.OUTLET,
        **identity,
    )
    exchanger = tf.homogeneous_fluid_heat_exchanger_component(
        "exchanger",
        thermodynamics=model,
        mole_fraction=(1.0,),
        conductance=100.0,
    )
    sink = tf.material_boundary_component(
        "sink",
        pressure=1.0e5,
        direction=tf.MaterialFlowDirection.INLET,
        **identity,
    )
    reservoir = tf.temperature_boundary_component("reservoir", temperature=350.0)
    process = tf.ThermofluidProcessPlan(
        (source, exchanger, sink, reservoir),
        (
            tf.ThermofluidConnection("source", "material", "exchanger", "inlet"),
            tf.ThermofluidConnection("exchanger", "outlet", "sink", "material"),
            tf.ThermofluidConnection("reservoir", "heat", "exchanger", "heat"),
        ),
    )
    _, _, jet = _initialize_process(
        process,
        {
            "exchanger.temperature": 310.0,
            "exchanger.molar_density": 40.0,
        },
    )
    outlet_state = model.evaluate(
        jet.value("exchanger.temperature"),
        jet.value("exchanger.molar_density"),
        jnp.asarray((1.0,)),
    )
    assert bool(outlet_state.evidence.successful)
    cp = inlet_state.molar_heat_capacity_pressure / inlet_state.molar_mass
    expected_temperature = (cp * 300.0 + 100.0 * 350.0) / (cp + 100.0)
    np.testing.assert_allclose(
        jet.value("exchanger.temperature"), expected_temperature, rtol=1e-7
    )
    heat = jet.value("exchanger.heat_flow")
    assert heat > 0
    np.testing.assert_allclose(jet.value("reservoir.heat_flow"), -heat, atol=1e-7)
    np.testing.assert_allclose(
        jet.value("sink.specific_enthalpy") - inlet_enthalpy, heat, atol=1e-7
    )
    np.testing.assert_allclose(jet.value("sink.mass_flow"), -1.0)


def test_heat_device_parameters_are_differentiable_and_retain_physical_support():
    tf = phx.applications.thermofluids

    def heating_energy(law):
        result = law.evaluate(jnp.asarray((100.0, 200.0)), 280.0, 320.0)
        return jnp.sum(result.delivered_heat)

    pump_gradient = eqx.filter_jit(eqx.filter_grad(heating_energy))(
        tf.ConstantCOPHeatPumpLaw(jnp.asarray(3.0))
    )
    np.testing.assert_allclose(pump_gradient.coefficient_of_performance, 300.0)

    def resistance_energy(efficiency):
        return heating_energy(tf.ResistiveHeatingLaw(efficiency))

    value, derivative = jax.jit(jax.value_and_grad(resistance_energy))(jnp.asarray(0.8))
    np.testing.assert_allclose(value, 240.0)
    np.testing.assert_allclose(derivative, 300.0)
    unsupported = eqx.tree_at(
        lambda law: law.efficiency,
        tf.ResistiveHeatingLaw(),
        jnp.asarray(1.2),
    ).evaluate(100.0, 280.0, 320.0)
    assert not bool(unsupported.successful)
