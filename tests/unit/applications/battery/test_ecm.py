#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from types import SimpleNamespace

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.battery._ecm import (
    PreparedThermalEquivalentCircuit,
    ThermalEquivalentCircuitAdapter,
    ThermalEquivalentCircuitInitialCondition,
    ThermalEquivalentCircuitParameters,
    ThermalEquivalentCircuitPlan,
    ThermalEquivalentCircuitState,
)
from phydrax.applications.battery._experiment import (
    BatteryDiffraxSolvePlan,
    BatteryExperimentPlan,
    BatteryOutputPlan,
    BatteryRuntimeInputs,
)
from phydrax.applications.battery._properties import (
    ConstantPropertyLaw,
    TabulatedPropertyLaw,
)
from phydrax.applications.battery._protocol import (
    BatteryProtocolPlan,
    BatteryProtocolValues,
    CurrentStepPlan,
    RestStepPlan,
)
from phydrax.applications.battery._qualification import (
    THERMAL_ECM_CANDIDATE,
    THERMAL_ECM_SUPPORT,
)
from phydrax.applications.battery._results import BatteryRunStatus
from phydrax.linalg import prepare_real_coordinate_tree
from phydrax.solver import solve_diffrax


def _identity_coordinates(state):
    maps = jax.tree.map(lambda _: None, state)
    return prepare_real_coordinate_tree(state, maps)


def _constant_law(value, support, *, quantity, unit):
    return ConstantPropertyLaw(
        value,
        support,
        quantity=quantity,
        coordinate="state_of_charge",
        value_unit=unit,
        coordinate_unit="1",
        source_id=f"test:{quantity}",
    )


def _parameters(
    *,
    resistances=(0.1, 0.2),
    capacitances=(10.0, 20.0),
    series_resistance=0.05,
    capacity=1000.0,
    heat_capacity=100.0,
    conductance=0.5,
    ambient_temperature=300.0,
    reference_temperature=300.0,
    ocv=3.7,
    entropic=0.0,
    support=(0.0, 1.0),
):
    return ThermalEquivalentCircuitParameters(
        series_resistance,
        jnp.asarray(resistances),
        jnp.asarray(capacitances),
        capacity,
        heat_capacity,
        conductance,
        ambient_temperature,
        reference_temperature,
        _constant_law(
            ocv,
            support,
            quantity="reference-open-circuit-voltage",
            unit="V",
        ),
        _constant_law(
            entropic,
            support,
            quantity="entropic-coefficient",
            unit="V/K",
        ),
    )


def _runtime(parameters, current, *, duration=10.0):
    protocol = BatteryProtocolPlan((CurrentStepPlan(duration),))
    values = BatteryProtocolValues(protocol, jnp.asarray((current,)))
    return BatteryRuntimeInputs(
        parameters,
        protocol.input_policy(values, node_side="right"),
        values.stop_thresholds,
        protocol_id=protocol.protocol_id,
        observation_input_policy=protocol.input_policy(values),
    )


def _problem(adapter, parameters, initial_condition, current, *, duration=10.0):
    prepared = adapter.prepare()
    state = adapter.initial_state(prepared, parameters, initial_condition)
    runtime = _runtime(parameters, current, duration=duration)
    return prepared, runtime, adapter.problem(prepared, state, runtime)


def test_constant_current_and_rest_states_match_closed_forms():
    parameters = _parameters()
    adapter = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(2))
    current = 2.0
    steady_polarization = parameters.branch_resistances_ohm * current
    irreversible = current**2 * (
        parameters.series_resistance_ohm + jnp.sum(parameters.branch_resistances_ohm)
    )
    steady_temperature = (
        parameters.ambient_temperature_k
        + irreversible / parameters.thermal_conductance_w_per_k
    )
    initial_temperature = 310.0
    initial = ThermalEquivalentCircuitInitialCondition(
        500.0,
        initial_temperature,
        steady_polarization,
    )
    prepared, runtime, problem = _problem(
        adapter, parameters, initial, current, duration=10.0
    )
    times = jnp.asarray((0.0, 2.0, 5.0, 10.0))
    solution = solve_diffrax(
        problem,
        save_times=times,
        solver=dfx.Tsit5(),
        state_coordinates=_identity_coordinates(problem.initial_state),
        dt0=0.01,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    expected_temperature = steady_temperature + (
        initial_temperature - steady_temperature
    ) * jnp.exp(
        -parameters.thermal_conductance_w_per_k * times / parameters.heat_capacity_j_per_k
    )
    np.testing.assert_allclose(
        solution.states.charge_c,
        500.0 + current * times,
        rtol=2.0e-7,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        solution.states.polarization_voltages_v,
        jnp.broadcast_to(steady_polarization, (times.size, 2)),
        rtol=2.0e-7,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        solution.states.temperature_k,
        expected_temperature,
        rtol=2.0e-7,
        atol=2.0e-6,
    )
    observed = adapter.observe(prepared, times, solution.states, runtime)
    assert bool(observed.domain_valid.all())

    rest_initial = ThermalEquivalentCircuitInitialCondition(
        600.0,
        315.0,
        jnp.asarray((0.2, -0.3)),
    )
    rest_prepared, _, rest_problem = _problem(
        adapter, parameters, rest_initial, 0.0, duration=8.0
    )
    rest_times = jnp.asarray((0.0, 1.0, 4.0, 8.0))
    rest_solution = solve_diffrax(
        rest_problem,
        save_times=rest_times,
        solver=dfx.Tsit5(),
        state_coordinates=_identity_coordinates(rest_problem.initial_state),
        dt0=0.01,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    time_constants = parameters.branch_resistances_ohm * parameters.branch_capacitances_f
    expected_relaxation = jnp.asarray((0.2, -0.3))[None, :] * jnp.exp(
        -rest_times[:, None] / time_constants[None, :]
    )
    np.testing.assert_allclose(rest_solution.states.charge_c, 600.0, atol=2.0e-6)
    np.testing.assert_allclose(
        rest_solution.states.polarization_voltages_v,
        expected_relaxation,
        rtol=3.0e-7,
        atol=3.0e-7,
    )
    assert isinstance(rest_prepared, PreparedThermalEquivalentCircuit)


def test_passive_sign_reversal_voltage_power_and_heat_are_explicit():
    parameters = _parameters(entropic=1.0e-3)
    adapter = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(2))
    prepared = adapter.prepare()
    state = ThermalEquivalentCircuitState(
        500.0,
        jnp.asarray((0.1, -0.02)),
        310.0,
    )
    positive_runtime = _runtime(parameters, 2.0)
    negative_runtime = _runtime(parameters, -2.0)
    positive = adapter.observe(prepared, jnp.asarray(0.0), state, positive_runtime).values
    negative = adapter.observe(prepared, jnp.asarray(0.0), state, negative_runtime).values

    ocv = 3.7 + (310.0 - 300.0) * 1.0e-3
    polarization_sum = 0.08
    np.testing.assert_allclose(positive[0], ocv + 0.1 + polarization_sum)
    np.testing.assert_allclose(negative[0], ocv - 0.1 + polarization_sum)
    np.testing.assert_allclose(positive[5], 2.0 * positive[0])
    np.testing.assert_allclose(negative[5], -2.0 * negative[0])
    np.testing.assert_allclose(positive[6], negative[6])
    np.testing.assert_allclose(positive[7], -negative[7])
    np.testing.assert_allclose(positive[9], positive[6] + positive[7] - positive[8])

    positive_rate = adapter.problem(prepared, state, positive_runtime).drift(
        jnp.asarray(0.0), state, positive_runtime
    )
    negative_rate = adapter.problem(prepared, state, negative_runtime).drift(
        jnp.asarray(0.0), state, negative_runtime
    )
    np.testing.assert_allclose(positive_rate.charge_c, 2.0)
    np.testing.assert_allclose(negative_rate.charge_c, -2.0)


def test_boundary_observation_side_is_separate_from_forward_execution_current():
    parameters = _parameters()
    adapter = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(2))
    prepared = adapter.prepare()
    state = ThermalEquivalentCircuitState(
        500.0,
        jnp.zeros((2,)),
        300.0,
    )
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(1.0), CurrentStepPlan(1.0)),
        node_side="left",
    )
    values = BatteryProtocolValues(protocol, jnp.asarray((1.0, -2.0)))
    runtime = BatteryRuntimeInputs(
        parameters,
        protocol.input_policy(values, node_side="right"),
        values.stop_thresholds,
        protocol_id=protocol.protocol_id,
        observation_input_policy=protocol.input_policy(values),
    )
    observed = adapter.observe(
        prepared,
        jnp.asarray(1.0),
        state,
        runtime,
    )
    np.testing.assert_allclose(observed.values[0], 3.75)
    rate = adapter.problem(prepared, state, runtime).drift(
        jnp.asarray(1.0), state, runtime
    )
    np.testing.assert_allclose(rate.charge_c, -2.0)


def test_property_support_refuses_without_clipping_and_supports_temperature_ocv():
    ocv = TabulatedPropertyLaw(
        jnp.asarray((0.2, 0.5, 0.8)),
        jnp.asarray((3.2, 3.7, 4.1)),
        quantity="reference-open-circuit-voltage",
        coordinate="state_of_charge",
        value_unit="V",
        coordinate_unit="1",
        source_id="test:bounded-ocv",
    )
    entropic = ConstantPropertyLaw(
        2.0e-3,
        jnp.asarray((0.2, 0.8)),
        quantity="entropic-coefficient",
        coordinate="state_of_charge",
        value_unit="V/K",
        coordinate_unit="1",
        source_id="test:bounded-entropic",
    )
    parameters = ThermalEquivalentCircuitParameters(
        0.05,
        jnp.asarray((0.1,)),
        jnp.asarray((10.0,)),
        1000.0,
        100.0,
        0.5,
        300.0,
        300.0,
        ocv,
        entropic,
    )
    adapter = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(1))
    prepared = adapter.prepare()
    runtime = _runtime(parameters, 1.0)

    supported_state = ThermalEquivalentCircuitState(500.0, jnp.asarray((0.0,)), 310.0)
    supported = adapter.observe(prepared, jnp.asarray(0.0), supported_state, runtime)
    assert bool(supported.domain_valid)
    np.testing.assert_allclose(supported.values[0], 3.7 + 0.02 + 0.05)

    unsupported_state = ThermalEquivalentCircuitState(100.0, jnp.asarray((0.0,)), 310.0)
    unsupported = adapter.observe(prepared, jnp.asarray(0.0), unsupported_state, runtime)
    assert not bool(unsupported.domain_valid)
    np.testing.assert_allclose(unsupported.values[0], 0.05)
    assert unsupported.values[0] != pytest.approx(3.2 + 0.02 + 0.05)

    mismatched_entropic = _constant_law(
        0.0,
        (0.0, 1.0),
        quantity="entropic-coefficient",
        unit="V/K",
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="identical SOC support",
    ):
        mismatched = ThermalEquivalentCircuitParameters(
            0.05,
            jnp.asarray((0.1,)),
            jnp.asarray((10.0,)),
            1000.0,
            100.0,
            0.5,
            300.0,
            300.0,
            ocv,
            mismatched_entropic,
        )
        jax.block_until_ready(mismatched.series_resistance_ohm)


def test_arbitrary_branch_topology_and_initial_polarization_are_shape_safe():
    plan = ThermalEquivalentCircuitPlan(3)
    adapter = ThermalEquivalentCircuitAdapter(plan)
    prepared = adapter.prepare()
    parameters = _parameters(
        resistances=(0.1, 0.2, 0.3),
        capacitances=(10.0, 20.0, 30.0),
    )
    explicit = ThermalEquivalentCircuitInitialCondition(
        500.0,
        300.0,
        jnp.asarray((0.1, -0.2, 0.3)),
    )
    explicit_state = adapter.initial_state(prepared, parameters, explicit)
    np.testing.assert_allclose(
        explicit_state.polarization_voltages_v,
        np.asarray((0.1, -0.2, 0.3)),
    )

    relaxed = ThermalEquivalentCircuitInitialCondition(
        500.0,
        300.0,
        relaxed=True,
    )
    relaxed_state = adapter.initial_state(prepared, parameters, relaxed)
    np.testing.assert_array_equal(relaxed_state.polarization_voltages_v, np.zeros((3,)))
    with pytest.raises(ValueError, match="explicitly request relaxed"):
        ThermalEquivalentCircuitInitialCondition(500.0, 300.0)
    with pytest.raises(ValueError, match="branch count"):
        adapter.initial_state(
            prepared,
            parameters,
            ThermalEquivalentCircuitInitialCondition(
                500.0, 300.0, jnp.asarray((0.1, 0.2))
            ),
        )

    duplicate = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(3))
    other = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(1))
    assert adapter.observable_names == (
        "voltage_v",
        "temperature_k",
        "state_of_charge",
        "charge_c",
        "polarization_voltage_sum_v",
        "terminal_power_w",
        "irreversible_heat_w",
        "reversible_heat_w",
        "cooling_power_w",
        "net_heat_w",
    )
    assert adapter.observable_units == (
        "V",
        "K",
        "1",
        "C",
        "V",
        "W",
        "W",
        "W",
        "W",
        "W",
    )
    assert adapter.model_id == "battery:ecm:thermal-prescribed-current"
    assert duplicate.model_id == adapter.model_id == other.model_id
    assert duplicate.prepare().prepared_id == prepared.prepared_id
    assert plan.prepare().prepared_id == prepared.prepared_id
    assert other.prepare().prepared_id != prepared.prepared_id


def test_ledger_integrates_charge_rc_dissipation_and_thermal_storage():
    parameters = _parameters(
        resistances=(0.1, 0.2),
        capacitances=(10.0, 20.0),
        entropic=0.0,
    )
    adapter = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(2))
    prepared = adapter.prepare()
    runtime = _runtime(parameters, 2.0, duration=2.0)
    times = jnp.asarray((0.0, 1.0, 2.0))
    current = 2.0
    polarization = parameters.branch_resistances_ohm * current
    irreversible = current**2 * (
        parameters.series_resistance_ohm + jnp.sum(parameters.branch_resistances_ohm)
    )
    temperature = [305.0]
    for _ in range(2):
        previous = temperature[-1]
        numerator = (
            (
                parameters.heat_capacity_j_per_k
                - 0.5 * parameters.thermal_conductance_w_per_k
            )
            * previous
            + irreversible
            + parameters.thermal_conductance_w_per_k * parameters.ambient_temperature_k
        )
        denominator = (
            parameters.heat_capacity_j_per_k
            + 0.5 * parameters.thermal_conductance_w_per_k
        )
        temperature.append(numerator / denominator)
    states = ThermalEquivalentCircuitState(
        500.0 + current * times,
        jnp.broadcast_to(polarization, (times.size, 2)),
        jnp.asarray(temperature),
    )
    native_solution = SimpleNamespace(
        times=times,
        states=states,
        valid=jnp.ones(times.shape, dtype=bool),
        backend_successful=jnp.asarray(True),
    )
    ledger = adapter.ledger(prepared, native_solution, runtime)
    assert bool(ledger.successful)
    np.testing.assert_allclose(ledger.charge_change_c, 4.0)
    np.testing.assert_allclose(ledger.terminal_charge_c, 4.0)
    np.testing.assert_allclose(ledger.charge_defect_c, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(ledger.rc_energy_change_j, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(ledger.resistive_dissipation_j, 2.0 * irreversible)
    np.testing.assert_allclose(ledger.irreversible_heat_j, ledger.resistive_dissipation_j)
    np.testing.assert_allclose(ledger.reversible_heat_j, 0.0)
    np.testing.assert_allclose(ledger.thermal_defect_j, 0.0, atol=2.0e-5)

    insufficient = SimpleNamespace(
        times=times,
        states=states,
        valid=jnp.asarray((True, False, False)),
        backend_successful=jnp.asarray(True),
    )
    failed = adapter.ledger(prepared, insufficient, runtime)
    assert not bool(failed.successful)
    np.testing.assert_array_equal(
        jnp.asarray(
            (
                failed.charge_change_c,
                failed.terminal_charge_c,
                failed.charge_defect_c,
                failed.rc_energy_change_j,
                failed.resistive_dissipation_j,
                failed.thermal_energy_change_j,
                failed.irreversible_heat_j,
                failed.reversible_heat_j,
                failed.ambient_heat_loss_j,
                failed.thermal_defect_j,
            )
        ),
        np.zeros((10,)),
    )

    nonfinite_states = ThermalEquivalentCircuitState(
        states.charge_c.at[1].set(jnp.nan),
        states.polarization_voltages_v,
        states.temperature_k,
    )
    nonfinite = SimpleNamespace(
        times=times,
        states=nonfinite_states,
        valid=jnp.ones(times.shape, dtype=bool),
        backend_successful=jnp.asarray(True),
    )
    assert not bool(adapter.ledger(prepared, nonfinite, runtime).successful)


def test_adapter_orchestration_uses_native_ode_and_exact_observables():
    parameters = _parameters(resistances=(0.1,), capacitances=(20.0,))
    adapter = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(1))
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(1.0), RestStepPlan(1.0)),
        node_side="right",
    )
    profile, support = THERMAL_ECM_CANDIDATE, THERMAL_ECM_SUPPORT
    output_names = (
        "voltage_v",
        "current_a",
        "temperature_k",
        "state_of_charge",
        "terminal_power_w",
    )
    experiment = BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(output_names),
        BatteryDiffraxSolvePlan(
            solver=dfx.Tsit5(),
            stepsize_controller=dfx.PIDController(rtol=1.0e-8, atol=1.0e-10),
            dt0=0.02,
        ),
        jnp.asarray((0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0)),
        profile,
        support,
    ).prepare()
    result = experiment.run(
        parameters,
        ThermalEquivalentCircuitInitialCondition(500.0, 300.0, relaxed=True),
        BatteryProtocolValues(protocol, jnp.asarray((2.0,))),
    )
    assert int(result.application_status) == int(BatteryRunStatus.SUCCESS)
    assert result.outputs.names == output_names
    assert result.outputs.units == ("V", "A", "K", "1", "W")
    assert result.problem_id == result.native_solution.problem_id
    assert result.model_id == adapter.model_id
    assert bool(result.ledger.successful)
    np.testing.assert_allclose(
        result.native_solution.times,
        np.asarray((0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
    )
    np.testing.assert_allclose(result.ledger.charge_change_c, 2.0, atol=3.0e-5)
    np.testing.assert_allclose(result.ledger.terminal_charge_c, 2.0)
    np.testing.assert_allclose(result.ledger.charge_defect_c, 0.0, atol=3.0e-5)
    np.testing.assert_allclose(result.ledger.thermal_defect_j, 0.0, atol=2.0e-3)
    np.testing.assert_allclose(
        result.native_solution.states.charge_c[-1],
        502.0,
        atol=3.0e-5,
    )


def test_observation_is_jittable_vmappable_and_has_fixed_path_derivatives():
    parameters = _parameters(resistances=(0.2,), capacitances=(5.0,))
    adapter = ThermalEquivalentCircuitAdapter(ThermalEquivalentCircuitPlan(1))
    prepared = adapter.prepare()
    runtime = _runtime(parameters, 1.5, duration=2.0)

    def voltage_at_charge(charge):
        state = ThermalEquivalentCircuitState(
            charge, jnp.asarray((0.1,)), jnp.asarray(305.0)
        )
        return adapter.observe(prepared, jnp.asarray(0.5), state, runtime).values[0]

    compiled = jax.jit(voltage_at_charge)
    np.testing.assert_allclose(compiled(jnp.asarray(500.0)), 3.875)
    mapped = jax.vmap(voltage_at_charge)(jnp.asarray((400.0, 500.0, 600.0)))
    np.testing.assert_allclose(mapped, np.full((3,), 3.875))

    state = ThermalEquivalentCircuitState(500.0, jnp.asarray((0.1,)), 305.0)

    def voltage_for_series_resistance(resistance):
        varied = eqx.tree_at(
            lambda profile: profile.series_resistance_ohm,
            parameters,
            resistance,
        )
        varied_runtime = eqx.tree_at(
            lambda inputs: inputs.parameters,
            runtime,
            varied,
        )
        return adapter.observe(prepared, jnp.asarray(0.5), state, varied_runtime).values[
            0
        ]

    np.testing.assert_allclose(
        jax.jacfwd(voltage_for_series_resistance)(parameters.series_resistance_ohm),
        1.5,
    )

    def final_polarization(initial_polarization):
        initial = ThermalEquivalentCircuitState(
            500.0,
            jnp.reshape(initial_polarization, (1,)),
            300.0,
        )
        rest_runtime = _runtime(parameters, 0.0, duration=2.0)
        problem = adapter.problem(prepared, initial, rest_runtime)
        solution = solve_diffrax(
            problem,
            save_times=jnp.asarray((0.0, 2.0)),
            solver=dfx.Tsit5(),
            state_coordinates=_identity_coordinates(problem.initial_state),
            adjoint=dfx.DirectAdjoint(),
            dt0=0.02,
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        return solution.states.polarization_voltages_v[-1, 0]

    derivative = jax.jacfwd(final_polarization)(jnp.asarray(0.1))
    expected = jnp.exp(
        -2.0
        / (parameters.branch_resistances_ohm[0] * parameters.branch_capacitances_f[0])
    )
    np.testing.assert_allclose(derivative, expected, rtol=2.0e-6, atol=2.0e-7)
