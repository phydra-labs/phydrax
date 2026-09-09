#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.battery._circuit_ecm import (
    _sampled_integral,
    CircuitConnectedEcmAdapter,
    CircuitConnectedEcmInitialCondition,
    CircuitConnectedEcmPlan,
)
from phydrax.applications.battery._ecm import (
    _thermal_ecm_drift,
    ThermalEquivalentCircuitParameters,
    ThermalEquivalentCircuitState,
)
from phydrax.applications.battery._experiment import (
    BatteryDAESolvePlan,
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
    VoltageStopGuard,
)
from phydrax.applications.battery._qualification import (
    CIRCUIT_ECM_CANDIDATE,
    CIRCUIT_ECM_SUPPORT,
)
from phydrax.circuit import CircuitElement, IndependentVoltageSourceLaw, Resistor
from phydrax.dynamics import TimeGrid
from phydrax.nonlinear import NonlinearTermination
from phydrax.solver import BDFMethod, DAEAdaptivePolicy, DAESolvePolicy, solve_dae


def _parameters():
    def law(value, quantity, unit):
        return ConstantPropertyLaw(
            value,
            (0.0, 1.0),
            quantity=quantity,
            coordinate="state_of_charge",
            value_unit=unit,
            coordinate_unit="1",
            source_id=f"test:circuit-ecm:{quantity}",
        )

    return ThermalEquivalentCircuitParameters(
        0.05,
        jnp.asarray((0.1,)),
        jnp.asarray((10.0,)),
        1000.0,
        100.0,
        0.5,
        300.0,
        300.0,
        law(3.7, "reference-open-circuit-voltage", "V"),
        law(0.0, "entropic-coefficient", "V/K"),
    )


def _runtime(parameters, current=2.0):
    protocol = BatteryProtocolPlan((CurrentStepPlan(0.2),))
    values = BatteryProtocolValues(protocol, jnp.asarray((current,)))
    return BatteryRuntimeInputs(
        parameters,
        protocol.input_policy(values, node_side="right"),
        values.stop_thresholds,
        protocol_id=protocol.protocol_id,
        observation_input_policy=protocol.input_policy(values),
    )


def _solve(*, boundary=None, polarization=0.2, adaptive=False):
    adapter = CircuitConnectedEcmAdapter(CircuitConnectedEcmPlan(1, boundary=boundary))
    prepared, parameters = adapter.prepare(), _parameters()
    initial = CircuitConnectedEcmInitialCondition(
        500.0, 310.0, jnp.asarray((polarization,))
    )
    state = adapter.initial_state(prepared, parameters, initial)
    runtime = _runtime(parameters)
    problem = adapter.problem(prepared, state, runtime)
    # The unscaled q/T state norm must not stop Newton before the thermal
    # residual converges: BDF2 amplifies temperature corrections by C * 1.5 / dt.
    # Match native DAE precision fixtures with residual-only termination.
    termination = NonlinearTermination(
        absolute_residual=1e-8,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=12,
    )
    policy = DAESolvePolicy(
        method=BDFMethod(2),
        nonlinear_termination=termination,
        adaptive=(
            DAEAdaptivePolicy(
                relative_tolerance=1e-9,
                absolute_tolerance=1e-11,
                initial_step=1e-3,
                maximum_step=2e-2,
                maximum_accepted_steps=768,
                maximum_attempts=1536,
            )
            if adaptive
            else None
        ),
    )
    solution = solve_dae(
        problem,
        TimeGrid(jnp.linspace(0.0, 0.2, 21), time_id="circuit-ecm-regression"),
        policy=policy,
    )
    return adapter, prepared, parameters, runtime, solution


def test_sampled_ledger_quadrature_is_quadratic_exact_on_terminal_prefix():
    times = jnp.asarray((0.0, 0.1, 0.4, 1.0, jnp.nan, jnp.nan))
    valid = jnp.asarray((True, True, True, True, False, False))
    values = jnp.where(valid, times**2, jnp.nan)
    np.testing.assert_allclose(
        _sampled_integral(times, values, valid), 1.0 / 3.0, atol=2e-7
    )
    np.testing.assert_allclose(
        _sampled_integral(times, values, jnp.asarray((True, True) + (False,) * 4)),
        0.0005,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        _sampled_integral(times, values, jnp.asarray((True,) + (False,) * 5)),
        0.0,
        atol=0.0,
    )


def test_charging_initialization_and_one_node_thermal_parity():
    adapter, prepared, p, runtime, solution = _solve()
    assert bool(jnp.all(solution.successful))
    view = prepared.state_view(solution.states)
    np.testing.assert_allclose(view.charge_c, 500.0 + 2.0 * solution.times, atol=1e-7)
    np.testing.assert_allclose(view.current_a, 2.0, atol=1e-8)
    np.testing.assert_allclose(view.polarization_voltages_v, 0.2, atol=1e-8)
    np.testing.assert_allclose(view.voltage_v, 4.0, atol=1e-8)
    equilibrium_temperature = 300.0 + (4.0 * 0.05 + 0.2**2 / 0.1) / 0.5
    expected_temperature = equilibrium_temperature + (
        310.0 - equilibrium_temperature
    ) * jnp.exp(-0.005 * solution.times)
    np.testing.assert_allclose(view.temperature_k, expected_temperature, atol=2e-7)
    fixed = jnp.asarray(prepared.initialization.fixed_state)
    np.testing.assert_array_equal(solution.initialization.state_correction[fixed], 0.0)
    physical = ThermalEquivalentCircuitState(
        view.charge_c[0], view.polarization_voltages_v[0], view.temperature_k[0]
    )
    standalone_rate = _thermal_ecm_drift(jnp.asarray(0.0), physical, runtime)
    native_rate = prepared.state_view(solution.state_rates[0])
    np.testing.assert_allclose(native_rate.charge_c, standalone_rate.charge_c, atol=1e-9)
    np.testing.assert_allclose(
        native_rate.polarization_voltages_v,
        standalone_rate.polarization_voltages_v,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        native_rate.temperature_k, standalone_rate.temperature_k, atol=1e-9
    )
    ledger = adapter.ledger(prepared, solution, runtime)
    assert bool(ledger.successful)
    np.testing.assert_allclose(ledger.maximum_power_defect_w, 0.0, atol=1e-8)
    # Native acceptance bounds the scaled RMS, not each residual component.
    # Thus |thermal residual| <= sqrt(row count) * scale_T * RMS tolerance.
    thermal_tolerance = (
        np.sqrt(prepared.system.residual_scale.size)
        * prepared.system.residual_scale[prepared.cell_stop - 2]
        * jnp.max(solution.residual_threshold)
    )
    assert bool(ledger.maximum_thermal_defect_w <= thermal_tolerance)


def test_voltage_clamp_conservation_and_load_current_are_circuit_owned():
    voltage_boundary = CircuitElement(
        IndependentVoltageSourceLaw(4.0), element_id="test-voltage"
    )
    adapter, prepared, _, runtime, solution = _solve(
        boundary=voltage_boundary, polarization=0.1, adaptive=True
    )
    assert bool(jnp.all(solution.successful))
    view = prepared.state_view(solution.states)
    np.testing.assert_allclose(view.voltage_v, 4.0, atol=1e-8)
    np.testing.assert_allclose(view.current_a[0], 4.0, atol=1e-8)
    assert float(view.charge_c[-1]) > float(view.charge_c[0])
    ledger = adapter.ledger(prepared, solution, runtime)
    assert bool(ledger.successful)
    np.testing.assert_allclose(ledger.charge_defect_c, 0.0, atol=2e-5)
    # Under the 4 V clamp, v_p = 0.2 - 0.1 exp(-3t), hence
    # i = 2 + 2 exp(-3t). This reference is independent of measured charge.
    duration = solution.times[-1]
    exact_terminal_charge = 2.0 * duration + (2.0 / 3.0) * (
        1.0 - jnp.exp(-3.0 * duration)
    )
    np.testing.assert_allclose(ledger.terminal_charge_c, exact_terminal_charge, atol=2e-5)

    adapter, prepared, _, runtime, solution = _solve(
        boundary=Resistor(2.0), polarization=0.1
    )
    assert bool(jnp.all(solution.successful))
    view = prepared.state_view(solution.states)
    np.testing.assert_allclose(view.current_a, -view.voltage_v / 2.0, atol=1e-8)
    np.testing.assert_allclose(view.current_a[0], -3.8 / 2.05, atol=1e-8)
    assert float(view.charge_c[-1]) < float(view.charge_c[0])
    observed = adapter.observe(prepared, solution.times, solution.states, runtime)
    assert bool(jnp.all(observed.values[:, 5] < 0.0))
    assert bool(adapter.ledger(prepared, solution, runtime).successful)


def test_finite_thermal_rate_defect_fails_model_ledger():
    adapter, prepared, _, runtime, solution = _solve()
    assert bool(adapter.ledger(prepared, solution, runtime).successful)
    rates = solution.state_rates.at[-1, prepared.cell_stop - 2].add(0.1)
    corrupted = eqx.tree_at(lambda value: value.state_rates, solution, rates)
    ledger = adapter.ledger(prepared, corrupted, runtime)
    assert not bool(ledger.successful)
    np.testing.assert_allclose(ledger.maximum_thermal_defect_w, 10.0, atol=1e-7)


def test_physical_initial_state_outside_support_is_refused():
    adapter = CircuitConnectedEcmAdapter(CircuitConnectedEcmPlan(1))
    with pytest.raises(Exception, match="outside property support"):
        adapter.initial_state(
            adapter.prepare(),
            _parameters(),
            CircuitConnectedEcmInitialCondition(1100.0, 300.0, relaxed=True),
        )


def test_masked_property_hole_has_interior_guard_and_refuses_initialization():
    p = _parameters()
    ocv = TabulatedPropertyLaw(
        (0.0, 0.3, 0.5, 0.7, 1.0),
        (3.7,) * 5,
        source_mask=(True, True, False, True, True),
        quantity="reference-open-circuit-voltage",
        coordinate="state_of_charge",
        value_unit="V",
        coordinate_unit="1",
        source_id="test:masked-ocv",
    )
    p = eqx.tree_at(lambda value: value.open_circuit_voltage, p, ocv)
    adapter = CircuitConnectedEcmAdapter(CircuitConnectedEcmPlan(1))
    prepared = adapter.prepare()
    supported = adapter.initial_state(
        prepared, p, CircuitConnectedEcmInitialCondition(200.0, 300.0, relaxed=True)
    )
    upper_guard = adapter.native_guards(prepared)[1]
    np.testing.assert_allclose(upper_guard.guard(0.0, supported, _runtime(p)), 0.1 - 1e-8)
    with pytest.raises(Exception, match="outside property support"):
        adapter.initial_state(
            prepared, p, CircuitConnectedEcmInitialCondition(500.0, 300.0, relaxed=True)
        )


def test_current_rest_restart_preserves_physics_and_one_sided_heat_integrals():
    adapter = CircuitConnectedEcmAdapter(CircuitConnectedEcmPlan(1))
    guard_ids = tuple(
        guard.guard_id for guard in adapter.native_guards(adapter.prepare())
    )
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(0.1), RestStepPlan(0.1)), node_side="left"
    )
    experiment = BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(("charge_c", "current_a", "voltage_v")),
        BatteryDAESolvePlan(
            DAESolvePolicy(
                method=BDFMethod(2),
                nonlinear_termination=NonlinearTermination(
                    absolute_step=0.0, relative_step=0.0
                ),
            ),
            guard_ids=guard_ids,
        ),
        jnp.linspace(0.0, 0.2, 21),
        CIRCUIT_ECM_CANDIDATE,
        CIRCUIT_ECM_SUPPORT,
    ).prepare()
    result = experiment.run(
        _parameters(),
        CircuitConnectedEcmInitialCondition(500.0, 310.0, jnp.asarray((0.2,))),
        BatteryProtocolValues(protocol, jnp.asarray((2.0,))),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.outputs.values[10, 1], 2.0, atol=1e-8)
    np.testing.assert_allclose(result.outputs.values[11:, 1], 0.0, atol=1e-8)
    np.testing.assert_allclose(result.ledger.terminal_charge_c, 0.2, atol=1e-8)
    np.testing.assert_allclose(result.ledger.charge_change_c, 0.2, atol=1e-7)
    expected_irreversible = (
        0.1 * (4.0 * 0.05 + 0.2**2 / 0.1) + 0.4 * (1.0 - np.exp(-0.2)) / 2.0
    )
    np.testing.assert_allclose(
        result.ledger.irreversible_heat_j, expected_irreversible, atol=2e-5
    )
    restart = result.native_solution.replay.restarts[0]
    assert bool(restart.differential_state_unchanged)
    np.testing.assert_array_equal(
        restart.initialization.state_correction[restart.initialization.fixed_state_mask],
        0.0,
    )


def test_terminal_guard_leaves_inactive_segment_physics_unevaluated():
    adapter = CircuitConnectedEcmAdapter(CircuitConnectedEcmPlan(1))
    guards = tuple(guard.guard_id for guard in adapter.native_guards(adapter.prepare()))
    protocol = BatteryProtocolPlan(
        (
            CurrentStepPlan(0.1, stop_guards=(VoltageStopGuard("above"),)),
            RestStepPlan(0.1),
        )
    )
    experiment = BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(("voltage_v", "charge_c")),
        BatteryDAESolvePlan(guard_ids=guards),
        jnp.asarray((0.0, 0.1, 0.2)),
        CIRCUIT_ECM_CANDIDATE,
        CIRCUIT_ECM_SUPPORT,
    ).prepare()
    result = experiment.run(
        _parameters(),
        CircuitConnectedEcmInitialCondition(500.0, 310.0, jnp.asarray((0.2,))),
        BatteryProtocolValues(protocol, jnp.asarray((2.0,)), jnp.asarray((3.9,))),
    )
    assert bool(result.successful)
    assert bool(result.ledger.successful)
    assert float(result.termination.time_s) == 0.0
    np.testing.assert_array_equal(
        result.native_solution.replay.segment_active, (True, False)
    )
    np.testing.assert_array_equal(result.outputs.valid, (True, False, False))
    np.testing.assert_allclose(result.outputs.values[0], (4.0, 500.0), atol=1e-8)
