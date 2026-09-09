#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.applications.battery import _qualification
from phydrax.applications.battery._experiment import (
    BatteryDAESolvePlan,
    BatteryDiffraxSolvePlan,
    BatteryExperimentPlan,
    BatteryOutputPlan,
    prepare_battery_experiment,
    run_battery_experiment,
)
from phydrax.applications.battery._protocol import (
    BatteryProtocolPlan,
    BatteryProtocolValues,
    CurrentStepPlan,
    CurrentStopGuard,
    RestStepPlan,
    VoltageStopGuard,
)
from phydrax.applications.battery._results import BatteryModelOutput, BatteryRunStatus
from phydrax.dynamics import DAEStructure, DifferentialAlgebraicSystem
from phydrax.qualification import CapabilityProfile, SupportTuple
from phydrax.solver import (
    DAEAdaptivePolicy,
    DAESolvePolicy,
    DAETerminationStatus,
    DifferentialAlgebraicProblem,
    DifferentialProblem,
    HybridGuardPlan,
    solve_diffrax,
)


class _ScalarLedger(StrictModule):
    successful: jax.Array


class RampAdapter(StrictModule, NonTrainableState):
    model_id: str = eqx.field(static=True)
    equation_form: str = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)
    domain_upper: float = eqx.field(static=True)
    voltage_current_gain: float = eqx.field(static=True)
    ledger_valid: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        equation_form="ode",
        domain_upper=jnp.inf,
        voltage_current_gain=0.0,
        ledger_valid=True,
    ):
        self.model_id = f"test:ramp:{equation_form}"
        self.equation_form = equation_form
        self.observable_names = (
            "voltage_v",
            "temperature_k",
            "stoichiometry:positive",
        )
        self.observable_units = ("V", "K", "1")
        self.domain_upper = float(domain_upper)
        self.voltage_current_gain = float(voltage_current_gain)
        self.ledger_valid = bool(ledger_valid)

    def prepare(self, /):
        return self.model_id

    def initial_state(self, prepared_model, parameters, initial_condition, /):
        assert prepared_model == self.model_id
        del parameters
        return {"charge": jnp.asarray(initial_condition)}

    def problem(self, prepared_model, initial_state, runtime_inputs, /):
        assert prepared_model == self.model_id

        def drift(time, state, runtime):
            return {
                "charge": jnp.asarray(
                    (runtime.current(time, state["charge"]),),
                    dtype=state["charge"].dtype,
                )
            }

        return DifferentialProblem(
            drift,
            initial_state,
            t0=runtime_inputs.input_policy.times[0],
            t1=runtime_inputs.input_policy.times[-1],
            args=runtime_inputs,
            problem_id=f"{self.model_id}:problem",
        )

    def observe(self, prepared_model, times_s, states, runtime_inputs, /):
        assert prepared_model == self.model_id
        charge = jnp.asarray(states["charge"])[..., 0]
        current = (
            runtime_inputs.observed_current(times_s)
            if jnp.asarray(times_s).shape == ()
            else jax.vmap(runtime_inputs.observed_current)(times_s)
        )
        voltage = charge + self.voltage_current_gain * current
        values = jnp.stack(
            (voltage, 300.0 + jnp.zeros_like(charge), charge / 10.0), axis=-1
        )
        return BatteryModelOutput(
            values, jnp.isfinite(charge) & (charge <= self.domain_upper)
        )

    def ledger(self, prepared_model, native_solution, runtime_inputs, /):
        assert prepared_model == self.model_id
        del runtime_inputs
        return _ScalarLedger(jnp.asarray(self.ledger_valid))


def _candidate(*, dae=False):
    model_id = "test:battery:algebraic-voltage" if dae else "test:ramp:ode"
    support = SupportTuple(
        "battery.simulation",
        {"model_id": model_id, "equation_form": "dae" if dae else "ode"},
    )
    return CapabilityProfile(
        f"battery.test-{'dae' if dae else 'ramp'}.candidate",
        "phydrax-tests",
        "candidate",
        (support,),
        released=False,
    ), support


@pytest.fixture(autouse=True)
def _register_synthetic_candidates(monkeypatch):
    monkeypatch.setattr(
        _qualification,
        "BATTERY_CANDIDATE_PROFILES",
        _qualification.BATTERY_CANDIDATE_PROFILES
        + (_candidate()[0], _candidate(dae=True)[0]),
    )


class AlgebraicVoltageAdapter(StrictModule, NonTrainableState):
    model_id: str = eqx.field(static=True, default="test:battery:algebraic-voltage")
    equation_form: str = eqx.field(static=True, default="dae")
    observable_names: tuple[str, ...] = eqx.field(
        static=True, default=("voltage_v", "charge_c")
    )
    observable_units: tuple[str, ...] = eqx.field(static=True, default=("V", "C"))
    native_guard: bool = eqx.field(static=True, default=False)

    def prepare(self, /):
        def residual(time, state, rate, runtime):
            current = runtime.current(time, state)
            return jnp.stack((rate[0] - current, state[1] - state[0] - 2.0 * current))

        return DifferentialAlgebraicSystem(
            residual,
            state_shape=(2,),
            structure=DAEStructure(("differential", "algebraic")),
            system_id="test:battery:algebraic-voltage",
        )

    def native_guards(self, prepared_model, /):
        del prepared_model
        if not self.native_guard:
            return ()
        return (
            HybridGuardPlan(
                lambda time, state, runtime: 5.5 - state[0],
                direction=-1,
                terminal=True,
                guard_id="test:battery:charge-upper",
            ),
        )

    def initial_state(self, prepared_model, parameters, initial_condition, /):
        del prepared_model, parameters
        charge = jnp.asarray(initial_condition)
        return jnp.stack((charge, charge))

    def problem(self, prepared_model, initial_state, runtime_inputs, /):
        return DifferentialAlgebraicProblem(
            prepared_model,
            initial_state,
            args=runtime_inputs,
            problem_id="test:battery:algebraic-voltage:problem",
        )

    def observe(self, prepared_model, times_s, states, runtime_inputs, /):
        del prepared_model, times_s, runtime_inputs
        return BatteryModelOutput(
            jnp.stack((states[..., 1], states[..., 0]), axis=-1),
            jnp.all(jnp.isfinite(states), axis=-1),
        )

    def ledger(self, prepared_model, native_solution, runtime_inputs, /):
        del prepared_model, runtime_inputs
        defect = (
            native_solution.states[:, 1]
            - native_solution.states[:, 0]
            - 2.0 * native_solution.forcing_currents_a
        )
        return _ScalarLedger(
            jnp.all((~native_solution.valid) | (jnp.abs(defect) < 1.0e-6))
        )


def _dae_experiment(protocol, save_times, *, native_guard=False, adaptive=False):
    adapter = AlgebraicVoltageAdapter(native_guard=native_guard)
    guard_ids = tuple(
        guard.guard_id for guard in adapter.native_guards(adapter.prepare())
    )
    profile, support = _candidate(dae=True)
    return BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(("charge_c", "voltage_v", "current_a")),
        BatteryDAESolvePlan(
            DAESolvePolicy(
                adaptive=DAEAdaptivePolicy() if adaptive else None,
            ),
            guard_ids=guard_ids,
        ),
        jnp.asarray(save_times),
        profile,
        support,
    ).prepare()


def _event_protocol(*, node_side="left"):
    return BatteryProtocolPlan(
        (
            CurrentStepPlan(1.0),
            CurrentStepPlan(
                2.0,
                stop_guards=(VoltageStopGuard("below"),),
            ),
        ),
        node_side=node_side,
    )


def _experiment(protocol, solve_plan, *, model=None, save_times=None):
    profile, support = _candidate()
    return BatteryExperimentPlan(
        RampAdapter() if model is None else model,
        protocol,
        BatteryOutputPlan(("voltage_v", "current_a", "temperature_k")),
        solve_plan,
        jnp.asarray(
            (0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0)
            if save_times is None
            else save_times
        ),
        profile,
        support,
    )


def _explicit_segmented_reference():
    def first(time, state, args):
        del time, args
        return jnp.ones_like(state)

    first_problem = DifferentialProblem(
        first,
        jnp.asarray((4.0,)),
        t0=0.0,
        t1=1.0,
        problem_id="test:ramp:segment-one",
    )
    first_solution = solve_diffrax(
        first_problem,
        save_times=jnp.asarray((0.0, 0.5, 1.0)),
        solver=dfx.Tsit5(),
        dt0=0.8,
    )

    def second(time, state, args):
        del time, args
        return -2.0 * jnp.ones_like(state)

    second_problem = DifferentialProblem(
        second,
        first_solution.states[-1],
        t0=1.0,
        t1=1.5,
        problem_id="test:ramp:segment-two",
    )
    second_solution = solve_diffrax(
        second_problem,
        save_times=jnp.asarray((1.0, 1.25, 1.5)),
        solver=dfx.Tsit5(),
        dt0=0.4,
    )
    return jnp.concatenate((first_solution.states[:, 0], second_solution.states[1:, 0]))


def test_adaptive_transition_clipping_matches_segments_and_localizes_later_guard():
    protocol = _event_protocol(node_side="left")
    solve_plan = BatteryDiffraxSolvePlan(
        solver=dfx.Tsit5(),
        stepsize_controller=dfx.PIDController(rtol=1.0e-8, atol=1.0e-10),
        dt0=2.5,
        event_relative_tolerance=1.0e-9,
        event_absolute_tolerance=1.0e-10,
    )
    prepared = prepare_battery_experiment(_experiment(protocol, solve_plan))
    values = BatteryProtocolValues(
        protocol, jnp.asarray((1.0, -2.0)), jnp.asarray((4.0,))
    )
    result = run_battery_experiment(prepared, (), jnp.asarray((4.0,)), values)

    assert isinstance(prepared.stepsize_controller, dfx.ClipStepSizeController)
    np.testing.assert_allclose(prepared.transition_times_s, np.asarray((0.0, 1.0, 3.0)))
    expected_prefix = _explicit_segmented_reference()
    np.testing.assert_allclose(result.outputs.values[:5, 0], expected_prefix, rtol=2.0e-6)
    np.testing.assert_allclose(result.outputs.values[2, 1], 1.0)
    np.testing.assert_allclose(result.termination.time_s, 1.5, atol=2.0e-6)
    assert result.termination.reason == "voltage-below"
    assert bool(result.native_solution.backend_successful)
    assert int(result.application_status) == int(BatteryRunStatus.SUCCESS)
    np.testing.assert_array_equal(
        result.outputs.valid,
        np.asarray((True, True, True, True, True, False, False, False)),
    )
    np.testing.assert_allclose(result.outputs.values[5:], 0.0)
    assert np.all(np.isinf(np.asarray(result.outputs.times_s[5:])))


def test_fixed_stepto_matches_guard_and_refuses_unaligned_transition():
    protocol = _event_protocol()
    fixed = BatteryDiffraxSolvePlan(
        solver=dfx.Tsit5(),
        stepsize_controller=dfx.StepTo(ts=jnp.asarray((0.0, 1.0, 2.0, 3.0))),
    )
    prepared = prepare_battery_experiment(_experiment(protocol, fixed))
    values = BatteryProtocolValues(
        protocol, jnp.asarray((1.0, -2.0)), jnp.asarray((4.0,))
    )
    result = prepared.run((), jnp.asarray((4.0,)), values)
    np.testing.assert_allclose(result.termination.time_s, 1.5, atol=2.0e-6)
    assert result.termination.reason == "voltage-below"

    unaligned = BatteryDiffraxSolvePlan(
        solver=dfx.Tsit5(),
        stepsize_controller=dfx.StepTo(ts=jnp.asarray((0.0, 0.75, 1.5, 3.0))),
    )
    with pytest.raises(ValueError, match="every protocol transition"):
        prepare_battery_experiment(_experiment(protocol, unaligned))


def test_fixed_left_boundary_guards_use_global_observation_side():
    protocol = BatteryProtocolPlan(
        (
            CurrentStepPlan(
                1.0,
                stop_guards=(
                    CurrentStopGuard("below"),
                    VoltageStopGuard("below"),
                ),
            ),
            CurrentStepPlan(1.0),
        ),
        node_side="left",
    )
    fixed = BatteryDiffraxSolvePlan(
        solver=dfx.Tsit5(),
        stepsize_controller=dfx.StepTo(ts=jnp.asarray((0.0, 1.0, 2.0))),
    )
    prepared = _experiment(
        protocol,
        fixed,
        model=RampAdapter(voltage_current_gain=1.0),
        save_times=(0.0, 1.0, 2.0),
    ).prepare()
    values = BatteryProtocolValues(
        protocol,
        jnp.asarray((1.0, -2.0)),
        jnp.asarray((0.0, 4.5)),
    )
    result = prepared.run((), jnp.asarray((4.0,)), values)
    assert not bool(result.termination.terminated)
    assert result.termination.reason == "completed"
    np.testing.assert_allclose(result.outputs.values[1, :2], np.asarray((6.0, 1.0)))
    np.testing.assert_allclose(result.native_solution.times, np.asarray((0.0, 1.0, 2.0)))


@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("threshold", (4.0, 4.5))
def test_ode_initial_equality_or_violation_stops_without_a_positive_crossing(
    fixed, threshold
):
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(1.0, stop_guards=(VoltageStopGuard("below"),)),)
    )
    solve_plan = (
        BatteryDiffraxSolvePlan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.StepTo(ts=jnp.asarray((0.0, 0.5, 1.0))),
        )
        if fixed
        else BatteryDiffraxSolvePlan(dt0=0.2)
    )
    result = (
        _experiment(
            protocol,
            solve_plan,
            save_times=(0.0, 0.5, 1.0),
        )
        .prepare()
        .run(
            (),
            jnp.asarray((4.0,)),
            BatteryProtocolValues(protocol, (1.0,), (threshold,)),
        )
    )
    assert bool(result.successful)
    assert result.termination.reason == "voltage-below"
    np.testing.assert_array_equal(result.termination.time_s, 0.0)
    np.testing.assert_array_equal(result.outputs.valid, (True, False, False))
    np.testing.assert_allclose(result.outputs.values[0], (4.0, 1.0, 300.0))
    assert not bool(result.termination.derivative_valid)
    assert int(result.native_solution.stats["num_accepted_steps"]) == 0


@pytest.mark.parametrize("threshold", (3.0, 3.5))
def test_fixed_ode_restart_guard_uses_new_forcing_and_preserves_left_output(threshold):
    protocol = BatteryProtocolPlan(
        (
            CurrentStepPlan(1.0),
            CurrentStepPlan(1.0, stop_guards=(VoltageStopGuard("below"),)),
        ),
        node_side="left",
    )
    solve_plan = BatteryDiffraxSolvePlan(
        solver=dfx.Euler(),
        stepsize_controller=dfx.StepTo(ts=jnp.asarray((0.0, 0.5, 1.0, 1.5, 2.0))),
    )
    result = (
        _experiment(
            protocol,
            solve_plan,
            model=RampAdapter(voltage_current_gain=2.0),
            save_times=(0.0, 0.5, 1.0, 1.5, 2.0),
        )
        .prepare()
        .run(
            (),
            jnp.asarray((4.0,)),
            BatteryProtocolValues(protocol, (1.0, -1.0), (threshold,)),
        )
    )
    assert bool(result.successful)
    assert result.termination.reason == "voltage-below"
    np.testing.assert_array_equal(result.termination.time_s, 1.0)
    np.testing.assert_array_equal(result.termination.protocol_step_index, 1)
    np.testing.assert_array_equal(result.outputs.valid, (True, True, True, False, False))
    np.testing.assert_allclose(result.outputs.values[2], (7.0, 1.0, 300.0))
    assert not bool(result.termination.derivative_valid)
    assert int(result.native_solution.stats["num_accepted_steps"]) == 2


@pytest.mark.parametrize("adaptive", (False, True))
@pytest.mark.parametrize(
    "node_side,boundary_voltage,boundary_current",
    (
        ("left", 7.0, 1.0),
        ("right", 5.0, 0.0),
    ),
)
def test_dae_transition_preserves_charge_and_reinitializes_voltage_and_rate(
    node_side, boundary_voltage, boundary_current, adaptive
):
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(1.0), RestStepPlan(1.0)),
        node_side=node_side,
    )
    prepared = _dae_experiment(protocol, (0.0, 0.5, 1.0, 1.5, 2.0), adaptive=adaptive)
    result = prepared.run((), jnp.asarray(4.0), BatteryProtocolValues(protocol, (1.0,)))
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.outputs.values,
        (
            (4.0, 6.0, 1.0),
            (4.5, 6.5, 1.0),
            (5.0, boundary_voltage, boundary_current),
            (5.0, 5.0, 0.0),
            (5.0, 5.0, 0.0),
        ),
        atol=1.0e-6,
    )
    restart = result.native_solution.replay.restarts[0]
    np.testing.assert_array_equal(
        restart.initialization.state[0], restart.state_before[0]
    )
    np.testing.assert_allclose(restart.initialization.state, (5.0, 5.0), atol=1.0e-6)
    np.testing.assert_allclose(restart.initialization.state_rate[0], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(restart.state_rate_before[0], 1.0, atol=1.0e-6)
    np.testing.assert_array_equal(restart.initialization.state_correction[0], 0.0)


@pytest.mark.parametrize("node_side", ("left", "right"))
def test_dae_linspace_transition_roundoff_preserves_saved_outputs(node_side):
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(0.1), RestStepPlan(0.1)),
        node_side=node_side,
    )
    save_times = jnp.linspace(0.0, 0.2, 21)
    prepared = _dae_experiment(protocol, save_times)
    result = prepared.run((), jnp.asarray(4.0), BatteryProtocolValues(protocol, (1.0,)))
    assert bool(result.successful)
    np.testing.assert_array_equal(result.outputs.times_s, save_times)
    expected_charge = 4.0 + np.minimum(np.asarray(save_times), 0.1)
    expected_current = (np.arange(21) < (11 if node_side == "left" else 10)).astype(float)
    np.testing.assert_allclose(
        result.outputs.values,
        np.stack(
            (expected_charge, expected_charge + 2.0 * expected_current, expected_current),
            axis=-1,
        ),
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        result.native_solution.replay.restarts[0].time_s,
        protocol.boundary_times_s[1],
    )


def test_transition_normalization_retains_distinct_nearby_save_nodes():
    protocol = BatteryProtocolPlan((CurrentStepPlan(0.1), RestStepPlan(0.1)))
    boundary = protocol.boundary_times_s[1]
    offset = 64.0 * np.finfo(boundary.dtype).eps * boundary
    save_times = jnp.asarray((0.0, boundary - offset, boundary + offset, 0.2))
    prepared = _dae_experiment(protocol, save_times)
    np.testing.assert_array_equal(
        prepared.integration_time_grid.times[prepared.save_indices],
        save_times,
    )
    np.testing.assert_array_equal(
        prepared.integration_time_grid.times,
        jnp.asarray((0.0, boundary - offset, boundary, boundary + offset, 0.2)),
    )


def test_dae_terminal_crossing_uses_native_success_with_invalid_suffix():
    protocol = BatteryProtocolPlan(
        (
            CurrentStepPlan(1.0),
            CurrentStepPlan(2.0, stop_guards=(VoltageStopGuard("below"),)),
        )
    )
    result = _dae_experiment(protocol, np.arange(0.0, 3.25, 0.25)).run(
        (), jnp.asarray(4.0), BatteryProtocolValues(protocol, (1.0, -1.0), (2.5,))
    )
    assert bool(result.successful)
    assert bool(result.native_solution.successful)
    assert not bool(jnp.all(result.native_solution.valid))
    assert int(result.native_solution.termination_status) == int(
        DAETerminationStatus.EVENT_TERMINATED
    )
    assert result.termination.reason == "voltage-below"
    np.testing.assert_allclose(result.termination.time_s, 1.5, atol=1.0e-6)
    np.testing.assert_allclose(result.outputs.values[6], (4.5, 2.5, -1.0), atol=1.0e-6)
    assert not bool(jnp.any(result.outputs.valid[7:]))


@pytest.mark.parametrize("adaptive", (False, True))
def test_dae_initial_equality_terminates_without_crossing_derivative_or_steps(adaptive):
    protocol = BatteryProtocolPlan(
        (
            CurrentStepPlan(1.0, stop_guards=(VoltageStopGuard("below"),)),
            RestStepPlan(1.0),
        )
    )
    result = _dae_experiment(protocol, (0.0, 0.5, 1.0, 1.5, 2.0), adaptive=adaptive).run(
        (), jnp.asarray(4.0), BatteryProtocolValues(protocol, (1.0,), (6.0,))
    )
    assert bool(result.successful)
    assert bool(result.termination.terminated)
    np.testing.assert_array_equal(result.termination.time_s, 0.0)
    np.testing.assert_array_equal(
        result.outputs.valid, (True, False, False, False, False)
    )
    np.testing.assert_allclose(result.outputs.values[0], (4.0, 6.0, 1.0), atol=1.0e-6)
    assert not bool(result.native_solution.replay.derivative_valid)
    assert int(result.native_solution.segments[0].step_history.count) == 0
    np.testing.assert_array_equal(
        result.native_solution.replay.segment_active, (True, False)
    )


def test_dae_model_guard_factory_localizes_a_native_terminal_event():
    protocol = BatteryProtocolPlan((CurrentStepPlan(2.0),))
    prepared = _dae_experiment(protocol, np.arange(0.0, 2.25, 0.25), native_guard=True)
    result = prepared.run((), jnp.asarray(4.0), BatteryProtocolValues(protocol, (1.0,)))
    assert bool(result.successful)
    assert result.termination.reason == prepared.native_guards[0].guard_id
    np.testing.assert_allclose(result.termination.time_s, 1.5, atol=1.0e-6)
    np.testing.assert_allclose(result.outputs.values[6], (5.5, 7.5, 1.0), atol=1.0e-6)


def test_dae_current_jump_equality_stops_after_preserving_the_left_output():
    protocol = BatteryProtocolPlan(
        (
            CurrentStepPlan(1.0),
            CurrentStepPlan(1.0, stop_guards=(VoltageStopGuard("below"),)),
        ),
        node_side="left",
    )
    result = _dae_experiment(protocol, (0.0, 0.5, 1.0, 1.5, 2.0)).run(
        (), jnp.asarray(4.0), BatteryProtocolValues(protocol, (1.0, -1.0), (3.0,))
    )
    assert bool(result.successful)
    np.testing.assert_array_equal(result.termination.time_s, 1.0)
    np.testing.assert_allclose(result.outputs.values[2], (5.0, 7.0, 1.0), atol=1.0e-6)
    event = result.native_solution.segments[1].events
    np.testing.assert_allclose(event.states_before[0], (5.0, 3.0), atol=1.0e-6)
    assert not bool(result.native_solution.replay.derivative_valid)
    assert not bool(jnp.any(result.outputs.valid[3:]))


def test_finite_native_trajectory_fails_when_model_ledger_fails():
    protocol = BatteryProtocolPlan((CurrentStepPlan(1.0),))
    result = (
        _experiment(
            protocol,
            BatteryDiffraxSolvePlan(dt0=0.2),
            model=RampAdapter(ledger_valid=False),
            save_times=(0.0, 0.5, 1.0),
        )
        .prepare()
        .run((), jnp.asarray((4.0,)), BatteryProtocolValues(protocol, (1.0,)))
    )
    assert bool(result.native_solution.backend_successful)
    assert bool(jnp.all(result.outputs.valid))
    assert int(result.application_status) == int(BatteryRunStatus.MODEL_LEDGER_FAILED)
    assert not bool(result.successful)


def test_domain_and_native_failures_are_distinct_fail_closed_statuses():
    protocol = BatteryProtocolPlan((CurrentStepPlan(1.0), RestStepPlan(2.0)))
    values = BatteryProtocolValues(protocol, jnp.asarray((1.0,)))
    domain_plan = BatteryDiffraxSolvePlan(dt0=0.2)
    domain_experiment = _experiment(
        protocol,
        domain_plan,
        model=RampAdapter(domain_upper=4.25),
        save_times=(0.0, 0.5, 1.0, 2.0, 3.0),
    ).prepare()
    domain_result = domain_experiment.run((), jnp.asarray((4.0,)), values)
    assert int(domain_result.application_status) == int(BatteryRunStatus.DOMAIN_ERROR)

    failure_plan = BatteryDiffraxSolvePlan(dt0=0.1, maximum_steps=1)
    failure_experiment = _experiment(
        protocol,
        failure_plan,
        save_times=(0.0, 0.5, 1.0, 2.0, 3.0),
    ).prepare()
    failure_result = failure_experiment.run((), jnp.asarray((4.0,)), values)
    assert int(failure_result.application_status) == int(
        BatteryRunStatus.NATIVE_SOLVE_FAILED
    )


def test_result_provenance_is_exactly_bound_to_preparation_and_candidate():
    protocol = BatteryProtocolPlan((CurrentStepPlan(1.0), RestStepPlan(2.0)))
    prepared = _experiment(
        protocol,
        BatteryDiffraxSolvePlan(dt0=0.2),
        save_times=(0.0, 3.0),
    ).prepare()
    parameters = jnp.asarray((1.0,))
    result = prepared.run(
        parameters,
        jnp.asarray((4.0,)),
        BatteryProtocolValues(protocol, jnp.asarray((0.5,))),
    )
    changed_current = prepared.run(
        parameters,
        jnp.asarray((4.0,)),
        BatteryProtocolValues(protocol, jnp.asarray((0.75,))),
    )
    changed_initial = prepared.run(
        parameters,
        jnp.asarray((4.5,)),
        BatteryProtocolValues(protocol, jnp.asarray((0.5,))),
    )
    changed_parameters = prepared.run(
        jnp.asarray((2.0,)),
        jnp.asarray((4.0,)),
        BatteryProtocolValues(protocol, jnp.asarray((0.5,))),
    )
    assert result.experiment_plan_id == prepared.plan.experiment_plan_id
    assert result.preparation_id == prepared.preparation_id
    assert result.protocol_id == protocol.protocol_id
    assert result.model_id == prepared.plan.model.model_id
    assert result.profile_id == prepared.plan.capability_profile.profile_id
    assert result.support_tuple_id == prepared.plan.support_tuple.support_tuple_id
    assert result.problem_id == result.native_solution.problem_id
    assert result.require_evidence_identity() == result.run_id
    assert result.protocol_values_digest != changed_current.protocol_values_digest
    assert result.initial_state_digest != changed_initial.initial_state_digest
    assert result.parameters_digest != changed_parameters.parameters_digest
    assert (
        len(
            {
                result.run_id,
                changed_current.run_id,
                changed_initial.run_id,
                changed_parameters.run_id,
            }
        )
        == 4
    )
    np.testing.assert_allclose(result.native_solution.times, np.asarray((0.0, 1.0, 3.0)))
    np.testing.assert_allclose(result.outputs.times_s, np.asarray((0.0, 3.0)))


def test_unsupported_controller_and_outputs_are_refused_before_execution():
    with pytest.raises(ValueError, match="adaptive controllers or explicit"):
        BatteryDiffraxSolvePlan(
            solver=dfx.Tsit5(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=0.1,
        )
    protocol = BatteryProtocolPlan((CurrentStepPlan(1.0),))
    profile, support = _candidate()
    plan = BatteryExperimentPlan(
        RampAdapter(),
        protocol,
        BatteryOutputPlan(("power_w",)),
        BatteryDiffraxSolvePlan(dt0=0.1),
        jnp.asarray((0.0, 1.0)),
        profile,
        support,
    )
    with pytest.raises(ValueError, match="power_w"):
        plan.prepare()


def test_failed_active_dae_segment_invalidates_replay_derivatives():
    protocol = BatteryProtocolPlan((CurrentStepPlan(1.0),))
    profile, support = _candidate(dae=True)
    prepared = BatteryExperimentPlan(
        AlgebraicVoltageAdapter(),
        protocol,
        BatteryOutputPlan(("charge_c", "voltage_v", "current_a")),
        BatteryDAESolvePlan(
            DAESolvePolicy(
                adaptive=DAEAdaptivePolicy(
                    initial_step=0.1,
                    maximum_step=0.1,
                    maximum_accepted_steps=1,
                    maximum_attempts=1,
                )
            ),
        ),
        jnp.asarray((0.0, 1.0)),
        profile,
        support,
    ).prepare()
    result = prepared.run((), jnp.asarray(4.0), BatteryProtocolValues(protocol, (1.0,)))
    assert bool(result.native_solution.initialization.valid)
    assert not bool(result.successful)
    assert not bool(result.native_solution.replay.successful)
    assert not bool(result.native_solution.replay.derivative_valid)
