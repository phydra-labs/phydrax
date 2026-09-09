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
from phydrax.applications.battery._current_control import (
    BatteryCurrentControlBounds,
    BatteryCurrentControlObjective,
    BatteryCurrentControlPlan,
    BatteryCurrentControlReplayStatus,
    BatteryCurrentControlTerminalTarget,
)
from phydrax.applications.battery._experiment import (
    BatteryDiffraxSolvePlan,
    BatteryExperimentPlan,
    BatteryOutputPlan,
    BatteryRuntimeInputs,
)
from phydrax.applications.battery._protocol import (
    BatteryProtocolPlan,
    CurrentStepPlan,
    RestStepPlan,
)
from phydrax.applications.battery._results import BatteryModelOutput
from phydrax.control import (
    compile_direct_collocation,
    DirectCollocationDerivativePolicy,
    DirectCollocationPlan,
)
from phydrax.discretization import TemporalMesh
from phydrax.qualification import CapabilityProfile, SupportTuple
from phydrax.solver import DifferentialProblem, ThetaMethod


class AnalyticCurrentParameters(StrictModule):
    runtime_current_limit_a: jax.Array
    ledger_current_limit_a: jax.Array


class AnalyticCurrentLedger(StrictModule):
    successful: jax.Array


class AnalyticCurrentAdapter(StrictModule, NonTrainableState):
    model_id: str = eqx.field(static=True)
    equation_form: str = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)

    def __init__(self):
        self.model_id = "test:battery:analytic-current"
        self.equation_form = "ode"
        self.observable_names = (
            "voltage_v",
            "stoichiometry:negative",
            "stoichiometry:positive",
            "temperature_k",
        )
        self.observable_units = ("V", "1", "1", "K")

    def prepare(self, /):
        return "test:battery:analytic-current:prepared-model"

    def initial_state(self, prepared_model, parameters, initial_condition, /):
        assert prepared_model == "test:battery:analytic-current:prepared-model"
        assert isinstance(parameters, AnalyticCurrentParameters)
        return jnp.asarray((initial_condition,))

    def problem(self, prepared_model, initial_state, runtime_inputs, /):
        assert prepared_model == "test:battery:analytic-current:prepared-model"
        assert isinstance(runtime_inputs, BatteryRuntimeInputs)

        def drift(time_s, state, runtime):
            current = runtime.current(time_s, state)
            parameters = runtime.parameters
            derivative = jnp.asarray((current,), dtype=state.dtype)
            return jnp.where(
                jnp.abs(current) <= parameters.runtime_current_limit_a,
                derivative,
                jnp.full_like(derivative, jnp.nan),
            )

        return DifferentialProblem(
            drift,
            initial_state,
            t0=runtime_inputs.input_policy.times[0],
            t1=runtime_inputs.input_policy.times[-1],
            args=runtime_inputs,
            problem_id="test:battery:analytic-current:problem",
        )

    def observe(self, prepared_model, times_s, states, runtime_inputs, /):
        assert prepared_model == "test:battery:analytic-current:prepared-model"
        charge = states[..., 0]
        flat_times = times_s.reshape((-1,))
        current = jax.vmap(runtime_inputs.observed_current)(flat_times).reshape(
            times_s.shape
        )
        voltage = 3.0 + jnp.exp(-(((charge - 0.5) / 0.12) ** 2))
        negative_stoichiometry = 0.5 * charge
        positive_stoichiometry = 1.0 - 0.5 * charge
        temperature = 300.0 + 0.1 * current**2
        values = jnp.stack(
            (
                voltage,
                negative_stoichiometry,
                positive_stoichiometry,
                temperature,
            ),
            axis=-1,
        )
        domain = (
            jnp.all(jnp.isfinite(values), axis=-1) & (charge >= 0.0) & (charge <= 2.0)
        )
        return BatteryModelOutput(values, domain)

    def ledger(self, prepared_model, native_solution, runtime_inputs, /):
        assert prepared_model == "test:battery:analytic-current:prepared-model"
        successful = (
            jnp.asarray(native_solution.backend_successful, dtype=bool)
            & jnp.all(native_solution.valid)
            & jnp.all(
                jnp.abs(runtime_inputs.input_policy.values[:, 0])
                <= runtime_inputs.parameters.ledger_current_limit_a
            )
        )
        return AnalyticCurrentLedger(successful)


def _support():
    support = SupportTuple(
        "battery.simulation",
        {
            "model_id": "test:battery:analytic-current",
            "equation_form": "ode",
            "control": "prescribed-current",
        },
    )
    profile = CapabilityProfile(
        "battery.analytic-current",
        "phydrax-tests",
        "candidate",
        (support,),
        released=False,
    )
    return profile, support


@pytest.fixture(autouse=True)
def _synthetic_candidate_registry(monkeypatch):
    profile, _ = _support()
    monkeypatch.setattr(
        _qualification,
        "BATTERY_CANDIDATE_PROFILES",
        (*_qualification.BATTERY_CANDIDATE_PROFILES, profile),
    )


def _prepared_experiment(
    *,
    runtime_limit=10.0,
    ledger_limit=10.0,
    protocol=None,
    save_times=(0.0, 0.5, 1.0, 1.5, 2.0),
):
    protocol = (
        BatteryProtocolPlan((CurrentStepPlan(1.0), CurrentStepPlan(1.0)))
        if protocol is None
        else protocol
    )
    profile, support = _support()
    experiment = BatteryExperimentPlan(
        AnalyticCurrentAdapter(),
        protocol,
        BatteryOutputPlan(
            (
                "current_a",
                "voltage_v",
                "stoichiometry:negative",
                "stoichiometry:positive",
                "temperature_k",
            )
        ),
        BatteryDiffraxSolvePlan(
            solver=dfx.Tsit5(),
            stepsize_controller=dfx.PIDController(rtol=1.0e-8, atol=1.0e-10),
            dt0=0.05,
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            maximum_steps=2048,
        ),
        jnp.asarray(save_times),
        profile,
        support,
    ).prepare()
    parameters = AnalyticCurrentParameters(
        jnp.asarray(runtime_limit), jnp.asarray(ledger_limit)
    )
    return experiment, parameters


def _prepared_control(
    *,
    voltage_upper=4.5,
    runtime_limit=10.0,
    ledger_limit=10.0,
    protocol=None,
    save_times=(0.0, 0.5, 1.0, 1.5, 2.0),
):
    experiment, parameters = _prepared_experiment(
        runtime_limit=runtime_limit,
        ledger_limit=ledger_limit,
        protocol=protocol,
        save_times=save_times,
    )
    bounds = BatteryCurrentControlBounds(
        current_a=(-2.0, 2.0),
        voltage_v=(2.5, voltage_upper),
        stoichiometry=(0.0, 1.0),
        temperature_k=(290.0, 320.0),
    )
    target = BatteryCurrentControlTerminalTarget(
        1.0,
        state_index=0,
        name="stored-charge-coordinate",
    )
    objective = BatteryCurrentControlObjective(
        current_squared_weight=0.5,
        terminal_target_squared_weight=2.0,
    )
    return BatteryCurrentControlPlan(
        experiment,
        parameters,
        jnp.asarray(0.0),
        bounds,
        target,
        objective,
        parameter_source_id="test:analytic-current:calibration",
        ledger_success=lambda ledger: ledger.successful,
        ledger_criterion_id="test:analytic-current:ledger-success",
        replay_constraint_tolerance=2.0e-6,
    ).prepare()


def test_piecewise_current_lowering_exposes_fixed_knots_and_dynamic_amplitudes():
    prepared = _prepared_control()
    amplitudes = jnp.asarray((1.0, 0.0))
    lowered = prepared.lower(amplitudes)

    assert lowered.knot_times_s == (0.0, 1.0, 2.0)
    np.testing.assert_allclose(lowered.amplitudes_a, amplitudes)
    np.testing.assert_allclose(lowered.coefficients, ((1.0,), (0.0,)))
    np.testing.assert_allclose(
        prepared.protocol_values(amplitudes).current_amplitudes_a,
        amplitudes,
    )
    assert prepared.parameterization.parameter_shape == (2, 1)
    assert prepared.control_problem.time_grid.time_id == prepared.phase_time_grid.time_id
    assert float(prepared.horizon_s) == 2.0
    leaves = jax.tree.leaves(lowered)
    assert any(leaf is lowered.amplitudes_a for leaf in leaves)


def test_native_control_terminal_equality_objective_gradient_and_constraint_signs():
    prepared = _prepared_control()
    amplitudes = jnp.asarray((1.0, 0.0))
    result = prepared.evaluate(amplitudes)

    assert bool(result.trajectory.successful)
    np.testing.assert_allclose(result.trajectory.states[-1, 0], 1.0, atol=2.0e-6)
    np.testing.assert_allclose(result.feasibility.terminal_residuals, 0.0, atol=2.0e-6)
    assert bool(result.feasibility.feasible)
    current_upper = prepared.path_constraint_names.index("current-a:upper")
    current_lower = prepared.path_constraint_names.index("current-a:lower")
    np.testing.assert_allclose(
        result.feasibility.path_residuals[:, current_upper], (-1.0, -2.0)
    )
    np.testing.assert_allclose(
        result.feasibility.path_residuals[:, current_lower], (-3.0, -2.0)
    )

    def loss(values):
        return prepared.evaluate(values).sampled_loss.total

    point = jnp.asarray((0.4, 0.6))
    value, gradient = jax.value_and_grad(loss)(point)
    np.testing.assert_allclose(value, 0.26, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(gradient, point, rtol=2.0e-4, atol=2.0e-5)
    compiled = jax.jit(loss)(point)
    np.testing.assert_allclose(compiled, value, rtol=2.0e-5, atol=2.0e-6)

    violating = prepared.evaluate(jnp.asarray((2.5, -1.5)))
    assert not bool(violating.feasibility.feasible)
    assert float(violating.feasibility.path_residuals[0, current_upper]) == 0.5


def test_replay_is_independent_finer_and_catches_between_knot_voltage_violation():
    prepared = _prepared_control(voltage_upper=3.5)
    amplitudes = jnp.asarray((1.0, 0.0))
    sampled = prepared.evaluate(amplitudes)
    replay = prepared.replay(amplitudes)

    assert bool(sampled.feasibility.feasible)
    assert replay.path_residuals.shape[0] == 5
    voltage_upper = replay.path_constraint_names.index("voltage_v:upper")
    assert float(replay.path_residuals[1, voltage_upper]) > 0.49
    assert not bool(replay.feasible)
    assert int(replay.status) == int(
        BatteryCurrentControlReplayStatus.PATH_CONSTRAINT_VIOLATED
    )
    assert not replay.experiment_result.termination.terminated
    assert replay.replay_plan_id == prepared.replay_plan_id


def test_failed_runtime_and_ledger_are_explicit_infeasible_constraints():
    runtime_failure = _prepared_control(runtime_limit=0.75)
    runtime_evidence = runtime_failure.replay(jnp.asarray((1.0, 0.0)))
    assert not bool(runtime_evidence.feasible)
    assert float(runtime_evidence.execution_residuals[0]) > 0.0
    assert int(runtime_evidence.status) == int(
        BatteryCurrentControlReplayStatus.RUNTIME_FAILED
    )

    ledger_failure = _prepared_control(ledger_limit=0.75)
    ledger_evidence = ledger_failure.replay(jnp.asarray((1.0, 0.0)))
    assert not bool(ledger_evidence.experiment_result.successful)
    assert not bool(ledger_evidence.feasible)
    assert float(ledger_evidence.execution_residuals[1]) > 0.0
    assert int(ledger_evidence.status) == int(
        BatteryCurrentControlReplayStatus.LEDGER_FAILED
    )


def test_fixed_topology_deterministic_identities_and_native_nlp_compilation():
    first = _prepared_control()
    second = _prepared_control()
    assert first.plan.control_plan_id == second.plan.control_plan_id
    assert first.preparation_id == second.preparation_id
    assert first.replay_plan_id == second.replay_plan_id
    assert first.control_problem.problem_id == second.control_problem.problem_id
    assert first.plan.parameter_source_id == "test:analytic-current:calibration"
    assert first.plan.experiment.plan.support_tuple.support_tuple_id

    collocation = DirectCollocationPlan(
        TemporalMesh(
            first.current_knot_times_s,
            role="collocation",
            mesh_id="test:battery-current-control:collocation-mesh",
        ),
        method=ThetaMethod(0.5, endpoint=False),
        derivatives=DirectCollocationDerivativePolicy(verify=False),
        variable_duration=False,
        plan_id="test:battery-current-control:direct-collocation",
    )
    compilation = compile_direct_collocation(
        first.control_problem,
        collocation,
        jnp.asarray(((0.0,), (1.0,), (1.0,))),
        jnp.asarray(((1.0,), (0.0,))),
    )
    assert not compilation.plan.variable_duration
    assert compilation.problem.problem_id == first.control_problem.problem_id
    assert compilation.structured_program.program_id.endswith("direct-collocation")
    np.testing.assert_allclose(
        compilation.initial_decision.controls,
        first.lower(jnp.asarray((1.0, 0.0))).coefficients,
    )


def test_control_construction_rejects_moving_or_underresolved_phase_topology():
    rest_protocol = BatteryProtocolPlan((CurrentStepPlan(1.0), RestStepPlan(1.0)))
    with pytest.raises(ValueError, match="Every fixed current-control phase"):
        _prepared_control(protocol=rest_protocol)

    with pytest.raises(ValueError, match="finer interior save time per phase"):
        _prepared_control(save_times=(0.0, 1.0, 2.0))
