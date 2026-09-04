#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.motor_units import PotvinFuglevand2017Plan
from phydrax.applications.skeletal_muscle.personalization import (
    SkeletalReplayObservationOperator,
    SkeletalSurrogateReplayPlan,
)
from phydrax.control import (
    ControlProblem,
    DiscreteControlDynamics,
    PiecewiseConstantControlParameterization,
    plan_sampling_mpc,
    solve_sampling_mpc,
)
from phydrax.dynamics import TimeGrid
from phydrax.optim import Bounds


def _motor_control_problem():
    model_plan = PotvinFuglevand2017Plan(
        central_adaptation=False,
        peripheral_fatigue=False,
    )
    runtime = model_plan.prepare()
    dynamics = DiscreteControlDynamics(model_plan.as_discrete_system())
    grid = TimeGrid(jnp.asarray((0.0, 0.1, 0.2)), time_id="motor-unit-control-time")
    parameterization = PiecewiseConstantControlParameterization(
        grid, (1,), parameterization_id="motor-unit-excitation"
    )
    initial = runtime.pack_state(runtime.initialize())
    target_force = runtime.evaluate(runtime.initialize(), 20.0).total_force

    def running_cost(time, state, control, parameters):
        del time, parameters
        typed = runtime.unpack_state(state)
        force = runtime.evaluate(typed, control[0]).total_force
        return ((force - target_force) / target_force) ** 2 + 1.0e-5 * (
            control[0] / 67.0
        ) ** 2

    problem = ControlProblem(
        dynamics,
        grid,
        initial,
        running_cost=running_cost,
        args=runtime.parameters,
        problem_id="hard-motor-unit-sampling-mpc",
    )
    return runtime, problem, parameterization, target_force


def test_surrogate_decision_requires_causal_exact_control_replay():
    runtime, problem, parameterization, target_force = _motor_control_problem()

    def exact_force(trajectory):
        return jax.vmap(
            lambda state, control: runtime.evaluate(
                runtime.unpack_state(state), control[0]
            ).total_force
        )(trajectory.states[:-1], trajectory.controls)

    replay = SkeletalSurrogateReplayPlan(
        problem,
        parameterization,
        SkeletalReplayObservationOperator(
            exact_force, "potvin-fuglevand-relative-force-observation"
        ),
        jnp.ones((2,), dtype=bool),
        "learned-force-surrogate",
        "relative_muscle_force",
        absolute_tolerance=0.05,
        relative_tolerance=0.02,
    )
    controls = jnp.full((2, 1), 20.0)
    accepted = replay.evaluate(controls, jnp.full((2,), target_force))
    rejected = replay.evaluate(controls, jnp.full((2,), 1.2 * target_force))

    assert bool(accepted.accepted)
    assert not bool(rejected.accepted)
    assert bool(accepted.exact_result.successful)
    assert accepted.source_problem_id == problem.problem_id
    assert accepted.source_dynamics_id == problem.dynamics.dynamics_id
    assert accepted.source_control_id == parameterization.parameterization_id
    assert accepted.active_sample_count == 2


def test_pure_relative_replay_handles_zero_exact_values_without_nan():
    _, problem, parameterization, _ = _motor_control_problem()
    replay = SkeletalSurrogateReplayPlan(
        problem,
        parameterization,
        SkeletalReplayObservationOperator(
            lambda trajectory: jnp.zeros(trajectory.controls.shape[:-1]),
            "zero-exact-relative-observation",
        ),
        jnp.ones((2,), dtype=bool),
        "zero-reference-surrogate",
        "zero_reference_quantity",
        absolute_tolerance=0.0,
        relative_tolerance=0.02,
    )
    controls = jnp.full((2, 1), 20.0)
    zero_values = jnp.zeros((2,))
    nonzero_values = jnp.ones((2,))

    zero_evidence = replay.evaluate(controls, zero_values)
    nonzero_evidence = replay.evaluate(controls, nonzero_values)
    relative_gradient = jax.grad(
        lambda values: replay.evaluate(controls, values).maximum_relative_error
    )

    assert zero_evidence.maximum_relative_error == 0.0
    assert not bool(jnp.isnan(zero_evidence.maximum_relative_error))
    assert bool(zero_evidence.finite)
    assert bool(zero_evidence.accepted)
    assert bool(jnp.isinf(nonzero_evidence.maximum_relative_error))
    assert not bool(jnp.isnan(nonzero_evidence.maximum_relative_error))
    assert not bool(nonzero_evidence.finite)
    assert not bool(nonzero_evidence.accepted)
    assert jnp.all(jnp.isfinite(relative_gradient(zero_values)))
    assert jnp.all(jnp.isfinite(relative_gradient(nonzero_values)))


def test_replay_promotes_integer_exact_values_before_comparison():
    _, problem, parameterization, _ = _motor_control_problem()
    replay = SkeletalSurrogateReplayPlan(
        problem,
        parameterization,
        SkeletalReplayObservationOperator(
            lambda trajectory: jnp.ones(
                trajectory.controls.shape[:-1], dtype=jnp.int32
            ),
            "integer-exact-observation",
        ),
        jnp.ones((2,), dtype=bool),
        "floating-surrogate",
        "shared_real_comparison",
        absolute_tolerance=0.0,
        relative_tolerance=0.1,
    )

    evidence = replay.evaluate(
        jnp.full((2, 1), 20.0),
        jnp.full((2,), 1.9),
    )

    assert jnp.issubdtype(evidence.exact_values.dtype, jnp.floating)
    assert jnp.issubdtype(evidence.surrogate_values.dtype, jnp.floating)
    assert jnp.isclose(evidence.maximum_absolute_error, 0.9)
    assert jnp.isclose(evidence.maximum_relative_error, 0.9)
    assert not bool(evidence.accepted)

def test_surrogate_replay_rejects_exact_but_physically_infeasible_control():
    runtime, problem, parameterization, target_force = _motor_control_problem()

    def exact_force(trajectory):
        return jax.vmap(
            lambda state, control: runtime.evaluate(
                runtime.unpack_state(state), control[0]
            ).total_force
        )(trajectory.states[:-1], trajectory.controls)

    constrained = ControlProblem(
        problem.dynamics,
        problem.time_grid,
        problem.initial_state,
        running_cost=problem.running_cost,
        path_constraints=(
            lambda time, state, control, args: control[0] - 10.0,
        ),
        args=problem.args,
        problem_id="infeasible-surrogate-replay-control",
    )
    replay = SkeletalSurrogateReplayPlan(
        constrained,
        parameterization,
        SkeletalReplayObservationOperator(
            exact_force, "potvin-fuglevand-relative-force-observation"
        ),
        jnp.ones((2,), dtype=bool),
        "infeasible-force-surrogate",
        "relative_muscle_force",
        absolute_tolerance=0.05,
        relative_tolerance=0.02,
    )
    result = replay.evaluate(
        jnp.full((2, 1), 20.0),
        jnp.full((2,), target_force),
    )

    assert not bool(result.exact_result.feasibility.feasible)
    assert not bool(result.exact_result.successful)
    assert not bool(result.accepted)


def test_exact_observation_operator_identity_prevents_replay_provenance_collision():
    _, problem, parameterization, _ = _motor_control_problem()
    first_state_coordinate = SkeletalReplayObservationOperator(
        lambda trajectory: trajectory.states[:-1, 0, 0],
        "motor-unit-first-state-coordinate-observation",
    )
    second_state_coordinate = SkeletalReplayObservationOperator(
        lambda trajectory: trajectory.states[:-1, 1, 0],
        "motor-unit-second-state-coordinate-observation",
    )
    state_replay = SkeletalSurrogateReplayPlan(
        problem,
        parameterization,
        first_state_coordinate,
        jnp.ones((2,), dtype=bool),
        "same-surrogate",
        "same-quantity",
        absolute_tolerance=0.05,
        relative_tolerance=0.02,
    )
    second_state_replay = SkeletalSurrogateReplayPlan(
        problem,
        parameterization,
        second_state_coordinate,
        jnp.ones((2,), dtype=bool),
        "same-surrogate",
        "same-quantity",
        absolute_tolerance=0.05,
        relative_tolerance=0.02,
    )
    controls = jnp.full((2, 1), 20.0)
    state_evidence = state_replay.evaluate(controls, jnp.zeros((2,)))
    second_state_evidence = second_state_replay.evaluate(controls, jnp.zeros((2,)))

    assert state_replay.plan_id != second_state_replay.plan_id
    assert (
        state_evidence.observation_operator_id == first_state_coordinate.operator_id
    )
    assert (
        second_state_evidence.observation_operator_id
        == second_state_coordinate.operator_id
    )
    assert state_evidence.replay_id != second_state_evidence.replay_id


def test_sampling_mpc_uses_exact_hard_motor_unit_rollouts():
    runtime, problem, parameterization, _ = _motor_control_problem()
    mpc = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=64,
        iteration_count=3,
        elite_count=8,
        bounds=Bounds(0.0, 67.0),
        minimum_standard_deviation=0.1,
    )
    state = mpc.initialize(jnp.full((2, 1), 10.0), jnp.full((2, 1), 8.0))
    result = solve_sampling_mpc(mpc, state, jax.random.key(7))

    assert bool(result.successful)
    assert 0.0 <= result.action[0] <= 67.0
    assert result.objective < 1.0
    assert result.evidence.model_rollout_valid.shape[-1] == 1
    assert runtime.plan.plan_id == problem.dynamics.dynamics_id
