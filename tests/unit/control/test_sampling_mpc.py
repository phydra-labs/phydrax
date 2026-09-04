#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.control import ControlProblem, PiecewiseConstantControlParameterization
from phydrax.control._sampling_mpc import (
    plan_sampling_mpc,
    SamplingMPCRealizations,
    SamplingMPCStatus,
    shift_sampling_mpc_state,
    solve_sampling_mpc,
)
from phydrax.dynamics import TimeGrid
from phydrax.optim import Bounds, CVaRRisk, EntropicRisk, MeanVarianceRisk
from tests._control_systems import make_discrete_control_dynamics


def _scalar_problem(
    *,
    initial_state=jnp.asarray([0.0]),
    horizon: int = 2,
    dynamics=None,
    running_cost=None,
    terminal_cost=None,
    args=None,
    problem_id: str = "sampling-mpc-scalar",
):
    grid = TimeGrid(
        jnp.linspace(0.0, float(horizon), horizon + 1),
        time_id=f"{problem_id}:time",
    )
    transition = (
        (lambda context, state, control, args: state + control)
        if dynamics is None
        else dynamics
    )
    discrete = make_discrete_control_dynamics(
        transition,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id=f"{problem_id}:dynamics",
    )
    parameterization = PiecewiseConstantControlParameterization(
        grid,
        (1,),
        parameterization_id=f"{problem_id}:controls",
    )
    problem = ControlProblem(
        discrete,
        grid,
        initial_state,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        args=args,
        problem_id=problem_id,
    )
    return problem, parameterization


def test_zero_noise_is_nominal_and_complete_fixed_work_is_jittable():
    problem, parameterization = _scalar_problem(
        terminal_cost=lambda time, state, args: state[0] ** 2,
    )
    plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=5,
        iteration_count=3,
        elite_count=2,
    )
    nominal = jnp.asarray([[0.25], [-0.5]])
    state = plan.initialize(nominal, 0.0)
    solve = eqx.filter_jit(lambda value, key: solve_sampling_mpc(plan, value, key))
    result = solve(state, jax.random.key(7))

    np.testing.assert_array_equal(
        result.evidence.candidate_controls,
        jnp.broadcast_to(nominal, (3, 5, 2, 1)),
    )
    np.testing.assert_array_equal(result.controls, nominal)
    np.testing.assert_array_equal(result.action, nominal[0])
    assert result.evidence.candidate_controls.shape == (3, 5, 2, 1)
    assert result.evidence.model_objectives.shape == (3, 5, 1)
    assert result.evidence.candidate_objectives.shape == (3, 5)
    assert result.evidence.mean_history.shape == (4, 2, 1)
    assert int(result.evidence.completed_iterations) == 3
    assert int(result.total_candidate_evaluations) == 15
    assert int(result.total_model_rollouts) == 16
    assert bool(result.completed)
    assert int(result.status) == SamplingMPCStatus.SUCCESS
    assert result.plan_id == plan.plan_id
    assert result.evidence.plan_id == plan.plan_id


def test_cem_updates_toward_a_better_deterministic_scalar_control():
    problem, parameterization = _scalar_problem(
        horizon=1,
        terminal_cost=lambda time, state, args: (state[0] - 1.0) ** 2,
        problem_id="sampling-mpc-cem",
    )
    plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=128,
        iteration_count=4,
        elite_count=16,
        update="cem",
        update_rate=1.0,
        minimum_standard_deviation=0.01,
    )
    state = plan.initialize(jnp.asarray([[-3.0]]), 2.0)

    result = solve_sampling_mpc(plan, state, jax.random.key(11))

    initial_objective = 16.0
    assert bool(result.successful)
    assert float(result.objective) < initial_objective
    assert abs(float(result.state.mean[0, 0]) - 1.0) < 0.5
    assert np.min(np.asarray(result.evidence.candidate_objectives[-1])) < np.min(
        np.asarray(result.evidence.candidate_objectives[0])
    )


def test_failed_rollout_mask_is_separate_and_failed_candidate_cannot_win():
    def transition(context, state, control, args):
        return jnp.where(control[0] < 0.0, jnp.full_like(state, jnp.nan), state)

    problem, parameterization = _scalar_problem(
        horizon=1,
        dynamics=transition,
        running_cost=lambda time, state, control, args: control[0],
        problem_id="sampling-mpc-failed-candidate",
    )
    plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=256,
        iteration_count=1,
        elite_count=32,
        update="predictive",
    )
    state = plan.initialize(jnp.zeros((1, 1)), 1.0)

    result = solve_sampling_mpc(plan, state, jax.random.key(3))
    objectives = np.asarray(result.evidence.candidate_objectives[0])
    valid = np.asarray(result.evidence.candidate_valid[0])
    rollout_valid = np.asarray(result.evidence.model_rollout_valid[0, :, 0])

    assert valid.any() and (~valid).any()
    np.testing.assert_array_equal(valid, rollout_valid)
    assert np.all(np.isfinite(objectives[~rollout_valid]))
    assert np.min(objectives[~valid]) < float(result.objective)
    assert bool(result.evidence.candidate_valid[0, result.selected_candidate])
    assert float(result.action[0]) >= 0.0


def test_warm_start_horizon_shift_is_exact_for_hold_and_zero_tail_policies():
    problem, parameterization = _scalar_problem(
        horizon=3,
        problem_id="sampling-mpc-warm-shift",
    )
    controls = jnp.asarray([[1.0], [2.0], [3.0]])
    deviations = jnp.asarray([[0.1], [0.2], [0.3]])
    hold_plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=2,
        iteration_count=1,
        warm_start_terminal="hold",
    )
    hold_state = hold_plan.initialize(controls, deviations)
    shifted = shift_sampling_mpc_state(hold_plan, hold_state)
    np.testing.assert_array_equal(shifted.mean[:, 0], [2.0, 3.0, 3.0])
    np.testing.assert_array_equal(shifted.standard_deviation[:, 0], [0.2, 0.3, 0.3])

    zero_plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=2,
        iteration_count=1,
        warm_start_terminal="zero",
    )
    zero_state = zero_plan.initialize(controls, deviations)
    zero_shifted = zero_plan.shift(zero_state)
    np.testing.assert_array_equal(zero_shifted.mean[:, 0], [2.0, 3.0, 0.0])

    warm_result = solve_sampling_mpc(
        hold_plan,
        hold_state,
        jax.random.key(0),
        warm_start=True,
    )
    np.testing.assert_array_equal(
        warm_result.evidence.candidate_controls[0, 0],
        shifted.mean,
    )


def test_expectation_worst_case_and_existing_risk_measure_aggregate_model_axis():
    problem, parameterization = _scalar_problem(
        initial_state=jnp.asarray([[0.0], [4.0]]),
        horizon=1,
        terminal_cost=lambda time, state, args: state[0] ** 2,
        problem_id="sampling-mpc-randomized-model",
    )
    common = dict(
        candidate_count=512,
        iteration_count=1,
        elite_count=32,
        update="predictive",
        model_weights=jnp.asarray([0.9, 0.1]),
    )
    expectation_plan = plan_sampling_mpc(
        problem,
        parameterization,
        risk="expectation",
        **common,
    )
    worst_plan = plan_sampling_mpc(
        problem,
        parameterization,
        risk="worst_case",
        **common,
    )
    mean_variance_plan = plan_sampling_mpc(
        problem,
        parameterization,
        risk=MeanVarianceRisk(0.5),
        **common,
    )
    nominal = jnp.zeros((1, 1))
    key = jax.random.key(19)
    expectation = solve_sampling_mpc(
        expectation_plan,
        expectation_plan.initialize(nominal, 3.0),
        key,
    )
    worst = solve_sampling_mpc(
        worst_plan,
        worst_plan.initialize(nominal, 3.0),
        key,
    )
    mean_variance = solve_sampling_mpc(
        mean_variance_plan,
        mean_variance_plan.initialize(nominal, 3.0),
        key,
    )

    assert expectation.evidence.model_objectives.shape == (1, 512, 2)
    assert float(expectation.action[0]) > float(worst.action[0]) + 0.75
    assert abs(float(expectation.action[0]) + 0.4) < 0.2
    assert abs(float(worst.action[0]) + 2.0) < 0.2
    assert bool(mean_variance.successful)
    assert mean_variance_plan.aggregation == "risk_measure"


def test_declared_clip_and_reject_bound_policies_are_auditable():
    problem, parameterization = _scalar_problem(
        horizon=1,
        terminal_cost=lambda time, state, args: (state[0] - 2.0) ** 2,
        problem_id="sampling-mpc-bounds",
    )
    common = dict(
        candidate_count=128,
        iteration_count=1,
        elite_count=16,
        update="predictive",
        bounds=Bounds(-0.25, 0.25),
    )
    clip_plan = plan_sampling_mpc(
        problem, parameterization, bound_policy="clip", **common
    )
    reject_plan = plan_sampling_mpc(
        problem, parameterization, bound_policy="reject", **common
    )
    key = jax.random.key(5)
    clip = solve_sampling_mpc(
        clip_plan, clip_plan.initialize(jnp.zeros((1, 1)), 3.0), key
    )
    reject = solve_sampling_mpc(
        reject_plan, reject_plan.initialize(jnp.zeros((1, 1)), 3.0), key
    )

    assert np.max(np.abs(np.asarray(clip.evidence.candidate_controls))) <= 0.25
    raw_reject = np.asarray(reject.evidence.candidate_controls[0, :, 0, 0])
    rejected = ~np.asarray(reject.evidence.candidate_valid[0])
    assert np.any(np.abs(raw_reject) > 0.25)
    assert np.all(rejected[np.abs(raw_reject) > 0.25])
    assert -0.25 <= float(reject.action[0]) <= 0.25


def _explicit_realization_problem(*, failing_unsupported: bool = False):
    def transition(context, state, control, args):
        candidate = state + args["gain"] * control
        if failing_unsupported:
            candidate = jnp.where(
                args["gain"] > 0.0,
                candidate,
                jnp.full_like(candidate, jnp.nan),
            )
        return candidate

    return _scalar_problem(
        horizon=1,
        dynamics=transition,
        terminal_cost=lambda time, state, args: (state[0] - args["target"]) ** 2,
        args={"target": jnp.asarray(0.0)},
        problem_id="sampling-mpc-explicit-realizations",
    )


def _bind_gain(base_args, realization_parameters):
    return {
        "target": base_args["target"],
        "gain": realization_parameters["gain"],
    }


def test_explicit_parameter_pytrees_change_rollouts_and_replay_both_states():
    problem, parameterization = _explicit_realization_problem()
    realizations = SamplingMPCRealizations(
        {"gain": jnp.asarray([1.0, 3.0, 9.0])},
        ("soft:low", "soft:high", "padding"),
        weights=jnp.asarray([0.25, 0.75, 0.0]),
        support_mask=jnp.asarray([True, True, False]),
        posterior_id="posterior:calibrated-rod",
        campaign_id="campaign:held-out-a",
        policy="fixed",
    )
    plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=1,
        iteration_count=1,
        update="predictive",
        realizations=realizations,
        realization_binding=_bind_gain,
        realization_binding_id="bind:gain+target",
    )

    result = plan.solve(
        plan.initialize(jnp.asarray([[1.0]]), 0.0),
        jax.random.key(4),
    )

    np.testing.assert_allclose(result.replay.states[:, -1, 0], [1.0, 3.0, 9.0])
    np.testing.assert_allclose(
        result.replay.transition_candidate_states[:, 0, 0],
        result.replay.transition_accepted_states[:, 0, 0],
    )
    np.testing.assert_array_equal(
        result.evidence.realization_support,
        [True, True, False],
    )
    np.testing.assert_allclose(
        result.evidence.realization_weights,
        [0.25, 0.75, 0.0],
    )
    assert result.evidence.realization_ids == realizations.realization_ids
    assert result.evidence.posterior_id == "posterior:calibrated-rod"
    assert result.evidence.campaign_id == "campaign:held-out-a"
    assert result.replay.realization_batch_id == realizations.batch_id
    assert bool(result.replay.accepted)


def test_zero_weight_failed_padding_cannot_contaminate_any_risk_or_acceptance():
    problem, parameterization = _explicit_realization_problem(failing_unsupported=True)
    realizations = SamplingMPCRealizations(
        {"gain": jnp.asarray([1.0, -1.0])},
        ("supported", "failed-padding"),
        weights=jnp.asarray([1.0, 0.0]),
        support_mask=jnp.asarray([True, False]),
        posterior_id="posterior:support-mask",
        campaign_id="campaign:support-mask",
    )
    risks = (
        "expectation",
        "worst_case",
        MeanVarianceRisk(0.5),
        CVaRRisk(0.5),
        EntropicRisk(1.0),
    )

    for risk in risks:
        plan = plan_sampling_mpc(
            problem,
            parameterization,
            candidate_count=1,
            iteration_count=1,
            update="predictive",
            risk=risk,
            realizations=realizations,
            realization_binding=_bind_gain,
            realization_binding_id="bind:gain+target",
        )
        result = plan.solve(
            plan.initialize(jnp.asarray([[1.0]]), 0.0),
            jax.random.key(5),
        )

        assert bool(result.successful)
        assert np.isfinite(float(result.objective))
        assert not bool(result.evidence.model_rollout_valid[0, 0, 1])
        assert bool(result.evidence.model_valid[0, 0, 1])
        assert bool(result.replay.accepted)


def test_positive_weight_rollout_failure_rejects_the_candidate():
    problem, parameterization = _explicit_realization_problem(failing_unsupported=True)
    realizations = SamplingMPCRealizations(
        {"gain": jnp.asarray([1.0, -1.0])},
        ("supported", "failed-supported"),
        weights=jnp.asarray([0.5, 0.5]),
        support_mask=jnp.asarray([True, True]),
        posterior_id="posterior:positive-failure",
        campaign_id="campaign:positive-failure",
    )
    plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=1,
        iteration_count=1,
        update="predictive",
        realizations=realizations,
        realization_binding=_bind_gain,
        realization_binding_id="bind:gain+target",
    )

    result = plan.solve(
        plan.initialize(jnp.asarray([[1.0]]), 0.0),
        jax.random.key(6),
    )

    assert not bool(result.successful)
    assert int(result.status) == SamplingMPCStatus.NO_VALID_CANDIDATE
    assert not bool(result.replay.accepted)


def test_mass_stiffness_and_friction_realizations_are_not_shared():
    problem, parameterization = _scalar_problem(
        initial_state=jnp.asarray([1.0]),
        horizon=1,
        dynamics=lambda context, state, control, args: (
            (1.0 - args["friction"]) * state + args["stiffness"] * control / args["mass"]
        ),
        terminal_cost=lambda time, state, args: jnp.square(state[0]),
        problem_id="sampling-mpc-mass-stiffness-friction",
    )
    realizations = SamplingMPCRealizations(
        {
            "mass": jnp.asarray([1.0, 2.0, 1.0]),
            "stiffness": jnp.asarray([1.0, 1.0, 2.0]),
            "friction": jnp.asarray([0.1, 0.1, 0.3]),
        },
        ("nominal", "heavy", "stiff-frictional"),
        posterior_id="posterior:mechanics",
        campaign_id="campaign:mechanics",
    )
    plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=1,
        iteration_count=1,
        update="predictive",
        realizations=realizations,
        realization_binding=lambda base, physical: physical,
        realization_binding_id="bind:mass-stiffness-friction",
    )

    result = plan.solve(
        plan.initialize(jnp.asarray([[1.0]]), 0.0),
        jax.random.key(9),
    )

    np.testing.assert_allclose(
        result.replay.states[:, -1, 0],
        [1.9, 1.4, 2.7],
    )
    assert np.unique(np.asarray(result.replay.model_objectives)).size == 3


def test_resample_policy_is_key_and_solve_count_reproducible_and_auditable():
    problem, parameterization = _explicit_realization_problem()
    realizations = SamplingMPCRealizations(
        {"gain": jnp.asarray([1.0, 2.0, 4.0, 8.0])},
        ("r0", "r1", "r2", "r3"),
        weights=jnp.asarray([0.7, 0.1, 0.1, 0.1]),
        support_mask=jnp.ones((4,), dtype=bool),
        posterior_id="posterior:draws",
        campaign_id="campaign:draws",
        policy="resample",
    )
    plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=1,
        iteration_count=2,
        update="predictive",
        realizations=realizations,
        realization_binding=_bind_gain,
        realization_binding_id="bind:gain+target",
    )
    state = plan.initialize(jnp.asarray([[0.5]]), 0.0)
    key = jax.random.key(10)
    first = plan.solve(state, key)
    repeated = plan.solve(state, key)
    next_solve = plan.solve(first.state, key)

    def expected_indices(solve_count):
        draw_key = jax.random.fold_in(
            jax.random.fold_in(key, jnp.asarray(solve_count, dtype=jnp.int32)),
            jnp.asarray(2**31 - 1, dtype=jnp.uint32),
        )
        return jax.random.categorical(
            draw_key,
            jnp.log(realizations.weights),
            shape=(realizations.count,),
        ).astype(jnp.int32)

    np.testing.assert_array_equal(
        first.evidence.realization_indices,
        repeated.evidence.realization_indices,
    )
    np.testing.assert_array_equal(
        first.evidence.realization_indices[0],
        first.evidence.realization_indices[1],
    )
    np.testing.assert_allclose(first.evidence.realization_weights, 0.25)
    np.testing.assert_array_equal(
        first.evidence.realization_support,
        jnp.ones((4,), dtype=bool),
    )
    assert first.evidence.realization_policy == "resample"
    np.testing.assert_array_equal(
        first.evidence.realization_indices[0],
        expected_indices(0),
    )
    np.testing.assert_array_equal(
        next_solve.evidence.realization_indices[0],
        expected_indices(1),
    )
