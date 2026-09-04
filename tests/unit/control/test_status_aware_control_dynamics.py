#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.control import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_SUCCESS,
    DiscreteControlDynamics,
    PiecewiseConstantControlParameterization,
)
from phydrax.dynamics import (
    DiscreteStepContext,
    DiscreteSystem,
    InputLayout,
    StateLayout,
    TimeGrid,
)
from phydrax.dynamics._system import DiscreteTransitionResult


def _grid_and_parameterization():
    grid = TimeGrid(jnp.asarray([0.0, 1.0, 2.0]), time_id="status-aware-grid")
    parameterization = PiecewiseConstantControlParameterization(
        grid,
        (1,),
        parameterization_id="status-aware-controls",
    )
    return grid, parameterization


def _dynamics(transition, *, system_id):
    return DiscreteControlDynamics(
        DiscreteSystem(
            transition,
            state_layout=StateLayout((1,)),
            input_layout=InputLayout((1,), roles="control"),
            system_id=system_id,
        )
    )


class _ScaledTransition(eqx.Module):
    scale: jax.Array

    def __call__(self, context, state, control, args):
        del context, args
        following = state + self.scale * control
        return DiscreteTransitionResult(
            following,
            following,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        )


def test_discrete_system_preserves_legacy_array_evaluation():
    system = DiscreteSystem(
        lambda context, state, control, args: state + context.duration * control,
        state_layout=StateLayout((1,)),
        input_layout=InputLayout((1,), roles="control"),
        system_id="legacy-array-transition",
    )
    context = DiscreteStepContext(0.0, 0.5, 0)
    state = jnp.asarray([2.0])
    control = jnp.asarray([3.0])

    result = jax.jit(
        lambda current, inputs: system.evaluate_result(
            context,
            current,
            inputs=inputs,
        )
    )(state, control)

    np.testing.assert_allclose(result.candidate_state, jnp.asarray([3.5]))
    np.testing.assert_allclose(result.accepted_state, jnp.asarray([3.5]))
    assert bool(result.successful)
    assert int(result.status) == 0
    np.testing.assert_allclose(
        system.evaluate(context, state, inputs=control),
        result.accepted_state,
    )
    np.testing.assert_allclose(
        system(context, state, inputs=control),
        result.accepted_state,
    )


def test_filtered_jit_keeps_transition_parameters_differentiable_and_refreshable():
    grid, parameterization = _grid_and_parameterization()
    system = DiscreteSystem(
        _ScaledTransition(jnp.asarray(1.5)),
        state_layout=StateLayout((1,)),
        input_layout=InputLayout((1,), roles="control"),
        system_id="trainable-filtered-transition",
    )
    coefficients = jnp.asarray([[1.0], [2.0]])

    def objective(candidate):
        trajectory = DiscreteControlDynamics(candidate).rollout(
            grid,
            jnp.asarray([2.0]),
            parameterization,
            coefficients,
            problem_id="trainable-filtered-transition",
        )
        return trajectory.final_state[0]

    compiled_value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(objective))
    value, gradient = compiled_value_and_grad(system)
    np.testing.assert_allclose(value, 6.5)
    np.testing.assert_allclose(gradient.transition.scale, 3.0)

    refreshed = eqx.tree_at(
        lambda candidate: candidate.transition.scale,
        system,
        jnp.asarray(2.0),
    )
    refreshed_value, refreshed_gradient = compiled_value_and_grad(refreshed)
    np.testing.assert_allclose(refreshed_value, 8.0)
    np.testing.assert_allclose(refreshed_gradient.transition.scale, 3.0)


def test_failed_finite_rollback_remains_invalid_and_preserves_backend_status():
    failure_status = 37

    def transition(context, state, control, args):
        del args
        failed = context.step_index == 0
        candidate = state + control + 100.0
        accepted = jnp.where(failed, state, state + control)
        return DiscreteTransitionResult(
            candidate,
            accepted,
            ~failed,
            jnp.where(failed, failure_status, 0),
        )

    grid, parameterization = _grid_and_parameterization()
    dynamics = _dynamics(transition, system_id="finite-rollback-transition")
    trajectory = jax.jit(
        lambda coefficients: dynamics.rollout(
            grid,
            jnp.asarray([2.0]),
            parameterization,
            coefficients,
            problem_id="finite-rollback",
        )
    )(jnp.asarray([[1.0], [2.0]]))

    np.testing.assert_array_equal(trajectory.valid, jnp.asarray([True, False, False]))
    assert int(trajectory.status) == CONTROL_DYNAMICS_FAILED
    assert int(trajectory.backend_status) == failure_status
    assert float(trajectory.states[1, 0]) == 2.0
    assert bool(jnp.isnan(trajectory.states[2, 0]))
    assert float(trajectory.controls[0, 0]) == 1.0
    assert bool(jnp.isnan(trajectory.controls[1, 0]))
    evidence = trajectory.transition_evidence
    assert evidence is not None
    np.testing.assert_allclose(evidence.candidate_states[0], jnp.asarray([103.0]))
    np.testing.assert_allclose(evidence.accepted_states[0], jnp.asarray([2.0]))
    assert bool(jnp.isnan(evidence.candidate_states[1, 0]))
    np.testing.assert_array_equal(evidence.successful, jnp.asarray([False, False]))
    np.testing.assert_array_equal(evidence.attempted, jnp.asarray([True, False]))
    np.testing.assert_array_equal(
        evidence.status,
        jnp.asarray([failure_status, 0], dtype=jnp.int32),
    )
    assert int(evidence.first_failure_step) == 0
    assert int(evidence.first_failure_status) == failure_status


def test_invalid_control_and_post_failure_steps_are_unattempted():
    grid, parameterization = _grid_and_parameterization()
    dynamics = _dynamics(
        lambda context, state, control, args: state + control,
        system_id="invalid-control-transition",
    )

    trajectory = dynamics.rollout(
        grid,
        jnp.asarray([2.0]),
        parameterization,
        jnp.asarray([[jnp.nan], [1.0]]),
        problem_id="invalid-control",
    )

    evidence = trajectory.transition_evidence
    assert evidence is not None
    np.testing.assert_array_equal(evidence.attempted, jnp.asarray([False, False]))
    np.testing.assert_array_equal(evidence.successful, jnp.asarray([False, False]))
    np.testing.assert_array_equal(
        evidence.status,
        jnp.zeros((2,), dtype=jnp.int32),
    )
    assert int(evidence.first_failure_step) == -1
    assert int(evidence.first_failure_status) == 0


def test_successful_result_and_legacy_rollouts_agree_under_batching_and_jit():
    def legacy_transition(context, state, control, args):
        del context, args
        return state + control

    def result_transition(context, state, control, args):
        del context, args
        accepted = state + control
        return DiscreteTransitionResult(
            accepted + 100.0,
            accepted,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        )

    grid, parameterization = _grid_and_parameterization()
    initial_state = jnp.asarray([[0.0], [10.0]])
    coefficients = jnp.asarray([[[1.0], [2.0]], [[-1.0], [-2.0]]])
    legacy = _dynamics(legacy_transition, system_id="legacy-batched-transition")
    status_aware = _dynamics(result_transition, system_id="result-batched-transition")

    legacy_trajectory, result_trajectory = jax.jit(
        lambda values: (
            legacy.rollout(
                grid,
                initial_state,
                parameterization,
                values,
                problem_id="legacy-batched",
            ),
            status_aware.rollout(
                grid,
                initial_state,
                parameterization,
                values,
                problem_id="result-batched",
            ),
        )
    )(coefficients)

    np.testing.assert_allclose(result_trajectory.states, legacy_trajectory.states)
    np.testing.assert_allclose(result_trajectory.controls, legacy_trajectory.controls)
    np.testing.assert_array_equal(result_trajectory.valid, legacy_trajectory.valid)
    np.testing.assert_array_equal(
        result_trajectory.status,
        jnp.full((2,), CONTROL_SUCCESS, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        result_trajectory.backend_status,
        jnp.zeros((2,), dtype=jnp.int32),
    )
    legacy_evidence = legacy_trajectory.transition_evidence
    result_evidence = result_trajectory.transition_evidence
    assert legacy_evidence is not None
    assert result_evidence is not None
    np.testing.assert_allclose(
        legacy_evidence.accepted_states,
        result_evidence.accepted_states,
    )
    np.testing.assert_array_equal(
        legacy_evidence.attempted,
        result_evidence.attempted,
    )
    assert bool(jnp.all(legacy_evidence.attempted))
    np.testing.assert_array_equal(
        legacy_evidence.successful,
        result_evidence.successful,
    )
    np.testing.assert_array_equal(legacy_evidence.status, result_evidence.status)


def test_batched_result_failures_preserve_per_case_validity_and_status():
    failure_status = 53

    def transition(context, state, control, args):
        del context, args
        successful = control[0] >= 0.0
        candidate = state + control + 100.0
        accepted = jnp.where(successful, state + control, state)
        return DiscreteTransitionResult(
            candidate,
            accepted,
            successful,
            jnp.where(successful, 0, failure_status),
        )

    grid, parameterization = _grid_and_parameterization()
    dynamics = _dynamics(transition, system_id="casewise-result-transition")
    trajectory = jax.jit(
        lambda coefficients: dynamics.rollout(
            grid,
            jnp.asarray([[0.0], [10.0]]),
            parameterization,
            coefficients,
            problem_id="casewise-result",
        )
    )(jnp.asarray([[[1.0], [2.0]], [[-1.0], [5.0]]]))

    np.testing.assert_allclose(
        trajectory.states[0, :, 0],
        jnp.asarray([0.0, 1.0, 3.0]),
    )
    assert float(trajectory.states[1, 1, 0]) == 10.0
    assert bool(jnp.isnan(trajectory.states[1, 2, 0]))
    np.testing.assert_array_equal(
        trajectory.valid,
        jnp.asarray([[True, True, True], [True, False, False]]),
    )
    np.testing.assert_array_equal(
        trajectory.status,
        jnp.asarray([CONTROL_SUCCESS, CONTROL_DYNAMICS_FAILED]),
    )
    np.testing.assert_array_equal(
        trajectory.backend_status,
        jnp.asarray([0, failure_status], dtype=jnp.int32),
    )
    evidence = trajectory.transition_evidence
    assert evidence is not None
    assert evidence.candidate_states.shape == (2, 2, 1)
    np.testing.assert_array_equal(
        evidence.first_failure_step,
        jnp.asarray([-1, 0], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        evidence.first_failure_status,
        jnp.asarray([0, failure_status], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        evidence.attempted,
        jnp.asarray([[True, True], [True, False]]),
    )
    np.testing.assert_array_equal(
        evidence.status,
        jnp.asarray([[0, 0], [failure_status, 0]], dtype=jnp.int32),
    )
