#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_layouts_distinguish_absent_and_scalar_inputs():
    state = phx.dynamics.StateLayout((), component_names=("temperature",))
    scalar_input = phx.dynamics.InputLayout(
        (), component_names=("forcing",), roles="forcing"
    )

    assert state.shape == ()
    assert state.axes == ()
    assert state.size == 1
    assert scalar_input.shape == ()
    assert scalar_input.size == 1
    assert scalar_input.roles == ("forcing",)

    autonomous = phx.dynamics.ContinuousSystem(
        lambda time, value, args: -value,
        state_layout=state,
        system_id="scalar-autonomous",
    )
    driven = phx.dynamics.ContinuousSystem(
        lambda time, value, inputs, args: -value + inputs,
        state_layout=state,
        input_layout=scalar_input,
        system_id="scalar-driven",
    )

    np.testing.assert_allclose(autonomous(0.0, jnp.asarray(2.0)), -2.0)
    np.testing.assert_allclose(
        driven(0.0, jnp.asarray(2.0), inputs=jnp.asarray(0.5)), -1.5
    )
    with pytest.raises(ValueError, match="requires explicit inputs"):
        driven(0.0, jnp.asarray(2.0))
    with pytest.raises(ValueError, match="does not accept inputs"):
        autonomous(0.0, jnp.asarray(2.0), inputs=jnp.asarray(0.5))


def test_discrete_evolution_rollout_and_jacobian_share_one_transition():
    state_layout = phx.dynamics.StateLayout((2,), component_names=("x", "y"))
    matrix = jnp.asarray([[1.1, 0.2], [0.0, 0.9]])
    system = phx.dynamics.DiscreteSystem(
        lambda step, state, args: matrix @ state,
        state_layout=state_layout,
        system_id="linear-map",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    grid = phx.dynamics.IterationGrid.from_steps(5, iteration_id="linear-map-steps")
    initial = jnp.asarray([0.4, -0.3])

    rollout = eqx.filter_jit(phx.dynamics.evolve)(evolution, initial, grid)
    expected = [initial]
    for _ in range(5):
        expected.append(matrix @ expected[-1])
    np.testing.assert_allclose(rollout.states, jnp.stack(expected), atol=1e-13)
    assert bool(rollout.successful)

    action = phx.dynamics.EvolutionJacobianAction(evolution, initial, 0, 1)
    np.testing.assert_allclose(action.as_dense(), matrix, atol=1e-13)
    vector = jnp.asarray([0.2, 0.7])
    np.testing.assert_allclose(action.mv(vector), matrix @ vector, atol=1e-13)
    np.testing.assert_allclose(action.transpose_mv(vector), matrix.T @ vector, atol=1e-13)


def test_input_policy_is_bound_and_differentiated_with_the_map():
    state_layout = phx.dynamics.StateLayout((1,))
    input_layout = phx.dynamics.InputLayout((1,), roles="control")
    system = phx.dynamics.DiscreteSystem(
        lambda step, state, inputs, args: state + inputs,
        state_layout=state_layout,
        input_layout=input_layout,
        system_id="feedback-map",
    )
    policy = phx.dynamics.CallableInputPolicy(
        lambda step, state, args: -0.25 * state,
        input_layout=input_layout,
        policy_id="quarter-feedback",
    )
    evolution = phx.dynamics.DiscreteEvolution(system, input_policy=policy)

    result = phx.dynamics.evolve(
        evolution,
        jnp.asarray([2.0]),
        phx.dynamics.IterationGrid.from_steps(3, iteration_id="feedback-steps"),
    )
    np.testing.assert_allclose(
        result.states[:, 0], jnp.asarray([2.0, 1.5, 1.125, 0.84375])
    )
    tangent = evolution.tangent_action(jnp.asarray([2.0]), jnp.asarray([1.0]), 0, 1)
    np.testing.assert_allclose(tangent.tangent, jnp.asarray([0.75]))


def test_nonfinite_map_result_is_invalid_without_repair():
    layout = phx.dynamics.StateLayout((1,))
    system = phx.dynamics.DiscreteSystem(
        lambda step, state, args: jnp.asarray([jnp.nan]),
        state_layout=layout,
        system_id="nonfinite-map",
    )
    result = phx.dynamics.DiscreteEvolution(system).advance(jnp.asarray([1.0]), 0, 1)

    assert not bool(result.valid)
    assert int(result.status) == phx.dynamics.EVOLUTION_NONFINITE
    assert bool(jnp.isnan(result.final_state[0]))



def test_failed_finite_discrete_rollback_invalidates_evolution_and_tangent():
    failure_status = 71

    def transition(context, state, args):
        del context, args
        return phx.dynamics.DiscreteTransitionResult(
            state + 100.0,
            state,
            jnp.asarray(False),
            jnp.asarray(failure_status, dtype=jnp.int32),
        )

    evolution = phx.dynamics.DiscreteEvolution(
        phx.dynamics.DiscreteSystem(
            transition,
            state_layout=phx.dynamics.StateLayout((1,)),
            system_id="failed-finite-evolution",
        )
    )
    trajectory = phx.dynamics.evolve(
        evolution,
        jnp.asarray([2.0]),
        phx.dynamics.IterationGrid.from_steps(
            3,
            iteration_id="failed-finite-evolution-grid",
        ),
    )

    np.testing.assert_allclose(trajectory.states[:, 0], jnp.asarray([2.0] * 4))
    np.testing.assert_array_equal(
        trajectory.valid,
        jnp.asarray([True, False, False, False]),
    )
    evidence = trajectory.transition_evidence
    assert evidence is not None
    np.testing.assert_allclose(evidence.candidate_states[0], jnp.asarray([102.0]))
    np.testing.assert_allclose(evidence.accepted_states[0], jnp.asarray([2.0]))
    assert bool(jnp.all(jnp.isnan(evidence.candidate_states[1:])))
    assert int(evidence.first_failure_step) == 0
    assert int(evidence.first_failure_status) == failure_status

    tangent = evolution.tangent_action(
        jnp.asarray([2.0]),
        jnp.asarray([1.0]),
        0,
        1,
    )
    assert not bool(tangent.valid)
    assert int(tangent.status) == phx.dynamics.EVOLUTION_BACKEND_FAILED
    assert bool(jnp.isnan(tangent.tangent[0]))


def test_diffrax_evolution_rollout_and_numerical_flow_jvp_share_system():
    layout = phx.dynamics.StateLayout((1,))
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, args: -args * state,
        state_layout=layout,
        system_id="exponential-decay",
    )
    evolution = phx.solver.DiffraxEvolution(system)
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 6), time_id="decay-save-times")

    trajectory = eqx.filter_jit(phx.dynamics.evolve)(
        evolution,
        jnp.asarray([2.0]),
        grid,
        args=jnp.asarray(0.7),
    )
    np.testing.assert_allclose(
        trajectory.states[:, 0],
        2.0 * jnp.exp(-0.7 * grid.times),
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert bool(trajectory.successful)

    tangent = eqx.filter_jit(evolution.tangent_action)(
        jnp.asarray([2.0]),
        jnp.asarray([1.0]),
        0.0,
        1.0,
        jnp.asarray(0.7),
    )
    np.testing.assert_allclose(
        tangent.tangent,
        jnp.asarray([jnp.exp(-0.7)]),
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert bool(tangent.valid)


def test_diffrax_evolution_reports_backend_failure_without_method_fallback():
    layout = phx.dynamics.StateLayout((1,))
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, args: state,
        state_layout=layout,
        system_id="step-limited-flow",
    )
    evolution = phx.solver.DiffraxEvolution(
        system,
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.01,
        max_steps=1,
    )

    result = evolution.advance(jnp.asarray([1.0]), 0.0, 1.0)

    assert not bool(result.valid)
    assert int(result.status) == phx.dynamics.EVOLUTION_BACKEND_FAILED


def test_declared_discrete_step_rejects_mismatched_and_nonfinite_intervals():
    def transition(coordinate, state, args):
        del coordinate, args
        return state

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((1,)),
        system_id="fixed-step",
        step_size=0.25,
        step_rtol=0.0,
        step_atol=1e-12,
    )
    evolution = phx.dynamics.DiscreteEvolution(system)

    assert jnp.array_equal(
        evolution.advance(jnp.asarray([1.0]), 2.0, 2.25).final_state,
        jnp.asarray([1.0]),
    )
    with pytest.raises((eqx.EquinoxRuntimeError, ValueError), match="step_size"):
        evolution.advance(jnp.asarray([1.0]), 2.0, 2.5)
    with pytest.raises((eqx.EquinoxRuntimeError, ValueError), match="finite"):
        evolution.advance(jnp.asarray([1.0]), jnp.nan, jnp.nan)


def test_discrete_step_metadata_validates_tolerances():
    layout = phx.dynamics.StateLayout((1,))
    transition = lambda coordinate, state, args: state

    with pytest.raises(ValueError, match="step_size"):
        phx.dynamics.DiscreteSystem(
            transition,
            state_layout=layout,
            system_id="bad-step",
            step_size=jnp.inf,
        )
    with pytest.raises(ValueError, match="step_rtol"):
        phx.dynamics.DiscreteSystem(
            transition,
            state_layout=layout,
            system_id="bad-tolerance",
            step_rtol=-1.0,
        )
