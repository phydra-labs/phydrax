import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _controlled_dae():
    return phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, control, args: jnp.asarray(
            (
                state_rate[0] - control[0],
                state[1] - args * state[0],
            )
        ),
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="controlled-dae-contract",
    )


def test_input_aware_dae_validates_inputs_and_differentiates():
    system = _controlled_dae()
    state = jnp.asarray((2.0, 6.0))
    state_rate = jnp.asarray((4.0, 0.0))
    control = jnp.asarray((1.5,))

    value = system.evaluate(
        0.25,
        state,
        state_rate,
        3.0,
        inputs=control,
    )
    assert jnp.allclose(value, jnp.asarray((2.5, 0.0)))
    derivative = jax.jacrev(
        lambda inputs: system.evaluate(
            0.25,
            state,
            state_rate,
            3.0,
            inputs=inputs,
        )
    )(control)
    assert jnp.allclose(derivative, jnp.asarray(((-1.0,), (0.0,))))

    with pytest.raises(ValueError, match="requires explicit inputs"):
        system.evaluate(0.25, state, state_rate, 3.0)
    with pytest.raises(ValueError, match="inputs must have shape"):
        system.evaluate(
            0.25,
            state,
            state_rate,
            3.0,
            inputs=jnp.ones((2,)),
        )


def test_autonomous_dae_rejects_extra_inputs_and_ivp_rejects_controlled_dae():
    autonomous = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, args: state_rate + state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="autonomous-dae-contract",
    )
    assert jnp.allclose(autonomous(0.0, jnp.ones(1), -jnp.ones(1)), 0.0)
    with pytest.raises(ValueError, match="does not accept inputs"):
        autonomous.evaluate(
            0.0,
            jnp.ones(1),
            -jnp.ones(1),
            inputs=jnp.ones(1),
        )
    with pytest.raises(ValueError, match="requires input_policy"):
        phx.solver.DifferentialAlgebraicProblem(
            _controlled_dae(),
            jnp.zeros(2),
            problem_id="controlled-ivp-rejected",
        )


def test_controlled_mass_matrix_constructor_preserves_input_role():
    system = phx.dynamics.DifferentialAlgebraicSystem.from_mass_matrix(
        jnp.eye(2),
        lambda time, state, control, args: args * state + control[0],
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "differential")),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="controlled-mass-matrix",
    )
    value = system.evaluate(
        0.0,
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((4.0, 7.0)),
        2.0,
        inputs=jnp.asarray((1.0,)),
    )
    assert jnp.allclose(value, jnp.asarray((1.0, 2.0)))


def test_trajectory_view_interpolates_states_and_holds_controls():
    view = phx.control.TrajectoryOptimizationView(
        jnp.asarray((0.0, 1.0, 2.0)),
        jnp.asarray(((0.0,), (2.0,), (6.0,))),
        jnp.asarray(((3.0,), (5.0,))),
        case_shape=(),
        state_shape=(1,),
        control_shape=(1,),
    )
    query = jnp.asarray((0.5, 1.5, 2.0))
    assert jnp.allclose(
        view.evaluate_state(query),
        jnp.asarray(((1.0,), (4.0,), (6.0,))),
    )
    assert jnp.allclose(
        view.evaluate_control(query),
        jnp.asarray(((3.0,), (5.0,), (5.0,))),
    )
    with pytest.raises(RuntimeError, match="inside the physical horizon"):
        view.evaluate_state(jnp.asarray((2.1,)))


def test_trajectory_view_retracts_between_equivalent_quaternion_poses():
    geometry = phx.metrix.QuaternionPoseStateGeometry()
    pose = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.2, -0.4, 0.7])
    equivalent = pose.at[:4].multiply(-1.0)
    view = phx.control.TrajectoryOptimizationView(
        jnp.asarray([0.0, 1.0]),
        jnp.stack((pose, equivalent)),
        jnp.zeros((1, 1)),
        case_shape=(),
        state_shape=(7,),
        control_shape=(1,),
        state_geometry=geometry,
    )

    midpoint = view.evaluate_state(0.5)

    assert bool(geometry.contains(midpoint))
    assert jnp.allclose(geometry.inverse_retract(pose, midpoint), 0.0)
    assert jnp.allclose(
        geometry.inverse_retract(equivalent, midpoint),
        0.0,
    )


def test_trajectory_problem_retains_cases_and_shared_parameter_space():
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: args * state + control,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="case-trajectory",
    )
    parameter_space = phx.linalg.ArraySpace((1,), dtype=jnp.float64)
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        case_shape=(2,),
        parameter_space=parameter_space,
        problem_id="two-case-trajectory",
    )
    assert problem.case_shape == (2,)
    assert problem.state_shape == (1,)
    assert problem.control_shape == (1,)
    assert problem.parameter_space is parameter_space

    with pytest.raises(ValueError, match="require explicit inputs"):
        phx.control.TrajectoryOptimizationProblem(
            phx.dynamics.ContinuousSystem(
                lambda time, state, args: state,
                state_layout=phx.dynamics.StateLayout((1,)),
                system_id="autonomous-trajectory",
            ),
            problem_id="invalid-autonomous-trajectory",
        )
