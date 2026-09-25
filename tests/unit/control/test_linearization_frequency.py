#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control import (
    continuous_transfer_function,
    descriptor_frequency_response,
    discrete_transfer_function,
    frequency_response,
    FREQUENCY_SINGULAR,
    FREQUENCY_UNSTABLE,
    linear_quadratic_problem_from_discrete_dynamics,
    linearize_differential_dynamics,
    linearize_discrete_dynamics,
    prepare_control_linearization,
)
from phydrax.control._dynamics import DiscreteControlDynamics
from phydrax.dynamics import (
    DiscreteSystem,
    InputLayout,
    LinearDescriptorSystem,
    StateLayout,
    TimeGrid,
)
from phydrax.dynamics._system import DiscreteTransitionResult
from phydrax.linalg import (
    ArraySpace,
    LinearCapabilityError,
    MaterializationPolicy,
    materialize,
)
from phydrax.metrix import QuaternionPoseStateGeometry
from tests._control_systems import (
    make_differential_control_dynamics,
    make_discrete_control_dynamics,
)


DENSE = MaterializationPolicy(max_entries=4096, max_bytes=32_768)


def test_quaternion_pose_discrete_linearization_is_six_dimensional_and_sign_invariant():
    geometry = QuaternionPoseStateGeometry()
    local_space = ArraySpace((6,), dtype=jnp.float32)
    state_layout = StateLayout(
        (7,),
        geometry=geometry,
        local_space=local_space,
        tangent_space=local_space,
        layout_id="test:quaternion-pose-linearization",
    )
    system = DiscreteSystem(
        lambda context, state, control, args: state,
        state_layout=state_layout,
        input_layout=InputLayout((1,), roles="control"),
        system_id="test:quaternion-pose-identity",
    )
    dynamics = DiscreteControlDynamics(system)
    pose = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.2, -0.4, 0.7])
    equivalent = pose.at[:4].multiply(-1.0)

    positive = linearize_discrete_dynamics(
        dynamics,
        0.0,
        pose,
        jnp.zeros((1,)),
        materialization=DENSE,
        target_time=1.0,
        step_index=0,
    )
    negative = linearize_discrete_dynamics(
        dynamics,
        0.0,
        equivalent,
        jnp.zeros((1,)),
        materialization=DENSE,
        target_time=1.0,
        step_index=0,
    )

    assert positive.state_local_size == 6
    assert positive.state_matrix.shape == (6, 6)
    assert positive.control_matrix.shape == (6, 1)
    np.testing.assert_allclose(positive.state_matrix, jnp.eye(6), atol=2.0e-6)
    np.testing.assert_allclose(negative.state_matrix, positive.state_matrix, atol=2.0e-6)
    np.testing.assert_allclose(negative.control_matrix, positive.control_matrix)
    np.testing.assert_allclose(positive.control_matrix, 0.0)
    np.testing.assert_allclose(positive.affine_offset, 0.0)
    assert jnp.all(jnp.isfinite(positive.state_matrix))
    assert jnp.all(jnp.isfinite(negative.state_matrix))
    assert bool(positive.valid)
    assert bool(negative.valid)


def test_nonlinear_input_output_linearization_has_affine_offsets():
    def vector_field(t, x, u, args):
        return jnp.array(
            [x[0] ** 2 + jnp.sin(x[1]) + args * u[0] + t, x[0] * x[1] + u[0] ** 2]
        )

    def output(t, x, u, args):
        del t, args
        return jnp.array([x[0] + u[0] ** 2, x[1] * u[0]])

    dynamics = make_differential_control_dynamics(
        vector_field,
        state_shape=(2,),
        control_shape=(1,),
        dynamics_id="analytic-nonlinear",
    )
    t = jnp.asarray(0.25)
    x = jnp.array([2.0, 0.5])
    u = jnp.array([1.5])
    result = linearize_differential_dynamics(
        dynamics, t, x, u, materialization=DENSE, args=3.0, output=output
    )

    expected_a = jnp.array([[4.0, jnp.cos(0.5)], [0.5, 2.0]])
    expected_b = jnp.array([[3.0], [3.0]])
    expected_c = jnp.array([[1.0, 0.0], [0.0, 1.5]])
    expected_d = jnp.array([[3.0], [0.5]])
    f0 = vector_field(t, x, u, 3.0)
    y0 = output(t, x, u, 3.0)
    np.testing.assert_allclose(result.A, expected_a)
    np.testing.assert_allclose(result.B, expected_b)
    np.testing.assert_allclose(result.C, expected_c)
    np.testing.assert_allclose(result.D, expected_d)
    np.testing.assert_allclose(result.affine_offset, f0 - expected_a @ x - expected_b @ u)
    np.testing.assert_allclose(result.output_offset, y0 - expected_c @ x - expected_d @ u)
    assert bool(result.valid)
    assert result.provenance.dynamics_id == "analytic-nonlinear"
    assert result.provenance.system_type == "continuous"


def test_discrete_linearization_preserves_batched_operating_points():
    def transition(context, x, u, args):
        return jnp.array([x[0] ** 2 + args * u[0] + context.source])

    dynamics = make_discrete_control_dynamics(
        transition,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="batched-map",
    )
    times = jnp.array([0.0, 0.5, 1.0])
    states = jnp.array([[1.0], [2.0], [3.0]])
    controls = jnp.array([[0.5], [1.0], [1.5]])
    result = linearize_discrete_dynamics(
        dynamics,
        times,
        states,
        controls,
        materialization=DENSE,
        args=2.0,
        target_time=times + 0.5,
        step_index=jnp.arange(times.size),
    )

    assert result.state_matrix.shape == (3, 1, 1)
    assert result.control_matrix.shape == (3, 1, 1)
    np.testing.assert_allclose(result.state_matrix[:, 0, 0], 2.0 * states[:, 0])
    np.testing.assert_allclose(result.control_matrix[:, 0, 0], 2.0)
    np.testing.assert_allclose(result.affine_offset[:, 0], times - states[:, 0] ** 2)
    np.testing.assert_allclose(result.output_matrix[:, 0, 0], 1.0)
    np.testing.assert_allclose(result.feedthrough_matrix[:, 0, 0], 0.0)
    assert bool(jnp.all(result.valid))
    assert result.provenance.system_type == "discrete"


def test_discrete_linearization_rejects_finite_failed_rollbacks():
    failure_status = 43

    def transition(context, state, control, args):
        del context, args
        successful = control[0] >= 0.0
        accepted = jnp.where(successful, state + control, state)
        return DiscreteTransitionResult(
            state + control + 100.0,
            accepted,
            successful,
            jnp.where(successful, 0, failure_status),
        )

    dynamics = make_discrete_control_dynamics(
        transition,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="failed-rollback-linearization",
    )
    result = linearize_discrete_dynamics(
        dynamics,
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([[2.0], [3.0]]),
        jnp.asarray([[1.0], [-1.0]]),
        materialization=DENSE,
        target_time=jnp.asarray([1.0, 1.0]),
        step_index=jnp.asarray([0, 1]),
    )

    np.testing.assert_array_equal(result.valid, jnp.asarray([True, False]))
    np.testing.assert_allclose(result.dynamics_value[0], jnp.asarray([3.0]))
    assert bool(jnp.isnan(result.dynamics_value[1, 0]))
    np.testing.assert_allclose(result.state_matrix[0], jnp.asarray([[1.0]]))
    np.testing.assert_allclose(result.control_matrix[0], jnp.asarray([[1.0]]))
    # The rollback is not a local model: no finite Jacobian is fabricated for it.
    assert bool(jnp.all(jnp.isnan(result.state_matrix[1])))
    assert bool(jnp.all(jnp.isnan(result.control_matrix[1])))
    assert bool(jnp.all(jnp.isnan(result.affine_offset[1])))


def test_linearization_marks_nonfinite_operating_time_invalid():
    dynamics = make_differential_control_dynamics(
        lambda time, state, control, args: jnp.ones((1,)),
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="time-independent-linearization",
    )

    result = linearize_differential_dynamics(
        dynamics,
        jnp.asarray(jnp.nan),
        jnp.ones((1,)),
        jnp.ones((1,)),
        materialization=DENSE,
    )

    assert not bool(result.valid)


def test_scalar_state_and_control_linearization_preserves_case_axes():
    discrete = make_discrete_control_dynamics(
        lambda context, state, control, args: state**2 + 3.0 * control + context.source,
        state_shape=(),
        control_shape=(),
        dynamics_id="scalar-discrete-map",
    )
    times = jnp.array([0.0, 0.5])
    states = jnp.array([1.0, 2.0])
    result = linearize_discrete_dynamics(
        discrete,
        times,
        states,
        jnp.asarray(0.25),
        materialization=DENSE,
        target_time=times + 0.5,
        step_index=jnp.arange(times.size),
    )

    assert result.operating_state.shape == (2,)
    assert result.operating_control.shape == (2,)
    assert result.dynamics_value.shape == (2,)
    assert result.state_matrix.shape == (2, 1, 1)
    assert result.control_matrix.shape == (2, 1, 1)
    np.testing.assert_allclose(result.state_matrix[:, 0, 0], 2.0 * states)
    np.testing.assert_allclose(result.control_matrix[:, 0, 0], 3.0)
    assert bool(jnp.all(result.valid))

    differential = make_differential_control_dynamics(
        lambda time, state, control, args: state * control,
        state_shape=(),
        control_shape=(),
        dynamics_id="scalar-differential-field",
    )
    scalar = linearize_differential_dynamics(
        differential,
        jnp.asarray(0.0),
        jnp.asarray(2.0),
        jnp.asarray(4.0),
        materialization=DENSE,
    )
    assert scalar.operating_state.shape == ()
    assert scalar.operating_control.shape == ()
    assert scalar.dynamics_value.shape == ()
    assert scalar.state_matrix.shape == (1, 1)
    assert scalar.control_matrix.shape == (1, 1)
    np.testing.assert_allclose(scalar.state_matrix, [[4.0]])
    np.testing.assert_allclose(scalar.control_matrix, [[2.0]])
    assert bool(scalar.valid)


def test_prepared_linearization_actions_match_the_dense_model_without_materializing():
    def vector_field(t, x, u, args):
        del args
        return jnp.array([x[0] * x[1] + jnp.sin(u[0]), x[1] ** 2 - t * u[1]])

    def output(t, x, u, args):
        del t, args
        return jnp.array([x[0] + u[0] * u[1], x[0] * x[1], jnp.cos(u[1])])

    dynamics = make_differential_control_dynamics(
        vector_field,
        state_shape=(2,),
        control_shape=(2,),
        dynamics_id="prepared-nonlinear",
    )
    t = jnp.asarray(0.5)
    x = jnp.array([1.5, -0.25])
    u = jnp.array([0.3, 0.8])
    prepared = prepare_control_linearization(dynamics, t, x, u, output=output)
    dense = linearize_differential_dynamics(
        dynamics, t, x, u, materialization=DENSE, output=output
    )
    joint_dynamics = jnp.concatenate((dense.A, dense.B), axis=-1)
    joint_output = jnp.concatenate((dense.C, dense.D), axis=-1)
    direction = jnp.array([0.7, -1.1, 0.4, 2.0])
    dynamics_cotangent = jnp.array([1.3, -0.6])
    output_cotangent = jnp.array([0.2, -1.0, 0.5])

    np.testing.assert_allclose(
        prepared.dynamics_jacobian.mv(direction), joint_dynamics @ direction
    )
    np.testing.assert_allclose(
        prepared.dynamics_jacobian.transpose_mv(dynamics_cotangent),
        joint_dynamics.T @ dynamics_cotangent,
    )
    np.testing.assert_allclose(
        prepared.output_jacobian.mv(direction), joint_output @ direction
    )
    np.testing.assert_allclose(
        prepared.output_jacobian.transpose_mv(output_cotangent),
        joint_output.T @ output_cotangent,
    )
    np.testing.assert_allclose(
        materialize(prepared.dynamics_jacobian, DENSE), joint_dynamics
    )
    np.testing.assert_allclose(prepared.dynamics_value, vector_field(t, x, u, None))
    np.testing.assert_allclose(prepared.output_value, output(t, x, u, None))
    assert prepared.output_shape == (3,)
    assert bool(prepared.valid)

    with pytest.raises(ValueError, match="one operating point"):
        prepare_control_linearization(dynamics, t, jnp.stack((x, x)), u)
    discrete = make_discrete_control_dynamics(
        lambda context, state, control, args: state + control,
        state_shape=(2,),
        control_shape=(2,),
        dynamics_id="prepared-discrete",
    )
    with pytest.raises(ValueError, match="target_time and step_index"):
        prepare_control_linearization(discrete, t, x, u)


def test_dense_linearization_is_bounded_by_the_required_materialization_policy():
    dynamics = make_discrete_control_dynamics(
        lambda context, state, control, args: state * control[0] + context.source,
        state_shape=(2,),
        control_shape=(1,),
        dynamics_id="budgeted-map",
    )
    times = jnp.zeros((3,))
    states = jnp.ones((3, 2))
    controls = jnp.ones((3, 1))

    def linearize(policy):
        return linearize_discrete_dynamics(
            dynamics,
            times,
            states,
            controls,
            materialization=policy,
            target_time=times + 1.0,
            step_index=jnp.arange(3),
        )

    # Three cases of a 2 x (2 + 1) Jacobian: 18 dense entries per family.
    exact = linearize(MaterializationPolicy(max_entries=18))
    assert exact.state_matrix.shape == (3, 2, 2)
    assert bool(jnp.all(exact.valid))
    with pytest.raises(LinearCapabilityError, match="18 entries"):
        linearize(MaterializationPolicy(max_entries=17))
    with pytest.raises(LinearCapabilityError, match="144 bytes"):
        linearize(MaterializationPolicy(max_bytes=143))
    with pytest.raises(TypeError, match="MaterializationPolicy"):
        linearize(None)


def test_discrete_bridge_builds_the_affine_lq_problem_along_an_operating_trajectory():
    step = 0.1

    def transition(context, state, control, args):
        del context, args
        return jnp.array(
            [
                state[0] + step * state[1],
                state[1] + step * (jnp.sin(state[0]) + control[0] * state[1]),
            ]
        )

    dynamics = make_discrete_control_dynamics(
        transition,
        state_shape=(2,),
        control_shape=(1,),
        dynamics_id="pendulum-like",
    )
    horizon = 3
    time_grid = TimeGrid(jnp.arange(horizon + 1) * step, time_id="bridge-grid")
    operating_states = jnp.array([[0.4, -0.2], [0.38, -0.1], [0.37, 0.05]])
    operating_controls = jnp.array([[0.5], [-0.3], [0.2]])
    problem = linear_quadratic_problem_from_discrete_dynamics(
        dynamics,
        time_grid,
        operating_states,
        operating_controls,
        operating_states[0],
        jnp.broadcast_to(jnp.eye(2), (horizon, 2, 2)),
        jnp.ones((horizon, 1, 1)),
        jnp.eye(2),
        materialization=DENSE,
        control_upper_bounds=jnp.ones((horizon, 1)),
        problem_id="bridge-problem",
    )

    for stage in range(horizon):
        x = operating_states[stage]
        u = operating_controls[stage]
        expected_a = jnp.array([[1.0, step], [step * jnp.cos(x[0]), 1.0 + step * u[0]]])
        expected_b = jnp.array([[0.0], [step * x[1]]])
        np.testing.assert_allclose(problem.dynamics_matrices[stage], expected_a)
        np.testing.assert_allclose(problem.control_matrices[stage], expected_b)
        # The affine model reproduces the nonlinear transition at its point.
        np.testing.assert_allclose(
            problem.dynamics_matrices[stage] @ x
            + problem.control_matrices[stage] @ u
            + problem.dynamics_bias[stage],
            transition(None, x, u, None),
        )
    assert problem.time_grid is time_grid
    assert problem.problem_id == "bridge-problem"
    assert problem.dynamics_id.startswith("pendulum-like:linearized:")
    np.testing.assert_allclose(problem.control_upper_bounds, 1.0)

    with pytest.raises(TypeError, match="dynamics_bias"):
        linear_quadratic_problem_from_discrete_dynamics(
            dynamics,
            time_grid,
            operating_states,
            operating_controls,
            operating_states[0],
            jnp.broadcast_to(jnp.eye(2), (horizon, 2, 2)),
            jnp.ones((horizon, 1, 1)),
            jnp.eye(2),
            materialization=DENSE,
            dynamics_bias=jnp.zeros((horizon, 2)),
        )


def test_discrete_bridge_refuses_failed_transitions_and_manifold_states():
    def transition(context, state, control, args):
        del args
        successful = context.step_index != 1
        accepted = jnp.where(successful, state + control, state)
        return DiscreteTransitionResult(
            state + control, accepted, successful, jnp.where(successful, 0, 7)
        )

    dynamics = make_discrete_control_dynamics(
        transition,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="fails-at-stage-one",
    )
    time_grid = TimeGrid(jnp.arange(3.0), time_id="failure-grid")
    with pytest.raises(ValueError, match=r"\[\[1\]\].*not repaired"):
        linear_quadratic_problem_from_discrete_dynamics(
            dynamics,
            time_grid,
            jnp.ones((2, 1)),
            jnp.ones((2, 1)),
            jnp.ones((1,)),
            jnp.ones((2, 1, 1)),
            jnp.ones((2, 1, 1)),
            jnp.ones((1, 1)),
            materialization=DENSE,
        )

    local_space = ArraySpace((6,), dtype=jnp.float64)
    pose = DiscreteControlDynamics(
        DiscreteSystem(
            lambda context, state, control, args: state,
            state_layout=StateLayout(
                (7,),
                geometry=QuaternionPoseStateGeometry(),
                local_space=local_space,
                tangent_space=local_space,
                layout_id="test:bridge-pose",
            ),
            input_layout=InputLayout((1,), roles="control"),
            system_id="test:bridge-pose",
        )
    )
    with pytest.raises(ValueError, match="Euclidean state geometry"):
        linear_quadratic_problem_from_discrete_dynamics(
            pose,
            time_grid,
            jnp.ones((2, 7)),
            jnp.ones((2, 1)),
            jnp.ones((7,)),
            jnp.ones((2, 7, 7)),
            jnp.ones((2, 1, 1)),
            jnp.ones((7, 7)),
            materialization=DENSE,
        )


def test_known_siso_continuous_and_discrete_resolvents():
    a = jnp.array([[-2.0]])
    b = jnp.array([[3.0]])
    c = jnp.array([[4.0]])
    d = jnp.array([[0.5]])
    s = jnp.array([0.0 + 0.0j, 0.0 + 2.0j])
    continuous = continuous_transfer_function(a, b, c, d, s)
    expected = 12.0 / (s + 2.0) + 0.5
    np.testing.assert_allclose(continuous.response[:, 0, 0], expected)
    np.testing.assert_allclose(continuous.state_response[:, 0, 0], 3.0 / (s + 2.0))
    assert bool(jnp.all(continuous.valid))

    ad = jnp.array([[0.5]])
    z = jnp.array([1.0 + 0.0j, 0.0 + 1.0j])
    discrete = discrete_transfer_function(ad, b, c, d, z)
    np.testing.assert_allclose(discrete.response[:, 0, 0], 12.0 / (z - 0.5) + 0.5)
    assert bool(jnp.all(discrete.valid))


def test_mimo_frequency_response_and_gradient():
    a = jnp.diag(jnp.array([-1.0, -3.0]))
    b = jnp.array([[1.0, 2.0], [0.5, -1.0]])
    c = jnp.array([[1.0, 0.25], [-2.0, 1.0]])
    d = jnp.array([[0.0, 0.1], [0.2, 0.0]])
    frequencies = jnp.array([0.0, 1.5])
    result = frequency_response(a, b, c, d, frequencies)
    expected = jax.vmap(lambda w: c @ jnp.linalg.solve(1j * w * jnp.eye(2) - a, b) + d)(
        frequencies
    )
    np.testing.assert_allclose(result.response, expected)
    assert result.response.shape == (2, 2, 2)

    def real_response(rate):
        scalar = frequency_response(
            jnp.array([[-rate]]),
            jnp.ones((1, 1)),
            jnp.ones((1, 1)),
            jnp.zeros((1, 1)),
            jnp.asarray(1.0),
        )
        return jnp.real(scalar.response[0, 0])

    np.testing.assert_allclose(jax.grad(real_response)(2.0), -0.12)


def test_descriptor_frequency_uses_i_omega_e_minus_a_resolvent():
    system = LinearDescriptorSystem(
        jnp.asarray([[2.0]]),
        jnp.asarray([[-3.0]]),
        jnp.asarray([[4.0]]),
        jnp.asarray([[5.0]]),
        jnp.asarray([[0.25]]),
        system_id="descriptor-reference",
    )
    frequency = jnp.asarray(1.5)
    result = descriptor_frequency_response(system, frequency)
    expected_state = 4.0 / (1j * frequency * 2.0 + 3.0)
    np.testing.assert_allclose(result.state_response[0, 0], expected_state)
    np.testing.assert_allclose(result.response[0, 0], 5.0 * expected_state + 0.25)
    assert bool(result.successful)


def test_unstable_and_singular_statuses_are_explicit():
    one = jnp.ones((1, 1))
    zero = jnp.zeros((1, 1))
    unstable = frequency_response(jnp.array([[0.25]]), one, one, zero, jnp.asarray(1.0))
    assert int(unstable.status) == FREQUENCY_UNSTABLE
    assert not bool(unstable.valid)

    singular = frequency_response(zero, one, one, zero, jnp.asarray(0.0))
    assert int(singular.status) == FREQUENCY_SINGULAR
    assert bool(singular.singular)
    assert not bool(singular.valid)
    assert np.isinf(float(singular.condition_number))
