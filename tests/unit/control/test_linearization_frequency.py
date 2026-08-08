#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.control import (
    continuous_transfer_function,
    DifferentialControlDynamics,
    discrete_transfer_function,
    DiscreteControlDynamics,
    frequency_response,
    FREQUENCY_SINGULAR,
    FREQUENCY_UNSTABLE,
    linearize_differential_dynamics,
    linearize_discrete_dynamics,
)


def test_nonlinear_input_output_linearization_has_affine_offsets():
    def vector_field(t, x, u, args):
        return jnp.array(
            [x[0] ** 2 + jnp.sin(x[1]) + args * u[0] + t, x[0] * x[1] + u[0] ** 2]
        )

    def output(t, x, u, args):
        del t, args
        return jnp.array([x[0] + u[0] ** 2, x[1] * u[0]])

    dynamics = DifferentialControlDynamics(
        vector_field,
        state_shape=(2,),
        control_shape=(1,),
        dynamics_id="analytic-nonlinear",
    )
    t = jnp.asarray(0.25)
    x = jnp.array([2.0, 0.5])
    u = jnp.array([1.5])
    result = linearize_differential_dynamics(dynamics, t, x, u, args=3.0, output=output)

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
    def transition(t, x, u, args):
        return jnp.array([x[0] ** 2 + args * u[0] + t])

    dynamics = DiscreteControlDynamics(
        transition,
        state_shape=(1,),
        control_shape=(1,),
        dynamics_id="batched-map",
    )
    times = jnp.array([0.0, 0.5, 1.0])
    states = jnp.array([[1.0], [2.0], [3.0]])
    controls = jnp.array([[0.5], [1.0], [1.5]])
    result = linearize_discrete_dynamics(dynamics, times, states, controls, args=2.0)

    assert result.state_matrix.shape == (3, 1, 1)
    assert result.control_matrix.shape == (3, 1, 1)
    np.testing.assert_allclose(result.state_matrix[:, 0, 0], 2.0 * states[:, 0])
    np.testing.assert_allclose(result.control_matrix[:, 0, 0], 2.0)
    np.testing.assert_allclose(result.affine_offset[:, 0], times - states[:, 0] ** 2)
    np.testing.assert_allclose(result.output_matrix[:, 0, 0], 1.0)
    np.testing.assert_allclose(result.feedthrough_matrix[:, 0, 0], 0.0)
    assert bool(jnp.all(result.valid))
    assert result.provenance.system_type == "discrete"


def test_linearization_marks_nonfinite_operating_time_invalid():
    dynamics = DifferentialControlDynamics(
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
    )

    assert not bool(result.valid)


def test_scalar_state_and_control_linearization_preserves_case_axes():
    discrete = DiscreteControlDynamics(
        lambda time, state, control, args: state**2 + 3.0 * control + time,
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
    )

    assert result.operating_state.shape == (2,)
    assert result.operating_control.shape == (2,)
    assert result.dynamics_value.shape == (2,)
    assert result.state_matrix.shape == (2, 1, 1)
    assert result.control_matrix.shape == (2, 1, 1)
    np.testing.assert_allclose(result.state_matrix[:, 0, 0], 2.0 * states)
    np.testing.assert_allclose(result.control_matrix[:, 0, 0], 3.0)
    assert bool(jnp.all(result.valid))

    differential = DifferentialControlDynamics(
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
    )
    assert scalar.operating_state.shape == ()
    assert scalar.operating_control.shape == ()
    assert scalar.dynamics_value.shape == ()
    assert scalar.state_matrix.shape == (1, 1)
    assert scalar.control_matrix.shape == (1, 1)
    np.testing.assert_allclose(scalar.state_matrix, [[4.0]])
    np.testing.assert_allclose(scalar.control_matrix, [[2.0]])
    assert bool(scalar.valid)


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
