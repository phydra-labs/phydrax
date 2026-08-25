import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

import phydrax as phx


def test_phi3_action_handles_zero_and_diagonal_arguments():
    operator = phx.linalg.DiagonalLinearOperator(jnp.asarray([-2.0]))
    vector = jnp.asarray([3.0])
    step = 0.2
    z = -0.4
    expected = vector * (jnp.exp(z) - 1.0 - z - 0.5 * z**2) / z**3

    actual = phx.linalg.matrix_phi3_action(operator, vector, step).value
    zero = phx.linalg.matrix_phi3_action(operator, vector, 0.0).value
    matrix = jnp.asarray([[-1.0, 0.3], [-0.2, -0.5]])
    dense_operator = phx.linalg.DenseLinearOperator(matrix)
    dense_vector = jnp.asarray([0.4, -0.7])
    scaled = step * matrix
    remainder = (
        jsp_linalg.expm(scaled) - jnp.eye(2) - scaled - 0.5 * (scaled @ scaled)
    ) @ dense_vector
    expected_dense = jnp.linalg.solve(
        scaled,
        jnp.linalg.solve(scaled, jnp.linalg.solve(scaled, remainder)),
    )
    dense = phx.linalg.matrix_phi3_action(
        dense_operator,
        dense_vector,
        step,
    ).value

    assert jnp.allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(zero, vector / 6.0, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(dense, expected_dense, rtol=1e-10, atol=1e-10)


def _semilinear_logistic(rate=-1.5):
    operator = phx.linalg.DiagonalLinearOperator(jnp.asarray([rate]))
    return phx.solver.SemilinearDrift(
        operator,
        lambda time, state, args: state**2,
        state_shape=(1,),
        operator_id=operator.operator_id,
    )


def _integrate_terminal(order, steps):
    method = phx.solver.ETDRKMethod(order)
    drift = _semilinear_logistic()
    times = jnp.linspace(0.0, 0.5, steps + 1)
    return phx.solver.solve_etdrk(
        method,
        drift,
        jnp.asarray([0.2]),
        times,
    ).states[-1, 0]


def test_etdrk_orders_converge_and_step_is_jittable():
    reference = _integrate_terminal(4, 512)
    error2_coarse = jnp.abs(_integrate_terminal(2, 8) - reference)
    error2_fine = jnp.abs(_integrate_terminal(2, 16) - reference)
    error4_coarse = jnp.abs(_integrate_terminal(4, 4) - reference)
    error4_fine = jnp.abs(_integrate_terminal(4, 8) - reference)
    method = phx.solver.ETDRKMethod(4)
    drift = _semilinear_logistic()
    eager = method.step(drift, 0.0, jnp.asarray([0.2]), 0.1, None)
    compiled = jax.jit(lambda state: method.step(drift, 0.0, state, 0.1, None))(
        jnp.asarray([0.2])
    )

    assert error2_fine < 0.35 * error2_coarse
    assert error4_fine < 0.1 * error4_coarse
    assert jnp.allclose(eager, compiled, rtol=1e-12, atol=1e-12)
