import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.nonlinear._domain_cond import domain_cond


def test_vmapped_domain_gradient_preserves_live_branches_under_jit() -> None:
    def residual(value: Array) -> Array:
        return domain_cond(
            value > 0.0,
            lambda operand: operand * operand,
            lambda operand: 3.0 * operand,
            value,
        )

    derivative = jax.jit(jax.vmap(jax.grad(residual)))
    actual = derivative(jnp.asarray([2.0, -2.0], dtype=jnp.float64))
    np.testing.assert_array_equal(actual, np.asarray([4.0, 3.0], dtype=np.float64))


def test_high_operation_domain_branch_preserves_value_and_derivative() -> None:
    def residual(value: Array) -> Array:
        return domain_cond(
            value > 0.0,
            lambda operand: jnp.ldexp(operand, 2),
            lambda operand: 3.0 * operand,
            value,
        )

    evaluate = jax.jit(jax.vmap(jax.value_and_grad(residual)))
    values, derivatives = evaluate(jnp.asarray([2.0, -2.0], dtype=jnp.float64))
    np.testing.assert_array_equal(values, np.asarray([8.0, -6.0], dtype=np.float64))
    np.testing.assert_array_equal(derivatives, np.asarray([4.0, 3.0], dtype=np.float64))
