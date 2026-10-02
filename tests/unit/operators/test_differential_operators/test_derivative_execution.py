import jax.numpy as jnp
from jax import Array

from phydrax.operators.differential import evaluate_fused_coordinate_derivatives


def test_fused_coordinate_derivatives_match_analytic_vector_derivatives() -> None:
    point = jnp.asarray([2.0, 0.5], dtype=jnp.float64)

    def function(value: Array) -> Array:
        x, y = value
        return jnp.asarray([x**2 * y, jnp.sin(y)])

    evaluated = evaluate_fused_coordinate_derivatives(
        function,
        point,
        first_axes=(0, 1),
        second_axes=(0, 1),
    )

    assert jnp.allclose(evaluated.value, jnp.asarray([2.0, jnp.sin(0.5)]))
    assert jnp.allclose(evaluated.first_derivatives[0], jnp.asarray([2.0, 0.0]))
    assert jnp.allclose(
        evaluated.first_derivatives[1],
        jnp.asarray([4.0, jnp.cos(0.5)]),
    )
    assert jnp.allclose(
        evaluated.diagonal_second_derivatives[0],
        jnp.asarray([1.0, 0.0]),
    )
    assert jnp.allclose(
        evaluated.diagonal_second_derivatives[1],
        jnp.asarray([0.0, -jnp.sin(0.5)]),
    )
    assert evaluated.plan is not None
