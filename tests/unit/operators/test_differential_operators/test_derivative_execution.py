import jax.numpy as jnp

from phydrax.operators.differential import (
    DerivativeRequest,
    evaluate_fused_coordinate_derivatives,
    plan_derivative_execution,
)


def test_derivative_execution_plan_preserves_exact_strategy_boundaries():
    first = (DerivativeRequest("u", "x", (0,)),)
    laplacian = (DerivativeRequest("u", "x", (), laplacian_count=1),)
    high_order = (DerivativeRequest("u", "x", (0, 0, 0)),)

    assert plan_derivative_execution(first).strategy == "reverse"
    assert (
        plan_derivative_execution(first, output_size=4, coordinate_size=2).strategy
        == "forward"
    )
    assert plan_derivative_execution(laplacian).strategy == "jvp"
    assert plan_derivative_execution(high_order).strategy == "jet"


def test_fused_coordinate_derivatives_match_analytic_vector_derivatives():
    point = jnp.asarray([2.0, 0.5])

    def function(value):
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
