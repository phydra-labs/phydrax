import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_linear_endpoint_interpolant_preserves_endpoints_and_jvp_velocity():
    interpolant = phx.transport.LinearEndpointInterpolant((2,))
    source = jnp.asarray([[0.0, 1.0], [2.0, -1.0]])
    target = jnp.asarray([[4.0, 3.0], [0.0, 5.0]])
    time = jnp.asarray([0.25, 0.75])

    evaluation = interpolant.evaluate(time, source, target)
    expected = (1.0 - time[:, None]) * source + time[:, None] * target
    _, velocity = jax.jvp(
        lambda current: interpolant.evaluate(current, source, target).state,
        (time,),
        (jnp.ones_like(time),),
    )

    assert jnp.allclose(evaluation.state, expected)
    assert jnp.allclose(evaluation.conditional_velocity, target - source)
    assert jnp.allclose(velocity, evaluation.conditional_velocity)
    assert jnp.all(evaluation.valid)

    start = interpolant.evaluate(0.0, source, target)
    end = interpolant.evaluate(1.0, source, target)
    assert jnp.allclose(start.state, source)
    assert jnp.allclose(end.state, target)


def test_linear_endpoint_interpolant_handles_event_rank_and_nonfinite_validity():
    interpolant = phx.transport.LinearEndpointInterpolant((2, 2))
    source = jnp.zeros((3, 2, 2)).at[1, 0, 0].set(jnp.nan)
    target = jnp.ones((3, 2, 2))
    evaluation = eqx.filter_jit(interpolant.evaluate)(0.5, source, target)

    assert evaluation.state.shape == (3, 2, 2)
    assert jnp.array_equal(evaluation.valid, jnp.asarray([True, False, True]))


def test_linear_endpoint_interpolant_rejects_shape_and_time_contract_violations():
    interpolant = phx.transport.LinearEndpointInterpolant(
        (2,), source_coordinate=2.0, target_coordinate=4.0
    )
    with pytest.raises(ValueError, match="shapes must match"):
        interpolant.evaluate(3.0, jnp.zeros((2, 2)), jnp.zeros((2,)))
    with pytest.raises(ValueError, match="leading shape"):
        interpolant.evaluate(jnp.ones((3,)), jnp.zeros((2, 2)), jnp.ones((2, 2)))
    with pytest.raises(eqx.EquinoxRuntimeError, match="outside"):
        eqx.filter_jit(interpolant.evaluate)(
            jnp.asarray([1.0, 3.0]),
            jnp.zeros((2, 2)),
            jnp.ones((2, 2)),
        )
