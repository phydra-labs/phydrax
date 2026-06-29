import jax.numpy as jnp

import phydrax.graph as vx


def test_segment_variance_and_normalize():
    data = jnp.array([1.0, 3.0, 2.0, 6.0])
    seg = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

    var = vx.segment_variance(data, seg, 2)
    assert jnp.allclose(var, jnp.array([1.0, 4.0]))

    norm = vx.segment_normalize(data, seg, 2)
    expected = jnp.array([-1.0, 1.0, -1.0, 1.0])
    assert jnp.allclose(norm, expected, atol=1e-5)


def test_segment_constant_variants():
    data = jnp.array([1.0, 2.0])
    seg = jnp.array([0, 0], dtype=jnp.int32)

    maxs = vx.segment_max_or_constant(data, seg, 3, constant=-7.0)
    mins = vx.segment_min_or_constant(data, seg, 3, constant=9.0)

    assert jnp.allclose(maxs, jnp.array([2.0, -7.0, -7.0]))
    assert jnp.allclose(mins, jnp.array([1.0, 9.0, 9.0]))
