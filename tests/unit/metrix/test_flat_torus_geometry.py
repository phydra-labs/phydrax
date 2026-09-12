#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_flat_torus_wrap_retraction_and_inverse_cross_the_seam():
    geometry = phx.metrix.FlatTorusStateGeometry(2.0 * jnp.pi)
    state = jnp.asarray([jnp.pi - 0.1, -jnp.pi + 0.2])
    local = jnp.asarray([0.3, -0.4])
    point = geometry.retract(state, local)
    recovered = geometry.inverse_retract(state, point)

    assert geometry.contains(state)
    assert geometry.contains(point)
    assert jnp.allclose(recovered, local)
    assert jnp.all(point >= -jnp.pi)
    assert jnp.all(point < jnp.pi)


def test_flat_torus_differential_and_transport_are_identity():
    geometry = phx.metrix.FlatTorusStateGeometry(4.0)
    state = jnp.asarray([0.2, -0.6])
    local = jnp.asarray([0.1, 0.3])
    velocity = jnp.asarray([-0.4, 0.2])
    point = geometry.retract(state, local)

    assert jnp.array_equal(geometry.retraction_jvp(state, local, velocity), velocity)
    assert jnp.array_equal(
        geometry.retraction_inverse_jvp(state, point, velocity), velocity
    )
    assert jnp.array_equal(geometry.retraction_vjp(state, local, velocity), velocity)
    assert jnp.array_equal(geometry.transport_tangent(state, point, velocity), velocity)
    assert jnp.array_equal(
        geometry.transport_cotangent_pullback(state, point, velocity), velocity
    )
    assert geometry.cut_locus_margin(state, point) > 0.0


def test_flat_torus_is_jittable_and_fail_closed():
    geometry = phx.metrix.FlatTorusStateGeometry(2.0)
    point = jax.jit(geometry.retract)(jnp.asarray([0.9]), jnp.asarray([0.3]))

    assert jnp.allclose(point, -0.8)
    assert not geometry.contains(jnp.asarray([1.0]))
    assert not geometry.contains(jnp.asarray([jnp.nan]))
    with pytest.raises(TypeError, match="real floating"):
        geometry.wrap(jnp.asarray([1], dtype=jnp.int32))
    with pytest.raises(ValueError, match="positive"):
        phx.metrix.FlatTorusStateGeometry(0.0)
