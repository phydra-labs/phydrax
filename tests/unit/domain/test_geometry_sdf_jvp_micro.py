#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.domain.geometry2d import Square
from phydrax.domain.geometry3d import Cube


def test_2d_sdf_jvp_vector_and_scalar_inputs():
    geom = Square(center=(0.0, 0.0), side=1.0)

    def f(p):
        return geom.adf(p)

    # Vector input near boundary, finite JVP
    val, tval = jax.jvp(f, (jnp.array([0.49, 0.0]),), (jnp.array([1.0, 0.0]),))
    assert jnp.isfinite(val)
    assert jnp.isfinite(tval)

    # Scalar input (broadcast inside JVP), finite JVP
    val_s, tval_s = jax.jvp(f, (jnp.array(0.1),), (jnp.array(0.0),))
    assert jnp.isfinite(val_s)
    assert jnp.isfinite(tval_s)


def test_3d_sdf_jvp_vector_and_scalar_inputs():
    geom = Cube(center=(0.0, 0.0, 0.0), side=1.0)

    def f(p):
        p = jnp.asarray(p)
        if p.ndim == 0:
            p = jnp.array([p, 0.0, 0.0])
        return geom.adf(p)

    # Vector input near boundary, finite JVP
    val, tval = jax.jvp(f, (jnp.array([0.49, 0.0, 0.0]),), (jnp.array([1.0, 0.0, 0.0]),))
    assert jnp.isfinite(val)
    assert jnp.isfinite(tval)

    # Scalar input (broadcast inside JVP), finite JVP
    val_s, tval_s = jax.jvp(f, (jnp.array(0.1),), (jnp.array(0.0),))
    assert jnp.isfinite(val_s)
    assert jnp.isfinite(tval_s)
