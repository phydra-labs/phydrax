#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import trimesh

from phydrax.domain.geometry2d import Square
from phydrax.domain.geometry3d import Cube, Geometry3DFromCAD


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


def test_compact_enforcement_gate_exact_zero_has_finite_linear_jet():
    geometry = Square(center=(0.0, 0.0), side=1.0)
    gate = geometry.make_enforcement_gate(method="compact")

    def profile(offset):
        return gate(jnp.array([0.5 + offset, 0.0]))

    derivatives = []
    derivative = profile
    for _ in range(4):
        derivative = jax.grad(derivative)
        derivatives.append(derivative(jnp.asarray(0.0)))

    assert profile(jnp.asarray(0.0)) == 0.0
    assert jnp.all(jnp.isfinite(jnp.stack(derivatives)))
    assert derivatives[0] < 0.0
    assert jnp.allclose(jnp.stack(derivatives[1:]), 0.0)


def test_3d_enforcement_gate_corner_has_finite_pseudoderivatives():
    geom = Cube(center=(0.0, 0.0, 0.0), side=1.0)
    gate = geom.make_enforcement_gate()
    corner = jnp.array([0.5, 0.5, 0.5])

    gradient = jax.grad(gate)(corner)
    hessian = jax.hessian(gate)(corner)

    assert gate(corner) == 0.0
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(jnp.isfinite(hessian))


def test_3d_enforcement_gate_vanishes_on_sliver_facet():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1e-3, 1e-2, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    geom = Geometry3DFromCAD(mesh, recenter=False)
    gate = geom.make_enforcement_gate()
    facet_point = jnp.asarray(np.array([0.026, 0.957, 0.017]) @ vertices[[0, 1, 2]])

    assert jnp.abs(gate(facet_point)) < 1e-10
