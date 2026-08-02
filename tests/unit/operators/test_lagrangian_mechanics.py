#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import opt_einsum as oe
import pytest

import phydrax as phx


def _oscillator_objects(*, dimension=1):
    lower = [-2.0] * dimension
    upper = [2.0] * dimension
    q_space = phx.domain.HyperRectangle(lower, upper, label="q")
    v_space = phx.domain.HyperRectangle(lower, upper, label="v")
    tangent = phx.domain.ProductDomain(q_space, v_space)
    time = phx.domain.TimeInterval(0.0, 2.0)
    return q_space, v_space, tangent, time


def test_canonical_momentum_and_euler_lagrange_harmonic_oscillator():
    _q_space, _v_space, tangent, time = _oscillator_objects()
    mass = 2.0
    stiffness = 8.0
    omega = jnp.sqrt(stiffness / mass)

    @tangent.Function("q", "v")
    def lagrangian(q, v):
        return 0.5 * mass * jnp.dot(v, v) - 0.5 * stiffness * jnp.dot(q, q)

    @time.Function("t")
    def trajectory(t):
        return jnp.asarray([jnp.cos(omega * t)])

    momentum = phx.operators.canonical_momentum(lagrangian)
    residual = phx.operators.euler_lagrange(trajectory, lagrangian)

    assert jnp.allclose(
        momentum.func(jnp.asarray([0.3]), jnp.asarray([-0.4])),
        jnp.asarray([-0.8]),
        atol=1e-12,
    )
    assert jnp.allclose(residual.func(0.37), jnp.zeros((1,)), atol=1e-10)


def test_euler_lagrange_matches_coupled_matrix_equation():
    _q_space, _v_space, tangent, time = _oscillator_objects(dimension=2)
    stiffness = jnp.asarray([[2.0, 0.0], [0.0, 3.0]])

    @tangent.Function("q", "v")
    def lagrangian(q, v):
        return 0.5 * jnp.dot(v, v) - 0.5 * oe.contract("i,ij,j->", q, stiffness, q)

    @time.Function("t")
    def trajectory(t):
        return jnp.asarray([jnp.cos(jnp.sqrt(2.0) * t), jnp.sin(jnp.sqrt(3.0) * t)])

    residual = phx.operators.euler_lagrange(trajectory, lagrangian)
    assert jnp.allclose(residual.func(0.41), jnp.zeros((2,)), atol=1e-10)


def test_euler_lagrange_generalized_force_sign():
    _q_space, _v_space, tangent, time = _oscillator_objects()

    @tangent.Function("q", "v")
    def lagrangian(q, v):
        return 0.5 * jnp.dot(v, v)

    @time.Function("t")
    def trajectory(t):
        return jnp.asarray([t])

    residual = phx.operators.euler_lagrange(
        trajectory,
        lagrangian,
        generalized_force=jnp.asarray([2.0]),
    )
    assert jnp.allclose(residual.func(0.5), jnp.asarray([-2.0]))


def test_euler_lagrange_rejects_mismatched_state_dimensions():
    q_space = phx.domain.HyperRectangle([-1.0], [1.0], label="q")
    v_space = phx.domain.HyperRectangle([-1.0, -1.0], [1.0, 1.0], label="v")
    tangent = phx.domain.ProductDomain(q_space, v_space)
    time = phx.domain.TimeInterval(0.0, 1.0)

    @tangent.Function("q", "v")
    def lagrangian(q, v):
        return jnp.sum(q) + jnp.sum(v)

    @time.Function("t")
    def trajectory(t):
        return jnp.asarray([t])

    with pytest.raises(ValueError, match="canonical dimensions must match"):
        phx.operators.euler_lagrange(trajectory, lagrangian)
