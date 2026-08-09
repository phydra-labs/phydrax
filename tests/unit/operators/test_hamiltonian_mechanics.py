#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_hamiltonian_vector_field_and_trajectory_residual():
    q_space = phx.domain.HyperRectangle([-2.0], [2.0], label="q")
    p_space = phx.domain.HyperRectangle([-2.0], [2.0], label="p")
    phase = phx.domain.ProductDomain(q_space, p_space)
    time = phx.domain.TimeInterval(0.0, 2.0)

    @phase.Function("q", "p")
    def hamiltonian(q, p):
        return 0.5 * jnp.dot(p, p) + 0.5 * jnp.dot(q, q)

    @time.Function("t")
    def q(t):
        return jnp.asarray([jnp.cos(t)])

    @time.Function("t")
    def p(t):
        return jnp.asarray([-jnp.sin(t)])

    vector_field = phx.operators.canonical_hamiltonian_vector_field(hamiltonian)
    residual = phx.operators.canonical_hamiltonian_residual(q, p, hamiltonian)

    assert jnp.allclose(
        vector_field.func(jnp.asarray([0.2]), jnp.asarray([0.3])),
        jnp.asarray([0.3, -0.2]),
        atol=1e-12,
    )
    assert jnp.allclose(residual.func(0.73), jnp.zeros((2,)), atol=1e-10)


def test_canonical_poisson_brackets():
    q_space = phx.domain.HyperRectangle([-1.0, -1.0], [1.0, 1.0], label="q")
    p_space = phx.domain.HyperRectangle([-1.0, -1.0], [1.0, 1.0], label="p")
    phase = phx.domain.ProductDomain(q_space, p_space)

    @phase.Function("q")
    def q0(q):
        return q[0]

    @phase.Function("q")
    def q1(q):
        return q[1]

    @phase.Function("p")
    def p0(p):
        return p[0]

    @phase.Function("p")
    def p1(p):
        return p[1]

    point_q = jnp.asarray([0.2, -0.4])
    point_p = jnp.asarray([0.7, 0.1])
    assert jnp.allclose(
        phx.operators.canonical_poisson_bracket(q0, p0).func(point_q, point_p), 1.0
    )
    assert jnp.allclose(
        phx.operators.canonical_poisson_bracket(q0, p1).func(point_q, point_p), 0.0
    )
    assert jnp.allclose(
        phx.operators.canonical_poisson_bracket(q0, q1).func(point_q), 0.0
    )
    assert jnp.allclose(
        phx.operators.canonical_poisson_bracket(p0, p1).func(point_p), 0.0
    )


def test_hamiltonian_self_bracket_is_zero():
    q_space = phx.domain.HyperRectangle([-1.0], [1.0], label="q")
    p_space = phx.domain.HyperRectangle([-1.0], [1.0], label="p")
    phase = phx.domain.ProductDomain(q_space, p_space)

    @phase.Function("q", "p")
    def hamiltonian(q, p):
        return jnp.exp(q[0]) + p[0] ** 4

    bracket = phx.operators.canonical_poisson_bracket(hamiltonian, hamiltonian)
    assert jnp.allclose(
        bracket.func(jnp.asarray([0.2]), jnp.asarray([-0.3])),
        0.0,
        atol=1e-12,
    )


def test_hamilton_jacobi_free_particle_solution():
    x_space = phx.domain.HyperRectangle([-1.0], [1.0], label="x")
    p_space = phx.domain.HyperRectangle([-2.0], [2.0], label="p")
    time = phx.domain.TimeInterval(0.5, 2.0)
    spacetime = phx.domain.ProductDomain(x_space, time)
    extended_phase = phx.domain.ProductDomain(x_space, p_space, time)

    @spacetime.Function("x", "t")
    def action(x, t):
        return 0.5 * x[0] ** 2 / t

    @extended_phase.Function("p")
    def hamiltonian(p):
        return 0.5 * p[0] ** 2

    residual = phx.operators.hamilton_jacobi_residual(action, hamiltonian)
    assert jnp.allclose(
        residual.func(jnp.asarray([0.3]), 1.2),
        0.0,
        atol=1e-12,
    )


def test_hamiltonian_operator_rejects_mismatched_canonical_dimensions():
    q_space = phx.domain.HyperRectangle([-1.0], [1.0], label="q")
    p_space = phx.domain.HyperRectangle([-1.0, -1.0], [1.0, 1.0], label="p")
    phase = phx.domain.ProductDomain(q_space, p_space)

    @phase.Function("q", "p")
    def hamiltonian(q, p):
        return jnp.sum(q) + jnp.sum(p)

    with pytest.raises(ValueError, match="phase-space dimensions must match"):
        phx.operators.canonical_hamiltonian_vector_field(hamiltonian)
