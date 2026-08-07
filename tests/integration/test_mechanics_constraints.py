#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_euler_lagrange_residual_runs_through_functional_solver():
    q_space = phx.domain.HyperRectangle([-2.0], [2.0], label="q")
    v_space = phx.domain.HyperRectangle([-2.0], [2.0], label="v")
    tangent = phx.domain.ProductDomain(q_space, v_space)
    time = phx.domain.TimeInterval(0.0, 1.0)

    @tangent.Function("q", "v")
    def lagrangian(q, v):
        return 0.5 * jnp.dot(v, v) - 0.5 * jnp.dot(q, q)

    @time.Function("t")
    def trajectory(t):
        return jnp.asarray([jnp.cos(t)])

    constraint = phx.constraints.ContinuousODEConstraint(
        "q",
        time,
        lambda q: phx.operators.euler_lagrange(q, lagrangian),
        sampling=phx.domain.PointSampling(32, layout=phx.domain.SampleLayout((("t",),))),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"q": trajectory},
        constraints=[constraint],
    )

    loss = eqx.filter_jit(lambda s, k: s.loss(key=k))(solver, jr.key(0))
    assert loss < 1e-18


def test_hamiltonian_residual_runs_through_multifield_constraint():
    q_space = phx.domain.HyperRectangle([-2.0], [2.0], label="q")
    p_space = phx.domain.HyperRectangle([-2.0], [2.0], label="p")
    phase = phx.domain.ProductDomain(q_space, p_space)
    time = phx.domain.TimeInterval(0.0, 1.0)

    @phase.Function("q", "p")
    def hamiltonian(q, p):
        return 0.5 * jnp.dot(q, q) + 0.5 * jnp.dot(p, p)

    @time.Function("t")
    def q(t):
        return jnp.asarray([jnp.cos(t)])

    @time.Function("t")
    def p(t):
        return jnp.asarray([-jnp.sin(t)])

    constraint = phx.constraints.FunctionalConstraint.from_operator(component=time.component(),
    operator=lambda q, p: phx.operators.canonical_hamiltonian_residual(
        q, p, hamiltonian
    ),
    constraint_vars=("q", "p"), sampling=phx.domain.PointSampling(32, layout=phx.domain.SampleLayout((("t",),))), )
    solver = phx.solver.FunctionalSolver(
        functions={"q": q, "p": p},
        constraints=[constraint],
    )

    loss = eqx.filter_jit(lambda s, k: s.loss(key=k))(solver, jr.key(1))
    assert loss < 1e-18
