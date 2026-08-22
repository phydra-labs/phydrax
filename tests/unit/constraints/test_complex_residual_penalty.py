#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _constant_residual_term(value):
    time = phx.domain.TimeInterval(0.0, 1.0)
    residual = time.Function()(jnp.asarray(value))
    condition = phx.conditions.Residual("u", time.component(), lambda _field: residual)
    source = phx.integration.per_step(
        phx.integration.mean_over(condition.on),
        phx.integration.MonteCarloPlan(16),
    )
    term = phx.terms.ResidualPenalty(condition, source)
    return time, term


def test_complex_scalar_residual_uses_absolute_square():
    time, term = _constant_residual_term(1.0 + 2.0j)
    solver = phx.solver.FunctionalSolver(
        functions={"u": time.Function()(0.0)}, terms=[term]
    )
    assert tuple(record.key.name for record in solver.discretization_bundle.records) == (
        "trial:u",
        "term:0",
    )
    assert solver.discretization_bundle.records[1].key.role == "residual"

    loss = eqx.filter_jit(lambda s, key: s.loss(key=key))(solver, jr.key(0))
    assert jnp.isrealobj(loss)
    assert jnp.allclose(loss, 5.0, atol=1e-12)


def test_complex_vector_residual_uses_frobenius_norm():
    time, term = _constant_residual_term(jnp.asarray([1.0 + 2.0j, 3.0 - 4.0j]))
    solver = phx.solver.FunctionalSolver(
        functions={"u": time.Function()(0.0)}, terms=[term]
    )

    loss = solver.loss(key=jr.key(1))
    assert jnp.isrealobj(loss)
    assert jnp.allclose(loss, 30.0, atol=1e-12)


def test_real_residual_behavior_is_unchanged():
    time, term = _constant_residual_term(jnp.asarray([3.0, 4.0]))
    solver = phx.solver.FunctionalSolver(
        functions={"u": time.Function()(0.0)}, terms=[term]
    )
    assert jnp.allclose(solver.loss(key=jr.key(2)), 25.0, atol=1e-12)
