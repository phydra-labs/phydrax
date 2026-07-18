#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _constant_residual_constraint(value):
    time = phx.domain.TimeInterval(0.0, 1.0)
    residual = time.Function()(jnp.asarray(value))
    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=time.component(),
        operator=lambda _field: residual,
        constraint_vars="u",
        num_points=16,
        structure=phx.domain.ProductStructure((("t",),)),
        reduction="mean",
    )
    return time, constraint


def test_complex_scalar_residual_uses_absolute_square():
    time, constraint = _constant_residual_constraint(1.0 + 2.0j)
    solver = phx.solver.FunctionalSolver(
        functions={"u": time.Function()(0.0)},
        constraints=[constraint],
    )

    loss = eqx.filter_jit(lambda s, key: s.loss(key=key))(solver, jr.key(0))
    assert jnp.isrealobj(loss)
    assert jnp.allclose(loss, 5.0, atol=1e-12)


def test_complex_vector_residual_uses_frobenius_norm():
    time, constraint = _constant_residual_constraint(
        jnp.asarray([1.0 + 2.0j, 3.0 - 4.0j])
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": time.Function()(0.0)},
        constraints=[constraint],
    )

    loss = solver.loss(key=jr.key(1))
    assert jnp.isrealobj(loss)
    assert jnp.allclose(loss, 30.0, atol=1e-12)


def test_real_residual_behavior_is_unchanged():
    time, constraint = _constant_residual_constraint(jnp.asarray([3.0, 4.0]))
    solver = phx.solver.FunctionalSolver(
        functions={"u": time.Function()(0.0)},
        constraints=[constraint],
    )
    assert jnp.allclose(solver.loss(key=jr.key(2)), 25.0, atol=1e-12)
