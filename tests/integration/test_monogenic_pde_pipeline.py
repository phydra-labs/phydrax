#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_monogenic_constant_boundary_field_recovers_without_interior_penalty():
    algebra = phx.metrix.clifford.CliffordAlgebraSpec((1, 1))
    basis = phx.equations.MonogenicPolynomialBasis(algebra, 0)
    domain = phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0))
    field = domain.Model("x")(phx.equations.LinearMonogenicField(basis))
    target_value = jnp.asarray([1.0, -0.2, 0.4, 0.3])
    target = domain.Function("x")(lambda x: target_value)
    boundary = domain.component({"x": phx.domain.Boundary()})
    condition = phx.conditions.Dirichlet("u", boundary, target=target)
    source = phx.integration.fixed(
        phx.integration.materialize(
            phx.integration.mean_over(boundary),
            phx.domain.PointSampling(64),
            key=jr.key(20),
        )
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(phx.terms.ResidualPenalty(condition, source),),
    )
    result = phx.solver.solve_linear_trial_space(solver, key=jr.key(21))

    assert bool(result.valid)
    assert float(result.final_residual_norm) < 1e-10
    interior = domain.component().sample(phx.domain.PointSampling(16), key=jr.key(22))
    assert bool(phx.equations.audit_trial_space(result.solver["u"], interior).valid)
    assert jnp.allclose(
        result.solver["u"](interior).data,
        jnp.broadcast_to(target_value, (16, 4)),
        atol=1e-10,
        rtol=1e-10,
    )
