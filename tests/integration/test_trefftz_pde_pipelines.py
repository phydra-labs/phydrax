#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_four_dimensional_harmonic_boundary_fit_and_interior_audit():
    dimension = 4
    domain = phx.domain.HyperRectangle(
        (-1.0,) * dimension,
        (1.0,) * dimension,
    )
    field = domain.Model("x")(
        phx.equations.LinearTrefftzField(
            phx.equations.HarmonicPolynomialBasis(dimension, 1)
        )
    )
    target = domain.Function("x")(
        lambda x: 0.3 + jnp.dot(jnp.asarray([0.2, -0.4, 0.7, 0.1]), x)
    )
    boundary = domain.component({"x": phx.domain.Boundary()})
    condition = phx.conditions.Dirichlet("u", boundary, target=target)
    source = phx.integration.fixed(
        phx.integration.materialize(
            phx.integration.mean_over(boundary),
            phx.domain.PointSampling(192),
            key=jr.key(10),
        )
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(phx.terms.ResidualPenalty(condition, source),),
    )
    result = phx.solver.solve_linear_trial_space(solver, key=jr.key(11))
    assert bool(result.valid)
    assert float(result.final_residual_norm) < 1e-9

    batch = domain.component().sample(phx.domain.PointSampling(32), key=jr.key(12))
    audit = phx.equations.audit_trial_space(result.solver["u"], batch)
    assert bool(audit.valid)
    predicted = jnp.asarray(result.solver["u"](batch).data)
    expected = jnp.asarray(target(batch).data)
    assert jnp.allclose(predicted, expected, atol=1e-9, rtol=1e-9)


def test_five_dimensional_almansi_and_eight_dimensional_helmholtz_audits():
    poly_domain = phx.domain.HyperRectangle((-1.0,) * 5, (1.0,) * 5)
    poly_field = poly_domain.Model("x")(
        phx.equations.LinearTrefftzField(
            phx.equations.PolyharmonicAlmansiBasis(5, 2, (2, 1)),
            initial_scale=0.05,
            key=jr.key(13),
        )
    )
    poly_batch = poly_domain.component().sample(
        phx.domain.PointSampling(12), key=jr.key(14)
    )
    assert bool(phx.equations.audit_trial_space(poly_field, poly_batch).valid)

    dimension = 8
    wave_domain = phx.domain.HyperRectangle((-0.5,) * dimension, (0.5,) * dimension)
    directions = jnp.eye(dimension)
    wave_field = wave_domain.Model("x")(
        phx.equations.LinearTrefftzField(
            phx.equations.HelmholtzPlaneWaveBasis(
                dimension,
                1.75,
                directions,
            ),
            initial_scale=0.05,
            key=jr.key(15),
        )
    )
    wave_batch = wave_domain.component().sample(
        phx.domain.PointSampling(12), key=jr.key(16)
    )
    assert bool(phx.equations.audit_trial_space(wave_field, wave_batch).valid)
