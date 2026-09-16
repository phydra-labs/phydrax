#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_slab_sweep_matches_upwind_absorber_and_global_balance():
    quadrature = phx.discretization.CertifiedSlabAngularQuadrature.gauss_legendre(4)
    cells = 80
    edges = jnp.linspace(0.0, 1.0, cells + 1)
    incident = jnp.zeros((1, quadrature.angle_count))
    incident = incident.at[0, quadrature.ordinates > 0.0].set(1.0)
    boundaries = phx.equations.SlabTransportBoundaryPlan(
        1,
        quadrature,
        left_kind="incident",
        left_incident=incident,
    )
    problem = phx.equations.MultigroupSlabTransportProblem(
        edges,
        jnp.ones((cells, 1)),
        jnp.zeros((cells, 1, 1)),
        jnp.zeros((cells, 1)),
        quadrature,
        boundaries,
    )
    result = phx.solver.DiscreteOrdinatesTransportPlan(
        problem, maximum_iterations=2, tolerance=1.0e-12
    ).solve()

    assert bool(result.evidence.successful)
    positive = np.flatnonzero(np.asarray(quadrature.ordinates) > 0.0)
    for angle in positive:
        mu = float(quadrature.ordinates[angle])
        expected = (1.0 / (1.0 + (1.0 / cells) / mu)) ** cells
        np.testing.assert_allclose(
            result.angular_flux[-1, 0, angle], expected, rtol=1.0e-11
        )
    np.testing.assert_allclose(result.evidence.global_balance_residual, 0.0, atol=1e-10)


def test_multigroup_groupsets_transport_downscatter_source():
    quadrature = phx.discretization.CertifiedSlabAngularQuadrature.gauss_legendre(4)
    cells = 16
    total = jnp.ones((cells, 2))
    scattering = jnp.zeros((cells, 2, 2)).at[:, 0, 1].set(0.4)
    source = jnp.zeros((cells, 2)).at[:, 0].set(1.0)
    boundaries = phx.equations.SlabTransportBoundaryPlan(2, quadrature)
    problem = phx.equations.MultigroupSlabTransportProblem(
        jnp.linspace(0.0, 1.0, cells + 1),
        total,
        scattering,
        source,
        quadrature,
        boundaries,
        group_sets=((0,), (1,)),
    )
    result = phx.solver.DiscreteOrdinatesTransportPlan(
        problem, maximum_iterations=32, tolerance=1.0e-10
    ).solve()

    assert bool(result.evidence.successful)
    assert jnp.all(result.scalar_flux[:, 0] > 0.0)
    assert jnp.all(result.scalar_flux[:, 1] > 0.0)
    assert jnp.max(jnp.abs(result.current)) < jnp.max(result.scalar_flux)


def test_discrete_ordinates_solve_is_jittable_and_rejects_invalid_initial_flux():
    quadrature = phx.discretization.CertifiedSlabAngularQuadrature.gauss_legendre(2)
    problem = phx.equations.MultigroupSlabTransportProblem(
        jnp.asarray((0.0, 0.5, 1.0)),
        jnp.ones((2, 1)),
        jnp.zeros((2, 1, 1)),
        jnp.ones((2, 1)),
        quadrature,
        phx.equations.SlabTransportBoundaryPlan(1, quadrature),
    )
    plan = phx.solver.DiscreteOrdinatesTransportPlan(
        problem, maximum_iterations=2, tolerance=1.0e-12
    )
    solve = jax.jit(lambda initial: plan.solve(initial))

    valid = solve(jnp.zeros((2, 1)))
    invalid = solve(-jnp.ones((2, 1)))

    assert bool(valid.evidence.successful)
    assert not bool(invalid.evidence.successful)
    assert not bool(invalid.evidence.nonnegative)
