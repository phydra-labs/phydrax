#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _quadrature():
    velocities = np.asarray(
        tuple(itertools.product((-2.0, -1.0, 0.0, 1.0, 2.0), repeat=3))
    )
    return phx.equations.MolecularVelocityQuadrature(
        jnp.asarray(velocities),
        jnp.ones((velocities.shape[0],)),
        1,
    )


def _equilibrium(quadrature):
    multipliers = jnp.asarray((-2.0, 0.1, -0.05, 0.02, -0.8))
    population = jnp.exp(quadrature.moment_features @ multipliers)
    return population, quadrature.moments(population)


def test_positive_discrete_maxwellian_recovers_all_five_moments():
    quadrature = _quadrature()
    expected, target = _equilibrium(quadrature)
    solved = phx.equations.PositiveDiscreteMaxwellianPlan(quadrature).solve(target)

    assert bool(solved.successful)
    assert bool(jnp.all(solved.population > 0.0))
    np.testing.assert_allclose(solved.target_moments, target, rtol=1.0e-10)
    np.testing.assert_allclose(solved.population, expected, rtol=1.0e-8)


def test_bgk_upwind_wall_and_breakdown_preserve_physical_semantics():
    quadrature = _quadrature()
    equilibrium, _ = _equilibrium(quadrature)
    population = equilibrium * (
        1.0 + 0.02 * jnp.sin(jnp.arange(quadrature.velocity_count))
    )
    collision = phx.equations.MonatomicBGKCollisionPlan(
        quadrature, dynamic_viscosity=0.01
    )
    result = collision.advance(population, jnp.asarray(1.0e-3))
    upwind = phx.equations.PopulationUpwindFluxPlan(quadrature)
    flux = upwind.numerical_flux(
        population,
        0.9 * population,
        jnp.asarray((1.0,)),
    )
    wall = phx.equations.MaxwellGasSurfaceBoundary(
        quadrature,
        jnp.asarray((1.0, 0.0, 0.0)),
        wall_temperature=1.0,
        accommodation=0.5,
    )
    exterior = wall.exterior_population(population)
    breakdown = phx.equations.KineticBreakdownPlan(
        collision, knudsen_threshold=1.0e-12
    ).evaluate(population, jnp.asarray(1.0))

    assert bool(result.successful)
    np.testing.assert_allclose(result.moment_defect, 0.0, atol=1.0e-9)
    assert flux.shape == population.shape
    assert bool(jnp.all(exterior > 0.0))
    assert bool(breakdown.kinetic_required)


def test_shakhov_and_synthetic_correction_have_explicit_evidence():
    quadrature = _quadrature()
    equilibrium, moments = _equilibrium(quadrature)
    shakhov = phx.equations.ShakhovCollisionPlan(
        quadrature,
        dynamic_viscosity=0.01,
        prandtl_number=1.0,
    )
    target, defect = shakhov.target(equilibrium)
    np.testing.assert_allclose(target, equilibrium, rtol=1.0e-8)
    np.testing.assert_allclose(defect, 0.0, atol=1.0e-9)

    synthetic = phx.solver.advanced.KineticSyntheticAccelerationPlan(shakhov.bgk)
    corrected = synthetic.correct(equilibrium, moments)
    residual = synthetic.residual(
        equilibrium,
        jnp.zeros((3, 3)),
        jnp.zeros((3,)),
    )
    assert bool(corrected.successful)
    np.testing.assert_allclose(corrected.moment_defect, 0.0, atol=1.0e-9)
    assert bool(residual.successful)
