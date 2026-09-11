#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.discrete_velocity._energy_equilibrium import (
    EnergyEquilibriumStatus,
    PositiveEnergyEquilibriumPlan,
)
from phydrax.discretization.discrete_velocity._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
    d2v17_quadrature,
)


jax.config.update("jax_enable_x64", True)


def _energy_flux(populations, velocities):
    return jnp.einsum("...q,qd->...d", populations, velocities)


def test_fixed_newton_recovers_known_duals_for_batched_targets():
    quadrature = d2v17_quadrature()
    plan = PositiveEnergyEquilibriumPlan(quadrature, residual_tolerance=1.0e-12)
    total_energy = jnp.asarray((2.5, 1.75))
    expected_dual = jnp.asarray(((0.18, -0.11), (-0.2, 0.16)))
    seed = plan.evaluate(
        total_energy,
        jnp.zeros_like(expected_dual),
        expected_dual,
    )
    target_flux = _energy_flux(seed.populations, quadrature.velocities)

    result = plan.solve(total_energy, target_flux)

    np.testing.assert_array_equal(
        result.status,
        jnp.full(total_energy.shape, int(EnergyEquilibriumStatus.SUCCESS), jnp.int32),
    )
    np.testing.assert_allclose(result.dual, expected_dual, rtol=2.0e-10, atol=2.0e-10)
    np.testing.assert_allclose(
        result.populations,
        seed.populations,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(result.evidence.residual_norm, 0.0, atol=1.0e-12)
    assert bool(jnp.all(result.evidence.iterations > 0))


def test_zero_dual_is_the_weighted_quadrature_reference_without_energy_rescaling():
    quadrature = d2v17_quadrature()
    plan = PositiveEnergyEquilibriumPlan(quadrature)
    identical_plan = PositiveEnergyEquilibriumPlan(quadrature)
    total_energy = jnp.asarray(3.25)
    expected = total_energy * quadrature.weights / jnp.sum(quadrature.weights)
    target_flux = _energy_flux(expected, quadrature.velocities)

    result = plan.evaluate(total_energy, target_flux, jnp.zeros((2,)))

    assert bool(result.successful)
    np.testing.assert_allclose(result.populations, expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(jnp.sum(result.populations), total_energy, atol=1.0e-13)
    assert plan.population_convention == "weight_absorbed_total_energy_sum"
    assert plan.plan_id == identical_plan.plan_id
    assert result.plan_id == plan.plan_id
    assert result.evidence.plan_id == plan.plan_id
    assert result.evidence.quadrature_id == quadrature.quadrature_id


def test_learned_dual_keeps_total_energy_exact_and_exposes_flux_error_separately():
    quadrature = d2v17_quadrature()
    plan = PositiveEnergyEquilibriumPlan(quadrature)
    total_energy = jnp.asarray(4.0)
    target_flux = jnp.asarray((0.2, 0.3))

    result = plan.evaluate(total_energy, target_flux, jnp.asarray((0.3, -0.2)))

    assert bool(result.successful)
    assert not bool(result.evidence.converged)
    np.testing.assert_allclose(
        result.evidence.recovered_total_energy,
        total_energy,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(result.evidence.total_energy_residual, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        result.evidence.recovered_flux - target_flux,
        result.evidence.flux_residual,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert float(result.evidence.flux_error_norm) > 1.0e-2


def test_solver_refuses_infeasible_nonfinite_and_nonpositive_batched_inputs():
    quadrature = d2v17_quadrature()
    plan = PositiveEnergyEquilibriumPlan(quadrature)
    total_energy = jnp.asarray((2.0, 2.0, 2.0, 0.0))
    target_flux = jnp.asarray(
        (
            (4.1, 0.0),
            (jnp.nan, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        )
    )

    result = plan.solve(total_energy, target_flux)

    np.testing.assert_array_equal(
        result.status,
        np.asarray(
            (
                int(EnergyEquilibriumStatus.INFEASIBLE_TARGET),
                int(EnergyEquilibriumStatus.NONFINITE_INPUT),
                int(EnergyEquilibriumStatus.SUCCESS),
                int(EnergyEquilibriumStatus.NONPOSITIVE_TOTAL_ENERGY),
            ),
            dtype=np.int32,
        ),
    )
    refused = result.populations[jnp.asarray((0, 1, 3))]
    np.testing.assert_array_equal(refused, jnp.zeros_like(refused))
    np.testing.assert_allclose(jnp.sum(result.populations[2]), total_energy[2])
    assert float(result.evidence.interior_margin[0]) < 0.0
    assert not bool(result.evidence.finite[1])
    assert not bool(result.evidence.positive[0])


def test_plan_rejects_velocity_support_that_does_not_span_the_plane():
    quadrature = CertifiedDiscreteVelocityQuadrature(
        "collinear",
        np.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))),
        np.asarray((0.25, 0.5, 0.25)),
        reference_temperature=0.5,
        certified_degree=0,
        transport_kind="integer_lattice",
    )

    with pytest.raises(ValueError, match="span two dimensions"):
        PositiveEnergyEquilibriumPlan(quadrature)


def test_interior_solve_has_finite_flux_consistent_autodiff():
    quadrature = d2v17_quadrature()
    plan = PositiveEnergyEquilibriumPlan(quadrature, residual_tolerance=1.0e-12)
    total_energy = jnp.asarray(1.7)
    target_flux = jnp.asarray((0.15, -0.08))
    solved = plan.solve(total_energy, target_flux)
    assert bool(solved.successful)

    population_jacobian = jax.jacrev(
        lambda flux: plan.solve(total_energy, flux).populations
    )(target_flux)
    flux_jacobian = quadrature.velocities.T @ population_jacobian

    assert bool(jnp.all(jnp.isfinite(population_jacobian)))
    np.testing.assert_allclose(flux_jacobian, jnp.eye(2), rtol=2.0e-9, atol=2.0e-9)
