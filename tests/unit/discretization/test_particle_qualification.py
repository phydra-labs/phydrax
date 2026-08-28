#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_dimensionless_constraint_residuals_reject_percent_level_density_error():
    residuals = phx.discretization.particle_constraint_residuals(
        jnp.asarray([1.012416, 1.0]),
        1.0,
        jnp.asarray([0.5, 0.5]),
    )
    profile = phx.discretization.ParticleQualificationProfile(
        density_linf_tolerance=1e-3,
        density_l2_tolerance=1e-3,
    )

    assert residuals.relative_density_linf > 0.01
    assert not profile.constraints_satisfied(residuals)


def test_execution_success_is_distinct_from_production_qualification():
    profile = phx.discretization.ParticleQualificationProfile()
    evidence = (
        phx.discretization.ParticleClaimEvidence(
            phx.discretization.ParticleQualificationClaim.FINITE_EXECUTION,
            "evidence:finite",
            True,
        ),
    )
    result = phx.discretization.ParticleQualificationResult(
        phx.discretization.ParticleMethodMaturity.EXPERIMENTAL,
        profile,
        evidence,
        True,
        False,
    )

    assert result.execution_successful
    assert not result.numerical_constraints_satisfied
    assert not result.production_gate_satisfied


def test_iisph_and_dfsph_lenient_execution_do_not_pass_production_gate():
    count = 6
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2,
        box=phx.discretization.ParticleBox([0.0], [1.0]),
    ).prepare(particles)
    kernel = phx.discretization.WendlandC2SPHKernel(1)
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    velocity = jnp.zeros_like(position)
    iisph = phx.discretization.PreparedIISPH(
        particles,
        neighborhood,
        kernel,
        1.25 * spacing,
        phx.discretization.IISPHMethodPlan(1.0, maximum_iterations=2, tolerance=1.0),
    )
    iisph_result = iisph.step_detailed(
        0.0, iisph.initialize_state(position, velocity), 0.001
    )
    dfsph = phx.discretization.PreparedDFSPH(
        particles,
        neighborhood,
        kernel,
        1.25 * spacing,
        phx.discretization.DFSPHMethodPlan(
            1.0,
            divergence_iterations=2,
            density_iterations=2,
            divergence_tolerance=1.0,
            density_tolerance=1.0,
        ),
    )
    dfsph_result = dfsph.step_detailed(
        0.0, dfsph.initialize_state(position, velocity), 0.001
    )

    assert iisph_result.successful
    assert not iisph_result.numerical_constraints_satisfied
    assert not iisph_result.production_qualified
    assert dfsph_result.successful
    assert not dfsph_result.numerical_constraints_satisfied
    assert not dfsph_result.production_qualified
