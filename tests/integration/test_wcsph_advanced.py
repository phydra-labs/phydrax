#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _compiled():
    count = 8
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    method = phx.discretization.WeaklyCompressibleSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(1),
        1.25 * spacing,
        density=phx.discretization.ContinuityDensityPlan(),
        physical_viscosity=phx.discretization.MorrisViscosityPlan(0.01),
        artificial_viscosity=phx.discretization.MonaghanArtificialViscosityPlan(0.1),
        density_diffusion=phx.discretization.MolteniColagrossiDensityDiffusionPlan(0.05),
        free_surface_detection=phx.discretization.FreeSurfaceDetectionPlan(
            completeness_threshold=0.7, normal_threshold=0.01
        ),
        free_surface_pressure=phx.discretization.FreeSurfacePressurePlan(0.0),
    )
    compiled = phx.equations.compile_weakly_compressible_sph_problem(
        phx.equations.WeaklyCompressibleFluidProblemIR(
            "advanced-wcsph", phx.equations.TaitBarotropicMaterial(1.0, 2.0)
        ),
        particles,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2, box=box
        ),
    )
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    velocity = -0.02 * (position - 0.5)
    return compiled, position, velocity


def test_wcsph_stabilizations_report_every_balance_term():
    compiled, position, velocity = _compiled()
    state = compiled.initialize_state(position, velocity)
    rate = compiled.dynamics(0.0, state, None)
    diagnostics = compiled.dynamics.diagnostics(0.0, state, None)

    assert jnp.all(jnp.isfinite(rate))
    assert diagnostics.viscous_dissipation_rate >= 0.0
    assert diagnostics.artificial_viscosity_dissipation >= 0.0
    assert jnp.isfinite(diagnostics.density_variance_rate)
    assert diagnostics.free_surface_count >= 0


def test_shepard_renormalization_runs_on_explicit_schedule():
    compiled, position, velocity = _compiled()
    density = 1.0 + 0.05 * jnp.sin(2.0 * jnp.pi * position[:, 0])
    state = compiled.initialize_state(position, velocity, density)
    transform = phx.solver.ShepardDensityRenormalizationTransform(
        compiled.dynamics,
        apply_every_steps=2,
        maximum_relative_correction=1.0,
    )
    method = phx.solver.SSPRK33FixedStepMethod(compiled.dynamics, transform=transform)
    solution = phx.solver.solve_fixed_step(
        phx.solver.FixedStepProblem(
            method,
            state,
            t0=0.0,
            t1=0.002,
            step_size=0.001,
        )
    )

    assert solution.successful
    assert jnp.array_equal(solution.transform_applied, jnp.asarray([False, True]))
    assert jnp.all(jnp.isfinite(solution.states))
