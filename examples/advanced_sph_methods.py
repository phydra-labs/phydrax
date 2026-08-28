#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


count = 8
spacing = 1.0 / count
particles = phx.discretization.ParticleSetPlan(
    jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
).prepare()
box = phx.discretization.ParticleBox([0.0], [1.0])
kernel = phx.discretization.WendlandC2SPHKernel(1)
neighborhood_plan = phx.discretization.DenseParticleNeighborhoodPlan(
    count * (count - 1) // 2, box=box
)
method = phx.discretization.WeaklyCompressibleSPHMethodPlan(
    kernel,
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
        "advanced-fluid", phx.equations.TaitBarotropicMaterial(1.0, 2.0)
    ),
    particles,
    method,
    neighborhood=neighborhood_plan,
)
position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
velocity = -0.02 * (position - 0.5)
initial = compiled.initialize_state(position, velocity)
transform = phx.solver.ShepardDensityRenormalizationTransform(
    compiled.dynamics, apply_every_steps=2, maximum_relative_correction=1.0
)
solution = phx.solver.solve_fixed_step(
    phx.solver.FixedStepProblem(
        phx.solver.SSPRK33FixedStepMethod(compiled.dynamics, transform=transform),
        initial,
        t0=0.0,
        t1=0.002,
        step_size=0.001,
    )
)
diagnostics = compiled.dynamics.diagnostics(0.002, solution.states[-1], None)

projection_neighborhood = neighborhood_plan.prepare(particles)
iisph = phx.discretization.PreparedIISPH(
    particles,
    projection_neighborhood,
    kernel,
    1.25 * spacing,
    phx.discretization.IISPHMethodPlan(1.0, maximum_iterations=3, tolerance=1.0),
)
iisph_result = iisph.step_detailed(
    0.0, iisph.initialize_state(position, jnp.zeros_like(position)), 0.001
)
dfsph = phx.discretization.PreparedDFSPH(
    particles,
    projection_neighborhood,
    kernel,
    1.25 * spacing,
    phx.discretization.DFSPHMethodPlan(
        1.0,
        divergence_iterations=3,
        density_iterations=3,
        divergence_tolerance=1.0,
        density_tolerance=1.0,
    ),
)
dfsph_result = dfsph.step_detailed(
    0.0, dfsph.initialize_state(position, jnp.zeros_like(position)), 0.001
)

print("advanced WCSPH successful", bool(solution.successful))
print("free-surface particles", int(diagnostics.free_surface_count))
print("viscous dissipation", float(diagnostics.viscous_dissipation_rate))
print("artificial dissipation", float(diagnostics.artificial_viscosity_dissipation))
print("density variance rate", float(diagnostics.density_variance_rate))
print("renormalization applications", int(jnp.sum(solution.transform_applied)))
print(
    "IISPH successful",
    bool(iisph_result.successful),
    "residual",
    float(iisph_result.residual),
)
print(
    "DFSPH successful",
    bool(dfsph_result.successful),
    "divergence",
    float(dfsph_result.divergence_residual),
    "density",
    float(dfsph_result.density_residual),
)
