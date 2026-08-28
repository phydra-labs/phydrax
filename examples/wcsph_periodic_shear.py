#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


resolution = 8
particle_count = resolution**2
spacing = 1.0 / resolution
kinematic_viscosity = 0.01
wave_number = 2.0 * jnp.pi
velocity_amplitude = 0.05
end_time = 0.01
x_axis = (jnp.arange(resolution, dtype=float) + 0.5) * spacing
x_grid, y_grid = jnp.meshgrid(x_axis, x_axis, indexing="ij")
initial_position = jnp.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
initial_velocity = jnp.stack(
    (
        jnp.zeros((particle_count,)),
        velocity_amplitude * jnp.sin(wave_number * initial_position[:, 0]),
    ),
    axis=-1,
)
particles = phx.discretization.ParticleSetPlan(
    jnp.arange(particle_count),
    jnp.full((particle_count,), spacing**2),
    ambient_dimension=2,
    name="periodic-shear-particles",
).prepare()
box = phx.discretization.ParticleBox([0.0, 0.0], [1.0, 1.0])
method = phx.discretization.WeaklyCompressibleSPHMethodPlan(phx.discretization.WendlandC2SPHKernel(2),
1.25 * spacing,
density=phx.discretization.ContinuityDensityPlan(), physical_viscosity=phx.discretization.MorrisViscosityPlan(kinematic_viscosity), )
compiled = phx.equations.compile_weakly_compressible_sph_problem(
    phx.equations.WeaklyCompressibleFluidProblemIR(
        "periodic-viscous-shear",
        phx.equations.TaitBarotropicMaterial(1.0, 10.0),
    ),
    particles,
    method,
    neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(
        method.kernel.support_factor * method.smoothing_length,
        16,
        20 * particle_count,
        box,
    ),
)
initial_state = compiled.initialize_state(initial_position, initial_velocity)
initial_diagnostics = compiled.dynamics.diagnostics(0.0, initial_state, None)
solution = phx.solver.solve_diffrax(
    compiled.as_differential_problem(
        initial_position,
        initial_velocity,
        t0=0.0,
        t1=end_time,
    ),
    save_times=jnp.asarray([0.0, end_time]),
    solver=phx.solver.SSPRK33(),
    dt0=5.0e-4,
    max_steps=64,
)
final_position, final_velocity, _ = compiled.dynamics.state_layout.unpack(
    solution.states[-1]
)
final_diagnostics = compiled.dynamics.diagnostics(
    solution.times[-1], solution.states[-1], None
)
expected_velocity = (
    velocity_amplitude
    * jnp.exp(-kinematic_viscosity * wave_number**2 * end_time)
    * jnp.sin(wave_number * final_position[:, 0])
)
velocity_error = jnp.sqrt(jnp.mean((final_velocity[:, 1] - expected_velocity) ** 2))
energy_decay = initial_diagnostics.kinetic_energy - final_diagnostics.kinetic_energy
neighborhood = compiled.dynamics.neighborhood.build(final_position)

print("solver", solution.resolved_method)
print("particle count", particle_count)
print("cell shape", compiled.dynamics.neighborhood.cell_shape)
print("pair count", int(neighborhood.pair_count))
print("maximum cell occupancy", int(neighborhood.maximum_cell_occupancy))
print("neighborhood successful", bool(neighborhood.successful))
print(
    "density range",
    float(final_diagnostics.density_minimum),
    float(final_diagnostics.density_maximum),
)
print("velocity L2 error", float(velocity_error))
print("kinetic energy decay", float(energy_decay))
print("viscous dissipation rate", float(final_diagnostics.viscous_dissipation_rate))
print(
    "momentum defect",
    float(
        jnp.max(
            jnp.abs(
                final_diagnostics.linear_momentum - initial_diagnostics.linear_momentum
            )
        )
    ),
)
print("bundle", solution.discretization_bundle_id)
