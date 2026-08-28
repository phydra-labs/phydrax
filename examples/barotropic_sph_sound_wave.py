#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


particle_count = 24
spacing = 1.0 / particle_count
particle_ids = jnp.arange(particle_count)
masses = jnp.full((particle_count,), spacing)
particles = phx.discretization.ParticleSetPlan(
    particle_ids,
    masses,
    ambient_dimension=1,
    name="sound-wave-particles",
).prepare()
box = phx.discretization.ParticleBox([0.0], [1.0])
neighborhood = phx.discretization.CellListParticleNeighborhoodPlan(
    search_radius=2.5 * spacing,
    maximum_particles_per_cell=8,
    maximum_pairs=4 * particle_count,
    box=box,
)
method = phx.discretization.BarotropicSPHMethodPlan(
    phx.discretization.WendlandC2SPHKernel(1),
    1.25 * spacing,
)
problem = phx.equations.BarotropicFluidProblemIR(
    "periodic-sound-wave",
    phx.equations.TaitBarotropicMaterial(1.0, 1.0),
)
compiled = phx.equations.compile_barotropic_sph_problem(
    problem,
    particles,
    method,
    neighborhood=neighborhood,
)

lattice = (jnp.arange(particle_count, dtype=float) + 0.5)[:, None] * spacing
initial_position = lattice + 1.0e-3 * jnp.sin(2.0 * jnp.pi * lattice)
initial_velocity = jnp.zeros_like(initial_position)
initial_neighborhood = compiled.dynamics.neighborhood_state(initial_position)
initial_phase = compiled.dynamics.pack_phase_state(initial_position, initial_velocity)
initial = compiled.dynamics.diagnostics(
    0.0,
    initial_position,
    initial_phase[:, 1:],
    None,
)
end_time = 0.02
ivp = compiled.as_differential_problem(
    initial_position,
    initial_velocity,
    t0=0.0,
    t1=end_time,
)
solution = phx.solver.solve_diffrax(
    ivp,
    save_times=jnp.asarray([0.0, end_time]),
    solver=phx.solver.StormerVerlet(1),
    dt0=2.0e-4,
    max_steps=128,
)
final_position, final_momentum, _ = compiled.dynamics.unpack_phase_state(
    solution.states[-1]
)
final = compiled.dynamics.diagnostics(
    solution.times[-1],
    final_position,
    final_momentum,
    None,
)
final_neighborhood = compiled.dynamics.neighborhood_state(final_position)
energy_scale = jnp.maximum(jnp.abs(initial.total_energy), jnp.finfo(float).tiny)

print("solver", solution.resolved_method)
print("particle count", particle_count)
print("cell shape", compiled.dynamics.neighborhood.cell_shape)
print("pair capacity", compiled.dynamics.neighborhood.pair_capacity)
print("initial candidate pairs", int(initial_neighborhood.pair_count))
print("final candidate pairs", int(final_neighborhood.pair_count))
print("maximum final cell occupancy", int(final_neighborhood.maximum_cell_occupancy))
print("neighborhood successful", bool(final_neighborhood.successful))
print("active final pairs", int(final.active_pairs))
print(
    "initial density range",
    float(initial.density_minimum),
    float(initial.density_maximum),
)
print("final density range", float(final.density_minimum), float(final.density_maximum))
print(
    "momentum defect",
    float(jnp.max(jnp.abs(final.linear_momentum - initial.linear_momentum))),
)
print(
    "relative energy defect",
    float(jnp.abs(final.total_energy - initial.total_energy) / energy_scale),
)
print("bundle", solution.discretization_bundle_id)
