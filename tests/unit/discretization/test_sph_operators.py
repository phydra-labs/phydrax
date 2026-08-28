#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx
from phydrax.discretization.particle._sph_operators import (
    sph_continuity_density_rate,
    sph_summation_density,
    sph_symmetric_pressure_gradient,
)


def _context(count=6):
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2, box=box
    ).prepare(particles)
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    position = position + 0.01 * spacing * jnp.sin(2.0 * jnp.pi * position)
    state = prepared.build(position)
    geometry = phx.discretization.particle_pair_geometry(
        position, state.pair_relation, box=box
    )
    kernel = phx.discretization.WendlandC2SPHKernel(1)
    smoothing_length = 1.25 * spacing
    physical = geometry.valid & (
        geometry.distance < kernel.support_radius(smoothing_length)
    )
    execution = phx.discretization.ParticleExecutionPolicy(accumulation="deterministic")
    precision = phx.discretization.ParticlePrecisionPolicy()
    return (
        particles,
        position,
        state.pair_relation,
        geometry,
        physical,
        kernel,
        smoothing_length,
        execution,
        precision,
    )


def test_shared_summation_density_matches_direct_all_particle_sum():
    (
        particles,
        position,
        pairs,
        geometry,
        physical,
        kernel,
        smoothing_length,
        execution,
        precision,
    ) = _context()
    density = sph_summation_density(
        particles.safe_masses,
        particles.active_mask,
        pairs,
        geometry,
        physical,
        kernel,
        smoothing_length,
        particle_count=particles.capacity,
        execution=execution,
        precision=precision,
    )
    displacement = position[:, None, :] - position[None, :, :]
    displacement = displacement - jnp.round(displacement)
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    direct = jnp.sum(
        particles.masses[None, :] * kernel.value(distance, smoothing_length), axis=1
    )

    assert jnp.allclose(density, direct, atol=2e-14)


def test_pair_once_continuity_rate_matches_direct_directed_sum():
    (
        particles,
        position,
        pairs,
        geometry,
        physical,
        kernel,
        smoothing_length,
        execution,
        precision,
    ) = _context()
    velocity = 0.03 * jnp.cos(2.0 * jnp.pi * position)
    rate = sph_continuity_density_rate(
        particles.safe_masses,
        velocity,
        pairs,
        geometry,
        physical,
        kernel,
        smoothing_length,
        particle_count=particles.capacity,
        execution=execution,
        precision=precision,
    )
    displacement = position[:, None, :] - position[None, :, :]
    displacement = displacement - jnp.round(displacement)
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    gradient = kernel.gradient(displacement, distance, smoothing_length)
    direct = jnp.sum(
        particles.masses[None, :]
        * jnp.sum((velocity[:, None, :] - velocity[None, :, :]) * gradient, axis=-1),
        axis=1,
    )

    assert jnp.allclose(rate, direct, rtol=2e-12, atol=2e-14)
    translated = sph_continuity_density_rate(
        particles.safe_masses,
        jnp.ones_like(velocity),
        pairs,
        geometry,
        physical,
        kernel,
        smoothing_length,
        particle_count=particles.capacity,
        execution=execution,
        precision=precision,
    )
    assert jnp.array_equal(translated, jnp.zeros_like(translated))


def test_shared_pressure_gradient_matches_conservative_barotropic_dynamics():
    (
        particles,
        position,
        pairs,
        geometry,
        physical,
        kernel,
        smoothing_length,
        execution,
        precision,
    ) = _context()
    material = phx.equations.TaitBarotropicMaterial(1.0, 1.0)
    density = sph_summation_density(
        particles.safe_masses,
        particles.active_mask,
        pairs,
        geometry,
        physical,
        kernel,
        smoothing_length,
        particle_count=particles.capacity,
        execution=execution,
        precision=precision,
    )
    pressure = material.pressure(density)
    gradient = sph_symmetric_pressure_gradient(
        particles.safe_masses,
        density,
        pressure,
        pairs,
        geometry,
        physical,
        kernel,
        smoothing_length,
        particle_count=particles.capacity,
        execution=execution,
        precision=precision,
    )
    compiled = phx.equations.compile_barotropic_sph_problem(
        phx.equations.BarotropicFluidProblemIR("fluid", material),
        particles,
        phx.discretization.BarotropicSPHMethodPlan(kernel, smoothing_length),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            particles.capacity * (particles.capacity - 1) // 2,
            box=phx.discretization.ParticleBox([0.0], [1.0]),
        ),
    )

    assert jnp.allclose(
        gradient,
        compiled.dynamics.internal_potential_gradient(position),
        rtol=2e-12,
        atol=2e-14,
    )
