#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _context(count=8):
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    kernel = phx.discretization.WendlandC2SPHKernel(1)
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    position = position + 0.002 * jnp.sin(2.0 * jnp.pi * position)
    neighborhood = (
        phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2, box=box
        )
        .prepare(particles)
        .build(position)
    )
    geometry = phx.discretization.particle_pair_geometry(
        position, neighborhood.pair_relation, box=box
    )
    physical = geometry.valid & (geometry.distance < 2.5 * spacing)
    return particles, kernel, position, neighborhood, geometry, physical, spacing


def test_kernel_and_first_order_corrections_return_explicit_evidence():
    particles, kernel, _, neighborhood, geometry, physical, spacing = _context()
    density = jnp.ones((particles.capacity,))
    execution = phx.discretization.ParticleExecutionPolicy()
    normalization = phx.discretization.sph_kernel_normalization(
        particles,
        density,
        neighborhood.pair_relation,
        geometry,
        physical,
        kernel,
        1.25 * spacing,
        execution,
    )
    correction = phx.discretization.sph_first_order_correction(
        phx.discretization.SPHFirstOrderGradientCorrectionPlan(maximum_condition=1e14),
        particles,
        density,
        neighborhood.pair_relation,
        geometry,
        physical,
        kernel,
        1.25 * spacing,
        execution,
    )

    assert jnp.all(normalization.completeness > 0.0)
    assert correction.correction_matrix.shape == (particles.capacity, 1, 1)
    assert jnp.all(jnp.isfinite(correction.residual_norm))


def test_delta_sph_artificial_viscosity_and_shepard_are_operational():
    particles, kernel, position, neighborhood, geometry, physical, spacing = _context()
    density = 1.0 + 0.02 * jnp.sin(2.0 * jnp.pi * position[:, 0])
    sound = jnp.ones_like(density)
    execution = phx.discretization.ParticleExecutionPolicy()
    diffusion = phx.discretization.sph_density_diffusion_rate(
        phx.discretization.AntuonoDeltaSPHDiffusionPlan(
            0.1,
            correction=phx.discretization.SPHFirstOrderGradientCorrectionPlan(
                maximum_condition=1e14
            ),
        ),
        particles,
        density,
        sound,
        neighborhood.pair_relation,
        geometry,
        physical,
        kernel,
        1.25 * spacing,
        execution,
    )
    velocity = -0.1 * (position - 0.5)
    artificial = phx.discretization.sph_artificial_viscosity_force(
        phx.discretization.MonaghanArtificialViscosityPlan(1.0, beta=2.0),
        particles,
        density,
        sound,
        velocity,
        neighborhood.pair_relation,
        geometry,
        physical,
        kernel,
        1.25 * spacing,
        execution,
    )
    renormalized, successful = phx.discretization.shepard_renormalized_density(
        particles,
        density,
        neighborhood.pair_relation,
        geometry,
        physical,
        kernel,
        1.25 * spacing,
        execution,
    )

    assert jnp.all(jnp.isfinite(diffusion.rate))
    assert artificial.dissipation_rate >= 0.0
    assert jnp.allclose(jnp.sum(artificial.force, axis=0), 0.0, atol=1e-13)
    assert jnp.all(successful)
    assert jnp.all(renormalized > 0.0)


def test_adaptive_h_computes_grad_h_and_variable_h_force():
    particles, kernel, _, neighborhood, geometry, _, spacing = _context()
    density = jnp.linspace(0.9, 1.1, particles.capacity)
    execution = phx.discretization.ParticleExecutionPolicy()
    adaptive = phx.discretization.adaptive_smoothing_state(
        phx.discretization.AlgebraicSmoothingLengthPlan(
            1.2, 0.8 * spacing, 2.0 * spacing
        ),
        particles,
        neighborhood.pair_relation,
        geometry,
        kernel,
        execution,
        density=density,
    )
    material = phx.equations.TaitBarotropicMaterial(1.0, 1.0)
    gradient = phx.discretization.variable_h_pressure_gradient(
        particles,
        density,
        material.pressure(density),
        adaptive,
        neighborhood.pair_relation,
        geometry,
        kernel,
        execution,
    )

    assert jnp.all(adaptive.smoothing_length > 0.0)
    assert jnp.all(jnp.isfinite(adaptive.omega))
    assert jnp.allclose(jnp.sum(-gradient, axis=0), 0.0, atol=1e-13)
