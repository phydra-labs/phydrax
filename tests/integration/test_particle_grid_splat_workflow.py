#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp

import phydrax as phx


def test_particle_grid_splat_mass_momentum_observation_workflow():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(5),
            phx.discretization.UniformAxisSpec(5),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    masses = jnp.asarray([1.0, 2.0, 1.5, 0.5])
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([10, 5, 30, 20]),
        masses,
        ambient_dimension=2,
    ).prepare()
    prepared = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    position = jnp.asarray([[0.2, 0.3], [0.7, 0.25], [0.35, 0.8], [0.8, 0.75]])
    velocity = jnp.asarray([[1.0, 0.0], [0.5, -1.0], [-0.25, 0.75], [1.5, 0.5]])
    state = prepared.build(position)
    mass = prepared.deposit_content(state, masses)
    momentum = prepared.deposit_content(state, masses[:, None] * velocity)
    reconstructed = prepared.reconstruct(state, velocity, masses)

    assert mass.successful and momentum.successful and reconstructed.successful
    assert mass.balance.closed_domain_conservation_valid
    assert momentum.balance.closed_domain_conservation_valid
    assert jnp.allclose(jnp.sum(mass.content), jnp.sum(masses))
    assert jnp.allclose(
        jnp.sum(momentum.content, axis=(0, 1)),
        jnp.sum(masses[:, None] * velocity, axis=0),
    )
    assert jnp.allclose(reconstructed.denominator, mass.content)
    assert jnp.allclose(reconstructed.numerator, momentum.content)
    integrated_density = jnp.sum(mass.density * grid.vertices().measure)
    assert jnp.allclose(integrated_density, jnp.sum(masses))

    x, y = grid.vertices().coordinates_by_axis
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    target_field = xx + 2.0 * yy
    observed = prepared.gather(state, target_field)
    assert jnp.all(observed.support)
    assert jnp.allclose(observed.values, position[:, 0] + 2.0 * position[:, 1])

    def observation_loss(current_position):
        current = prepared.build(current_position)
        density = prepared.deposit_content(current, masses).density
        return jnp.mean(density**2)

    gradient = jax.jit(jax.grad(observation_loss))(position)
    assert gradient.shape == position.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.sum(gradient * gradient) > 0.0


def test_batched_quadratic_bspline_cell_transfer_workflow():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(16, periodic=True),
            phx.discretization.UniformCellAxisSpec(16, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    masses = jnp.asarray([1.0, 2.0, 1.5, 0.5])
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([10, 5, 30, 20]),
        masses,
        ambient_dimension=2,
    ).prepare()
    prepared = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    positions = jnp.asarray(
        [
            [[0.2, 0.3], [0.7, 0.25], [0.35, 0.8], [0.8, 0.75]],
            [[0.25, 0.35], [0.75, 0.3], [0.4, 0.85], [0.85, 0.8]],
        ]
    )
    momentum = masses[:, None] * jnp.asarray(
        [[1.0, 0.0], [0.5, -1.0], [-0.25, 0.75], [1.5, 0.5]]
    )

    def transfer(position):
        state = prepared.build(position)
        mass = prepared.deposit_content(state, masses)
        vector = prepared.deposit_content(state, momentum)
        return (
            mass.content,
            vector.content,
            mass.balance.maximum_absolute_balance_defect,
            state.first_moments,
        )

    mass_content, momentum_content, defects, first_moments = jax.jit(jax.vmap(transfer))(
        positions
    )

    assert jnp.allclose(jnp.sum(mass_content, axis=(1, 2)), jnp.sum(masses))
    assert jnp.allclose(
        jnp.sum(momentum_content, axis=(1, 2)),
        jnp.sum(momentum, axis=0),
    )
    assert jnp.max(defects) < 1e-12
    assert jnp.max(jnp.abs(first_moments)) < 1e-12
