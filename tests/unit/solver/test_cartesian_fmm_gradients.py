from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.solver import (
    CartesianExpansionSpace,
    ParticleOctreePlan3D,
    UniformFMMPlan,
)


def _fixture():
    positions = jnp.asarray(
        [
            [0.10, 0.10, 0.10],
            [0.14, 0.09, 0.12],
            [0.45, 0.75, 0.20],
            [0.50, 0.78, 0.18],
            [0.82, 0.22, 0.80],
            [0.88, 0.25, 0.84],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.asarray([1.0, 0.7, 1.2, 0.8, 1.1, 0.6])
    weights = jnp.arange(18.0, dtype=positions.dtype).reshape((6, 3)) / 17
    tree_plan = ParticleOctreePlan3D((1.0, 1.0, 1.0), 6)
    fmm = UniformFMMPlan(
        1.0,
        CartesianExpansionSpace(5),
        softening=0.03,
        opening_angle=0.45,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    )
    return positions, masses, weights, tree_plan, fmm


def test_whole_fmm_custom_vjp_matches_native_reference_pullback() -> None:
    positions, masses, weights, tree_plan, fmm = _fixture()

    def custom_loss(position, mass):
        tree = tree_plan.prepare(position, mass)
        return jnp.sum(fmm.evaluate(tree).acceleration * weights)

    def reference_loss(position, mass):
        tree = tree_plan.prepare(position, mass)
        return jnp.sum(fmm._evaluate_impl(tree).acceleration * weights)

    custom = jax.grad(custom_loss, argnums=(0, 1))(positions, masses)
    reference = jax.grad(reference_loss, argnums=(0, 1))(positions, masses)
    np.testing.assert_allclose(custom[0], reference[0], rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(custom[1], reference[1], rtol=2e-11, atol=2e-11)
    assert bool(jnp.all(jnp.isfinite(custom[0])))
    assert bool(jnp.all(jnp.isfinite(custom[1])))


def test_whole_fmm_position_and_mass_gradients_track_direct_force() -> None:
    positions, masses, weights, tree_plan, fmm = _fixture()
    softening = fmm.softening

    def fmm_loss(position, mass):
        tree = tree_plan.prepare(position, mass)
        return jnp.sum(fmm.evaluate(tree).acceleration * weights)

    def direct_loss(position, mass):
        displacement = position[None, :, :] - position[:, None, :]
        radius_squared = jnp.sum(displacement * displacement, axis=-1) + softening**2
        valid = ~jnp.eye(position.shape[0], dtype=bool)
        acceleration = jnp.sum(
            jnp.where(
                valid[..., None],
                mass[None, :, None] * displacement / radius_squared[..., None] ** 1.5,
                0.0,
            ),
            axis=1,
        )
        return jnp.sum(acceleration * weights)

    approximate = jax.grad(fmm_loss, argnums=(0, 1))(positions, masses)
    direct = jax.grad(direct_loss, argnums=(0, 1))(positions, masses)
    position_error = jnp.sqrt(jnp.sum((approximate[0] - direct[0]) ** 2))
    mass_error = jnp.sqrt(jnp.sum((approximate[1] - direct[1]) ** 2))
    assert float(position_error) < 0.2
    assert float(mass_error) < 0.1
