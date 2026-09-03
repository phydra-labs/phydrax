from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications import cosmology


def _direct(positions, masses, softening):
    displacement = positions[None, :, :] - positions[:, None, :]
    squared = jnp.sum(displacement**2, axis=-1) + softening**2
    mask = ~jnp.eye(positions.shape[0], dtype=bool)
    return jnp.sum(
        jnp.where(
            mask[..., None],
            masses[None, :, None] * displacement / squared[..., None] ** 1.5,
            0.0,
        ),
        axis=1,
    )


def _cloud(count=32):
    key = jax.random.key(17)
    return 0.05 + 0.9 * jax.random.uniform(key, (count, 3))


def test_zero_opening_tree_matches_direct_without_dense_tree_storage() -> None:
    positions = _cloud()
    masses = jnp.linspace(0.5, 1.5, positions.shape[0])
    tree = cosmology.ParticleOctreePlan3D(
        (1.0, 1.0, 1.0), 5, target_leaf_occupancy=2
    ).prepare(positions, masses)
    result = cosmology.BarnesHutGravityPlan(
        1.0,
        softening=0.02,
        opening_angle=0.0,
        use_quadrupole=True,
    ).evaluate(tree)
    np.testing.assert_allclose(
        result.acceleration,
        _direct(positions, masses, 0.02),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert bool(result.evidence.traversal_complete)
    assert int(result.evidence.direct_particle_interactions) == 32 * 31
    assert tree.leaf_mass.size == (tree.depth + 1) * positions.shape[0]
    assert tree.leaf_mass.size < 8**tree.depth


def test_barnes_hut_accepts_far_nodes_and_jits() -> None:
    positions = _cloud(64)
    masses = jnp.ones((64,))
    tree = cosmology.ParticleOctreePlan3D(
        (1.0, 1.0, 1.0), 6, target_leaf_occupancy=1
    ).prepare(positions, masses)
    evaluate = eqx.filter_jit(
        cosmology.BarnesHutGravityPlan(
            1.0,
            softening=0.02,
            opening_angle=0.7,
            use_quadrupole=True,
        ).evaluate
    )
    result = evaluate(tree)
    assert bool(result.successful)
    assert int(result.evidence.accepted_leaf_interactions) > 0
    assert int(result.evidence.direct_particle_interactions) < 64 * 63
    reference = _direct(positions, masses, 0.02)
    relative_error = jnp.linalg.norm(result.acceleration - reference) / jnp.linalg.norm(
        reference
    )
    assert relative_error < 0.15


def test_barnes_hut_fixed_topology_position_gradient_is_finite() -> None:
    positions = _cloud(8)
    masses = jnp.ones((8,))
    plan = cosmology.ParticleOctreePlan3D((1.0, 1.0, 1.0), 4, target_leaf_occupancy=1)
    gravity = cosmology.BarnesHutGravityPlan(
        1.0,
        softening=0.05,
        opening_angle=0.4,
        use_quadrupole=False,
    )

    def objective(current):
        tree = plan.prepare(current, masses)
        acceleration = gravity.evaluate(tree).acceleration
        return jnp.sum(acceleration**2)

    gradient = jax.grad(objective)(positions)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_inactive_nonfinite_particles_are_ignored() -> None:
    positions = _cloud(8).at[-1].set(jnp.asarray((jnp.nan, jnp.nan, jnp.nan)))
    masses = jnp.ones((8,)).at[-1].set(jnp.nan)
    active = jnp.arange(8) < 7
    tree = cosmology.ParticleOctreePlan3D((1.0, 1.0, 1.0), 4).prepare(
        positions, masses, active
    )
    result = cosmology.BarnesHutGravityPlan(
        1.0, softening=0.05, opening_angle=0.5
    ).evaluate(tree)
    assert bool(result.successful)
    np.testing.assert_array_equal(result.acceleration[-1], 0.0)
