from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.backends import (
    pallas_spatial_availability,
    spatial_pair_acceleration,
    spatial_squared_norm,
)
from phydrax.discretization.spatial import MortonAddressPlan, MortonNeighborQueryPlan


def test_pallas_squared_norm_matches_jax_and_has_native_jvp() -> None:
    relative = jnp.arange(30.0).reshape((5, 2, 3)) / 10
    expected = spatial_squared_norm(relative, backend="jax")
    actual = jax.jit(
        lambda value: spatial_squared_norm(
            value,
            backend="pallas",
            pallas_interpret=True,
            block_size=4,
        )
    )(relative)
    gradient = jax.grad(
        lambda value: jnp.sum(
            spatial_squared_norm(
                value,
                backend="pallas",
                pallas_interpret=True,
                block_size=4,
            )
        )
    )(relative)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(gradient, 2 * relative)
    assert pallas_spatial_availability(interpret=True).available


def test_morton_query_pallas_leaf_kernel_matches_jax() -> None:
    source = jnp.asarray([[0.1], [0.3], [0.6], [0.9]])
    target = jnp.asarray([[0.2], [0.8]])
    address = MortonAddressPlan((0.0,), (1.0,), 16)
    reference = MortonNeighborQueryPlan(
        address,
        4,
        2,
        2,
        maximum_leaf_occupancy=2,
        target_top_nodes=1,
    ).query(source, target)
    accelerated = MortonNeighborQueryPlan(
        address,
        4,
        2,
        2,
        maximum_leaf_occupancy=2,
        target_top_nodes=1,
        distance_backend="pallas",
        pallas_interpret=True,
    ).query(source, target)
    assert bool(accelerated.evidence.successful)
    np.testing.assert_array_equal(accelerated.source_indices, reference.source_indices)
    np.testing.assert_array_equal(accelerated.valid, reference.valid)


def test_pallas_particle_pair_acceleration_matches_jax_and_gradients() -> None:
    relative = jnp.asarray([[0.2, -0.1, 0.4], [0.7, 0.3, -0.2], [-0.5, 0.6, 0.1]])
    mass = jnp.asarray([1.0, 0.7, 1.3])
    valid = jnp.asarray([True, False, True])

    def evaluate(value, backend):
        return spatial_pair_acceleration(
            value,
            mass,
            valid,
            softening=0.04,
            coefficient=1.7,
            backend=backend,
            pallas_interpret=backend == "pallas",
            block_size=2,
        )

    reference = evaluate(relative, "jax")
    accelerated = evaluate(relative, "pallas")
    reference_gradient = jax.grad(lambda value: jnp.sum(evaluate(value, "jax")))(relative)
    accelerated_gradient = jax.grad(lambda value: jnp.sum(evaluate(value, "pallas")))(
        relative
    )
    np.testing.assert_allclose(accelerated, reference)
    np.testing.assert_allclose(accelerated_gradient, reference_gradient)


def test_pallas_uniform_fmm_near_kernel_matches_jax() -> None:
    positions = jnp.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.8, 0.8, 0.8],
            [0.9, 0.8, 0.8],
        ]
    )
    masses = jnp.asarray([1.0, 0.7, 1.2, 0.9])
    tree = phx.solver.ParticleOctreePlan3D((1.0, 1.0, 1.0), 4).prepare(positions, masses)

    def evaluate(backend):
        return phx.solver.UniformFMMPlan(
            1.0,
            phx.solver.CartesianExpansionSpace(2),
            softening=0.02,
            opening_angle=0.5,
            maximum_leaf_occupancy=2,
            coarsening_factor=2,
            target_top_nodes=1,
            execution_backend=backend,
            pallas_interpret=backend == "pallas",
        ).evaluate(tree)

    reference = evaluate("jax")
    accelerated = evaluate("pallas")
    assert bool(accelerated.successful)
    np.testing.assert_allclose(accelerated.acceleration, reference.acceleration)
