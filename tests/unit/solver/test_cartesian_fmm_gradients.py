from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from phydrax.backends import pallas_spatial_availability
from phydrax.discretization.spatial import SpatialDistanceBackend
from phydrax.solver import (
    CartesianExpansionSpace,
    ParticleOctreePlan3D,
    UniformFMMPlan,
)


def _fixture(
    execution_backend: SpatialDistanceBackend,
    pallas_interpret: bool,
    dtype: str,
) -> Any:
    positions = jnp.asarray(
        [
            [0.10, 0.10, 0.10],
            [0.14, 0.09, 0.12],
            [0.45, 0.75, 0.20],
            [0.50, 0.78, 0.18],
            [0.82, 0.22, 0.80],
            [0.88, 0.25, 0.84],
        ],
        dtype=dtype,
    )
    masses = jnp.asarray([1.0, 0.7, 1.2, 0.8, 1.1, 0.6], dtype=dtype)
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
        execution_backend=execution_backend,
        pallas_interpret=pallas_interpret,
    )
    return positions, masses, weights, tree_plan, fmm


@pytest.mark.parametrize(
    ("execution_backend", "pallas_interpret"),
    (
        pytest.param("jax", False, id="jax"),
        pytest.param("pallas", True, id="pallas-interpret"),
        pytest.param("pallas", False, id="pallas-compiled"),
    ),
)
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_whole_fmm_position_and_mass_gradients_track_direct_force(
    execution_backend: SpatialDistanceBackend, pallas_interpret: bool, dtype: str
) -> None:
    if execution_backend == "pallas":
        availability = pallas_spatial_availability(interpret=pallas_interpret)
        if not availability.available or not availability.capabilities.supports(
            "particle_fmm.p2p"
        ):
            pytest.skip(availability.reason)
    positions, masses, weights, tree_plan, fmm = _fixture(
        execution_backend, pallas_interpret, dtype
    )
    softening = fmm.softening

    def fmm_loss(position: Any, mass: Any) -> Any:
        tree = tree_plan.prepare(position, mass)
        return jnp.sum(fmm.evaluate(tree).acceleration * weights)

    def direct_loss(position: Any, mass: Any) -> Any:
        displacement = position[None, :, :] - position[:, None, :]
        radius_squared = jnp.sum(displacement * displacement, axis=-1) + softening**2
        valid = ~jnp.eye(position.shape[0], dtype="bool")
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


@pytest.mark.parametrize("backend", ("jax", "pallas"))
def test_streamed_monopole_potential_geometry_and_mixed_derivatives(
    backend: SpatialDistanceBackend,
) -> None:
    positions = jnp.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.25, 0.3, 0.4],
            [0.5, 0.4, 0.2],
            [0.7, 0.8, 0.6],
        ],
        dtype=jnp.float64,
    )
    strengths = jnp.asarray([1.0, -2.0, 0.5, 1.5, -0.75], dtype=jnp.float64)
    plan = UniformFMMPlan(
        1.0,
        CartesianExpansionSpace(1),
        softening=0.03,
        maximum_leaf_occupancy=5,
        execution_backend=backend,
        pallas_interpret=backend == "pallas",
        accumulation="compensated",
    )

    def field(position: jax.Array, strength: jax.Array) -> jax.Array:
        structure = plan.prepare_structure(position, box_size=(1.0, 1.0, 1.0), depth=6)
        return plan.evaluate_monopole(structure, strength).potential

    def direct(position: jax.Array, strength: jax.Array) -> jax.Array:
        delta = position[None, :, :] - position[:, None, :]
        radius = jnp.sqrt(jnp.sum(delta**2, axis=-1) + plan.softening**2)
        pair = ~jnp.eye(position.shape[0], dtype=jnp.bool_)
        return jnp.sum(jnp.where(pair, strength[None, :] / radius, 0.0), axis=1)

    weights = jnp.asarray([0.2, -0.3, 0.4, 0.7, -0.2], dtype=jnp.float64)
    direction = jnp.arange(15, dtype=jnp.float64).reshape(5, 3) / 50

    def loss(position: jax.Array, strength: jax.Array) -> jax.Array:
        return jnp.vdot(weights, field(position, strength))

    def reference_loss(position: jax.Array, strength: jax.Array) -> jax.Array:
        return jnp.vdot(weights, direct(position, strength))

    actual = jax.grad(loss, argnums=(0, 1))(positions, strengths)
    expected = jax.grad(reference_loss, argnums=(0, 1))(positions, strengths)
    assert jnp.allclose(actual[0], expected[0], rtol=1e-11, atol=1e-11)
    assert jnp.allclose(actual[1], expected[1], rtol=1e-11, atol=1e-11)
    mixed = jax.jvp(
        lambda position: jax.grad(loss, argnums=1)(position, strengths),
        (positions,),
        (direction,),
    )[1]
    reference_mixed = jax.jvp(
        lambda position: jax.grad(reference_loss, argnums=1)(position, strengths),
        (positions,),
        (direction,),
    )[1]
    assert jnp.allclose(mixed, reference_mixed, rtol=1e-11, atol=1e-11)
