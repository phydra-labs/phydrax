from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.backends import (
    pallas_spatial_availability,
    spatial_pair_acceleration,
    spatial_squared_norm,
)
from phydrax.discretization.spatial import (
    MortonAddressPlan,
    MortonNeighborQueryPlan,
    MortonRadiusRelationPlan,
    SpatialDistanceBackend,
)


@pytest.fixture(params=(True, False), ids=("interpret", "compiled"))
def pallas_interpret(request: pytest.FixtureRequest) -> bool:
    interpret = bool(request.param)
    availability = pallas_spatial_availability(interpret=interpret)
    if not availability.available:
        pytest.skip(availability.reason)
    return interpret


def test_spatial_backend_scenario_1(pallas_interpret: bool) -> None:
    relative = jnp.arange(30.0).reshape((5, 2, 3)) / 10
    expected = spatial_squared_norm(relative, backend="jax")
    actual = jax.jit(
        lambda value: spatial_squared_norm(
            value,
            backend="pallas",
            pallas_interpret=pallas_interpret,
            block_size=4,
        )
    )(relative)
    gradient = jax.grad(
        lambda value: jnp.sum(
            spatial_squared_norm(
                value,
                backend="pallas",
                pallas_interpret=pallas_interpret,
                block_size=4,
            )
        )
    )(relative)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(gradient, 2 * relative)
    assert pallas_spatial_availability(interpret=True).available
    source = jnp.asarray([[0.1], [0.3], [0.6], [0.9]])
    target = jnp.asarray([[0.2], [0.8]])
    address = MortonAddressPlan((0.0,), (1.0,), 16)
    reference = MortonNeighborQueryPlan(
        address,
        4,
        2,
        2,
    ).query(source, target)
    accelerated = MortonNeighborQueryPlan(
        address,
        4,
        2,
        2,
        distance_backend="pallas",
        pallas_interpret=pallas_interpret,
    ).query(source, target)
    assert bool(accelerated.evidence.successful)
    np.testing.assert_array_equal(accelerated.source_indices, reference.source_indices)
    np.testing.assert_array_equal(accelerated.valid, reference.valid)


def test_pallas_particle_pair_acceleration_matches_jax_and_gradients(
    pallas_interpret: bool,
) -> None:
    relative = jnp.asarray([[0.2, -0.1, 0.4], [0.7, 0.3, -0.2], [-0.5, 0.6, 0.1]])
    mass = jnp.asarray([1.0, 0.7, 1.3])
    valid = jnp.asarray([True, False, True])

    def evaluate(value: Any, backend: Any) -> Any:
        return spatial_pair_acceleration(
            value,
            mass,
            valid,
            softening=0.04,
            coefficient=1.7,
            backend=backend,
            pallas_interpret=pallas_interpret,
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
    relative_tangent = jnp.asarray([[0.1, 0.2, -0.3], [-0.4, 0.1, 0.2], [0.3, -0.2, 0.1]])
    mass_tangent = jnp.asarray([0.2, -0.1, 0.4])

    def evaluate_both(current_relative: Any, current_mass: Any, backend: Any) -> Any:
        return spatial_pair_acceleration(
            current_relative,
            current_mass,
            valid,
            softening=0.04,
            coefficient=1.7,
            backend=backend,
            pallas_interpret=pallas_interpret,
            block_size=2,
        )

    _, reference_tangent = jax.jvp(
        lambda current_relative, current_mass: evaluate_both(
            current_relative,
            current_mass,
            "jax",
        ),
        (relative, mass),
        (relative_tangent, mass_tangent),
    )
    _, accelerated_tangent = jax.jvp(
        lambda current_relative, current_mass: evaluate_both(
            current_relative,
            current_mass,
            "pallas",
        ),
        (relative, mass),
        (relative_tangent, mass_tangent),
    )
    np.testing.assert_allclose(accelerated_tangent, reference_tangent)

    cotangent = jnp.arange(relative.size, dtype=relative.dtype).reshape(relative.shape)
    _, pullback = jax.vjp(
        lambda current_relative, current_mass: evaluate_both(
            current_relative,
            current_mass,
            "pallas",
        ),
        relative,
        mass,
    )
    relative_cotangent, mass_cotangent = pullback(cotangent)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, accelerated_tangent),
        jnp.vdot(relative_cotangent, relative_tangent)
        + jnp.vdot(mass_cotangent, mass_tangent),
    )


def test_pallas_uniform_fmm_near_kernel_matches_jax(pallas_interpret: bool) -> None:
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

    def evaluate(backend: Any) -> Any:
        return phx.solver.UniformFMMPlan(
            1.0,
            phx.solver.CartesianExpansionSpace(2),
            softening=0.02,
            opening_angle=0.5,
            maximum_leaf_occupancy=2,
            coarsening_factor=2,
            target_top_nodes=1,
            execution_backend=backend,
            pallas_interpret=pallas_interpret,
        ).evaluate(tree)

    reference = evaluate("jax")
    accelerated = evaluate("pallas")
    assert bool(accelerated.successful)
    np.testing.assert_allclose(accelerated.acceleration, reference.acceleration)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_pallas_spatial_3d_padding_and_consumers(
    pallas_interpret: bool, dtype: str
) -> None:
    relative = jnp.asarray(np.random.default_rng(9).normal(size=(257, 3)).astype(dtype))
    mass = jnp.linspace(0.2, 1.3, relative.shape[0], dtype=relative.dtype)
    valid = jnp.arange(relative.shape[0]) % 3 != 0
    relative = jnp.where(valid[:, None], relative, 0.0)

    def pairs(backend: SpatialDistanceBackend) -> jax.Array:
        return spatial_pair_acceleration(
            relative,
            mass,
            valid,
            softening=0.0,
            coefficient=1.7,
            backend=backend,
            pallas_interpret=pallas_interpret,
            block_size=127,
        )

    actual = jax.jit(lambda: pairs("pallas"))()
    np.testing.assert_allclose(actual, pairs("jax"), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual[~valid], 0.0)
    expected = spatial_squared_norm(relative)
    actual_norm = jax.jit(
        lambda value: spatial_squared_norm(
            value,
            backend="pallas",
            pallas_interpret=pallas_interpret,
            block_size=3,
        )
    )(relative)
    np.testing.assert_allclose(actual_norm, expected, rtol=2e-6, atol=2e-6)

    points = jnp.asarray(
        [[0.05, 0.4, 0.5], [0.95, 0.4, 0.5], [0.45, 0.5, 0.5], [0.6, 0.5, 0.5]],
        dtype=relative.dtype,
    )
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=(True, False, False)
    )
    particles = phx.discretization.ParticleSetPlan(
        np.asarray([40, 10, 30, 20], dtype=np.int64),
        np.ones(4, dtype=np.float64),
        ambient_dimension=3,
    ).prepare()
    box = phx.discretization.ParticleBox(
        jnp.zeros(3, dtype=relative.dtype),
        jnp.ones(3, dtype=relative.dtype),
        periodic_axes=(True, False, False),
    )
    outputs = []
    for backend in ("jax", "pallas"):
        neighbors = MortonNeighborQueryPlan(
            address,
            4,
            4,
            2,
            distance_backend=backend,
            pallas_interpret=pallas_interpret,
        )
        radius = MortonRadiusRelationPlan(
            address,
            4,
            4,
            6,
            maximum_candidates=4,
            distance_backend=backend,
            pallas_interpret=pallas_interpret,
        )
        neighborhood = phx.discretization.MortonTreeParticleNeighborhoodPlan(
            0.2,
            6,
            box,
            maximum_candidates=4,
            distance_backend=backend,
            pallas_interpret=pallas_interpret,
        ).prepare(particles)
        query = eqx.filter_jit(neighbors.query)(points, points, exclude_self=True)
        relation = eqx.filter_jit(radius.query)(
            points, points, 0.2, exclude_self=True, pair_once=True
        )
        state = eqx.filter_jit(neighborhood.build)(points)
        assert bool(query.evidence.successful)
        assert bool(relation.evidence.successful)
        assert bool(state.successful)
        assert int(relation.evidence.required_pairs) == 2
        outputs.append((query, relation, state))
    # Static plan identities intentionally differ by execution policy.
    for reference, accelerated in zip(
        jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1]), strict=True
    ):
        np.testing.assert_array_equal(accelerated, reference)


@pytest.mark.skipif(
    not pallas_spatial_availability().available,
    reason=pallas_spatial_availability().reason,
)
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_compiled_pallas_full_fmm_matches_jax_direct_and_strength_derivatives(
    dtype: str,
) -> None:
    generator = np.random.default_rng(291)
    centers = np.asarray(
        [[a, b, c] for a in (0.25, 0.75) for b in (0.25, 0.75) for c in (0.25, 0.75)]
    )
    positions = (
        centers[np.arange(64) % 8] + 0.02 * generator.normal(size=(64, 3))
    ).astype(dtype)
    strengths = jnp.asarray(generator.uniform(0.5, 1.5, size=64).astype(dtype))
    refreshed = jnp.asarray(generator.normal(size=64).astype(dtype))
    weights = jnp.asarray(generator.normal(size=(64, 3)).astype(dtype))
    tree = phx.solver.ParticleOctreePlan3D((1.0, 1.0, 1.0), 8).prepare(
        positions, strengths
    )
    results = []
    for backend in ("jax", "pallas"):
        plan = phx.solver.UniformFMMPlan(
            1.0,
            phx.solver.CartesianExpansionSpace(3),
            softening=0.02,
            opening_angle=0.6,
            maximum_leaf_occupancy=8,
            coarsening_factor=2,
            target_top_nodes=1,
            execution_backend=backend,
            pallas_interpret=False,
        )
        acceleration = eqx.filter_jit(plan.evaluate)(tree)
        structure = eqx.filter_jit(plan.prepare_structure)(
            positions, box_size=(1.0, 1.0, 1.0), depth=8
        )
        field = eqx.filter_jit(plan.evaluate_monopole)
        first = field(structure, strengths)
        second = field(structure, refreshed)

        def values(current: jax.Array) -> tuple[jax.Array, jax.Array]:
            result = plan.evaluate_monopole(structure, current)
            return result.potential, result.gradient

        _, tangent = jax.jvp(values, (strengths,), (refreshed,))
        cotangent = jax.grad(
            lambda current: (
                jnp.sum(values(current)[0]) + jnp.sum(values(current)[1] * weights)
            )
        )(strengths)
        assert bool(acceleration.successful)
        assert bool(acceleration.evidence.successful)
        assert bool(acceleration.evidence.traversal_complete)
        assert acceleration.fmm_evidence is not None
        assert bool(acceleration.fmm_evidence.successful)
        assert bool(first.successful) and bool(second.successful)
        assert int(first.evidence.p2p_count) > 0
        assert int(first.evidence.required_far) > 0
        assert int(first.evidence.m2m_count) > 0
        assert int(first.evidence.l2l_count) > 0
        assert int(acceleration.evidence.accepted_leaf_interactions) == int(
            first.evidence.required_far
        )
        assert int(acceleration.evidence.direct_particle_interactions) == int(
            first.evidence.p2p_count
        )
        weighted_acceleration = np.asarray(strengths, dtype=np.float64)[:, None] * (
            np.asarray(acceleration.acceleration, dtype=np.float64)
        )
        # Net force nearly cancels; bound its reduction roundoff by the
        # absolute force sum, not by the small residual's relative magnitude.
        force_roundoff = (
            8
            * np.finfo(np.dtype(dtype)).eps
            * np.sum(np.abs(weighted_acceleration), axis=0)
        )
        assert np.all(
            np.abs(
                np.asarray(acceleration.evidence.net_force)
                - np.sum(weighted_acceleration, axis=0)
            )
            <= force_roundoff
        )
        np.testing.assert_allclose(
            acceleration.evidence.maximum_acceleration,
            np.max(np.linalg.norm(np.asarray(acceleration.acceleration), axis=-1)),
            rtol=2e-6,
        )
        for gravity_evidence, field_evidence in zip(
            jax.tree.leaves(acceleration.fmm_evidence),
            jax.tree.leaves(first.evidence),
            strict=True,
        ):
            np.testing.assert_array_equal(gravity_evidence, field_evidence)
        np.testing.assert_allclose(
            acceleration.acceleration, first.gradient, rtol=2e-6, atol=2e-5
        )
        for derivative, value in zip(
            tangent, (second.potential, second.gradient), strict=True
        ):
            np.testing.assert_allclose(derivative, value, rtol=2e-5, atol=2e-4)
        results.append((acceleration.acceleration, first, second, tangent, cotangent))
    for reference, accelerated in zip(
        jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True
    ):
        np.testing.assert_allclose(accelerated, reference, rtol=2e-5, atol=2e-4)

    displacement = positions[None, :, :].astype(np.float64) - positions[
        :, None, :
    ].astype(np.float64)
    squared = np.sum(displacement**2, axis=-1) + 0.02**2
    pair = ~np.eye(64, dtype=np.bool_)
    potential = np.where(pair, squared**-0.5, 0.0) @ np.asarray(strengths)
    gradient = np.einsum(
        "ij,ijk->ik",
        np.where(pair, squared**-1.5, 0.0) * np.asarray(strengths)[None, :],
        displacement,
    )
    accelerated = results[1][1]
    potential_error = np.max(np.abs(np.asarray(accelerated.potential) - potential))
    gradient_error = np.max(
        np.linalg.norm(np.asarray(accelerated.gradient) - gradient, axis=-1)
    )
    assert potential_error / np.max(np.abs(potential)) < 0.01
    assert gradient_error / np.max(np.linalg.norm(gradient, axis=-1)) < 0.01
