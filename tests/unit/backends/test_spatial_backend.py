from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._execution_device import device_platform
from phydrax.backends import (
    pallas_spatial_availability,
    spatial_pair_acceleration,
    spatial_squared_norm,
)
from phydrax.backends.spatial import spatial_leaf_pair_fragment
from phydrax.discretization.spatial import (
    MortonAddressPlan,
    MortonNeighborQueryPlan,
    MortonRadiusRelationPlan,
    MortonSourcePlan,
    SpatialDistanceBackend,
)


@pytest.fixture(params=(True, False), ids=("interpret", "compiled"))
def pallas_interpret(request: pytest.FixtureRequest) -> bool:
    interpret = bool(request.param)
    availability = pallas_spatial_availability(interpret=interpret)
    if not availability.available or not availability.capabilities.supports(
        "particle_fmm.p2p"
    ):
        pytest.skip(availability.reason)
    return interpret


def test_tpu_pallas_float32_distance_and_morton_leaves() -> None:
    if device_platform(jax.devices()[0]) != "tpu":
        pytest.skip("Requires a real TPU runtime.")
    availability = pallas_spatial_availability()
    availability.require("spatial.distance_squared")
    relative = jnp.asarray(
        np.arange(128 * 2, dtype=np.float32).reshape((128, 2)) / np.float32(100)
    )
    norm = jax.jit(lambda value: spatial_squared_norm(value, backend="pallas"))
    np.testing.assert_allclose(norm(relative), jnp.sum(relative * relative, axis=-1))
    np.testing.assert_allclose(
        jax.grad(lambda value: jnp.sum(norm(value)))(relative), 2 * relative
    )
    points = jnp.asarray([[0.1], [0.3], [0.6], [0.9]], dtype=jnp.float32)
    targets = jnp.asarray([[0.2], [0.8]], dtype=jnp.float32)
    sources = MortonSourcePlan(MortonAddressPlan((0.0,), (1.0,), 16), 4).prepare(points)
    reference = MortonNeighborQueryPlan(sources.source_plan, 2, 2).query(sources, targets)
    accelerated = MortonNeighborQueryPlan(
        sources.source_plan, 2, 2, distance_backend="pallas"
    ).query(sources, targets)
    assert bool(accelerated.evidence.successful)
    np.testing.assert_array_equal(accelerated.source_indices, reference.source_indices)
    np.testing.assert_array_equal(accelerated.valid, reference.valid)


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
    sources = MortonSourcePlan(address, 4).prepare(source)
    reference = MortonNeighborQueryPlan(
        sources.source_plan,
        2,
        2,
    ).query(sources, target)
    accelerated = MortonNeighborQueryPlan(
        sources.source_plan,
        2,
        2,
        distance_backend="pallas",
        pallas_interpret=pallas_interpret,
    ).query(sources, target)
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
    sources = MortonSourcePlan(address, 4).prepare(points)
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
            sources.source_plan,
            4,
            2,
            distance_backend=backend,
            pallas_interpret=pallas_interpret,
        )
        radius = MortonRadiusRelationPlan(
            sources.source_plan,
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
        query = eqx.filter_jit(neighbors.query)(sources, points, exclude_self=True)
        relation = eqx.filter_jit(radius.query)(
            sources, points, 0.2, exclude_self=True, pair_once=True
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
    not pallas_spatial_availability().available
    or not pallas_spatial_availability().capabilities.supports("particle_fmm.p2p"),
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


def _leaf_fragment_case(
    dtype: str, accumulation: phx.sparse.RelationAccumulation
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
    phx.sparse.PreparedStreamedRelation,
]:
    rng = np.random.default_rng(712)
    positions = rng.normal(size=(5, 3, 3)).astype(dtype)
    positions[1, 0] = positions[0, 0]  # distinct IDs at coincident coordinates
    strengths = rng.normal(size=(5, 3)).astype(dtype)
    mask = np.ones((5, 3), dtype=np.bool_)
    mask[1, 2] = False
    mask[4] = False
    positions[4] = np.nan
    strengths[4] = np.nan
    ids = np.arange(15, dtype=np.int64).reshape((5, 3))
    sources = (
        jnp.asarray(positions),
        jnp.asarray(strengths),
        jnp.asarray(mask),
        jnp.asarray(ids),
    )
    receiver_rows = np.asarray([0, 2, 4], dtype=np.int32)
    receivers = (
        sources[0][receiver_rows],
        sources[2][receiver_rows],
        sources[3][receiver_rows],
    )
    sender = np.asarray([0, 1, 2, 3, 4, 1, 3, 0, 1, 2, 4], dtype=np.int32)
    target = np.asarray([0] * 8 + [1, 1, 2], dtype=np.int32)
    relation = phx.sparse.EdgeRelation(sender, target, source_size=5, target_size=3)
    prepared = phx.sparse.StreamedRelationPlan(
        receiver_tile=3, edge_tile=5, channel_capacity=12, accumulation=accumulation
    ).prepare(relation, owner_id="test-leaf-fragments")
    return sources, receivers, prepared


def _direct_leaf_field(
    sources: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    receivers: tuple[jax.Array, jax.Array, jax.Array],
    prepared: phx.sparse.PreparedStreamedRelation,
    split: float | None,
    cutoff: float | None,
) -> np.ndarray:
    from scipy.special import erfc

    x, q, active, ids = jax.device_get(sources)
    y, target_active, target_ids = jax.device_get(receivers)
    output = np.zeros((3, 3, 4), dtype=x.dtype)
    fragments = jax.device_get(prepared.fragments())
    for tile in range(prepared.schedule.tile_count):
        for lane in np.flatnonzero(fragments.lane_use[tile]):
            source = fragments.lane_sources[tile, lane]
            target = fragments.receiver_ids[tile, fragments.lane_slots[tile, lane]]
            for i in range(3):
                for j in range(3):
                    if not target_active[target, i] or not active[source, j]:
                        continue
                    if target_ids[target, i] == ids[source, j]:
                        continue
                    displacement = x[source, j] - y[target, i]
                    radius = np.sqrt(np.dot(displacement, displacement) + 0.2**2)
                    if cutoff is not None and radius > cutoff:
                        continue
                    potential_factor = (
                        1.0 if split is None else erfc(radius / (2 * split))
                    )
                    gradient_factor = potential_factor
                    if split is not None:
                        gradient_factor += (
                            radius
                            / (split * np.sqrt(np.pi))
                            * np.exp(-((radius / (2 * split)) ** 2))
                        )
                    output[target, i, :3] += (
                        1.3 * q[source, j] * displacement * gradient_factor / radius**3
                    )
                    output[target, i, 3] += 1.3 * q[source, j] * potential_factor / radius
    return output


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("accumulation", ("fast", "deterministic", "compensated"))
@pytest.mark.parametrize("split,cutoff", ((None, None), (0.7, 2.0)))
def test_leaf_fragments_match_direct_and_seeded_reference(
    pallas_interpret: bool,
    dtype: str,
    accumulation: phx.sparse.RelationAccumulation,
    split: float | None,
    cutoff: float | None,
) -> None:
    sources, receivers, prepared = _leaf_fragment_case(dtype, accumulation)
    fragments = prepared.fragments()
    rng = np.random.default_rng(32)
    seed = phx.sparse.KeyGroupAccumulation(
        jnp.asarray(rng.normal(size=(3, 3, 4)), dtype=sources[0].dtype),
        jnp.full((3, 3, 4), 0.125, dtype=sources[0].dtype),
    )
    tolerance = 4e-6 if dtype == "float32" else 2e-12
    for tile in range(prepared.schedule.tile_count):
        fragment = jax.tree.map(lambda value: value[tile], fragments)
        gathered = jax.tree.map(lambda value: value[fragment.receiver_ids], receivers)

        def run(
            initial: phx.sparse.KeyGroupAccumulation, backend: SpatialDistanceBackend
        ) -> phx.sparse.KeyGroupAccumulation:
            return spatial_leaf_pair_fragment(
                sources,
                gathered,
                fragment,
                initial,
                softening=0.2,
                coefficient=1.3,
                short_range_scale=split,
                short_range_cutoff=cutoff,
                backend=backend,
                pallas_interpret=pallas_interpret,
            )

        expected = run(seed, "jax")
        actual = eqx.filter_jit(lambda initial: run(initial, "pallas"))(seed)
        np.testing.assert_allclose(
            actual.high, expected.high, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            actual.correction, expected.correction, rtol=tolerance, atol=tolerance
        )
        absent = np.asarray(fragment.receiver_lane_counts) == 0
        np.testing.assert_array_equal(actual.high[absent], seed.high[absent])
        np.testing.assert_array_equal(actual.correction[absent], seed.correction[absent])
        masked = eqx.tree_at(
            lambda f: f.lane_use, fragment, jnp.zeros_like(fragment.lane_use)
        )
        empty = spatial_leaf_pair_fragment(
            sources,
            gathered,
            masked,
            seed,
            softening=0.2,
            coefficient=1.3,
            backend="pallas",
            pallas_interpret=pallas_interpret,
        )
        np.testing.assert_array_equal(empty.high, seed.high)
        np.testing.assert_array_equal(empty.correction, seed.correction)

    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((3, 4), sources[0].dtype),
        jax.ShapeDtypeStruct((3, 4), sources[0].dtype),
    )

    def aggregate(
        p: dict[str, jax.Array],
        s: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        r: tuple[jax.Array, jax.Array, jax.Array],
        e: jax.Array,
        f: phx.sparse.StreamedFragment,
        initial: phx.sparse.KeyGroupAccumulation,
    ) -> phx.sparse.KeyGroupAccumulation:
        return spatial_leaf_pair_fragment(
            s,
            r,
            f,
            initial,
            softening=0.2,
            coefficient=1.3,
            short_range_scale=split,
            short_range_cutoff=cutoff,
            backend="pallas",
            pallas_interpret=pallas_interpret,
        )

    result = prepared.evaluate_fragments(
        payload,
        aggregate,
        lambda p, r, value: value,
        {},
        sources,
        receivers,
        jnp.zeros((11,), dtype=sources[0].dtype),
    )
    assert bool(result.evidence.successful)
    assert int(result.evidence.fragmented_receivers) > 0
    np.testing.assert_allclose(
        result.receiver_outputs,
        _direct_leaf_field(sources, receivers, prepared, split, cutoff),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("accumulation", ("fast", "deterministic", "compensated"))
def test_leaf_fragment_jvp_vjp_potential_and_mixed_derivatives(
    pallas_interpret: bool,
    accumulation: phx.sparse.RelationAccumulation,
) -> None:
    sources, receivers, prepared = _leaf_fragment_case("float64", accumulation)
    fragment = jax.tree.map(lambda value: value[0], prepared.fragments())
    receivers = jax.tree.map(lambda value: value[fragment.receiver_ids], receivers)
    rng = np.random.default_rng(513)
    initial = jnp.asarray(rng.normal(size=(3, 3, 4)), dtype=jnp.float64)
    correction = jnp.full_like(initial, 0.125)
    inputs = (sources[0], sources[1], receivers[0], initial, correction)
    tangents = tuple(jnp.asarray(rng.normal(size=value.shape)) for value in inputs)

    def field(
        values: tuple[jax.Array, ...], backend: SpatialDistanceBackend
    ) -> jax.Array:
        x, q, y, high, residual = values
        return spatial_leaf_pair_fragment(
            (x, q, sources[2], sources[3]),
            (y, receivers[1], receivers[2]),
            fragment,
            phx.sparse.KeyGroupAccumulation(high, residual),
            softening=0.2,
            coefficient=1.3,
            short_range_scale=0.7,
            short_range_cutoff=2.0,
            backend=backend,
            pallas_interpret=pallas_interpret,
        ).value

    expected, expected_jvp = jax.jvp(lambda v: field(v, "jax"), (inputs,), (tangents,))
    actual, actual_jvp = jax.jvp(lambda v: field(v, "pallas"), (inputs,), (tangents,))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2e-12, atol=2e-12)
    for potential_only in (True, False):
        cotangent = jnp.asarray(rng.normal(size=actual.shape))
        if potential_only:
            cotangent = cotangent.at[..., :3].set(0.0)
        _, pullback = jax.vjp(lambda v: field(v, "pallas"), inputs)
        adjoint = pullback(cotangent)[0]
        np.testing.assert_allclose(
            jnp.vdot(actual_jvp, cotangent),
            sum(
                jnp.vdot(tangent, gradient)
                for tangent, gradient in zip(tangents, adjoint, strict=True)
            ),
            rtol=2e-12,
            atol=2e-12,
        )

        def loss(v: tuple[jax.Array, ...], backend: SpatialDistanceBackend) -> jax.Array:
            return jnp.vdot(field(v, backend), cotangent)

        expected_grad = jax.grad(lambda v: loss(v, "jax"))(inputs)
        actual_grad = jax.grad(lambda v: loss(v, "pallas"))(inputs)
        _, expected_mixed = jax.jvp(
            jax.grad(lambda v: loss(v, "jax")), (inputs,), (tangents,)
        )
        _, actual_mixed = jax.jvp(
            jax.grad(lambda v: loss(v, "pallas")), (inputs,), (tangents,)
        )
        for a, b in zip(
            jax.tree.leaves((actual_grad, actual_mixed)),
            jax.tree.leaves((expected_grad, expected_mixed)),
            strict=True,
        ):
            assert bool(jnp.all(jnp.isfinite(a)))
            np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("split", (0.25, 0.05))
def test_leaf_fragment_inclusive_cutoff_and_true_split_potential(
    pallas_interpret: bool, split: float
) -> None:
    x = jnp.asarray([[[0.0, 0.0, 0.0]]], dtype=jnp.float64)
    mask = jnp.ones((1, 1), dtype=jnp.bool_)
    relation = phx.sparse.EdgeRelation(
        jnp.zeros(1, dtype=jnp.int32),
        jnp.zeros(1, dtype=jnp.int32),
        source_size=1,
        target_size=1,
    )
    fragment = jax.tree.map(
        lambda a: a[0],
        phx.sparse.StreamedRelationPlan(receiver_tile=1, edge_tile=1)
        .prepare(relation, owner_id="test-inclusive")
        .fragments(),
    )
    seed = phx.sparse.KeyGroupAccumulation(
        jnp.zeros((1, 1, 4), dtype=jnp.float64), jnp.zeros((1, 1, 4), dtype=jnp.float64)
    )
    sources = (
        x,
        jnp.ones((1, 1), dtype=jnp.float64),
        mask,
        jnp.zeros((1, 1), dtype=jnp.int64),
    )
    receivers = (x, mask, jnp.ones((1, 1), dtype=jnp.int64))
    actual = spatial_leaf_pair_fragment(
        sources,
        receivers,
        fragment,
        seed,
        softening=0.5,
        coefficient=1.0,
        short_range_scale=split,
        short_range_cutoff=0.5,
        backend="pallas",
        pallas_interpret=pallas_interpret,
    ).value
    np.testing.assert_array_equal(actual[..., :3], 0.0)
    np.testing.assert_allclose(
        actual[..., 3],
        2.0 * jax.lax.erfc(jnp.asarray(0.5 / (2.0 * split))),
        rtol=2e-12,
        atol=0.0,
    )


def test_leaf_fragment_refuses_unsupported_routes() -> None:
    sources, receivers, prepared = _leaf_fragment_case("float32", "compensated")
    fragment = jax.tree.map(lambda a: a[0], prepared.fragments())
    receivers = jax.tree.map(lambda a: a[fragment.receiver_ids], receivers)
    seed = phx.sparse.KeyGroupAccumulation(
        jnp.zeros((3, 3, 4), dtype=jnp.float32), jnp.zeros((3, 3, 4), dtype=jnp.float32)
    )
    with pytest.raises(ValueError, match="backend"):
        spatial_leaf_pair_fragment(
            sources,
            receivers,
            fragment,
            seed,
            softening=0.2,
            coefficient=1.0,
            backend="unknown",  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(TypeError, match="float32 and float64"):
        spatial_leaf_pair_fragment(
            (
                sources[0].astype(jnp.float16),
                sources[1].astype(jnp.float16),
                sources[2],
                sources[3],
            ),
            receivers,
            fragment,
            seed,
            softening=0.2,
            coefficient=1.0,
        )
    with pytest.raises(TypeError, match="share one dtype"):
        spatial_leaf_pair_fragment(
            (sources[0], sources[1].astype(jnp.float64), sources[2], sources[3]),
            receivers,
            fragment,
            seed,
            softening=0.2,
            coefficient=1.0,
        )
    if jax.default_backend() == "cpu":
        with pytest.raises(ValueError, match="require CUDA"):
            spatial_leaf_pair_fragment(
                sources,
                receivers,
                fragment,
                seed,
                softening=0.2,
                coefficient=1.0,
                backend="pallas",
                pallas_interpret=False,
            )


@pytest.mark.parametrize("edge_tile", (1, 4))
@pytest.mark.parametrize(
    "accumulation, expected", (("deterministic", 1.0), ("compensated", 2.0))
)
def test_leaf_fragment_cancellation_carries_uncollapsed_seed(
    pallas_interpret: bool,
    edge_tile: int,
    accumulation: phx.sparse.RelationAccumulation,
    expected: float,
) -> None:
    relation = phx.sparse.EdgeRelation(
        jnp.arange(4, dtype=jnp.int32),
        jnp.zeros(4, dtype=jnp.int32),
        source_size=4,
        target_size=1,
    )
    prepared = phx.sparse.StreamedRelationPlan(
        receiver_tile=1, edge_tile=edge_tile, accumulation=accumulation
    ).prepare(relation, owner_id="test-leaf-cancellation")
    x = jnp.tile(jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float64), (4, 1, 1))
    sources = (
        x,
        jnp.asarray([[1e16], [1.0], [-1e16], [1.0]], dtype=jnp.float64),
        jnp.ones((4, 1), dtype=jnp.bool_),
        jnp.arange(4, dtype=jnp.int64)[:, None],
    )
    receivers = (
        jnp.zeros((1, 1, 3), dtype=jnp.float64),
        jnp.ones((1, 1), dtype=jnp.bool_),
        jnp.full((1, 1), 4, dtype=jnp.int64),
    )
    seed = phx.sparse.KeyGroupAccumulation(
        jnp.zeros((1, 1, 4), dtype=jnp.float64), jnp.zeros((1, 1, 4), dtype=jnp.float64)
    )
    fragments = prepared.fragments()
    for tile in range(prepared.schedule.tile_count):
        fragment = jax.tree.map(lambda a: a[tile], fragments)
        seed = spatial_leaf_pair_fragment(
            sources,
            receivers,
            fragment,
            seed,
            softening=0.0,
            coefficient=1.0,
            backend="pallas",
            pallas_interpret=pallas_interpret,
        )
    np.testing.assert_array_equal(seed.value, [[[expected, 0.0, 0.0, expected]]])
    if accumulation == "compensated":
        np.testing.assert_array_equal(seed.high, [[[1.0, 0.0, 0.0, 1.0]]])
        np.testing.assert_array_equal(seed.correction, [[[1.0, 0.0, 0.0, 1.0]]])


def test_leaf_fragment_empty_source_capacity_keeps_seed(pallas_interpret: bool) -> None:
    sources, receivers, prepared = _leaf_fragment_case("float64", "compensated")
    fragment = jax.tree.map(lambda a: a[0], prepared.fragments())
    receivers = jax.tree.map(lambda a: a[fragment.receiver_ids], receivers)
    empty_sources = jax.tree.map(lambda a: a[:0], sources)
    fragment = eqx.tree_at(
        lambda f: f.lane_use, fragment, jnp.zeros_like(fragment.lane_use)
    )
    seed = phx.sparse.KeyGroupAccumulation(
        jnp.full((3, 3, 4), -0.0, dtype=jnp.float64),
        jnp.full((3, 3, 4), 0.125, dtype=jnp.float64),
    )
    for backend in ("jax", "pallas"):
        actual = spatial_leaf_pair_fragment(
            empty_sources,
            receivers,
            fragment,
            seed,
            softening=0.2,
            coefficient=1.0,
            backend=backend,
            pallas_interpret=pallas_interpret,
        )
        np.testing.assert_array_equal(actual.high, seed.high)
        np.testing.assert_array_equal(jnp.signbit(actual.high), jnp.signbit(seed.high))
        np.testing.assert_array_equal(actual.correction, seed.correction)
