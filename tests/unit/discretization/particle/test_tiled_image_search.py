#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalar/tiled image reference: exact dense equivalence with bounded tiles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jax_core

from phydrax.discretization.particle import (
    DenseImageRouteSearch,
    ImageReceiverSubset,
    ImageRouteSearchResult,
    ImageTilePolicy,
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    TiledImageRouteSearch,
)
from phydrax.discretization.particle._image_neighborhood import AbstractImageRouteSearch
from phydrax.discretization.particle._image_relation import ParticleImageHalfView
from phydrax.discretization.particle._tiled_image_search import (
    _receiver_degree,
    _route_targets,
    _row_prefix,
    _saturating_scan,
)
from tests._support.image_lattice import (
    assert_resolved_cutoff,
    assert_routes_match,
    lattice_routes,
)


TRICLINIC = np.asarray([[2.0, 0.0, 0.0], [0.7, 2.1, 0.0], [0.3, -0.4, 2.3]])
RADIUS = 2.6
CAPACITY = ParticleImageCapacity(
    maximum_particles_per_cell=16,
    maximum_edges=8192,
    maximum_degree=1024,
    maximum_images=4096,
)
SEGMENTED = ParticleImageCapacity(
    maximum_particles_per_cell=16,
    maximum_edges=8192,
    maximum_degree=1024,
    maximum_images=4096,
    maximum_case_edges=4096,
)
TILES = {
    "scalar": ImageTilePolicy(
        receiver_width=1, source_width=1, maximum_tile_candidates=10**6
    ),
    "rectangular": ImageTilePolicy(
        receiver_width=2, source_width=3, maximum_tile_candidates=10**6
    ),
    "ragged": ImageTilePolicy(
        receiver_width=4, source_width=5, maximum_tile_candidates=10**6
    ),
    "wide": ImageTilePolicy(
        receiver_width=64, source_width=64, maximum_tile_candidates=10**7
    ),
}
PERIODIC = {
    "full": (True, True, True),
    "slab": (True, True, False),
    "wire": (False, True, False),
    "finite": (False, False, False),
}


def _pair(
    vectors: np.ndarray,
    periodic: np.ndarray,
    particles: int,
    *,
    tiles: ImageTilePolicy = TILES["rectangular"],
    capacity: ParticleImageCapacity = CAPACITY,
) -> tuple[TiledImageRouteSearch, DenseImageRouteSearch]:
    tiled = TiledImageRouteSearch(
        RADIUS, vectors, periodic, capacity, particle_capacity=particles, tiles=tiles
    )
    dense = DenseImageRouteSearch(
        RADIUS,
        vectors,
        periodic,
        capacity,
        particle_capacity=particles,
        maximum_dense_routes=10**8,
    )
    return tiled, dense


def _routes(
    search: AbstractImageRouteSearch,
    positions: np.ndarray,
    vectors: np.ndarray,
    periodic: np.ndarray,
    ids: np.ndarray,
    *,
    active: np.ndarray | None = None,
    receivers: ImageReceiverSubset | None = None,
) -> ImageRouteSearchResult:
    value = jnp.asarray(positions)
    mask = np.ones(positions.shape[:2], dtype=np.bool_) if active is None else active
    return search.routes(
        value,
        jnp.asarray(mask),
        jnp.asarray(ids),
        jnp.asarray(vectors, dtype=value.dtype),
        jnp.zeros((positions.shape[0], positions.shape[-1]), dtype=value.dtype),
        jnp.asarray(periodic),
        support_id="support",
        relation_schema_id="schema",
        receivers=receivers,
    )


def _assert_identical(
    left: ImageRouteSearchResult | ParticleImageHalfView,
    right: ImageRouteSearchResult | ParticleImageHalfView,
) -> None:
    """Every relation, receiver-row and evidence leaf is equal, slot by slot."""
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for first, second in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def _positions(periodic: tuple[bool, ...], count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fractional = rng.uniform(0.02, 0.98, size=(count, 3))
    axes = [axis for axis, value in enumerate(periodic) if value]
    fractional[:, axes] = rng.uniform(-0.4, 1.4, size=(count, len(axes)))
    return fractional @ TRICLINIC


@pytest.mark.parametrize("tiles", TILES.values(), ids=TILES.keys())
@pytest.mark.parametrize("periodic", PERIODIC.values(), ids=PERIODIC.keys())
@pytest.mark.parametrize("capacity", [CAPACITY, SEGMENTED], ids=["shared", "segmented"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["fp64", "fp32"])
def test_tiled_routes_equal_dense_and_lattice_oracle(
    tiles: ImageTilePolicy,
    periodic: tuple[bool, ...],
    capacity: ParticleImageCapacity,
    dtype: type[np.floating],
) -> None:
    positions = _positions(periodic, 7, 7)
    near = lattice_routes(positions, TRICLINIC, periodic, RADIUS + 1e-3, np.arange(7))
    assert_resolved_cutoff(near, positions, TRICLINIC, RADIUS, np.arange(7), dtype)
    ids = np.asarray([[31, 4, 17, 9, 22, 12, 40]])
    vectors = TRICLINIC[None]
    mask = np.asarray([periodic])
    tiled, dense = _pair(vectors, mask, 7, tiles=tiles, capacity=capacity)
    value = positions[None].astype(dtype)
    result = _routes(tiled, value, vectors, mask, ids)
    _assert_identical(result, _routes(dense, value, vectors, mask, ids))
    assert bool(result.evidence.successful[0])
    if any(periodic):
        # Nonzero self images of the short triclinic cell are retained.
        shifts = np.asarray(result.relation.image_shifts)[
            np.asarray(result.relation.valid)
        ]
        sources = np.asarray(result.relation.source_particle_ids)
        receivers = np.asarray(result.relation.receiver_particle_ids)
        valid = np.asarray(result.relation.valid)
        assert np.any(sources[valid] == receivers[valid])
        assert not np.any(
            (sources[valid] == receivers[valid]) & np.all(shifts == 0, axis=1)
        )
    if dtype is np.float64:
        expected = lattice_routes(positions, TRICLINIC, periodic, RADIUS, ids[0])
        assert_routes_match(result.relation, positions, TRICLINIC, expected)
        half = result.relation.canonical_half(capacity=4096)
        dense_relation = _routes(dense, value, vectors, mask, ids).relation
        reference = dense_relation.canonical_half(capacity=4096)
        _assert_identical(half, reference)
        assert bool(np.all(half.successful))
        reversal = result.relation.reversal_map()
        valid = np.asarray(result.relation.valid)
        assert bool(np.all(np.asarray(reversal.paired)[valid]))


def test_heterogeneous_cases_subsets_and_inactive_particles_equal_dense() -> None:
    second = 1.3 * TRICLINIC
    vectors = np.stack((TRICLINIC, second))
    periodic = np.asarray([[True, True, True], [True, False, True]])
    rng = np.random.default_rng(5)
    case0 = _positions((True,) * 3, 6, 3)
    case1 = rng.uniform(0.05, 0.95, size=(6, 3)) @ second
    positions = np.stack((case0, case1))
    ids = np.stack(
        (np.asarray([5, 9, 2, 14, 7, 3]), np.asarray([41, 40, 43, 42, 45, 44]))
    )
    active = np.ones((2, 6), dtype=np.bool_)
    active[1, 5] = False
    subset = ImageReceiverSubset(
        np.asarray([[14, 5, 0], [40, 43, 42]]),
        np.asarray([[True, True, False], [True, True, True]]),
    )
    for capacity in (CAPACITY, SEGMENTED):
        for tiles in TILES.values():
            tiled, dense = _pair(vectors, periodic, 6, tiles=tiles, capacity=capacity)
            for receivers in (None, subset):
                result = _routes(
                    tiled,
                    positions,
                    vectors,
                    periodic,
                    ids,
                    active=active,
                    receivers=receivers,
                )
                _assert_identical(
                    result,
                    _routes(
                        dense,
                        positions,
                        vectors,
                        periodic,
                        ids,
                        active=active,
                        receivers=receivers,
                    ),
                )
                assert bool(np.all(result.evidence.successful))
    flat = positions.reshape((-1, 3))
    tiled, _ = _pair(vectors, periodic, 6, tiles=TILES["scalar"], capacity=SEGMENTED)
    result = _routes(tiled, positions, vectors, periodic, ids, active=active)
    expected = lattice_routes(case0, TRICLINIC, (True,) * 3, RADIUS, ids[0]) | (
        lattice_routes(
            case1, second, (True, False, True), RADIUS, ids[1], case=1, active=active[1]
        )
    )
    assert_routes_match(result.relation, flat, vectors, expected)


def _failure_capacities() -> dict[str, ParticleImageCapacity]:
    return {
        "shared-edges": ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=5,
            maximum_degree=1024,
            maximum_images=4096,
        ),
        "case-edges": ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=64,
            maximum_degree=1024,
            maximum_images=4096,
            maximum_case_edges=5,
        ),
        "degree": ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=8192,
            maximum_degree=2,
            maximum_images=4096,
            maximum_case_edges=4096,
        ),
    }


@pytest.mark.parametrize("name", list(_failure_capacities()))
@pytest.mark.parametrize("tiles", TILES.values(), ids=TILES.keys())
def test_capacity_failures_count_every_route_and_match_dense(
    name: str, tiles: ImageTilePolicy
) -> None:
    capacity = _failure_capacities()[name]
    vectors = np.stack((TRICLINIC, TRICLINIC))
    periodic = np.ones((2, 3), dtype=np.bool_)
    positions = np.stack((_positions((True,) * 3, 6, 11), _positions((True,) * 3, 6, 12)))
    ids = np.stack((np.arange(6) + 3, np.arange(6) + 30))
    tiled, dense = _pair(vectors, periodic, 6, tiles=tiles, capacity=capacity)
    result = _routes(tiled, positions, vectors, periodic, ids)
    _assert_identical(result, _routes(dense, positions, vectors, periodic, ids))
    complete = _routes(_pair(vectors, periodic, 6)[0], positions, vectors, periodic, ids)
    evidence = result.evidence
    # Required counts are independent of the stored capacity.
    np.testing.assert_array_equal(
        evidence.required_edges, complete.evidence.required_edges
    )
    np.testing.assert_array_equal(
        evidence.required_degree, complete.evidence.required_degree
    )
    assert bool(np.all(evidence.capacity_failure))
    assert not bool(np.any(evidence.scientific_failure))
    np.testing.assert_array_equal(evidence.successful, [False, False])
    assert not bool(np.any(evidence.tile_overflow))
    if name != "degree":
        assert bool(np.all(evidence.edge_overflow))
        assert bool(np.all(evidence.stored_edges < evidence.required_edges))
    grown = ParticleImageCapacity(
        maximum_particles_per_cell=32,
        maximum_edges=16384,
        maximum_degree=2048,
        maximum_images=8192,
        maximum_case_edges=None if capacity.maximum_case_edges is None else 8192,
    )
    selected = ParticleImageCapacityLadder((capacity, grown)).select(evidence, capacity)
    assert selected.capacity_id == grown.capacity_id


THREE_CASES = ParticleImageCapacity(
    maximum_particles_per_cell=16,
    maximum_edges=3 * 2048,
    maximum_degree=1024,
    maximum_images=4096,
    maximum_case_edges=2048,
)


def test_scientific_and_receiver_failures_match_dense() -> None:
    vectors = np.stack((TRICLINIC, TRICLINIC, TRICLINIC))
    periodic = np.asarray([[True, True, True], [False, True, False], [True, True, True]])
    positions = np.stack(
        (
            _positions((True,) * 3, 6, 21),
            _positions((False, True, False), 6, 22),
            _positions((True,) * 3, 6, 23),
        )
    )
    # Case 1 leaves its nonperiodic domain; case 2 has a non-finite coordinate.
    positions[1, 0] = -0.5 * TRICLINIC[0] + 0.5 * TRICLINIC[2]
    positions[2, 3, 1] = np.nan
    ids = np.stack((np.arange(6) + 1, np.arange(6) + 11, np.arange(6) + 21))
    subset = ImageReceiverSubset(np.asarray([[1, 1], [999, 12], [21, 22]]))
    for tiles in TILES.values():
        tiled, dense = _pair(vectors, periodic, 6, tiles=tiles, capacity=THREE_CASES)
        for receivers in (None, subset):
            result = _routes(
                tiled, positions, vectors, periodic, ids, receivers=receivers
            )
            _assert_identical(
                result,
                _routes(dense, positions, vectors, periodic, ids, receivers=receivers),
            )
            evidence = result.evidence
            assert bool(evidence.domain_violation[1])
            assert bool(evidence.nonfinite[2])
            np.testing.assert_array_equal(
                evidence.successful, [receivers is None, False, False]
            )
            if receivers is not None:
                np.testing.assert_array_equal(
                    evidence.receiver_violation, [True, True, False]
                )
        # A compressed current cell outgrows the prepared stencil.
        compressed = 0.6 * vectors
        clean = np.stack(
            [_positions((True,) * 3, 6, 30 + case) * 0.6 for case in range(3)]
        )
        all_periodic = np.ones((3, 3), dtype=np.bool_)
        tiled, dense = _pair(vectors, all_periodic, 6, tiles=tiles, capacity=THREE_CASES)
        shrunk = _routes(tiled, clean, compressed, all_periodic, ids)
        _assert_identical(shrunk, _routes(dense, clean, compressed, all_periodic, ids))
        assert bool(np.all(shrunk.evidence.stencil_overflow))
        assert bool(np.all(shrunk.evidence.scientific_failure))


def _jaxpr_summary(jaxpr: jax_core.Jaxpr) -> tuple[int, set[str]]:
    """Largest intermediate element count and every primitive, sub-jaxprs included."""
    largest = 0
    names: set[str] = set()
    pending = [jaxpr]
    while pending:
        current = pending.pop()
        for equation in current.eqns:
            names.add(equation.primitive.name)
            for variable in equation.outvars:
                shape = getattr(variable.aval, "shape", ())
                largest = max(largest, int(np.prod(shape, dtype=np.int64)))
            for value in equation.params.values():
                for item in value if isinstance(value, (tuple, list)) else (value,):
                    if isinstance(item, jax_core.ClosedJaxpr):
                        pending.append(item.jaxpr)
                    elif isinstance(item, jax_core.Jaxpr):
                        pending.append(item)
    return largest, names


def test_compiled_tile_traversal_is_bounded_and_matches_eager() -> None:
    cell = 3.0 * TRICLINIC
    particles = 32
    rng = np.random.default_rng(17)
    positions = (rng.uniform(0.0, 1.0, size=(particles, 3)) @ cell)[None]
    ids = rng.permutation(particles)[None] + 100
    vectors = cell[None]
    periodic = np.ones((1, 3), dtype=np.bool_)
    capacity = ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=4096,
        maximum_degree=512,
        maximum_images=4096,
        maximum_case_edges=4096,
    )
    tiles = ImageTilePolicy(
        receiver_width=4, source_width=8, maximum_tile_candidates=4096
    )
    tiled, dense = _pair(vectors, periodic, particles, tiles=tiles, capacity=capacity)
    images = int(tiled.stencil.shape[0])
    all_pairs = particles * particles * images
    assert (
        tiled.candidate_slot_count == 4 * 8 * images < capacity.maximum_edges < all_pairs
    )
    schedule = tiled.tile_schedule()
    assert schedule.tile_steps == 2 * (particles // 4) * (particles // 8)
    assert schedule.workspace_candidates == tiled.candidate_slot_count
    assert schedule.traversed_candidates == 2 * all_pairs

    def search(
        value: jax.Array, current: AbstractImageRouteSearch
    ) -> ImageRouteSearchResult:
        return current.routes(
            value,
            jnp.ones((1, particles), dtype=jnp.bool_),
            jnp.asarray(ids),
            jnp.asarray(vectors),
            jnp.zeros((1, 3)),
            jnp.asarray(periodic),
            support_id="support",
            relation_schema_id="schema",
        )

    traced = jax.make_jaxpr(lambda value: search(value, tiled))(jnp.asarray(positions))
    largest, primitives = _jaxpr_summary(traced.jaxpr)
    # The static tile trip count lowers to a scan (or while) loop over tiles.
    assert primitives & {"scan", "while"}
    assert largest < all_pairs
    dense_traced = jax.make_jaxpr(lambda value: search(value, dense))(
        jnp.asarray(positions)
    )
    assert _jaxpr_summary(dense_traced.jaxpr)[0] >= all_pairs
    compiled = jax.jit(lambda value: search(value, tiled))(jnp.asarray(positions))
    eager = _routes(tiled, positions, vectors, periodic, ids)
    _assert_identical(compiled, eager)
    _assert_identical(eager, _routes(dense, positions, vectors, periodic, ids))
    assert bool(eager.evidence.successful[0])
    assert int(eager.evidence.required_edges[0]) > 0


def test_pair_energy_derivatives_flow_through_displacements_like_dense() -> None:
    positions = _positions((True, True, False), 6, 41)[None]
    vectors = TRICLINIC[None]
    periodic = np.asarray([[True, True, False]])
    ids = np.arange(6)[None] + 7
    tiled, dense = _pair(vectors, periodic, 6, tiles=TILES["scalar"], capacity=SEGMENTED)

    def energy(
        value: jax.Array, cell: jax.Array, search: AbstractImageRouteSearch
    ) -> jax.Array:
        result = search.routes(
            value,
            jnp.ones((1, 6), dtype=jnp.bool_),
            jnp.asarray(ids),
            cell,
            jnp.zeros((1, 3), dtype=value.dtype),
            jnp.asarray(periodic),
            support_id="support",
            relation_schema_id="schema",
        )
        displacement = result.relation.displacement(value.reshape((-1, 3)), cell)
        squared = jnp.sum(displacement * displacement, axis=-1)
        weight = jnp.where(result.relation.valid, 1.0, 0.0)
        return jnp.sum(weight * (RADIUS * RADIUS - squared) ** 2)

    value, cell = jnp.asarray(positions), jnp.asarray(vectors)
    tiled_force, tiled_virial = jax.grad(energy, argnums=(0, 1))(value, cell, tiled)
    dense_force, dense_virial = jax.grad(energy, argnums=(0, 1))(value, cell, dense)
    np.testing.assert_allclose(tiled_force, dense_force, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tiled_virial, dense_virial, rtol=1e-12, atol=1e-12)
    assert float(jnp.max(jnp.abs(tiled_force))) > 0.0
    step = 1e-6
    direction = np.random.default_rng(3).normal(size=positions.shape)
    forward = energy(value + step * direction, cell, tiled)
    backward = energy(value - step * direction, cell, tiled)
    np.testing.assert_allclose(
        (forward - backward) / (2.0 * step),
        float(jnp.sum(tiled_force * direction)),
        rtol=1e-6,
    )


def test_tile_policy_and_preparation_refusals() -> None:
    assert TILES["scalar"].scalar and not TILES["rectangular"].scalar
    with pytest.raises(ValueError, match="receiver_width"):
        ImageTilePolicy(receiver_width=0, source_width=1, maximum_tile_candidates=1)
    with pytest.raises(ValueError, match="maximum_tile_candidates"):
        ImageTilePolicy(receiver_width=1, source_width=1, maximum_tile_candidates=-1)
    vectors = TRICLINIC[None]
    periodic = np.ones((1, 3), dtype=np.bool_)
    with pytest.raises(ValueError, match="maximum_tile_candidates"):
        TiledImageRouteSearch(
            RADIUS,
            vectors,
            periodic,
            CAPACITY,
            particle_capacity=6,
            tiles=ImageTilePolicy(
                receiver_width=4, source_width=4, maximum_tile_candidates=10
            ),
        )
    with pytest.raises(ValueError, match="maximum_images"):
        TiledImageRouteSearch(
            RADIUS,
            vectors,
            periodic,
            ParticleImageCapacity(
                maximum_particles_per_cell=16,
                maximum_edges=64,
                maximum_degree=64,
                maximum_images=8,
            ),
            particle_capacity=6,
            tiles=TILES["scalar"],
        )
    with pytest.raises(TypeError, match="ImageTilePolicy"):
        TiledImageRouteSearch(
            RADIUS,
            vectors,
            periodic,
            CAPACITY,
            particle_capacity=6,
            tiles=CAPACITY,  # ty: ignore[invalid-argument-type]
        )
    first = TiledImageRouteSearch(
        RADIUS, vectors, periodic, CAPACITY, particle_capacity=6, tiles=TILES["scalar"]
    )
    second = TiledImageRouteSearch(
        RADIUS, vectors, periodic, CAPACITY, particle_capacity=6, tiles=TILES["ragged"]
    )
    assert first.search_id != second.search_id
    assert first.candidate_slot_count == int(first.stencil.shape[0])


LIMIT = 2**31 - 1


def _int32(values: list[list[int]] | list[int]) -> jax.Array:
    return jnp.asarray(np.asarray(values, dtype=np.int32))


def test_saturating_counts_report_exact_int32_overflow() -> None:
    value, over = _saturating_scan(
        _int32([LIMIT - 5, 3, 4]), jnp.zeros((3,), dtype=jnp.bool_), axis=0
    )
    np.testing.assert_array_equal(value, [LIMIT - 5, LIMIT - 2, LIMIT])
    np.testing.assert_array_equal(over, [False, False, True])
    # Reaching exactly 2**31 - 1 is representable, not an overflow.
    value, over = _saturating_scan(
        _int32([LIMIT - 5, 5]), jnp.zeros((2,), dtype=jnp.bool_), axis=0
    )
    np.testing.assert_array_equal(value, [LIMIT - 5, LIMIT])
    np.testing.assert_array_equal(over, [False, False])
    value, over = _saturating_scan(
        _int32([LIMIT - 1, 2, 3, 4]),
        jnp.asarray([True, False, True, False]),
        axis=0,
    )
    np.testing.assert_array_equal(value, [LIMIT - 1, LIMIT, 3, 7])
    np.testing.assert_array_equal(over, [False, True, False, False])
    # Any associative regrouping of the scan agrees with exact integer sums.
    rng = np.random.default_rng(9)
    values = rng.integers(0, LIMIT // 6, size=37, dtype=np.int64)
    starts = rng.uniform(size=37) < 0.15
    starts[0] = True
    exact, flags, running = [], [], 0
    for count, start in zip(values, starts, strict=True):
        running = int(count) if start else running + int(count)
        exact.append(min(running, LIMIT))
        flags.append(running > LIMIT)
    value, over = _saturating_scan(
        jnp.asarray(values.astype(np.int32)), jnp.asarray(starts), axis=0
    )
    np.testing.assert_array_equal(value, exact)
    np.testing.assert_array_equal(over, flags)


def test_row_prefix_and_degree_saturate_with_case_overflow_bits() -> None:
    first, total, overflow = _row_prefix(_int32([[LIMIT - 10, 6, 7], [1, 2, 3]]))
    np.testing.assert_array_equal(first, [[0, LIMIT - 10, LIMIT - 4], [0, 1, 3]])
    np.testing.assert_array_equal(total, [LIMIT, 6])
    np.testing.assert_array_equal(overflow, [True, False])
    # Repeated rows of one slot sum (as the dense reference does) and saturate.
    degree = _receiver_degree(
        _int32([[2, 0, 2, 1], [0, 0, 1, 1]]),
        _int32([[LIMIT - 3, 5, 9, 4], [3, 4, 0, 0]]),
        3,
    )
    np.testing.assert_array_equal(degree, [[5, 4, LIMIT], [7, 0, 0]])


def test_route_targets_keep_exact_slots_under_saturation() -> None:
    candidate = jnp.asarray([[[True, False, True, True, True]]])
    target, advanced = _route_targets(
        candidate, _int32([[0]]), _int32([[LIMIT - 3]]), buffer=LIMIT, segment=None
    )
    # Exact slots LIMIT - 3, LIMIT - 2, LIMIT - 1 fit; exact slot 2**31 does not.
    np.testing.assert_array_equal(target, [LIMIT - 3, LIMIT, LIMIT - 2, LIMIT - 1, LIMIT])
    np.testing.assert_array_equal(advanced, [[4]])
    target, advanced = _route_targets(
        jnp.ones((3, 1, 2), dtype=jnp.bool_),
        _int32([[2], [0], [0]]),
        _int32([[0], [1], [LIMIT]]),
        buffer=9,
        segment=3,
    )
    # Case 0 overflows its segment after slot 2; case 2's saturated base drops.
    np.testing.assert_array_equal(target, [2, 9, 4, 5, 9, 9])
    np.testing.assert_array_equal(advanced, [[4], [2], [2]])
