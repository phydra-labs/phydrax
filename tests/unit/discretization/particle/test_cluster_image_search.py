#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.particle import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
)
from phydrax.discretization.particle._cluster_image_search import (
    ClusterTileImageRouteSearch,
)
from phydrax.discretization.particle._image_neighborhood import (
    AbstractImageRouteSearch,
    CellListImageRouteSearch,
    DenseImageRouteSearch,
    ImageReceiverSubset,
    ImageRouteSearchResult,
)
from tests._support.image_lattice import (
    assert_resolved_cutoff,
    assert_routes_match,
    lattice_routes,
    relation_routes,
)


TRICLINIC = np.asarray([[2.0, 0.0, 0.0], [0.7, 2.1, 0.0], [0.3, -0.4, 2.3]])
RADIUS = 2.6
CAPACITY = ParticleImageCapacity(
    maximum_particles_per_cell=16,
    maximum_edges=8192,
    maximum_degree=1024,
    maximum_images=4096,
)
TILES = ParticleImageCapacity(
    maximum_particles_per_cell=16,
    maximum_edges=2 * 8192,
    maximum_degree=4096,
    maximum_images=4096,
    maximum_case_edges=8192,
)


def _diameter(vectors: np.ndarray) -> float:
    """Longest parallelepiped diagonal: bounds every in-cell cluster radius."""
    signs = np.asarray([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=np.float64)
    return float(np.max(np.linalg.norm(signs @ vectors, axis=-1)))


def _cluster(
    vectors: np.ndarray,
    particles: int,
    *,
    width: int = 2,
    cluster_radius: float | None = None,
    tiles: ParticleImageCapacity = TILES,
    capacity: ParticleImageCapacity = CAPACITY,
) -> ClusterTileImageRouteSearch:
    return ClusterTileImageRouteSearch(
        RADIUS,
        vectors,
        capacity,
        particle_capacity=particles,
        cluster_width=width,
        maximum_cluster_radius=(
            _diameter(vectors[0]) if cluster_radius is None else cluster_radius
        ),
        tile_capacity=tiles,
        maximum_candidate_slots=10**8,
    )


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
        jnp.zeros((positions.shape[0], 3), dtype=value.dtype),
        jnp.asarray(periodic),
        support_id="support",
        relation_schema_id="schema",
        receivers=receivers,
    )


def _positions(periodic: tuple[bool, ...], count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fractional = rng.uniform(0.02, 0.98, size=(count, 3))
    axes = [axis for axis, value in enumerate(periodic) if value]
    fractional[:, axes] = rng.uniform(-0.4, 1.4, size=(count, len(axes)))
    return fractional @ TRICLINIC


@pytest.mark.parametrize(
    "periodic",
    [(True, True, True), (True, True, False), (False, True, False)],
    ids=["full", "slab", "wire"],
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["fp64", "fp32"])
def test_cluster_routes_match_dense_cell_list_and_lattice_oracle(
    periodic: tuple[bool, ...], dtype: type[np.floating]
) -> None:
    positions = _positions(periodic, 6, 7)
    near = lattice_routes(positions, TRICLINIC, periodic, RADIUS + 1e-3, np.arange(6))
    assert_resolved_cutoff(near, positions, TRICLINIC, RADIUS, np.arange(6), dtype)
    ids = np.asarray([[31, 4, 17, 9, 22, 12]])
    vectors = TRICLINIC[None]
    mask = np.asarray([periodic])
    searches = {
        "cluster": _cluster(vectors, 6),
        "cell": CellListImageRouteSearch(
            RADIUS, vectors, CAPACITY, particle_capacity=6, maximum_candidate_slots=10**7
        ),
        "dense": DenseImageRouteSearch(
            RADIUS,
            vectors,
            mask,
            CAPACITY,
            particle_capacity=6,
            maximum_dense_routes=10**7,
        ),
    }
    value = positions[None].astype(dtype)
    results = {
        name: _routes(search, value, vectors, mask, ids)
        for name, search in searches.items()
    }
    for name, result in results.items():
        assert bool(result.evidence.successful[0]), name
    multisets = {
        name: relation_routes(result.relation, positions, TRICLINIC)[0]
        for name, result in results.items()
    }
    assert all(count == 1 for count in multisets["cluster"].values())
    assert multisets["cluster"] == multisets["cell"] == multisets["dense"]
    assert int(results["cluster"].evidence.required_tile_pairs[0]) > 0
    if dtype is np.float64:
        expected = lattice_routes(positions, TRICLINIC, periodic, RADIUS, ids[0])
        assert_routes_match(results["cluster"].relation, positions, TRICLINIC, expected)


def test_skewed_occupancy_and_empty_cases_are_exact() -> None:
    corner = np.asarray(
        [[0.05, 0.05, 0.05], [0.1, 0.07, 0.04], [0.08, 0.12, 0.09], [0.11, 0.1, 0.13]]
    )
    case0 = np.concatenate((corner, [[0.6, 0.55, 0.5], [0.9, 0.2, 0.7]])) @ TRICLINIC
    case1 = np.random.default_rng(3).uniform(0.0, 1.0, size=(6, 3)) @ TRICLINIC
    positions = np.stack((case0, case1))
    vectors = np.stack((TRICLINIC, TRICLINIC))
    periodic = np.ones((2, 3), dtype=np.bool_)
    ids = np.stack((np.arange(6) + 1, np.arange(6) + 40))
    active = np.stack((np.ones(6, dtype=np.bool_), np.zeros(6, dtype=np.bool_)))
    result = _routes(
        _cluster(vectors, 6), positions, vectors, periodic, ids, active=active
    )
    assert bool(np.all(result.evidence.successful))
    np.testing.assert_array_equal(result.evidence.active_particles, [6, 0])
    np.testing.assert_array_equal(result.evidence.required_edges[1], 0)
    expected = lattice_routes(case0, TRICLINIC, (True,) * 3, RADIUS, ids[0])
    assert_routes_match(result.relation, positions.reshape((-1, 3)), vectors, expected)
    crowded = ParticleImageCapacity(
        maximum_particles_per_cell=2,
        maximum_edges=8192,
        maximum_degree=1024,
        maximum_images=4096,
    )
    cell = CellListImageRouteSearch(
        RADIUS, vectors, crowded, particle_capacity=6, maximum_candidate_slots=10**7
    )
    overflowed = _routes(cell, positions, vectors, periodic, ids, active=active)
    assert bool(overflowed.evidence.cell_overflow[0])
    clustered = _routes(
        _cluster(vectors, 6, capacity=crowded),
        positions,
        vectors,
        periodic,
        ids,
        active=active,
    )
    assert bool(np.all(clustered.evidence.successful))


def test_tile_budget_overflow_is_separate_from_edges_and_not_ladder_repairable() -> None:
    positions = _positions((True,) * 3, 6, 11)[None]
    vectors = TRICLINIC[None]
    periodic = np.ones((1, 3), dtype=np.bool_)
    ids = np.arange(6)[None] + 3
    tiny = ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=4,
        maximum_degree=4096,
        maximum_images=4096,
        maximum_case_edges=4,
    )
    result = _routes(_cluster(vectors, 6, tiles=tiny), positions, vectors, periodic, ids)
    evidence = result.evidence
    assert bool(evidence.tile_overflow[0])
    assert bool(evidence.capacity_failure[0])
    assert not bool(evidence.scientific_failure[0])
    assert not bool(evidence.edge_overflow[0])
    assert int(evidence.required_tile_pairs[0]) > 4
    bigger = ParticleImageCapacity(
        maximum_particles_per_cell=32,
        maximum_edges=16384,
        maximum_degree=2048,
        maximum_images=8192,
    )
    with pytest.raises(ValueError, match="tile"):
        ParticleImageCapacityLadder((CAPACITY, bigger)).select(evidence, CAPACITY)


def test_cluster_extent_beyond_prepared_envelope_is_scientific() -> None:
    positions = _positions((True,) * 3, 6, 13)[None]
    vectors = TRICLINIC[None]
    result = _routes(
        _cluster(vectors, 6, width=3, cluster_radius=0.01),
        positions,
        vectors,
        np.ones((1, 3), dtype=np.bool_),
        np.arange(6)[None],
    )
    assert bool(result.evidence.cluster_extent_overflow[0])
    assert bool(result.evidence.scientific_failure[0])
    assert not bool(result.evidence.successful[0])


def test_cluster_preparation_refuses_unbounded_or_unsupported_envelopes() -> None:
    vectors = TRICLINIC[None]
    shared_tiles = ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=8192,
        maximum_degree=4096,
        maximum_images=4096,
    )
    with pytest.raises(ValueError, match="maximum_case_edges"):
        _cluster(vectors, 6, tiles=shared_tiles)
    with pytest.raises(ValueError, match="maximum_candidate_slots"):
        ClusterTileImageRouteSearch(
            RADIUS,
            vectors,
            CAPACITY,
            particle_capacity=6,
            cluster_width=2,
            maximum_cluster_radius=1.0,
            tile_capacity=TILES,
            maximum_candidate_slots=1000,
        )
    with pytest.raises(ValueError, match="rank"):
        _cluster(np.eye(4)[None] * 3.0, 6, cluster_radius=1.0)


def test_cluster_receiver_subset_and_compiled_search_match_cell_list() -> None:
    positions = _positions((True, True, False), 6, 17)[None]
    vectors = TRICLINIC[None]
    periodic = np.asarray([[True, True, False]])
    ids = np.asarray([[5, 9, 2, 14, 7, 3]])
    subset = ImageReceiverSubset(np.asarray([[14, 5]]))
    cluster = _cluster(vectors, 6)
    cell = CellListImageRouteSearch(
        RADIUS, vectors, CAPACITY, particle_capacity=6, maximum_candidate_slots=10**7
    )
    eager = _routes(cluster, positions, vectors, periodic, ids, receivers=subset)
    reference = _routes(cell, positions, vectors, periodic, ids, receivers=subset)
    compiled = jax.jit(
        lambda x: cluster.routes(
            x,
            jnp.ones((1, 6), dtype=jnp.bool_),
            jnp.asarray(ids),
            jnp.asarray(vectors),
            jnp.zeros((1, 3)),
            jnp.asarray(periodic),
            support_id="support",
            relation_schema_id="schema",
            receivers=subset,
        )
    )(jnp.asarray(positions))
    flat = positions.reshape((-1, 3))
    expected = relation_routes(reference.relation, flat, vectors)[0]
    assert relation_routes(eager.relation, flat, vectors)[0] == expected
    assert relation_routes(compiled.relation, flat, vectors)[0] == expected
    assert {key[2] for key in expected} == {14, 5}
    assert bool(eager.evidence.successful[0])
