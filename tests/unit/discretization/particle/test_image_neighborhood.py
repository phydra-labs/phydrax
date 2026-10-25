#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import lattice_measure, PeriodicCell
from phydrax.discretization.particle import (
    CellListParticleImageNeighborhoodPlan,
    DenseParticleImageNeighborhoodPlan,
    ImageVerletParticleNeighborhoodPlan,
    ParticleDiscretization,
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    ParticleImageNeighborhoodState,
    ParticleImageRelation,
    ParticleSetPlan,
    PreparedImageVerletParticleNeighborhood,
)
from phydrax.discretization.particle._image_neighborhood import (
    _host_reach,
    CellListImageRouteSearch,
    DenseImageRouteSearch,
    ImageNeighborSelection,
    ImageNeighborSelectionPolicy,
    ImageReceiverSubset,
    ImageRouteSearchResult,
    ImageSelectionCapacity,
    PreparedImageNeighborSelection,
)
from phydrax.sparse import EdgeRelation
from tests._support.image_lattice import (
    assert_resolved_cutoff,
    assert_routes_match,
    lattice_routes,
    relation_routes,
)


Route = tuple[int, int, tuple[int, ...]]


TRICLINIC = np.asarray([[2.0, 0.0, 0.0], [0.7, 2.1, 0.0], [0.3, -0.4, 2.3]])
CAPACITY = ParticleImageCapacity(
    maximum_particles_per_cell=8,
    maximum_edges=4096,
    maximum_degree=1024,
    maximum_images=4096,
)


def _particles(count: int, ids: np.ndarray | None = None) -> ParticleDiscretization:
    stable = np.arange(count) * 7 + 3 if ids is None else ids
    return ParticleSetPlan(stable, np.ones(count), ambient_dimension=3).prepare()


def _oracle(
    positions: np.ndarray,
    vectors: np.ndarray,
    periodic: tuple[bool, ...],
    radius: float,
    ids: np.ndarray,
) -> dict[tuple[int, int, tuple[int, ...]], np.ndarray]:
    """Independent brute-force lattice sum over a generous integer box."""
    extent = 8
    choices = [range(-extent, extent + 1) if axis else (0,) for axis in periodic]
    routes = {}
    for shift in product(*choices):
        translation = np.asarray(shift, dtype=np.float64) @ vectors
        for source in range(positions.shape[0]):
            for receiver in range(positions.shape[0]):
                if source == receiver and not any(shift):
                    continue
                displacement = positions[receiver] - positions[source] + translation
                if np.linalg.norm(displacement) < radius:
                    routes[(int(ids[source]), int(ids[receiver]), shift)] = displacement
    return routes


def _routes(
    state: ParticleImageNeighborhoodState, positions: np.ndarray, vectors: np.ndarray
) -> dict[Route, np.ndarray]:
    relation = state.relation
    valid = np.asarray(relation.valid)
    displacement = np.asarray(
        relation.displacement(jnp.asarray(positions), jnp.asarray(vectors))
    )
    return {
        (int(source), int(receiver), tuple(int(value) for value in shift)): row
        for source, receiver, shift, row in zip(
            np.asarray(relation.source_particle_ids)[valid],
            np.asarray(relation.receiver_particle_ids)[valid],
            np.asarray(relation.image_shifts)[valid],
            displacement[valid],
            strict=True,
        )
    }


def _within(routes: dict[Route, np.ndarray], radius: float) -> set[Route]:
    return {key for key, row in routes.items() if np.linalg.norm(row) < radius}


def _assert_matches_oracle(
    state: ParticleImageNeighborhoodState,
    positions: np.ndarray,
    vectors: np.ndarray,
    periodic: tuple[bool, ...],
    radius: float,
    ids: np.ndarray,
) -> None:
    expected = _oracle(positions, vectors, periodic, radius, ids)
    actual = _routes(state, positions, vectors)
    assert bool(state.successful)
    assert set(actual) == set(expected)
    for key, row in expected.items():
        np.testing.assert_allclose(actual[key], row, atol=1e-12)


@pytest.mark.parametrize(
    "periodic",
    [(True, True, True), (True, True, False), (False, True, False)],
    ids=["full", "slab", "wire"],
)
@pytest.mark.parametrize("backend", ["cell_list", "dense"])
def test_image_routes_match_brute_force_lattice_beyond_unique_image(
    periodic: tuple[bool, ...], backend: str
) -> None:
    rng = np.random.default_rng(11)
    count = 5
    fractional = rng.uniform(0.02, 0.98, size=(count, 3))
    fractional[:, list(periodic)] = rng.uniform(-0.4, 1.4, size=(count, sum(periodic)))
    positions = fractional @ TRICLINIC
    radius = 3.1
    cell = PeriodicCell(TRICLINIC, periodic_axes=periodic)
    assert radius > cell.unique_image_radius
    plan = (
        CellListParticleImageNeighborhoodPlan(radius, cell, CAPACITY)
        if backend == "cell_list"
        else DenseParticleImageNeighborhoodPlan(
            radius, cell, CAPACITY, maximum_dense_routes=10**6
        )
    )
    ids = np.asarray([31, 4, 17, 9, 22])
    state = plan.prepare(_particles(count, ids)).build(jnp.asarray(positions))
    _assert_matches_oracle(state, positions, TRICLINIC, periodic, radius, ids)
    np.testing.assert_array_equal(
        np.asarray(state.relation.image_shifts)[:, [not axis for axis in periodic]], 0
    )


def test_one_atom_cell_has_every_nonzero_self_image_and_reversal_negates_shift() -> None:
    cell = PeriodicCell(2.0 * np.eye(3))
    state = (
        CellListParticleImageNeighborhoodPlan(2.5, cell, CAPACITY)
        .prepare(_particles(1))
        .build(jnp.full((1, 3), 0.3))
    )
    routes = _routes(state, np.full((1, 3), 0.3), 2.0 * np.eye(3))
    expected_shifts = {
        tuple(int(value) for value in row)
        for row in np.concatenate((np.eye(3, dtype=int), -np.eye(3, dtype=int)))
    }
    assert {shift for (_, _, shift) in routes} == expected_shifts
    for (_, _, shift), row in routes.items():
        np.testing.assert_allclose(row, 2.0 * np.asarray(shift))
    reversed_relation = state.relation.reversed()
    valid = np.asarray(state.relation.valid)
    np.testing.assert_array_equal(
        np.asarray(reversed_relation.image_shifts)[valid],
        -np.asarray(state.relation.image_shifts)[valid],
    )
    np.testing.assert_allclose(
        np.asarray(
            reversed_relation.displacement(jnp.full((1, 3), 0.3), 2.0 * jnp.eye(3))
        )[valid],
        -np.asarray(state.relation.displacement(jnp.full((1, 3), 0.3), 2.0 * jnp.eye(3)))[
            valid
        ],
    )


def test_route_identity_is_stable_under_storage_reorder_and_inactive_padding() -> None:
    cell = PeriodicCell(TRICLINIC)
    rng = np.random.default_rng(3)
    positions = rng.uniform(0.0, 1.0, size=(4, 3)) @ TRICLINIC
    ids = np.asarray([5, 9, 2, 14])
    plan = CellListParticleImageNeighborhoodPlan(2.6, cell, CAPACITY)
    first = plan.prepare(_particles(4, ids)).build(jnp.asarray(positions))
    order = np.asarray([2, 0, 3, 1], dtype=np.int64)
    reordered_positions = np.take(positions, order, axis=0)
    second = plan.prepare(_particles(4, np.take(ids, order))).build(
        jnp.asarray(reordered_positions)
    )
    assert set(_routes(first, positions, TRICLINIC)) == set(
        _routes(second, reordered_positions, TRICLINIC)
    )
    padded_positions = np.concatenate((positions, np.full((1, 3), 1.0e3)))
    padded = ParticleSetPlan(
        np.concatenate((ids, [99])),
        np.ones(5),
        ambient_dimension=3,
        active_mask=np.asarray([True, True, True, True, False]),
    ).prepare()
    padded_state = plan.prepare(padded).build(jnp.asarray(padded_positions))
    assert bool(padded_state.successful)
    assert set(_routes(padded_state, padded_positions, TRICLINIC)) == set(
        _routes(first, positions, TRICLINIC)
    )


def test_capacity_failures_are_separate_and_ladder_selects_covering_entry() -> None:
    cell = PeriodicCell(TRICLINIC)
    positions = np.random.default_rng(5).uniform(0.0, 1.0, size=(4, 3)) @ TRICLINIC
    small = ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=6,
        maximum_degree=1024,
        maximum_images=4096,
    )
    plan = CellListParticleImageNeighborhoodPlan(3.0, cell, small)
    state = plan.prepare(_particles(4)).build(jnp.asarray(positions))
    evidence = state.evidence
    assert bool(state.capacity_failure)
    assert not bool(state.scientific_failure)
    assert bool(evidence.edge_overflow[0])
    assert not bool(evidence.degree_overflow[0])
    required = int(evidence.required_edges[0])
    assert required > 6
    middle = ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=required - 1,
        maximum_degree=1024,
        maximum_images=4096,
    )
    ladder = ParticleImageCapacityLadder((small, middle, CAPACITY))
    assert ladder.select(evidence, small).capacity_id == CAPACITY.capacity_id
    grown = (
        plan.with_capacity(CAPACITY).prepare(_particles(4)).build(jnp.asarray(positions))
    )
    assert bool(grown.successful)
    assert int(grown.evidence.stored_edges[0]) == required
    with pytest.raises(RuntimeError, match="exhausted"):
        ParticleImageCapacityLadder((small, middle)).select(evidence, small)
    degree_limited = ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=4096,
        maximum_degree=1,
        maximum_images=4096,
    )
    degree_state = (
        plan.with_capacity(degree_limited)
        .prepare(_particles(4))
        .build(jnp.asarray(positions))
    )
    assert bool(degree_state.evidence.degree_overflow[0])
    assert not bool(degree_state.evidence.edge_overflow[0])


def test_domain_failure_is_not_capacity_retryable() -> None:
    cell = PeriodicCell(TRICLINIC, periodic_axes=(True, True, False))
    positions = np.asarray([[0.1, 0.2, 0.5], [0.4, 0.4, 1.5]]) @ TRICLINIC
    plan = CellListParticleImageNeighborhoodPlan(2.0, cell, CAPACITY)
    state = plan.prepare(_particles(2)).build(jnp.asarray(positions))
    assert bool(state.scientific_failure)
    bigger = ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=8192,
        maximum_degree=2048,
        maximum_images=8192,
    )
    with pytest.raises(ValueError, match="not capacity-only"):
        ParticleImageCapacityLadder((CAPACITY, bigger)).select(state.evidence, CAPACITY)


def _self_relation(valid: list[bool], shifts: list[list[int]]) -> ParticleImageRelation:
    count = len(valid)
    edge = EdgeRelation(
        jnp.zeros((count,), dtype=jnp.int32),
        jnp.zeros((count,), dtype=jnp.int32),
        source_size=1,
        target_size=1,
        valid=jnp.asarray(valid),
    )
    return ParticleImageRelation(
        edge,
        jnp.zeros((count,), dtype=jnp.int64),
        jnp.zeros((count,), dtype=jnp.int64),
        jnp.asarray(shifts, dtype=jnp.int64),
        jnp.zeros((count,), dtype=jnp.int32),
        case_count=1,
        particle_capacity=1,
        support_id="support",
        relation_schema_id="schema",
    )


def test_valid_zero_self_route_is_refused_on_host_and_under_jit() -> None:
    padded = _self_relation([True, False], [[1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(padded.valid, [True, False])
    with pytest.raises(ValueError, match="zero translation"):
        _self_relation([True, True], [[1, 0, 0], [0, 0, 0]])

    def construct(valid: jax.Array) -> jax.Array:
        relation = ParticleImageRelation(
            EdgeRelation(
                jnp.zeros((2,), dtype=jnp.int32),
                jnp.zeros((2,), dtype=jnp.int32),
                source_size=1,
                target_size=1,
                valid=valid,
            ),
            jnp.zeros((2,), dtype=jnp.int64),
            jnp.zeros((2,), dtype=jnp.int64),
            jnp.asarray([[1, 0, 0], [0, 0, 0]], dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
            case_count=1,
            particle_capacity=1,
            support_id="support",
            relation_schema_id="schema",
        )
        return relation.valid

    traced = jax.jit(construct)
    np.testing.assert_array_equal(traced(jnp.asarray([True, False])), [True, False])
    with pytest.raises(Exception, match="zero translation"):
        jax.block_until_ready(traced(jnp.asarray([True, True])))


def test_dense_reference_and_cell_list_refuse_unbounded_preparation() -> None:
    cell = PeriodicCell(TRICLINIC)
    with pytest.raises(ValueError, match="maximum_dense_routes"):
        DenseParticleImageNeighborhoodPlan(
            3.0, cell, CAPACITY, maximum_dense_routes=10
        ).prepare(_particles(4))
    with pytest.raises(ValueError, match="maximum_candidate_slots"):
        CellListParticleImageNeighborhoodPlan(
            3.0, cell, CAPACITY, maximum_candidate_slots=10
        ).prepare(_particles(4))


def test_image_stencil_is_complete_against_brute_force_and_bounded() -> None:
    cell = PeriodicCell(TRICLINIC)
    radius = 4.4
    stencil = cell.image_stencil(radius, maximum_image_count=10_000)
    shifts = {tuple(int(value) for value in row) for row in np.asarray(stencil.shifts)}
    box = np.asarray(tuple(product(range(-8, 9), repeat=3)), dtype=np.float64)
    rng = np.random.default_rng(13)
    endpoints = rng.uniform(0.0, 1.0, size=(200, 2, 3)) @ TRICLINIC
    separation = endpoints[:, 1] - endpoints[:, 0]
    displacement = separation[:, None, :] + (box @ TRICLINIC)[None]
    reachable = np.any(np.linalg.norm(displacement, axis=-1) < radius, axis=0)
    needed = {
        tuple(int(value) for value in box[index]) for index in np.flatnonzero(reachable)
    }
    assert needed <= shifts
    with pytest.raises(ValueError, match="maximum_image_count"):
        cell.image_stencil(radius, maximum_image_count=8)
    margin = cell.image_stencil_margin(stencil, TRICLINIC, radius, jnp.ones((3,)))
    assert bool(jnp.all(margin > 0.0))
    shrunk = 0.5 * TRICLINIC
    assert bool(
        jnp.any(cell.image_stencil_margin(stencil, shrunk, radius, jnp.ones((3,))) <= 0)
    )
    singular_values = np.linalg.svd(TRICLINIC, compute_uv=False)
    certified = PeriodicCell(
        TRICLINIC,
        maximum_condition_number=1.5 * float(singular_values[0] / singular_values[-1]),
    )
    with pytest.raises(ValueError, match="condition certificate"):
        certified.image_stencil(
            1.0,
            maximum_image_count=100,
            cell_vectors=np.asarray([[2.0, 0.0, 0.0], [1.99, 0.05, 0.0], [0, 0, 2.0]]),
        )


def test_lattice_measure_reports_volume_and_singular_status() -> None:
    volume, successful = lattice_measure(jnp.asarray(TRICLINIC))
    np.testing.assert_allclose(volume, abs(np.linalg.det(TRICLINIC)))
    assert bool(successful)
    _, singular = lattice_measure(jnp.asarray([[1.0, 0, 0], [2.0, 0, 0], [0, 0, 1.0]]))
    assert not bool(singular)
    gradient = jax.grad(lambda h: lattice_measure(h)[0])(jnp.asarray(TRICLINIC))
    np.testing.assert_allclose(
        gradient,
        np.linalg.det(TRICLINIC) * np.linalg.inv(TRICLINIC).T,
        rtol=1e-10,
        atol=1e-12,
    )


def _verlet(
    cell: PeriodicCell,
    count: int,
    interaction: float,
    skin: float,
    deformation_margin: float = 0.0,
) -> PreparedImageVerletParticleNeighborhood:
    base = CellListParticleImageNeighborhoodPlan(
        interaction + skin, cell, CAPACITY, deformation_margin=deformation_margin
    )
    return ImageVerletParticleNeighborhoodPlan(base, interaction, skin).prepare(
        _particles(count)
    )


def test_face_crossing_rewrap_reuses_routes_with_exact_shift_update() -> None:
    cell = PeriodicCell(TRICLINIC)
    ids = np.arange(4) * 7 + 3
    fractional = np.asarray(
        [[0.98, 0.5, 0.5], [0.02, 0.4, 0.6], [0.5, 0.97, 0.1], [0.3, 0.2, 0.99]]
    )
    unwrapped = fractional @ TRICLINIC
    prepared = _verlet(cell, 4, 2.4, 0.4)
    wrapped, counts = cell.wrap(jnp.asarray(unwrapped))
    state = prepared.initialize(wrapped, image_counts=counts)
    velocity = np.asarray([[0.05, 0, 0], [-0.05, 0, 0], [0, 0.04, 0], [0, 0, 0.03]])
    moved = unwrapped + velocity
    moved_wrapped, moved_counts = cell.wrap(jnp.asarray(moved))
    assert bool(jnp.any(moved_counts != counts))
    updated = jax.jit(
        lambda x, c, previous: prepared.update(x, previous, image_counts=c)
    )(moved_wrapped, moved_counts, state)
    assert not bool(updated.rebuilt)
    assert bool(updated.successful)
    expected = _oracle(np.asarray(moved_wrapped), TRICLINIC, (True,) * 3, 2.4, ids)
    cached = _routes(updated.neighborhood, np.asarray(moved_wrapped), TRICLINIC)
    assert _within(cached, 2.4) == set(expected)
    for key, row in expected.items():
        np.testing.assert_allclose(cached[key], row, atol=1e-12)
    inferred = prepared.update(moved_wrapped, state)
    assert not bool(inferred.rebuilt)
    inferred_routes = _routes(inferred.neighborhood, np.asarray(moved_wrapped), TRICLINIC)
    assert _within(inferred_routes, 2.4) == set(expected)


def test_cell_shrink_admits_image_from_outside_old_stencil_by_rebuilding() -> None:
    cell = PeriodicCell(3.0 * np.eye(3))
    positions = jnp.full((1, 3), 0.25)
    shrunk = 0.95 * np.eye(3)
    prepared = _verlet(cell, 1, 2.0, 0.4, deformation_margin=2.5)
    state = prepared.initialize(positions)
    assert not _routes(state.neighborhood, np.asarray(positions), 3.0 * np.eye(3))
    updated = prepared.update(positions, state, cell_vectors=shrunk)
    assert bool(updated.rebuilt)
    assert bool(updated.successful)
    expected = _oracle(np.asarray(positions), shrunk, (True,) * 3, 2.0, np.asarray([3]))
    assert any(max(abs(value) for value in shift) == 2 for (_, _, shift) in expected)
    cached = _routes(updated.neighborhood, np.asarray(positions), shrunk)
    assert _within(cached, 2.0) == set(expected)
    unenveloped = _verlet(cell, 1, 2.0, 0.4)
    refused = unenveloped.update(
        positions, unenveloped.initialize(positions), cell_vectors=shrunk
    )
    assert bool(refused.rebuilt)
    assert not bool(refused.successful)
    assert bool(refused.scientific_failure)
    assert bool(refused.reference.evidence.stencil_overflow[0])


def test_small_shear_reuses_cache_and_keeps_every_route_within_cutoff() -> None:
    cell = PeriodicCell(TRICLINIC)
    ids = np.arange(5) * 7 + 3
    positions = np.random.default_rng(17).uniform(0.0, 1.0, size=(5, 3)) @ TRICLINIC
    prepared = _verlet(cell, 5, 2.4, 0.6)
    state = prepared.initialize(jnp.asarray(positions))
    sheared = TRICLINIC + np.asarray(
        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.01, 0.0]]
    )
    updated = prepared.update(jnp.asarray(positions), state, cell_vectors=sheared)
    assert not bool(updated.rebuilt)
    assert float(updated.maximum_cell_deformation) > 0.0
    assert float(updated.certificate_margin) >= 0.0
    expected = _oracle(positions, sheared, (True,) * 3, 2.4, ids)
    assert _within(_routes(updated.neighborhood, positions, sheared), 2.4) == set(
        expected
    )


def _cased_relation(
    valid: list[bool] | jax.Array,
    sources: list[int],
    receivers: list[int],
    cases: list[int],
) -> ParticleImageRelation:
    count = len(valid)
    return ParticleImageRelation(
        EdgeRelation(
            jnp.asarray(sources, dtype=jnp.int32),
            jnp.asarray(receivers, dtype=jnp.int32),
            source_size=2,
            target_size=2,
            valid=jnp.asarray(valid),
        ),
        jnp.zeros((count,), dtype=jnp.int64),
        jnp.zeros((count,), dtype=jnp.int64),
        jnp.tile(jnp.asarray([[1, 0, 0]], dtype=jnp.int32), (count, 1)),
        jnp.asarray(cases, dtype=jnp.int32),
        case_count=2,
        particle_capacity=1,
        support_id="support",
        relation_schema_id="schema",
    )


@pytest.mark.parametrize(
    "sources, receivers, cases",
    [([0], [0], [1]), ([0], [0], [2]), ([0], [0], [-1]), ([0], [1], [0])],
    ids=["wrong-case", "above-range", "negative", "cross-case-endpoints"],
)
def test_valid_route_case_must_own_both_endpoints(
    sources: list[int], receivers: list[int], cases: list[int]
) -> None:
    with pytest.raises(ValueError, match="case"):
        _cased_relation([True], sources, receivers, cases)

    def construct(valid: jax.Array) -> jax.Array:
        return _cased_relation(valid, sources, receivers, cases).valid

    with pytest.raises(Exception, match="case"):
        jax.block_until_ready(jax.jit(construct)(jnp.asarray([True])))
    np.testing.assert_array_equal(jax.jit(construct)(jnp.asarray([False])), [False])


def test_case_owned_routes_translate_with_their_own_lattice() -> None:
    relation = _cased_relation([True, True], [0, 1], [0, 1], [0, 1])
    vectors = jnp.stack((2.0 * jnp.eye(3), 5.0 * jnp.eye(3)))
    np.testing.assert_allclose(
        relation.displacement(jnp.zeros((2, 3)), vectors), [[2.0, 0, 0], [5.0, 0, 0]]
    )


def test_certifies_is_the_update_reuse_predicate_for_moves_wraps_and_cell_changes() -> (
    None
):
    cell = PeriodicCell(TRICLINIC)
    prepared = _verlet(cell, 4, 2.4, 0.4)
    unwrapped = (
        np.asarray(
            [[0.98, 0.5, 0.5], [0.02, 0.4, 0.6], [0.5, 0.97, 0.1], [0.3, 0.2, 0.99]]
        )
        @ TRICLINIC
    )
    wrapped, counts = cell.wrap(jnp.asarray(unwrapped))
    state = prepared.initialize(wrapped, image_counts=counts)
    moved_wrapped, moved_counts = cell.wrap(jnp.asarray(unwrapped + 0.05))
    cases = [
        ("wrapped-within-skin", moved_wrapped, {"image_counts": moved_counts}, True),
        ("far-move", jnp.asarray(unwrapped + 0.5), {}, False),
        ("shrunk-cell", wrapped, {"cell_vectors": 0.55 * TRICLINIC}, False),
        (
            "active-change",
            wrapped,
            {"active_mask": jnp.asarray([True, True, True, False])},
            False,
        ),
        ("nonfinite", wrapped.at[0, 0].set(jnp.nan), {}, False),
    ]
    certify = jax.jit(
        lambda x, previous, kwargs: prepared.certifies(x, previous, **kwargs)
    )
    for name, positions, kwargs, expected in cases:
        verdict = bool(certify(positions, state, kwargs))
        updated = prepared.update(positions, state, **kwargs)
        assert verdict is expected, name
        assert bool(updated.rebuilt) is (not verdict), name


_INT32_SYMMETRIC = 2**31 - 1
CUBE = 2.0 * np.eye(3)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("lattice", ["singular", "nonfinite"])
def test_lattice_failure_is_refused_even_without_active_particles(
    lattice: str, compiled: bool
) -> None:
    prepared = CellListParticleImageNeighborhoodPlan(0.3, PeriodicCell(CUBE), CAPACITY)
    prepared = prepared.prepare(_particles(1))
    vectors = (
        jnp.zeros((3, 3)) if lattice == "singular" else jnp.eye(3).at[0, 0].set(jnp.nan)
    )

    def build(x: jax.Array, h: jax.Array) -> ParticleImageNeighborhoodState:
        return prepared.build(x, active_mask=jnp.asarray([False]), cell_vectors=h)

    state = (jax.jit(build) if compiled else build)(jnp.full((1, 3), 0.3), vectors)
    assert not bool(state.successful)
    assert bool(state.scientific_failure)
    assert not bool(state.capacity_failure)


def test_resource_quotas_refuse_before_enumerating_radius_sized_stencils() -> None:
    tiny = ParticleImageCapacity(
        maximum_particles_per_cell=1,
        maximum_edges=10,
        maximum_degree=10,
        maximum_images=10,
    )
    tracemalloc.start()
    try:
        # 61**3 offsets / 71**3 shifts would be enumerated by an eager stencil.
        with pytest.raises(ValueError, match="maximum_candidate_slots"):
            CellListParticleImageNeighborhoodPlan(
                60.0, PeriodicCell(CUBE), tiny, maximum_candidate_slots=10
            ).prepare(_particles(1))
        with pytest.raises(ValueError, match="maximum_image_count"):
            PeriodicCell(CUBE, maximum_condition_number=20.0, maximum_image_count=27)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 2**20


def _pair_state(
    positions: np.ndarray, compiled: bool
) -> tuple[ParticleImageNeighborhoodState, jax.Array]:
    prepared = CellListParticleImageNeighborhoodPlan(
        0.3, PeriodicCell(CUBE), CAPACITY
    ).prepare(_particles(2))
    value = jnp.asarray(positions)

    def build(x: jax.Array) -> ParticleImageNeighborhoodState:
        return prepared.build(x)

    return (jax.jit(build) if compiled else build)(value), value


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_translated_systems_keep_exact_routes_up_to_the_int32_image_bound(
    compiled: bool,
) -> None:
    base = np.asarray([[0.1, 0.5, 0.5], [0.2, 0.5, 0.5]])
    for cells in (0, _INT32_SYMMETRIC - 1, -(_INT32_SYMMETRIC - 1)):
        positions = base + np.asarray([2.0 * cells, 0.0, 0.0])
        state, value = _pair_state(positions, compiled)
        assert bool(state.successful), cells
        np.testing.assert_array_equal(state.wrap_counts[:, 0], [cells, cells])
        routes = _routes(state, np.asarray(value), CUBE)
        assert sorted(np.round(row[0], 6) for row in routes.values()) == [-0.1, 0.1]


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "positions",
    [
        # The first wrap count 5e9 exceeds int32.
        [[1.0e10 + 0.1, 0.5, 0.5], [0.2, 0.5, 0.5]],
        # Wrap counts +-2e9 fit, but the route shifts +-4e9 do not.
        [[4.0e9 + 0.1, 0.5, 0.5], [-4.0e9 + 0.2, 0.5, 0.5]],
    ],
    ids=["wrap-count", "shift-difference"],
)
def test_unrepresentable_image_counts_refuse_instead_of_wrapping(
    positions: list[list[float]], compiled: bool
) -> None:
    state, _ = _pair_state(np.asarray(positions), compiled)
    assert not bool(state.successful)
    assert bool(state.evidence.representation_overflow[0])
    assert bool(state.scientific_failure)
    assert not bool(state.capacity_failure)


def test_representation_offsets_are_exact_or_refused() -> None:
    base = np.asarray([[0.1, 0.5, 0.5], [0.2, 0.5, 0.5]])
    state, _ = _pair_state(base, compiled=False)
    offsets = jnp.asarray([[7, -3, 0], [7, -3, 0]], dtype=jnp.int32)
    moved = base - np.asarray(offsets) @ CUBE
    reexpressed = state.with_representation_offsets(offsets)
    rebuilt, _ = _pair_state(moved, compiled=False)
    assert bool(reexpressed.successful)
    assert set(_routes(reexpressed, moved, CUBE)) == set(_routes(rebuilt, moved, CUBE))
    np.testing.assert_array_equal(reexpressed.wrap_counts, rebuilt.wrap_counts)
    np.testing.assert_allclose(reexpressed.reference_positions, moved)
    apart = jnp.asarray([[2_000_000_000, 0, 0], [-2_000_000_000, 0, 0]], dtype=jnp.int32)
    wide = jnp.asarray([[2**32, 0, 0], [0, 0, 0]], dtype=jnp.int64)
    for name, bad in (("difference", apart), ("wide", wide)):
        refused = jax.jit(lambda s, o: s.with_representation_offsets(o))(state, bad)
        assert not bool(refused.successful), name
        assert bool(refused.evidence.representation_overflow[0]), name
        with pytest.raises(ValueError, match="int32"):
            state.relation.with_representation_offsets(bad)


def test_wide_image_shifts_are_refused_before_int32_storage() -> None:
    with pytest.raises(ValueError, match="int32"):
        _self_relation([True], [[2**32 + 1, 0, 0]])  # would wrap to n = (1, 0, 0)
    with pytest.raises(ValueError, match="int32"):
        _self_relation([True], [[-(2**31), 0, 0]])  # has no int32 negation
    reversed_relation = _self_relation([True], [[_INT32_SYMMETRIC, 0, 0]]).reversed()
    np.testing.assert_array_equal(
        reversed_relation.image_shifts, [[-_INT32_SYMMETRIC, 0, 0]]
    )


def test_image_verlet_refuses_unrepresentable_counts_and_rebuilds_on_overflow() -> None:
    prepared = _verlet(PeriodicCell(CUBE), 2, 0.3, 0.2)
    base = jnp.asarray([[0.1, 0.5, 0.5], [0.2, 0.5, 0.5]])
    initialize = jax.jit(lambda x, c: prepared.initialize(x, image_counts=c))
    update = jax.jit(lambda x, s, c: prepared.update(x, s, image_counts=c))
    certifies = jax.jit(lambda x, s, c: prepared.certifies(x, s, image_counts=c))
    wide = jnp.asarray([[2**33, 0, 0], [0, 0, 0]], dtype=jnp.int64)
    refused = initialize(base, wide)
    assert not bool(refused.successful)
    assert bool(refused.scientific_failure)
    counts = jnp.asarray([[2_000_000_000, 0, 0], [-2_000_000_000, 0, 0]], jnp.int64)
    state = initialize(base, counts)
    assert bool(state.successful)
    # The count differences 4e9 overflow int32; the cache must not be reused.
    assert not bool(certifies(base, state, -counts))
    flipped = update(base, state, -counts)
    assert bool(flipped.rebuilt)
    assert bool(flipped.successful)
    rewrapped = base.at[0, 0].add(-2.0)
    reused = update(rewrapped, state, counts.at[0, 0].add(1))
    assert not bool(reused.rebuilt)
    assert bool(reused.neighborhood.successful)
    routes = _routes(reused.neighborhood, np.asarray(rewrapped), CUBE)
    assert sorted(np.round(row[0], 6) for row in routes.values()) == [-0.1, 0.1]


@pytest.mark.parametrize("rank", (1, 2, 4, 5))
@pytest.mark.parametrize("condition", (1.0, 1e4))
def test_host_lattice_coordinates_and_reach_match_dense_reference(
    rank: int, condition: float
) -> None:
    rng = np.random.default_rng(210 + rank)
    rotation, _ = np.linalg.qr(rng.normal(size=(rank + 1, rank + 1)))
    vectors = np.geomspace(1.0, condition, rank)[:, None] * rotation[:rank]
    cell = PeriodicCell(vectors, periodic_axes=(False,) * rank)
    reference = vectors.T @ np.linalg.inv(vectors @ vectors.T)
    reach = np.sqrt(np.sum(reference**2, axis=0))
    np.testing.assert_allclose(cell.inverse_vectors, reference, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        cell.reciprocal_vectors, 2 * np.pi * reference.T, rtol=1e-9
    )
    stencil = cell.image_stencil(1.3, maximum_image_count=1)
    np.testing.assert_allclose(stencil.axis_reach, reach, rtol=1e-9)
    np.testing.assert_allclose(_host_reach(vectors[None]), reach[None], rtol=1e-9)
    singular = vectors.copy()
    singular[-1] = 0.0
    with pytest.raises(ValueError, match="full row rank"):
        PeriodicCell(singular, periodic_axes=(False,) * rank)
    with pytest.raises(ValueError, match="rank deficient"):
        cell.image_stencil(1.3, maximum_image_count=1, cell_vectors=singular)
    with pytest.raises(ValueError, match="rank deficient"):
        _host_reach(singular[None])


@pytest.mark.parametrize(
    "vectors",
    (np.asarray([[2.0, 0.0, 0.0]]), TRICLINIC[:2], TRICLINIC),
)
def test_image_stencil_integer_reaches_match_inverse_at_cutoff_boundaries(
    vectors: np.ndarray,
) -> None:
    cell = PeriodicCell(vectors)
    inverse = vectors.T @ np.linalg.inv(vectors @ vectors.T)
    reach = np.sqrt(np.sum(inverse * inverse, axis=0))
    excursion = jnp.ones((vectors.shape[0],))
    for axis_reach in reach:
        boundary = 1.0 / axis_reach
        for radius in (
            np.nextafter(boundary, 0.0),
            boundary,
            np.nextafter(boundary, np.inf),
        ):
            stencil = cell.image_stencil(float(radius), maximum_image_count=100_000)
            exact = np.floor(1.0 + radius * reach).astype(int)
            extents = np.asarray(stencil.extents)
            # Sizing headroom never removes a required shell; at these O(1)
            # reaches it adds at most one. The evidence keeps the raw reach.
            assert np.all(extents >= exact) and np.all(extents <= exact + 1)
            np.testing.assert_allclose(stencil.axis_reach, reach, rtol=1e-12)
            assert bool(
                jnp.all(
                    cell.image_stencil_margin(stencil, vectors, float(radius), excursion)
                    > 0.0
                )
            )
            expected_images = np.asarray(
                tuple(
                    product(*(range(-extent, extent + 1) for extent in stencil.extents))
                ),
                dtype=np.int32,
            )
            np.testing.assert_array_equal(stencil.shifts, expected_images)


def test_reference_image_stencil_keeps_a_positive_margin_on_its_own_lattice() -> None:
    # A float64 reference whose native margin was exactly zero on the first axis.
    vectors = np.asarray(
        [
            [3.2290427473489114, -0.09052985842208917, -0.04311943261795318],
            [-0.4039972258294502, 3.900855550855561, -0.17304261525498835],
            [0.6645999033289766, 0.04515732264558436, 6.134571701957268],
        ]
    )
    radius = 3.19046819940291
    cell = PeriodicCell(vectors)
    stencil = cell.image_stencil(radius, maximum_image_count=4096)
    inverse = vectors.T @ np.linalg.inv(vectors @ vectors.T)
    reach = np.sqrt(np.sum(inverse * inverse, axis=0))
    np.testing.assert_allclose(stencil.axis_reach, reach, rtol=1e-12)
    assert stencil.radius == radius
    assert np.all(np.asarray(stencil.extents) >= np.floor(1.0 + radius * reach))
    excursion = jnp.ones((3,))
    margin = cell.image_stencil_margin(stencil, vectors, radius, excursion)
    assert bool(jnp.all(margin > 0.0))
    # Independent complete-image oracle over wrapped endpoints.
    shifts = {tuple(int(value) for value in row) for row in np.asarray(stencil.shifts)}
    box = np.asarray(tuple(product(range(-6, 7), repeat=3)), dtype=np.float64)
    endpoints = np.random.default_rng(29).uniform(0.0, 1.0, size=(400, 2, 3)) @ vectors
    separation = endpoints[:, 1] - endpoints[:, 0]
    displacement = separation[:, None, :] + (box @ vectors)[None]
    reachable = np.any(np.linalg.norm(displacement, axis=-1) < radius, axis=0)
    needed = {
        tuple(int(value) for value in box[index]) for index in np.flatnonzero(reachable)
    }
    assert needed <= shifts
    # Genuine over-deformation still leaves the prepared stencil.
    shrunk = 0.5 * vectors
    assert bool(
        jnp.any(cell.image_stencil_margin(stencil, shrunk, radius, excursion) <= 0.0)
    )


def _route_search(
    backend: str,
    radius: float,
    vectors: np.ndarray,
    capacity: ParticleImageCapacity,
    particles: int,
) -> CellListImageRouteSearch | DenseImageRouteSearch:
    if backend == "cell_list":
        return CellListImageRouteSearch(
            radius,
            vectors,
            capacity,
            particle_capacity=particles,
            maximum_candidate_slots=10**7,
        )
    return DenseImageRouteSearch(
        radius,
        vectors,
        np.ones(vectors.shape[:2], dtype=np.bool_),
        capacity,
        particle_capacity=particles,
        maximum_dense_routes=10**7,
    )


def _batched_routes(
    search: CellListImageRouteSearch | DenseImageRouteSearch,
    positions: np.ndarray,
    vectors: np.ndarray,
    ids: np.ndarray,
    *,
    active: np.ndarray | None = None,
    receivers: ImageReceiverSubset | None = None,
) -> ImageRouteSearchResult:
    cases = positions.shape[0]
    mask = np.ones(positions.shape[:2], dtype=np.bool_) if active is None else active
    value = jnp.asarray(positions)
    return search.routes(
        value,
        jnp.asarray(mask),
        jnp.asarray(ids),
        jnp.asarray(vectors, dtype=value.dtype),
        jnp.zeros((cases, positions.shape[-1]), dtype=value.dtype),
        jnp.ones(vectors.shape[:2], dtype=jnp.bool_),
        support_id="support",
        relation_schema_id="schema",
        receivers=receivers,
    )


# The charged AIMNet2 family cell: its third height is exactly twice the PME
# real-space search radius ``1.6 + 0.3``.
BOUNDARY_CELL = np.asarray([[3.6, 0.0, 0.0], [0.3, 3.7, 0.0], [0.2, 0.1, 3.8]])
BOUNDARY_RADIUS = 1.6 + 0.3


def _boundary_lattice(name: str) -> np.ndarray:
    if name == "boundary-cell":
        return BOUNDARY_CELL
    rotation, _ = np.linalg.qr(np.random.default_rng(97).normal(size=(3, 3)))
    return TRICLINIC @ rotation


def _boundary_radii(vectors: np.ndarray, /) -> tuple[float, ...]:
    """Radii putting ``radius * reach`` or ``radius * 2 * reach`` on an integer."""
    reach = _host_reach(vectors[None])[0]
    radii = [BOUNDARY_RADIUS] if vectors is BOUNDARY_CELL else []
    for axis_reach in reach:
        radii.extend((1.0 / axis_reach, 0.5 / axis_reach))
    widest = 1.0 / float(np.max(reach))
    radii.extend((np.nextafter(widest, 0.0), np.nextafter(widest, np.inf)))
    return tuple(float(radius) for radius in radii)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("backend", ["cell_list", "dense"])
@pytest.mark.parametrize("lattice", ["boundary-cell", "rotated-triclinic"])
def test_reference_lattice_at_integer_stencil_boundaries_is_admitted_and_complete(
    lattice: str, backend: str, dtype: type[np.floating]
) -> None:
    vectors = _boundary_lattice(lattice)
    count = 6
    positions = (
        np.random.default_rng(71).uniform(0.0, 1.0, size=(count, 3)) @ vectors
    ).astype(dtype)
    stored = vectors.astype(dtype)
    exact_positions = positions.astype(np.float64)
    exact_vectors = stored.astype(np.float64)
    ids = np.asarray([12, 5, 30, 7, 21, 2])
    for radius in _boundary_radii(vectors):
        search = _route_search(backend, radius, vectors[None], CAPACITY, count)
        result = _batched_routes(search, positions[None], stored[None], ids[None])
        evidence = result.evidence
        assert not bool(evidence.stencil_overflow[0]), radius
        assert bool(evidence.successful[0]), radius
        expected = lattice_routes(
            exact_positions, exact_vectors, (True,) * 3, radius, ids
        )
        assert_resolved_cutoff(
            expected, exact_positions, exact_vectors, radius, ids, dtype
        )
        assert_routes_match(result.relation, exact_positions, exact_vectors, expected)


@pytest.mark.parametrize("backend", ["cell_list", "dense"])
def test_lattice_deformed_beyond_the_prepared_stencil_is_a_scientific_refusal(
    backend: str,
) -> None:
    count = 6
    search = _route_search(backend, BOUNDARY_RADIUS, BOUNDARY_CELL[None], CAPACITY, count)
    shrunk = 0.4 * BOUNDARY_CELL
    positions = np.random.default_rng(73).uniform(0.0, 1.0, size=(count, 3)) @ shrunk
    result = _batched_routes(
        search, positions[None], shrunk[None], np.arange(1, count + 1)[None]
    )
    evidence = result.evidence
    bound = (
        search.offset_extents
        if isinstance(search, CellListImageRouteSearch)
        else search.stencil_extents_bound
    )
    assert np.any(np.asarray(evidence.required_offset_extents[0]) > np.asarray(bound))
    assert bool(evidence.stencil_overflow[0])
    assert bool(evidence.scientific_failure[0])
    assert not bool(evidence.successful[0])


@pytest.mark.parametrize("backend", ["cell_list", "dense"])
def test_strongly_skewed_float64_reference_lattice_stays_admitted_and_complete(
    backend: str,
) -> None:
    # Condition number ~2**11 with ``radius * grid * reach`` on an integer:
    # stencil headroom must not narrow native float64 admission.
    vectors = np.asarray([[1.0, 0.0], [1.0, 2.0**-10]])
    radius = 2.0**-12
    first = np.asarray([0.31, 0.42]) @ vectors
    second = np.asarray([0.6, 0.0])
    positions = np.stack(
        (
            first,
            first + np.asarray([1.0e-4, 0.5e-4]),
            second,
            second + np.asarray([-0.7e-4, 0.9e-4]),
        )
    )
    ids = np.asarray([3, 8, 1, 6])
    search = _route_search(backend, radius, vectors[None], CAPACITY, 4)
    result = _batched_routes(search, positions[None], vectors[None], ids[None])
    assert not bool(result.evidence.stencil_overflow[0])
    assert bool(result.evidence.successful[0])
    expected = lattice_routes(positions, vectors, (True, True), radius, ids)
    assert len(expected) == 4
    assert_resolved_cutoff(expected, positions, vectors, radius, ids, np.float64)
    assert_routes_match(result.relation, positions, vectors, expected)


def test_canonical_half_is_an_exact_reversal_half_with_paired_self_images() -> None:
    vectors = 2.0 * np.eye(3)
    positions = np.asarray([[0.3, 0.4, 0.5], [1.1, 0.2, 1.7]])
    state = (
        CellListParticleImageNeighborhoodPlan(2.5, PeriodicCell(vectors), CAPACITY)
        .prepare(_particles(2))
        .build(jnp.asarray(positions))
    )
    relation = state.relation
    valid = np.asarray(relation.valid)
    reversal = relation.reversal_map()
    assert bool(np.all(reversal.closed))
    assert not bool(np.any(reversal.identity_conflict))
    np.testing.assert_array_equal(np.asarray(reversal.paired), valid)
    partner = np.asarray(reversal.partner)[valid]
    np.testing.assert_array_equal(
        np.asarray(relation.source_particle_ids)[partner],
        np.asarray(relation.receiver_particle_ids)[valid],
    )
    np.testing.assert_array_equal(
        np.asarray(relation.image_shifts)[partner],
        -np.asarray(relation.image_shifts)[valid],
    )
    half = relation.canonical_half(capacity=CAPACITY.maximum_edges)
    assert bool(np.all(half.successful))
    assert half.counting_convention == "canonical-half"
    assert half.directed_multiplicity == 2
    full_routes, _ = relation_routes(relation, positions, vectors)
    half_routes, _ = relation_routes(half.relation, positions, vectors)
    mirrored = Counter(
        (case, receiver, source, tuple(-value for value in shift))
        for (case, source, receiver, shift) in half_routes
    )
    assert half_routes + mirrored == full_routes
    assert not set(half_routes) & set(mirrored)
    self_routes = [key for key in full_routes if key[1] == key[2]]
    half_self = np.asarray(half.self_image)[np.asarray(half.relation.valid)]
    assert 2 * int(half_self.sum()) == len(self_routes) > 0
    half_valid = np.asarray(half.relation.valid)
    full_index = np.asarray(half.full_routes)[half_valid]
    np.testing.assert_array_equal(
        np.asarray(relation.image_shifts)[full_index],
        np.asarray(half.relation.image_shifts)[half_valid],
    )
    np.testing.assert_array_equal(
        np.asarray(relation.image_shifts)[np.asarray(half.partner_routes)[half_valid]],
        -np.asarray(half.relation.image_shifts)[half_valid],
    )


def test_self_image_pair_energy_has_zero_force_but_nonzero_cell_response() -> None:
    x = jnp.asarray([[0.2, 0.3, 0.4]])
    vectors = jnp.asarray(TRICLINIC)
    state = (
        CellListParticleImageNeighborhoodPlan(3.0, PeriodicCell(TRICLINIC), CAPACITY)
        .prepare(_particles(1))
        .build(x)
    )
    relation = state.relation
    half = relation.canonical_half(case_capacity=CAPACITY.maximum_edges)
    assert int(np.sum(np.asarray(relation.valid))) > 0

    def energy(
        routes: ParticleImageRelation, weight: float
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        def value(positions: jax.Array, cell: jax.Array) -> jax.Array:
            displacement = routes.displacement(positions, cell)
            squared = jnp.sum(displacement * displacement, axis=-1)
            return weight * jnp.sum(jnp.where(routes.valid, jnp.exp(-squared), 0.0))

        return value

    full = energy(relation, 0.5)
    halved = energy(half.relation, 1.0)
    np.testing.assert_allclose(full(x, vectors), halved(x, vectors), rtol=1e-13)
    force, strain = jax.grad(full, argnums=(0, 1))(x, vectors)
    half_force, half_strain = jax.grad(halved, argnums=(0, 1))(x, vectors)
    np.testing.assert_allclose(force, 0.0, atol=1e-12)
    np.testing.assert_allclose(half_force, 0.0, atol=1e-12)
    assert float(jnp.max(jnp.abs(strain))) > 1e-6
    np.testing.assert_allclose(half_strain, strain, rtol=1e-12, atol=1e-15)


def test_half_view_reports_truncation_overflow_and_identity_conflicts() -> None:
    positions = np.random.default_rng(5).uniform(0.0, 1.0, size=(4, 3)) @ TRICLINIC
    small = ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=6,
        maximum_degree=1024,
        maximum_images=4096,
    )
    truncated = (
        CellListParticleImageNeighborhoodPlan(3.0, PeriodicCell(TRICLINIC), small)
        .prepare(_particles(4))
        .build(jnp.asarray(positions))
    )
    assert bool(truncated.evidence.edge_overflow[0])
    open_half = truncated.relation.canonical_half(capacity=64)
    assert bool(open_half.reversal_open[0])
    assert bool(open_half.scientific_failure[0])
    complete = (
        CellListParticleImageNeighborhoodPlan(3.0, PeriodicCell(TRICLINIC), CAPACITY)
        .prepare(_particles(4))
        .build(jnp.asarray(positions))
    )
    clipped = complete.relation.canonical_half(capacity=1)
    assert bool(clipped.overflow[0])
    assert bool(clipped.capacity_failure[0])
    assert int(clipped.required_routes[0]) == int(complete.evidence.stored_edges[0]) // 2
    ambiguous = ParticleImageRelation(
        EdgeRelation(
            jnp.asarray([0, 1], dtype=jnp.int32),
            jnp.asarray([1, 0], dtype=jnp.int32),
            source_size=2,
            target_size=2,
        ),
        jnp.asarray([5, 5]),
        jnp.asarray([5, 5]),
        jnp.asarray([[1, 0, 0], [-1, 0, 0]], dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        case_count=1,
        particle_capacity=2,
        support_id="support",
        relation_schema_id="schema",
    )
    assert bool(ambiguous.reversal_map().identity_conflict[0])
    assert not bool(ambiguous.canonical_half(capacity=2).successful[0])


def test_segmented_layout_owns_case_segments_and_charges_overflow_per_case() -> None:
    rng = np.random.default_rng(31)
    vectors = np.stack((TRICLINIC, 1.6 * TRICLINIC))
    ids = np.stack((np.arange(5) * 3 + 1, np.arange(5) * 5 + 2))
    positions = np.stack(
        [rng.uniform(0.0, 1.0, size=(5, 3)) @ vectors[case] for case in range(2)]
    )
    radius = 2.2
    oracle = [
        lattice_routes(
            positions[case], vectors[case], (True,) * 3, radius, ids[case], case=case
        )
        for case in range(2)
    ]
    counts = [len(routes) for routes in oracle]
    fits = int(np.argmin(counts))
    over = 1 - fits
    assert counts[fits] < counts[over]
    width = counts[fits]
    capacity = ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=2 * width,
        maximum_degree=1024,
        maximum_images=4096,
        maximum_case_edges=width,
    )
    result = _batched_routes(
        _route_search("cell_list", radius, vectors, capacity, 5), positions, vectors, ids
    )
    evidence = result.evidence
    assert not bool(evidence.edge_overflow[fits])
    assert bool(evidence.edge_overflow[over])
    assert int(evidence.required_edges[over]) == counts[over]
    assert int(evidence.stored_edges[over]) == width
    assert bool(evidence.successful[fits])
    assert bool(evidence.capacity_failure[over])
    assert not bool(evidence.scientific_failure[over])
    np.testing.assert_array_equal(
        np.asarray(result.relation.route_cases).reshape((2, width)),
        np.repeat(np.arange(2)[:, None], width, axis=1),
    )
    multiset, displacement = relation_routes(
        result.relation, positions.reshape((-1, 3)), vectors
    )
    assert {key for key in multiset if key[0] == fits} == set(oracle[fits])
    for key, row in oracle[fits].items():
        np.testing.assert_allclose(displacement[key], row, atol=1e-12)
    with pytest.raises(ValueError, match="maximum_edges"):
        _route_search(
            "cell_list",
            radius,
            vectors,
            ParticleImageCapacity(
                maximum_particles_per_cell=8,
                maximum_edges=width,
                maximum_degree=1024,
                maximum_images=4096,
                maximum_case_edges=width,
            ),
            5,
        )


@pytest.mark.parametrize("backend", ["cell_list", "dense"])
def test_receiver_subset_returns_exact_rows_and_refuses_bad_requests(
    backend: str,
) -> None:
    rng = np.random.default_rng(41)
    positions = rng.uniform(0.0, 1.0, size=(1, 6, 3)) @ TRICLINIC
    ids = np.asarray([[11, 4, 27, 8, 15, 30]])
    radius = 2.6
    search = _route_search(backend, radius, TRICLINIC[None], CAPACITY, 6)
    wanted = [27, 4]
    subset = ImageReceiverSubset(
        np.asarray([wanted + [0]]), np.asarray([[True, True, False]])
    )
    result = _batched_routes(search, positions, TRICLINIC[None], ids, receivers=subset)
    expected = lattice_routes(
        positions[0], TRICLINIC, (True,) * 3, radius, ids[0], receivers=set(wanted)
    )
    assert_routes_match(result.relation, positions[0], TRICLINIC, expected)
    assert bool(result.evidence.successful[0])
    rows = result.receiver_rows
    assert rows is not None
    valid = np.asarray(result.relation.valid)
    np.testing.assert_array_equal(
        np.asarray(wanted)[np.asarray(rows.route_rows)[valid]],
        np.asarray(result.relation.receiver_particle_ids)[valid],
    )
    np.testing.assert_array_equal(np.asarray(rows.valid), [True, True, False])
    np.testing.assert_array_equal(np.asarray(rows.particle_indices)[:2], [2, 1])
    for name, request in (
        ("absent", [[99]]),
        ("repeated", [[4, 4]]),
        ("inactive", [[8]]),
    ):
        active = np.ones((1, 6), dtype=np.bool_)
        if name == "inactive":
            active[0, 3] = False
        refused = _batched_routes(
            search,
            positions,
            TRICLINIC[None],
            ids,
            active=active,
            receivers=ImageReceiverSubset(np.asarray(request)),
        )
        assert bool(refused.evidence.receiver_violation[0]), name
        assert bool(refused.evidence.scientific_failure[0]), name
        assert not bool(refused.evidence.capacity_failure[0]), name


def _selection_oracle(
    routes: dict, radius: float, neighbors: int, tolerance: float, strict: bool
) -> set:
    """Independent per-receiver form of the declared closed-radius selection."""
    groups = defaultdict(list)
    for key, row in routes.items():
        squared = float(row @ row)
        if squared <= radius * radius:
            groups[(key[0], key[2])].append((squared, key[1], key[3], key))
    chosen = set()
    for items in groups.values():
        items.sort()
        if len(items) <= neighbors:
            chosen.update(item[-1] for item in items)
        elif strict:
            chosen.update(item[-1] for item in items[:neighbors])
        else:
            anchor = items[neighbors][0]
            chosen.update(item[-1] for item in items if item[0] <= anchor + tolerance)
    return chosen


@pytest.mark.parametrize("strict", [False, True], ids=["band", "strict"])
def test_declared_nearest_selection_matches_rule_and_storage_is_separate(
    strict: bool,
) -> None:
    positions = np.random.default_rng(53).uniform(0.0, 1.0, size=(8, 3)) @ TRICLINIC
    state = (
        DenseParticleImageNeighborhoodPlan(
            3.0, PeriodicCell(TRICLINIC), CAPACITY, maximum_dense_routes=10**7
        )
        .prepare(_particles(8))
        .build(jnp.asarray(positions))
    )
    policy = ImageNeighborSelectionPolicy(
        2.8,
        5,
        degeneracy_tolerance=0.3,
        measure="squared-distance",
        strict=strict,
        radius_boundary="closed",
    )

    def select(capacity: ImageSelectionCapacity) -> ImageNeighborSelection:
        return PreparedImageNeighborSelection(
            policy, capacity, source_radius=3.0, case_count=1
        ).select(
            state.relation,
            jnp.asarray(positions),
            jnp.asarray(TRICLINIC),
            evidence=state.evidence,
        )

    selection = select(ImageSelectionCapacity(maximum_routes=512, maximum_degree=64))
    _, candidates = relation_routes(state.relation, positions, TRICLINIC)
    expected = _selection_oracle(candidates, 2.8, 5, 0.3, strict)
    multiset, _ = relation_routes(selection.relation, positions, TRICLINIC)
    assert all(count == 1 for count in multiset.values())
    assert set(multiset) == expected
    evidence = selection.evidence
    assert bool(evidence.successful[0])
    assert int(evidence.capped_receivers[0]) > 0
    if strict:
        assert int(evidence.selected_degree[0]) == 5
    else:
        assert int(evidence.selected_degree[0]) >= 6
    chosen = np.asarray(selection.relation.valid)
    parent = np.asarray(selection.parent_routes)[chosen]
    np.testing.assert_array_equal(
        np.asarray(state.relation.image_shifts)[parent],
        np.asarray(selection.relation.image_shifts)[chosen],
    )
    tight = select(ImageSelectionCapacity(maximum_routes=4, maximum_degree=64))
    assert bool(tight.evidence.route_overflow[0])
    assert bool(tight.evidence.capacity_failure[0])
    assert not bool(tight.evidence.scientific_failure[0])
    assert int(tight.evidence.selected_routes[0]) == len(expected)
    narrow = select(ImageSelectionCapacity(maximum_routes=512, maximum_degree=2))
    assert bool(narrow.evidence.degree_overflow[0])
    assert not bool(narrow.evidence.route_overflow[0])


def test_selection_ties_use_stable_identity_and_the_declared_band() -> None:
    vectors = 2.0 * np.eye(3)
    x = jnp.full((1, 3), 0.5)
    state = (
        CellListParticleImageNeighborhoodPlan(2.5, PeriodicCell(vectors), CAPACITY)
        .prepare(_particles(1))
        .build(x)
    )

    def select(
        neighbors: int, strict: bool
    ) -> tuple[set[tuple[int, ...]], ImageNeighborSelection]:
        policy = ImageNeighborSelectionPolicy(
            2.2,
            neighbors,
            degeneracy_tolerance=0.0,
            measure="distance",
            strict=strict,
            radius_boundary="open",
        )
        result = PreparedImageNeighborSelection(
            policy,
            ImageSelectionCapacity(maximum_routes=64, maximum_degree=64),
            source_radius=2.5,
            case_count=1,
        ).select(state.relation, x, jnp.asarray(vectors))
        multiset, _ = relation_routes(result.relation, np.asarray(x), vectors)
        return {key[3] for key in multiset}, result

    strict_routes, _ = select(4, True)
    assert strict_routes == {(-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)}
    banded, result = select(4, False)
    assert len(banded) == 6
    assert int(result.evidence.capped_receivers[0]) == 0


def test_selection_refuses_uncovered_radius_and_carries_source_status() -> None:
    capacity = ImageSelectionCapacity(maximum_routes=64, maximum_degree=64)
    closed = ImageNeighborSelectionPolicy(
        2.0,
        4,
        degeneracy_tolerance=0.01,
        measure="squared-distance",
        strict=False,
        radius_boundary="closed",
    )
    with pytest.raises(ValueError, match="cover"):
        PreparedImageNeighborSelection(closed, capacity, source_radius=2.0, case_count=1)
    positions = np.random.default_rng(5).uniform(0.0, 1.0, size=(4, 3)) @ TRICLINIC
    small = ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=6,
        maximum_degree=1024,
        maximum_images=4096,
    )
    truncated = (
        CellListParticleImageNeighborhoodPlan(3.0, PeriodicCell(TRICLINIC), small)
        .prepare(_particles(4))
        .build(jnp.asarray(positions))
    )
    prepared = PreparedImageNeighborSelection(
        closed, capacity, source_radius=3.0, case_count=1
    )
    carried = prepared.select(
        truncated.relation,
        jnp.asarray(positions),
        jnp.asarray(TRICLINIC),
        evidence=truncated.evidence,
        certified=jnp.asarray([False]),
    )
    assert bool(carried.evidence.source_capacity_failure[0])
    assert bool(carried.evidence.source_uncertified[0])
    assert not bool(carried.evidence.successful[0])
