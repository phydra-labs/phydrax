#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import tracemalloc
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
from phydrax.sparse import EdgeRelation


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
