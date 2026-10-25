#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.particle import ParticleImageCapacity
from phydrax.discretization.particle._image_neighborhood import (
    CellListImageRouteSearch,
    image_certificate,
)
from phydrax.discretization.particle._image_radius_tiers import (
    ImageRadiusTier,
    ImageRadiusTierPlan,
)
from phydrax.discretization.particle._verlet import (
    SegmentedImageVerletPlan,
    SegmentedImageVerletState,
)
from tests._support.image_lattice import lattice_routes, relation_routes


TRICLINIC = np.asarray(
    [[2.0, 0.0, 0.0], [0.7, 2.1, 0.0], [0.3, -0.4, 2.3]], dtype=np.float64
)
VECTORS = np.stack((TRICLINIC, 1.2 * TRICLINIC))
IDS = np.stack(
    (np.arange(5, dtype=np.int64) * 3 + 1, np.arange(5, dtype=np.int64) * 5 + 2)
)
RADIUS = 2.0
SKIN = 0.4
WIDTH = 2048


def _capacity(width: int = WIDTH) -> ParticleImageCapacity:
    return ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=2 * width,
        maximum_degree=1024,
        maximum_images=4096,
        maximum_case_edges=width,
    )


def _plan(width: int = WIDTH) -> SegmentedImageVerletPlan:
    search = CellListImageRouteSearch(
        RADIUS + SKIN,
        VECTORS,
        _capacity(width),
        particle_capacity=5,
        maximum_candidate_slots=10**7,
    )
    return SegmentedImageVerletPlan(search, RADIUS, SKIN, support_id="support")


class _ImageFrame(TypedDict):
    active: jax.Array
    stable_ids: jax.Array
    vectors: jax.Array
    origin: jax.Array
    periodic: jax.Array


def _frame(vectors: np.ndarray = VECTORS) -> _ImageFrame:
    return {
        "active": jnp.ones((2, 5), dtype=jnp.bool_),
        "stable_ids": jnp.asarray(IDS),
        "vectors": jnp.asarray(vectors),
        "origin": jnp.zeros((2, 3), dtype=jnp.float64),
        "periodic": jnp.ones((2, 3), dtype=jnp.bool_),
    }


def _positions(seed: int = 19) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.uniform(0.0, 1.0, size=(5, 3)) @ VECTORS[case] for case in range(2)]
    )


def _segment(state: SegmentedImageVerletState, case: int) -> tuple[np.ndarray, ...]:
    relation = state.reference.relation
    return tuple(
        np.asarray(value).reshape((2, WIDTH) + np.asarray(value).shape[1:])[case]
        for value in (
            relation.source_indices,
            relation.receiver_indices,
            relation.image_shifts,
            relation.valid,
        )
    )


def _assert_cutoff_routes(
    state: SegmentedImageVerletState, positions: np.ndarray, vectors: np.ndarray
) -> None:
    """Cached routes within the interaction radius equal the lattice oracle."""
    neighborhood = state.neighborhood
    multiset, displacement = relation_routes(
        neighborhood.relation, positions.reshape((-1, 3)), vectors
    )
    assert all(count == 1 for count in multiset.values())
    within = {key for key, row in displacement.items() if np.linalg.norm(row) < RADIUS}
    expected = set()
    for case in range(2):
        expected |= set(
            lattice_routes(
                positions[case], vectors[case], (True,) * 3, RADIUS, IDS[case], case=case
            )
        )
    assert within == expected


def test_selective_rebuild_retains_certified_case_bitwise() -> None:
    plan = _plan()
    x0 = _positions()
    state = plan.initialize(jnp.asarray(x0), **_frame())
    np.testing.assert_array_equal(state.successful, [True, True])
    np.testing.assert_array_equal(state.case_epochs, [0, 0])
    moved = x0.copy()
    moved[0] += 0.5
    moved[1] += 0.01
    updated = plan.update(jnp.asarray(moved), state, **_frame())
    np.testing.assert_array_equal(updated.rebuilt, [True, False])
    np.testing.assert_array_equal(updated.case_epochs, [1, 0])
    np.testing.assert_array_equal(updated.rebuild_counts, [2, 1])
    np.testing.assert_array_equal(updated.successful, [True, True])
    for before, after in zip(_segment(state, 1), _segment(updated, 1), strict=True):
        np.testing.assert_array_equal(before, after)
    _assert_cutoff_routes(updated, moved, VECTORS)
    compiled = jax.jit(lambda x, previous: plan.update(x, previous, **_frame()))(
        jnp.asarray(moved), state
    )
    np.testing.assert_array_equal(compiled.rebuilt, updated.rebuilt)
    for case in range(2):
        for eager, traced in zip(
            _segment(updated, case), _segment(compiled, case), strict=True
        ):
            np.testing.assert_array_equal(eager, traced)


def test_disallowed_rebuild_is_stale_and_never_successful() -> None:
    plan = _plan()
    x0 = _positions()
    state = plan.initialize(jnp.asarray(x0), **_frame())
    moved = x0.copy()
    moved[0] += 0.5
    kept = plan.update(
        jnp.asarray(moved), state, **_frame(), allow_rebuild=jnp.asarray([False, True])
    )
    np.testing.assert_array_equal(kept.stale, [True, False])
    np.testing.assert_array_equal(kept.successful, [False, True])
    np.testing.assert_array_equal(kept.rebuilt, [False, False])
    np.testing.assert_array_equal(kept.case_epochs, [0, 0])
    certificate = plan.certify(jnp.asarray(moved), state, **_frame())
    np.testing.assert_array_equal(certificate.certified, [False, True])
    np.testing.assert_array_equal(certificate.motion_exceeded, [True, False])
    recovered = plan.update(jnp.asarray(moved), kept, **_frame())
    np.testing.assert_array_equal(recovered.rebuilt, [True, False])
    np.testing.assert_array_equal(recovered.successful, [True, True])
    forced = plan.update(
        jnp.asarray(x0), state, **_frame(), force_rebuild=jnp.asarray([False, True])
    )
    np.testing.assert_array_equal(forced.rebuilt, [False, True])
    np.testing.assert_array_equal(forced.case_epochs, [0, 1])


def test_uninitialized_cases_cannot_be_reused_until_rebuilt() -> None:
    plan = _plan()
    x0 = _positions()
    partial = plan.initialize(
        jnp.asarray(x0), **_frame(), cases=jnp.asarray([True, False])
    )
    np.testing.assert_array_equal(partial.initialized, [True, False])
    np.testing.assert_array_equal(partial.successful, [True, False])
    assert not np.any(_segment(partial, 1)[3])
    certificate = plan.certify(jnp.asarray(x0), partial, **_frame())
    np.testing.assert_array_equal(certificate.certified, [True, False])
    np.testing.assert_array_equal(certificate.uninitialized, [False, True])
    kept = plan.update(
        jnp.asarray(x0), partial, **_frame(), allow_rebuild=jnp.asarray([True, False])
    )
    np.testing.assert_array_equal(kept.initialized, [True, False])
    np.testing.assert_array_equal(kept.stale, [False, True])
    np.testing.assert_array_equal(kept.successful, [True, False])
    filled = plan.update(jnp.asarray(x0), partial, **_frame())
    np.testing.assert_array_equal(filled.rebuilt, [False, True])
    np.testing.assert_array_equal(filled.initialized, [True, True])
    np.testing.assert_array_equal(filled.successful, [True, True])
    for before, after in zip(_segment(partial, 0), _segment(filled, 0), strict=True):
        np.testing.assert_array_equal(before, after)
    _assert_cutoff_routes(filled, x0, VECTORS)


@pytest.mark.parametrize("exact_counts", [True, False], ids=["counts", "inferred"])
def test_rewrap_is_exact_and_cell_deformation_rebuilds_only_its_case(
    exact_counts: bool,
) -> None:
    plan = _plan()
    x0 = _positions()
    counts = np.zeros((2, 5, 3), dtype=np.int32)
    state = plan.initialize(
        jnp.asarray(x0),
        **_frame(),
        image_counts=jnp.asarray(counts) if exact_counts else None,
    )
    rewrapped = x0.copy()
    rewrapped[1, 0] -= VECTORS[1, 0]
    counts[1, 0, 0] += 1
    reused = plan.update(
        jnp.asarray(rewrapped),
        state,
        **_frame(),
        image_counts=jnp.asarray(counts) if exact_counts else None,
    )
    np.testing.assert_array_equal(reused.rebuilt, [False, False])
    np.testing.assert_array_equal(reused.representation_offsets[1, 0], [1, 0, 0])
    _assert_cutoff_routes(reused, rewrapped, VECTORS)
    deformed = VECTORS.copy()
    deformed[1] = 0.9 * VECTORS[1]
    rebuilt = plan.update(
        jnp.asarray(rewrapped),
        reused,
        **_frame(deformed),
        image_counts=jnp.asarray(counts) if exact_counts else None,
    )
    np.testing.assert_array_equal(rebuilt.rebuilt, [False, True])
    np.testing.assert_array_equal(rebuilt.successful, [True, True])
    _assert_cutoff_routes(rebuilt, rewrapped, deformed)


def test_capacity_rejection_keeps_the_accepted_case_epoch() -> None:
    x0 = _positions()
    probe = _plan().initialize(jnp.asarray(x0), **_frame())
    width = int(np.max(np.asarray(probe.reference.evidence.required_edges)))
    plan = SegmentedImageVerletPlan(
        CellListImageRouteSearch(
            RADIUS + SKIN,
            VECTORS,
            _capacity(width),
            particle_capacity=5,
            maximum_candidate_slots=10**7,
        ),
        RADIUS,
        SKIN,
        support_id="support",
    )
    state = plan.initialize(jnp.asarray(x0), **_frame())
    np.testing.assert_array_equal(state.successful, [True, True])
    # Shrinking case 0's cell at fixed fractional coordinates roughly doubles
    # its routes, overflowing its own segment only.
    shrunk = VECTORS.copy()
    shrunk[0] = 0.8 * VECTORS[0]
    squeezed = x0.copy()
    squeezed[0] = 0.8 * x0[0]
    candidate = plan.update(jnp.asarray(squeezed), state, **_frame(shrunk))
    assert bool(candidate.rebuilt[0])
    assert bool(candidate.capacity_failure[0])
    assert not bool(candidate.scientific_failure[0])
    np.testing.assert_array_equal(candidate.successful, [False, True])
    accepted = plan.select_cases(candidate.successful, candidate, state)
    np.testing.assert_array_equal(accepted.case_epochs, state.case_epochs)
    relation = accepted.reference.relation
    previous = state.reference.relation
    for after, before in (
        (relation.source_indices, previous.source_indices),
        (relation.image_shifts, previous.image_shifts),
        (relation.valid, previous.valid),
    ):
        np.testing.assert_array_equal(
            np.asarray(after).reshape((2, width, -1))[0],
            np.asarray(before).reshape((2, width, -1))[0],
        )
    np.testing.assert_array_equal(accepted.successful, [True, True])


def test_segmented_plan_refuses_shared_layout_and_uncovered_radius() -> None:
    shared = ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=4096,
        maximum_degree=1024,
        maximum_images=4096,
    )
    search = CellListImageRouteSearch(
        RADIUS + SKIN, VECTORS, shared, particle_capacity=5, maximum_candidate_slots=10**7
    )
    with pytest.raises(ValueError, match="segmented"):
        SegmentedImageVerletPlan(search, RADIUS, SKIN, support_id="support")
    with pytest.raises(ValueError, match="cover"):
        SegmentedImageVerletPlan(_plan().search, RADIUS, 2.0 * SKIN, support_id="support")


def test_tier_views_of_a_segmented_epoch_are_nested_and_independent() -> None:
    plan = _plan()
    x0 = _positions()
    state = plan.initialize(jnp.asarray(x0), **_frame())
    tiers = ImageRadiusTierPlan(
        (
            ImageRadiusTier(
                "short", 1.2, maximum_edges=8, maximum_degree=64, maximum_case_edges=4
            ),
            ImageRadiusTier("long", 2.0, maximum_edges=4096, maximum_degree=1024),
        ),
        skin=SKIN,
    )
    assert tiers.outer_radius == pytest.approx(RADIUS + SKIN)
    views = tiers.prepare(source_radius=RADIUS + SKIN, case_count=2).views(
        state.reference
    )
    flat = x0.reshape((-1, 3))
    short, long = views.view("short"), views.view("long")
    long_routes = relation_routes(long.relation, flat, VECTORS)[0]
    expected_long = set()
    expected_short = []
    for case in range(2):
        expected_long |= set(
            lattice_routes(
                x0[case], VECTORS[case], (True,) * 3, 2.4, IDS[case], case=case
            )
        )
        expected_short.append(
            lattice_routes(
                x0[case], VECTORS[case], (True,) * 3, 1.6, IDS[case], case=case
            )
        )
    assert set(long_routes) == expected_long
    assert bool(np.all(long.evidence.successful))
    np.testing.assert_array_equal(
        short.evidence.required_edges, [len(routes) for routes in expected_short]
    )
    overflowed = np.asarray(short.evidence.edge_overflow)
    np.testing.assert_array_equal(
        overflowed, [len(routes) > 4 for routes in expected_short]
    )
    short_routes = relation_routes(short.relation, flat, VECTORS)[0]
    assert set(short_routes) <= set(long_routes)
    for case in range(2):
        if not overflowed[case]:
            assert {key for key in short_routes if key[0] == case} == set(
                expected_short[case]
            )
    moved = x0 + 0.05
    mask = np.asarray(long.cutoff_mask(jnp.asarray(moved.reshape((-1, 3))), VECTORS))
    displacement = np.asarray(
        long.relation.displacement(jnp.asarray(moved.reshape((-1, 3))), VECTORS)
    )
    physical = set()
    for case in range(2):
        physical |= set(
            lattice_routes(
                moved[case], VECTORS[case], (True,) * 3, 2.0, IDS[case], case=case
            )
        )
    kept = {
        (int(case), int(source), int(receiver), tuple(int(v) for v in shift))
        for case, source, receiver, shift in zip(
            np.asarray(long.relation.route_cases)[mask],
            np.asarray(long.relation.source_particle_ids)[mask],
            np.asarray(long.relation.receiver_particle_ids)[mask],
            np.asarray(long.relation.image_shifts)[mask],
            strict=True,
        )
    }
    assert kept == physical
    assert np.all(np.linalg.norm(displacement[mask], axis=-1) < 2.0)
    certificate = image_certificate(
        x0,
        moved,
        VECTORS,
        VECTORS,
        np.zeros((2, 3)),
        np.asarray(state.reference.wrap_counts).reshape((2, 5, 3)),
        state.reference.stencil_extents,
        np.ones((2, 5), dtype=np.bool_),
        np.ones((2, 3), dtype=np.bool_),
        2.0,
    )
    np.testing.assert_array_equal(
        views.complete(certificate),
        [~overflowed, [True, True]],
    )
    with pytest.raises(ValueError, match="cover"):
        tiers.prepare(source_radius=RADIUS, case_count=2)
    with pytest.raises(ValueError, match="increasing"):
        ImageRadiusTierPlan(
            (
                ImageRadiusTier("a", 2.0, maximum_edges=8, maximum_degree=8),
                ImageRadiusTier("b", 1.0, maximum_edges=8, maximum_degree=8),
            ),
            skin=0.1,
        )
