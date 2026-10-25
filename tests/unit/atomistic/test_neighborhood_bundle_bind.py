#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frozen-epoch binding of current geometry to an accepted neighborhood bundle."""

import json
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jax_core

import phydrax as phx
from phydrax.atomistic._neighborhood_bundle import (
    AbstractAtomisticImageSearchPolicy,
    AtomisticCellListImageSearch,
    AtomisticClusterTileImageSearch,
    AtomisticCompletePairSource,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodBundleState,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticNeighborhoodSource,
    AtomisticNeighborhoodView,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
    PreparedAtomisticNeighborhoodBundle,
)
from phydrax.atomistic._sites import AtomisticSiteDomain
from phydrax.atomistic._system import PreparedAtomisticSystem
from phydrax.discretization import (
    ImageNeighborSelectionPolicy,
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)


Role = AtomisticNeighborhoodRole
Requirement = AtomisticNeighborhoodRequirement
ATOMS = AtomisticSiteDomain.PHYSICAL_ATOMS
CELL = np.asarray([[6.0, 0.0, 0.0], [0.5, 6.2, 0.0], [0.3, -0.4, 6.4]])
IDS = np.asarray([3, 7, 11, 20])
FRACTIONS = np.asarray(
    [
        [0.10, 0.20, 0.30],
        [0.17, 0.24, 0.33],
        [0.55, 0.62, 0.71],
        [0.83, 0.12, 0.88],
    ]
)
SEARCH = AtomisticCellListImageSearch(
    maximum_candidate_slots=10**7, deformation_margin=1.0
)
GRAPH = Requirement("mace", Role.DIRECTED_IMAGE_GRAPH, 3.0)
COORDINATION = Requirement("d3", Role.COORDINATION, 2.5)
PAIRS = Requirement("lj", Role.CLASSICAL_MINIMUM_IMAGE_PAIRS, 2.0)
# A large cube keeps the selected-graph fixture free of periodic images.
BOX = 20.0 * np.eye(3)
CENTER = np.full((3,), 10.0)
STAR = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.4]])


def _system(cell: np.ndarray) -> PreparedAtomisticSystem:
    return phx.atomistic.AtomisticSystemPlan(
        IDS,
        np.ones(IDS.shape, dtype=np.int32),
        np.ones(IDS.shape),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros(IDS.shape, dtype=np.int32),
        cell=PeriodicCell(cell),
    ).prepare()


def _capacity(width: int = 256) -> ParticleImageCapacity:
    return ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=width,
        maximum_degree=width,
        maximum_images=4096,
        maximum_case_edges=width,
    )


def _plan(
    requirements: tuple[Requirement, ...],
    *,
    skin: float,
    search: AbstractAtomisticImageSearchPolicy = SEARCH,
    frame: PeriodicCell | None = None,
) -> AtomisticNeighborhoodBundlePlan:
    bounded = tuple(value for value in requirements if not value.complete)
    complete = tuple(value for value in requirements if value.complete)
    sources: list[AtomisticNeighborhoodSource | AtomisticCompletePairSource] = []
    if bounded:
        sources.append(
            AtomisticNeighborhoodSource(
                "images",
                ATOMS,
                bounded,
                search,
                ParticleImageCapacityLadder((_capacity(),)),
                skin=skin,
            )
        )
    if complete:
        sources.append(
            AtomisticCompletePairSource("complete", ATOMS, complete, maximum_pairs=6)
        )
    ladders = tuple(
        AtomisticRoleCapacityLadder(
            value, (AtomisticRoleCapacity(maximum_routes=256, maximum_degree=256),)
        )
        for value in bounded
    )
    return AtomisticNeighborhoodBundlePlan(
        requirements, tuple(sources), ladders, frame=frame
    )


def _routes(
    view: AtomisticNeighborhoodView, positions: np.ndarray, vectors: np.ndarray
) -> dict[tuple[int, int, tuple[int, ...]], np.ndarray]:
    """Masked view routes by material identity with their displacements."""
    displacement = np.asarray(
        view.displacement(jnp.asarray(positions.reshape((-1, 3))), jnp.asarray(vectors))
    )
    relation = view.relation
    mask = np.asarray(view.mask)
    keys = [
        (int(source), int(receiver), tuple(int(value) for value in shift))
        for source, receiver, shift in zip(
            np.asarray(relation.source_particle_ids)[mask],
            np.asarray(relation.receiver_particle_ids)[mask],
            np.asarray(relation.image_shifts)[mask],
            strict=True,
        )
    ]
    assert all(count == 1 for count in Counter(keys).values())
    return dict(zip(keys, displacement[mask], strict=True))


def _assert_same_views(
    prepared: PreparedAtomisticNeighborhoodBundle,
    first: AtomisticNeighborhoodBundleState,
    second: AtomisticNeighborhoodBundleState,
    requirements: tuple[Requirement, ...],
    positions: np.ndarray,
    vectors: np.ndarray,
) -> None:
    for requirement in requirements:
        left = _routes(prepared.view(first, requirement), positions, vectors)
        right = _routes(prepared.view(second, requirement), positions, vectors)
        assert set(left) == set(right)
        for key, value in left.items():
            np.testing.assert_allclose(value, right[key], atol=1e-12)


def _primitives(jaxpr: jax_core.Jaxpr) -> set[str]:
    names: set[str] = set()
    pending = [jaxpr]
    while pending:
        current = pending.pop()
        for equation in current.eqns:
            names.add(equation.primitive.name)
            for value in equation.params.values():
                for item in value if isinstance(value, (tuple, list)) else (value,):
                    if isinstance(item, jax_core.ClosedJaxpr):
                        pending.append(item.jaxpr)
                    elif isinstance(item, jax_core.Jaxpr):
                        pending.append(item)
    return names


def test_bind_matches_refresh_on_certified_motion_and_strain_without_rebuild() -> None:
    requirements = (GRAPH, COORDINATION, PAIRS)
    prepared = _plan(requirements, skin=0.4).prepare(_system(CELL))
    positions = FRACTIONS @ CELL
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    strain = np.asarray([[0.002, 0.001, 0.0], [0.0, -0.001, 0.0005], [0.0, 0.0, 0.001]])
    strained = CELL @ (np.eye(3) + strain).T
    moved = FRACTIONS @ strained + 0.01 * np.asarray(
        [[1.0, -1.0, 0.5], [0.0, 1.0, -0.5], [-1.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )
    geometry = prepared.geometry(
        jnp.asarray(moved[None]), cell_vectors=jnp.asarray(strained[None])
    )
    bound = prepared.bind(state, geometry)
    refreshed = prepared.refresh(state, geometry)
    np.testing.assert_array_equal(refreshed.rebuilt, [[False]])
    np.testing.assert_array_equal(bound.successful, [True])
    np.testing.assert_array_equal(bound.rebuilt, [[False]])
    np.testing.assert_array_equal(bound.epochs, state.epochs)
    np.testing.assert_array_equal(bound.graph_epochs, state.graph_epochs)
    np.testing.assert_array_equal(bound.selection_changed, [False])
    _assert_same_views(prepared, bound, refreshed, requirements, moved, strained[None])
    # The bound geometry is the exact point a program evaluates.
    np.testing.assert_array_equal(bound.geometry.positions[0], moved)
    np.testing.assert_array_equal(bound.geometry.cell_vectors[0], strained)


def test_bind_reexpresses_an_exact_rewrap_as_the_same_physical_routes() -> None:
    requirements = (GRAPH, PAIRS)
    prepared = _plan(requirements, skin=0.4).prepare(_system(CELL))
    positions = FRACTIONS @ CELL
    counts = np.zeros((1, 4, 3), dtype=np.int32)
    state = prepared.initialize(
        prepared.geometry(jnp.asarray(positions[None]), image_counts=counts)
    )
    wrapped = positions.copy()
    wrapped[1] -= CELL[0]
    wrapped_counts = counts.copy()
    wrapped_counts[0, 1, 0] = 1
    geometry = prepared.geometry(
        jnp.asarray(wrapped[None]), image_counts=jnp.asarray(wrapped_counts)
    )
    bound = prepared.bind(state, geometry)
    np.testing.assert_array_equal(bound.successful, [True])
    _assert_same_views(
        prepared,
        bound,
        prepared.refresh(state, geometry),
        requirements,
        wrapped,
        CELL[None],
    )
    original = _routes(prepared.view(state, GRAPH), positions, CELL[None])
    rewrapped = _routes(prepared.view(bound, GRAPH), wrapped, CELL[None])
    assert set(original) != set(rewrapped)
    np.testing.assert_allclose(
        np.sort(np.linalg.norm(np.stack(list(original.values())), axis=-1)),
        np.sort(np.linalg.norm(np.stack(list(rewrapped.values())), axis=-1)),
        atol=1e-12,
    )


def test_bind_never_rebuilds_an_uncertified_epoch() -> None:
    prepared = _plan((GRAPH, COORDINATION), skin=0.3).prepare(_system(CELL))
    positions = FRACTIONS @ CELL
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    shrunk = 0.7 * CELL
    geometry = prepared.geometry(
        jnp.asarray((FRACTIONS @ shrunk)[None]), cell_vectors=jnp.asarray(shrunk[None])
    )
    bound = prepared.bind(state, geometry)
    np.testing.assert_array_equal(bound.stale, [True])
    np.testing.assert_array_equal(bound.successful, [False])
    np.testing.assert_array_equal(bound.capacity_failure, [False])
    np.testing.assert_array_equal(bound.capacity_only, [False])
    np.testing.assert_array_equal(bound.rebuilt, [[False]])
    np.testing.assert_array_equal(bound.epochs, state.epochs)
    np.testing.assert_array_equal(bound.graph_epochs, state.graph_epochs)
    # The ordinary lifecycle may rebuild the same geometry instead.
    np.testing.assert_array_equal(prepared.refresh(state, geometry).successful, [True])


def _selected(
    policy: ImageNeighborSelectionPolicy,
) -> tuple[PreparedAtomisticNeighborhoodBundle, AtomisticNeighborhoodRequirement]:
    requirement = Requirement("uma", Role.DIRECTED_IMAGE_GRAPH, 3.0, selection=policy)
    prepared = _plan((requirement,), skin=1.0).prepare(_system(BOX))
    return prepared, requirement


def _selection_outcome(
    prepared: PreparedAtomisticNeighborhoodBundle,
    requirement: AtomisticNeighborhoodRequirement,
    state: AtomisticNeighborhoodBundleState,
    moved: np.ndarray,
) -> bool:
    """Bind ``moved``; cross-check against native reselection; return the change."""
    geometry = prepared.geometry(jnp.asarray(moved[None]))
    bound = prepared.bind(state, geometry)
    refreshed = prepared.refresh(state, geometry)
    np.testing.assert_array_equal(refreshed.rebuilt, [[False]])
    np.testing.assert_array_equal(refreshed.successful, [True])
    changed = bool(bound.selection_changed[0])
    reselected = int(refreshed.graph_epochs[0]) != int(state.graph_epochs[0])
    assert changed == reselected
    np.testing.assert_array_equal(bound.capacity_failure, [False])
    np.testing.assert_array_equal(bound.scientific_failure, [False])
    if changed:
        # A lifecycle refusal: stale and unsuccessful, never a stale cap.
        np.testing.assert_array_equal(bound.stale, [True])
        np.testing.assert_array_equal(bound.successful, [False])
        np.testing.assert_array_equal(bound.capacity_only, [False])
    else:
        np.testing.assert_array_equal(bound.successful, [True])
        _assert_same_views(prepared, bound, refreshed, (requirement,), moved, BOX[None])
    np.testing.assert_array_equal(bound.graph_epochs, state.graph_epochs)
    return changed


def test_strict_selection_is_certified_unchanged_or_refused_when_it_changes() -> None:
    prepared, requirement = _selected(
        ImageNeighborSelectionPolicy(
            3.0,
            2,
            degeneracy_tolerance=0.0,
            measure="squared-distance",
            strict=True,
            radius_boundary="open",
        )
    )
    positions = CENTER + STAR
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    assert prepared.view(state, requirement).selected
    assert int(prepared.view(state, requirement).evidence.capped_receivers[0]) > 0
    kept = positions.copy()
    kept[2, 1] += 0.1
    assert not _selection_outcome(prepared, requirement, state, kept)
    # The third neighbor of the central atom now precedes the moved one.
    swapped = positions.copy()
    swapped[2, 1] += 0.3
    assert _selection_outcome(prepared, requirement, state, swapped)


def test_degeneracy_band_selection_is_certified_without_sorting() -> None:
    prepared, requirement = _selected(
        ImageNeighborSelectionPolicy(
            3.0,
            1,
            degeneracy_tolerance=0.25,
            measure="squared-distance",
            strict=False,
            radius_boundary="open",
        )
    )
    positions = CENTER + STAR
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    kept = positions.copy()
    kept[2, 1] += 0.1
    assert not _selection_outcome(prepared, requirement, state, kept)
    joined = positions.copy()
    joined[3, 2] -= 0.15
    assert _selection_outcome(prepared, requirement, state, joined)


def test_bind_traces_no_search_reselection_or_sort() -> None:
    prepared, _ = _selected(
        ImageNeighborSelectionPolicy(
            3.0,
            2,
            degeneracy_tolerance=0.0,
            measure="squared-distance",
            strict=True,
            radius_boundary="open",
        )
    )
    positions = CENTER + STAR
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))

    def bind(values: jax.Array) -> jax.Array:
        return prepared.bind(state, prepared.geometry(values[None])).successful

    def refresh(values: jax.Array) -> jax.Array:
        return prepared.refresh(state, prepared.geometry(values[None])).successful

    assert "sort" in _primitives(jax.make_jaxpr(refresh)(jnp.asarray(positions)).jaxpr)
    assert "sort" not in _primitives(jax.make_jaxpr(bind)(jnp.asarray(positions)).jaxpr)
    np.testing.assert_array_equal(jax.jit(bind)(jnp.asarray(positions)), [True])


def test_bundle_plan_recipe_replays_through_constructors() -> None:
    policy = ImageNeighborSelectionPolicy(
        3.0,
        2,
        degeneracy_tolerance=0.1,
        measure="distance",
        strict=False,
        radius_boundary="closed",
        coincidence_threshold=1.0e-6,
    )
    selected = Requirement("uma", Role.DIRECTED_IMAGE_GRAPH, 3.0, selection=policy)
    complete = Requirement("coulomb", Role.CLASSICAL_MINIMUM_IMAGE_PAIRS, None)
    frame = PeriodicCell(
        30.0 * np.eye(3), origin=np.full((3,), -15.0), periodic_axes=(False,) * 3
    )
    tiles = AtomisticClusterTileImageSearch(
        cluster_width=4,
        maximum_cluster_radius=1.5,
        tile_capacity=_capacity(64),
        maximum_candidate_slots=4096,
        deformation_margin=0.2,
    )
    for plan in (
        _plan((GRAPH, COORDINATION, PAIRS), skin=0.4),
        _plan((selected, complete), skin=0.5, search=tiles, frame=frame),
    ):
        recipe = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
        restored = AtomisticNeighborhoodBundlePlan.from_dict(recipe)
        assert restored.plan_id == plan.plan_id
        assert restored.to_dict() == plan.to_dict()
    forged = plan.to_dict()
    forged["plan_id"] = "0" * 64
    with pytest.raises(ValueError, match="not reproduced"):
        AtomisticNeighborhoodBundlePlan.from_dict(forged)
    extended = {**plan.to_dict(), "skin": 0.4}
    with pytest.raises(ValueError, match="not canonical"):
        AtomisticNeighborhoodBundlePlan.from_dict(extended)
