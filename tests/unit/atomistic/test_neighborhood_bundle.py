#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections import Counter
from collections.abc import Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticCompletePairSource,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticNeighborhoodSource,
    AtomisticNeighborhoodView,
    AtomisticPairCounting,
    AtomisticPairExclusions,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
    PreparedAtomisticNeighborhoodBundle,
)
from phydrax.atomistic._sites import (
    AtomisticCoordinateMapPlan,
    AtomisticInteractionSitePlan,
    AtomisticSiteDomain,
    VirtualSiteKind,
    VirtualSiteRule,
)
from phydrax.atomistic._system import PreparedAtomisticSystem
from phydrax.atomistic._topology import MolecularTopologyPlan
from phydrax.discretization import (
    ImageNeighborSelectionPolicy,
    ImageReceiverSubset,
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    ParticleImageRelation,
    PeriodicCell,
)
from phydrax.sparse import StreamedRelationPlan
from tests._support.image_lattice import lattice_routes, Route


Role = AtomisticNeighborhoodRole
Requirement = AtomisticNeighborhoodRequirement
ATOMS = AtomisticSiteDomain.PHYSICAL_ATOMS
SITES = AtomisticSiteDomain.INTERACTION_SITES
CELL = np.asarray([[6.0, 0.0, 0.0], [0.5, 6.2, 0.0], [0.3, -0.4, 6.4]])
CASE_CELLS = np.stack((CELL, CELL))
IDS = np.asarray([3, 7, 11, 20])
SITE_IDS = np.asarray([3, 7, 11, 20, 50])
FRACTIONS = np.asarray(
    [
        [0.10, 0.20, 0.30],
        [0.17, 0.24, 0.33],
        [0.55, 0.62, 0.71],
        [0.83, 0.12, 0.88],
    ]
)
PERIODIC = (True, True, True)
SEARCH = AtomisticCellListImageSearch(
    maximum_candidate_slots=10**7, deformation_margin=1.0
)
LJ = Requirement("lj", Role.CLASSICAL_MINIMUM_IMAGE_PAIRS, 2.0)


def _system(
    cell: PeriodicCell | None = None,
    *,
    finite: bool = False,
    coordinate_map: AtomisticCoordinateMapPlan | None = None,
    topology: MolecularTopologyPlan | None = None,
) -> PreparedAtomisticSystem:
    """Prepared system in ``CELL`` by default, or explicitly finite."""
    resolved = None if finite else PeriodicCell(CELL) if cell is None else cell
    return phx.atomistic.AtomisticSystemPlan(
        IDS,
        np.ones(IDS.shape, dtype=np.int32),
        np.ones(IDS.shape),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros(IDS.shape, dtype=np.int32),
        cell=resolved,
        coordinate_map=coordinate_map,
        topology=topology,
    ).prepare()


def _virtual_site_map() -> AtomisticCoordinateMapPlan:
    sites = AtomisticInteractionSitePlan(
        SITE_IDS,
        np.asarray([1, 1, 1, 1, 0]),
        np.asarray([0, 0, 0, 0, 1]),
        np.asarray([0.0, 0.0, 0.0, 0.0, -1.0]),
    )
    return AtomisticCoordinateMapPlan(
        IDS,
        sites,
        np.asarray([0, 1, 2, 3, -1]),
        virtual_rules=(
            VirtualSiteRule(
                VirtualSiteKind.WEIGHTED, 50, np.asarray([3, 7]), np.asarray([0.5, 0.5])
            ),
        ),
    )


def _virtual_site_system(
    cell: PeriodicCell | None = None, *, finite: bool = False
) -> PreparedAtomisticSystem:
    return _system(cell, finite=finite, coordinate_map=_virtual_site_map())


def _role_capacity(routes: int = 256, degree: int = 256) -> AtomisticRoleCapacity:
    return AtomisticRoleCapacity(maximum_routes=routes, maximum_degree=degree)


def _search_capacity(width: int, cases: int = 1) -> ParticleImageCapacity:
    return ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=width * cases,
        maximum_degree=width,
        maximum_images=4096,
        maximum_case_edges=width,
    )


def _source(
    name: str,
    serves: Sequence[AtomisticNeighborhoodRequirement],
    *,
    widths: tuple[int, ...] = (256,),
    cases: int = 1,
    entity: AtomisticSiteDomain = ATOMS,
    skin: float = 0.4,
    search_radius: float | None = None,
    receiver_group: str | None = None,
    receiver_rows: int | None = None,
) -> AtomisticNeighborhoodSource:
    return AtomisticNeighborhoodSource(
        name,
        entity,
        tuple(serves),
        SEARCH,
        ParticleImageCapacityLadder(
            tuple(_search_capacity(width, cases) for width in widths)
        ),
        skin=skin,
        search_radius=search_radius,
        receiver_group=receiver_group,
        receiver_rows=receiver_rows,
    )


def _plan(
    requirements: Sequence[AtomisticNeighborhoodRequirement],
    sources: Sequence[AtomisticNeighborhoodSource | AtomisticCompletePairSource],
    *,
    capacities: Mapping[str, tuple[AtomisticRoleCapacity, ...]] | None = None,
    frame: PeriodicCell | None = None,
    case_count: int = 1,
) -> AtomisticNeighborhoodBundlePlan:
    """Bind one explicit role capacity ladder per bounded requirement."""
    ladders = tuple(
        AtomisticRoleCapacityLadder(
            requirement,
            (capacities or {}).get(requirement.requirement_id, (_role_capacity(),)),
        )
        for requirement in requirements
        if not requirement.complete
    )
    return AtomisticNeighborhoodBundlePlan(
        tuple(requirements), tuple(sources), ladders, frame=frame, case_count=case_count
    )


def _canonical(routes: dict[Route, np.ndarray]) -> dict[Route, np.ndarray]:
    """Independent canonical half: one of every reversal pair."""
    kept = {}
    for key, value in routes.items():
        _, source, receiver, shift = key
        first = next((component for component in shift if component != 0), 0)
        if source < receiver or (source == receiver and first > 0):
            kept[key] = value
    return kept


def _view_routes(
    view: AtomisticNeighborhoodView, positions: ArrayLike, vectors: ArrayLike
) -> dict[Route, np.ndarray]:
    """Masked view routes with their displacements; each route stored once."""
    flat = np.asarray(positions).reshape((-1, 3))
    displacement = np.asarray(view.displacement(jnp.asarray(flat), jnp.asarray(vectors)))
    relation = view.relation
    mask = np.asarray(view.mask)
    keys = [
        (int(case), int(source), int(receiver), tuple(int(value) for value in shift))
        for case, source, receiver, shift in zip(
            np.asarray(relation.route_cases)[mask],
            np.asarray(relation.source_particle_ids)[mask],
            np.asarray(relation.receiver_particle_ids)[mask],
            np.asarray(relation.image_shifts)[mask],
            strict=True,
        )
    ]
    assert all(count == 1 for count in Counter(keys).values()), "duplicated route"
    return dict(zip(keys, displacement[mask], strict=True))


def _assert_routes(
    actual: dict[Route, np.ndarray], expected: dict[Route, np.ndarray]
) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=1e-10)


def _oracle(
    positions: ArrayLike,
    vectors: ArrayLike,
    cutoff: float | None,
    *,
    ids: np.ndarray = IDS,
    periodic: tuple[bool, ...] = PERIODIC,
    case: int = 0,
    active: np.ndarray | None = None,
    receivers: set[int] | None = None,
) -> dict[Route, np.ndarray]:
    assert cutoff is not None
    return lattice_routes(
        np.asarray(positions),
        np.asarray(vectors),
        periodic,
        cutoff,
        ids,
        case=case,
        extent=3,
        active=active,
        receivers=receivers,
    )


def _source_index(prepared: PreparedAtomisticNeighborhoodBundle, name: str) -> int:
    return [proof.source for proof in prepared.proofs].index(name)


def _leaves_match(first: object, second: object) -> None:
    left, right = jax.tree.leaves(first), jax.tree.leaves(second)
    assert len(left) == len(right)
    for one, two in zip(left, right, strict=True):
        one, two = np.asarray(one), np.asarray(two)
        if np.issubdtype(one.dtype, np.floating):
            np.testing.assert_allclose(one, two, rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_array_equal(one, two)


MACE = Requirement("mace", Role.DIRECTED_IMAGE_GRAPH, 6.5)
EWALD = Requirement("ewald", Role.IMAGE_PAIR_SUM, 6.5)
D3 = Requirement("d3", Role.COORDINATION, 5.0)


def _mixed_plan() -> AtomisticNeighborhoodBundlePlan:
    return _plan(
        (LJ, MACE, EWALD, D3),
        (_source("pairs", (LJ,)), _source("images", (MACE, EWALD, D3))),
    )


def test_mixed_pair_and_image_resources_are_independently_valid() -> None:
    plan = _mixed_plan()
    prepared = plan.prepare(_system())
    positions = FRACTIONS @ CELL
    vectors = CELL[None]
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    np.testing.assert_array_equal(state.successful, [True])
    np.testing.assert_array_equal(state.capacity_failure, [False])
    np.testing.assert_array_equal(state.scientific_failure, [False])

    pairs = prepared.view(state, LJ)
    assert pairs.requirement_id == LJ.requirement_id
    expected_pairs = _canonical(_oracle(positions, CELL, 2.0))
    assert expected_pairs
    assert pairs.counting is AtomisticPairCounting.MINIMUM_IMAGE_HALF
    assert pairs.directed_multiplicity == 2
    _assert_routes(_view_routes(pairs, positions, vectors), expected_pairs)
    np.testing.assert_array_equal(pairs.evidence.repeated_minimum_image, [False])
    assert pairs.pair_relation is not None
    valid = np.asarray(pairs.pair_relation.valid)
    assert np.all(
        np.asarray(pairs.pair_relation.left_particle_ids)[valid]
        < np.asarray(pairs.pair_relation.right_particle_ids)[valid]
    )

    graph = prepared.view(state, MACE)
    expected_graph = _oracle(positions, CELL, 6.5)
    # The graph cutoff exceeds the shortest lattice vector: repeated and
    # nonzero self images are explicit routes, not a minimum-image relation.
    assert any(source == receiver for _, source, receiver, _ in expected_graph)
    _assert_routes(_view_routes(graph, positions, vectors), expected_graph)
    assert graph.directed_multiplicity == 1

    pair_sum = prepared.view(state, EWALD)
    actual_sum = _view_routes(pair_sum, positions, vectors)
    _assert_routes(actual_sum, _canonical(expected_graph))
    self_image = np.asarray(pair_sum.self_image)[np.asarray(pair_sum.mask)]
    np.testing.assert_array_equal(self_image, [key[1] == key[2] for key in actual_sum])
    assert np.any(self_image)
    directed = sum(np.exp(-np.linalg.norm(row)) for row in expected_graph.values())
    half = sum(np.exp(-np.linalg.norm(row)) for row in actual_sum.values())
    np.testing.assert_allclose(pair_sum.directed_multiplicity * half, directed)

    _assert_routes(
        _view_routes(prepared.view(state, D3), positions, vectors),
        _oracle(positions, CELL, 5.0),
    )

    images = prepared.proofs[_source_index(prepared, "images")]
    assert images.shared and len(images.served) == 3
    assert set(images.served) == {
        MACE.requirement_id,
        EWALD.requirement_id,
        D3.requirement_id,
    }
    assert all(margin is not None and margin >= 0.0 for margin in images.coverage_margins)
    assert images.search_radius is not None
    np.testing.assert_allclose(images.search_radius, 6.9)
    assert not prepared.proofs[_source_index(prepared, "pairs")].shared


def test_same_role_requirements_keep_distinct_identities() -> None:
    inner = Requirement("mace", Role.DIRECTED_IMAGE_GRAPH, 4.0)
    plan = _plan((inner, MACE), (_source("images", (inner, MACE)),))
    assert inner.requirement_id != MACE.requirement_id
    assert plan.slot(inner) != plan.slot(MACE)
    prepared = plan.prepare(_system())
    positions = FRACTIONS @ CELL
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    for requirement in (inner, MACE):
        view = prepared.view(state, requirement)
        assert view.requirement_id == requirement.requirement_id
        _assert_routes(
            _view_routes(view, positions, CELL[None]),
            _oracle(positions, CELL, requirement.cutoff),
        )
    with pytest.raises(KeyError):
        plan.slot(Requirement("mace", Role.DIRECTED_IMAGE_GRAPH, 5.0))
    # Storage is bound by the plan; the scientific identity never changes.
    grown = _plan(
        (inner, MACE),
        (_source("images", (inner, MACE)),),
        capacities={MACE.requirement_id: (_role_capacity(64), _role_capacity(512))},
    )
    assert grown.plan_id != plan.plan_id
    assert [value.requirement_id for value in grown.requirements] == [
        value.requirement_id for value in plan.requirements
    ]


def test_finite_system_searches_its_explicit_frame() -> None:
    frame = PeriodicCell(10.0 * np.eye(3), periodic_axes=(False, False, False))
    system = phx.atomistic.AtomisticSystemPlan(
        IDS,
        np.ones(IDS.shape, dtype=np.int32),
        np.ones(IDS.shape),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros(IDS.shape, dtype=np.int32),
    ).prepare()
    sources = (_source("pairs", (LJ,)),)
    with pytest.raises(ValueError, match="frame"):
        _plan((LJ,), sources).prepare(system)
    plan = _plan((LJ,), sources, frame=frame)
    prepared = plan.prepare(system)
    positions = np.asarray(
        [[4.0, 4.0, 4.0], [5.2, 4.3, 4.1], [6.0, 5.4, 4.5], [8.5, 8.5, 8.5]]
    )
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    np.testing.assert_array_equal(state.successful, [True])
    expected = _canonical(
        _oracle(positions, frame.vectors, 2.0, periodic=(False, False, False))
    )
    assert len(expected) == 2
    _assert_routes(_view_routes(state.views[0], positions, frame.vectors[None]), expected)
    with pytest.raises(ValueError, match="own prepared cell"):
        plan.prepare(_system())


def test_sharing_refusal_and_canonical_identity() -> None:
    plan = _mixed_plan()
    shuffled = _plan(
        (D3, EWALD, MACE, LJ),
        (_source("images", (D3, EWALD, MACE)), _source("pairs", (LJ,))),
    )
    assert shuffled.plan_id == plan.plan_id
    assert shuffled.slot(EWALD) == plan.slot(EWALD)

    # A receiver-row subset is not reversal closed: no pair-once view.
    with pytest.raises(ValueError, match="reversal-closed"):
        _plan((LJ,), (_source("rows", (LJ,), receiver_group="probe", receiver_rows=2),))
    # Coverage of cutoff + skin is required, not merely an equal cutoff.
    with pytest.raises(ValueError, match="does not cover"):
        _plan((LJ,), (_source("pairs", (LJ,), search_radius=2.1),))
    # An interaction-site source cannot serve physical-atom roles.
    with pytest.raises(ValueError, match="requires physical-atoms"):
        _plan((LJ,), (_source("s", (LJ,), entity=SITES),))
    with pytest.raises(ValueError, match="not served"):
        _plan((LJ, D3), (_source("pairs", (LJ,)),))
    with pytest.raises(ValueError, match="two sources"):
        _plan((LJ,), (_source("a", (LJ,)), _source("b", (LJ,))))
    with pytest.raises(ValueError, match="no role capacity ladder"):
        AtomisticNeighborhoodBundlePlan((LJ,), (_source("pairs", (LJ,)),), ())
    probe = Requirement("d3", Role.COORDINATION, 3.0, receiver_group="probe")
    with pytest.raises(ValueError, match="different receiver"):
        _plan((probe,), (_source("rows", (probe,)),))
    with pytest.raises(ValueError, match="directed roles"):
        Requirement("ewald", Role.IMAGE_PAIR_SUM, 3.0, receiver_group="probe")
    policy = ImageNeighborSelectionPolicy(
        3.0,
        4,
        degeneracy_tolerance=0.0,
        measure="squared-distance",
        strict=True,
        radius_boundary="open",
    )
    with pytest.raises(ValueError, match="directed image graph"):
        Requirement("d3", Role.COORDINATION, 3.0, selection=policy)

    # A classical minimum-image cutoff beyond the unique-image radius is an
    # invalid classical image relation; explicit image sums are the remedy.
    wide = Requirement("lj", Role.CLASSICAL_MINIMUM_IMAGE_PAIRS, 3.5)
    with pytest.raises(ValueError, match="unique-image"):
        _plan((wide,), (_source("pairs", (wide,)),)).prepare(_system())

    # Interaction-site pairs may reuse the atom search only under a proven
    # identity map; an independent site source is always admissible.
    tip = Requirement("tip", Role.INTERACTION_SITE_PAIRS, 2.0)
    aliased_plan = _plan((tip,), (_source("atoms", (tip,)),))
    with pytest.raises(ValueError, match="proven identity image"):
        aliased_plan.prepare(_virtual_site_system())
    independent = _plan((tip,), (_source("sites", (tip,), entity=SITES),))
    assert not independent.prepare(_virtual_site_system()).proofs[0].site_alias
    aliased = aliased_plan.prepare(_system())
    assert aliased.proofs[0].site_alias
    positions = FRACTIONS @ CELL
    state = aliased.initialize(aliased.geometry(jnp.asarray(positions[None])))
    _assert_routes(
        _view_routes(state.views[0], positions, CELL[None]),
        _canonical(_oracle(positions, CELL, 2.0)),
    )


GRAPH4 = Requirement("mace", Role.DIRECTED_IMAGE_GRAPH, 4.0)


def _two_source_plan(
    pair_skin: float, image_skin: float
) -> AtomisticNeighborhoodBundlePlan:
    return _plan(
        (LJ, GRAPH4),
        (
            _source("pairs", (LJ,), skin=pair_skin),
            _source("images", (GRAPH4,), skin=image_skin),
        ),
    )


def test_roles_rebuild_independently_and_refresh_traces() -> None:
    prepared = _two_source_plan(pair_skin=0.2, image_skin=1.0).prepare(_system())
    pairs = _source_index(prepared, "pairs")
    images = _source_index(prepared, "images")
    positions = FRACTIONS @ CELL
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    moved = positions + np.asarray([0.15, 0.0, 0.0])
    geometry = prepared.geometry(jnp.asarray(moved[None]))
    certificates = prepared.certify(state, geometry)
    pair_certificate, image_certificate = certificates[pairs], certificates[images]
    assert pair_certificate is not None and image_certificate is not None
    np.testing.assert_array_equal(pair_certificate.certified, [False])
    np.testing.assert_array_equal(pair_certificate.motion_exceeded, [True])
    np.testing.assert_array_equal(image_certificate.certified, [True])

    refreshed = prepared.refresh(state, geometry)
    np.testing.assert_array_equal(refreshed.successful, [True])
    np.testing.assert_array_equal(refreshed.rebuilt[pairs], [True])
    np.testing.assert_array_equal(refreshed.rebuilt[images], [False])
    np.testing.assert_array_equal(refreshed.epochs[pairs], [1])
    np.testing.assert_array_equal(refreshed.epochs[images], [0])
    before_source, after_source = state.sources[images], refreshed.sources[images]
    assert isinstance(before_source, phx.discretization.SegmentedImageVerletState)
    assert isinstance(after_source, phx.discretization.SegmentedImageVerletState)
    before = before_source.reference.relation
    after = after_source.reference.relation
    for one, two in (
        (before.source_indices, after.source_indices),
        (before.receiver_indices, after.receiver_indices),
        (before.image_shifts, after.image_shifts),
        (before.valid, after.valid),
    ):
        np.testing.assert_array_equal(one, two)
    _assert_routes(
        _view_routes(prepared.view(refreshed, LJ), moved, CELL[None]),
        _canonical(_oracle(moved, CELL, 2.0)),
    )
    _assert_routes(
        _view_routes(prepared.view(refreshed, GRAPH4), moved, CELL[None]),
        _oracle(moved, CELL, 4.0),
    )
    compiled = jax.jit(lambda previous, current: prepared.refresh(previous, current))(
        state, geometry
    )
    _leaves_match(refreshed, compiled)


def test_exact_wrapping_reexpresses_image_shifts_without_rebuild() -> None:
    prepared = _two_source_plan(pair_skin=0.4, image_skin=0.4).prepare(_system())
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
    assert all(
        value is not None and bool(value.certified[0])
        for value in prepared.certify(state, geometry)
    )
    refreshed = prepared.refresh(state, geometry)
    np.testing.assert_array_equal(refreshed.rebuilt, [[False], [False]])
    np.testing.assert_array_equal(refreshed.epochs, [[0], [0]])
    np.testing.assert_array_equal(refreshed.successful, [True])
    for requirement, half in ((LJ, True), (GRAPH4, False)):
        expected = _oracle(wrapped, CELL, requirement.cutoff)
        actual = _view_routes(prepared.view(refreshed, requirement), wrapped, CELL[None])
        _assert_routes(actual, _canonical(expected) if half else expected)
        original = _view_routes(prepared.view(state, requirement), positions, CELL[None])
        # The same physical routes: identical displacements, shifts moved by
        # exactly the receiver-minus-source rewrap count of particle ID 7.
        for (case, source, receiver, shift), value in original.items():
            change = (receiver == 7) - (source == 7)
            moved = (case, source, receiver, (shift[0] + change,) + shift[1:])
            np.testing.assert_allclose(actual[moved], value, atol=1e-10)


def test_deformed_cell_certificate_covers_absent_images() -> None:
    graph = Requirement("mace", Role.DIRECTED_IMAGE_GRAPH, 3.0)
    coordination = Requirement("d3", Role.COORDINATION, 2.5)
    plan = _plan(
        (graph, coordination), (_source("images", (graph, coordination), skin=0.3),)
    )
    prepared = plan.prepare(_system())
    positions = FRACTIONS @ CELL
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))

    strain = np.asarray(
        [[0.002, 0.001, 0.0], [0.0, -0.001, 0.0005], [0.0003, 0.0, 0.0015]]
    )
    small = CELL @ (np.eye(3) + strain).T
    small_positions = FRACTIONS @ small
    retained = prepared.refresh(
        state,
        prepared.geometry(
            jnp.asarray(small_positions[None]), cell_vectors=jnp.asarray(small[None])
        ),
    )
    np.testing.assert_array_equal(retained.rebuilt, [[False]])
    np.testing.assert_array_equal(retained.successful, [True])
    for requirement in (graph, coordination):
        _assert_routes(
            _view_routes(
                prepared.view(retained, requirement), small_positions, small[None]
            ),
            _oracle(small_positions, small, requirement.cutoff),
        )

    shrunk = 0.7 * CELL
    shrunk_positions = FRACTIONS @ shrunk
    geometry = prepared.geometry(
        jnp.asarray(shrunk_positions[None]), cell_vectors=jnp.asarray(shrunk[None])
    )
    expected = _oracle(shrunk_positions, shrunk, 3.0)
    source = state.sources[0]
    assert isinstance(source, phx.discretization.SegmentedImageVerletState)
    stored = source.reference.relation
    stored_keys = {
        (int(source), int(receiver), tuple(int(value) for value in shift))
        for source, receiver, shift, valid in zip(
            np.asarray(stored.source_particle_ids),
            np.asarray(stored.receiver_particle_ids),
            np.asarray(stored.image_shifts),
            np.asarray(stored.valid),
            strict=True,
        )
        if valid
    }
    # The deformation brings images within the cutoff that the epoch never stored.
    assert {key[1:] for key in expected} - stored_keys
    certificate = prepared.certify(state, geometry)[0]
    assert certificate is not None
    np.testing.assert_array_equal(certificate.certified, [False])
    np.testing.assert_array_equal(certificate.motion_exceeded, [True])
    rebuilt = prepared.refresh(state, geometry)
    np.testing.assert_array_equal(rebuilt.rebuilt, [[True]])
    np.testing.assert_array_equal(rebuilt.successful, [True])
    _assert_routes(
        _view_routes(prepared.view(rebuilt, graph), shrunk_positions, shrunk[None]),
        expected,
    )
    # Without a permitted rebuild the uncertified epoch is stale, never reused.
    stale = prepared.refresh(state, geometry, allow_rebuild=False)
    np.testing.assert_array_equal(stale.stale, [True])
    np.testing.assert_array_equal(stale.successful, [False])
    np.testing.assert_array_equal(stale.capacity_failure, [False])


SELECTED = Requirement(
    "uma",
    Role.DIRECTED_IMAGE_GRAPH,
    5.5,
    selection=ImageNeighborSelectionPolicy(
        5.5,
        3,
        degeneracy_tolerance=0.0,
        measure="squared-distance",
        strict=True,
        radius_boundary="open",
    ),
)


def _selection_plan(degree: int) -> AtomisticNeighborhoodBundlePlan:
    return _plan(
        (SELECTED,),
        (_source("images", (SELECTED,)),),
        capacities={SELECTED.requirement_id: (_role_capacity(256, degree),)},
    )


def test_selected_graph_is_scientific_selection_not_overflow() -> None:
    positions = FRACTIONS @ CELL
    candidates = _oracle(positions, CELL, 5.5)
    by_receiver: dict[int, list] = {}
    for key, value in candidates.items():
        by_receiver.setdefault(key[2], []).append((np.linalg.norm(value), key))
    assert any(len(rows) > 3 for rows in by_receiver.values())
    expected = {}
    for rows in by_receiver.values():
        for _, key in sorted(rows)[:3]:
            expected[key] = candidates[key]

    prepared = _selection_plan(degree=64).prepare(_system())
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    view = prepared.view(state, SELECTED)
    assert view.selected
    evidence = view.evidence
    np.testing.assert_array_equal(state.successful, [True])
    assert int(evidence.capped_receivers[0]) > 0
    np.testing.assert_array_equal(evidence.capacity_failure, [False])
    np.testing.assert_array_equal(evidence.scientific_failure, [False])
    _assert_routes(_view_routes(view, positions, CELL[None]), expected)

    narrow = _selection_plan(degree=1).prepare(_system())
    overflow = narrow.initialize(narrow.geometry(jnp.asarray(positions[None])))
    evidence = overflow.views[0].evidence
    np.testing.assert_array_equal(evidence.degree_overflow, [True])
    np.testing.assert_array_equal(overflow.capacity_only, [True])
    np.testing.assert_array_equal(overflow.scientific_failure, [False])


def test_receiver_selected_sources_keep_declared_rows() -> None:
    probe = Requirement("cn", Role.COORDINATION, 3.0, receiver_group="probe")
    prepared = _plan(
        (probe,),
        (_source("rows", (probe,), receiver_group="probe", receiver_rows=2),),
    ).prepare(_system())
    positions = FRACTIONS @ CELL
    geometry = prepared.geometry(jnp.asarray(positions[None]))
    with pytest.raises(ValueError, match="Receiver subsets"):
        prepared.initialize(geometry)
    receivers = {"probe": ImageReceiverSubset(jnp.asarray([[7, 20]]))}
    state = prepared.initialize(geometry, receivers=receivers)
    view = prepared.view(state, probe)
    _assert_routes(
        _view_routes(view, positions, CELL[None]),
        _oracle(positions, CELL, 3.0, receivers={7, 20}),
    )
    mask = np.asarray(view.mask)
    rows = view.receiver_rows
    assert rows is not None
    routed = np.asarray(rows.particle_indices)[np.asarray(rows.route_rows)[mask]]
    np.testing.assert_array_equal(
        routed, np.asarray(view.relation.receiver_indices)[mask]
    )


def _replay_fixture() -> tuple[
    PreparedAtomisticNeighborhoodBundle, np.ndarray, np.ndarray, np.ndarray
]:
    cell = PeriodicCell(6.0 * np.eye(3))
    prepared = _plan(
        (LJ,),
        (_source("pairs", (LJ,), widths=(4, 64)),),
        capacities={LJ.requirement_id: (_role_capacity(1, 4), _role_capacity(16, 16))},
    ).prepare(_system(cell))
    spread = np.asarray(
        [[3.0, 3.0, 3.0], [4.0, 3.0, 3.0], [0.6, 0.6, 0.6], [0.6, 3.0, 5.4]]
    )
    clustered = np.asarray(
        [[3.0, 3.0, 3.0], [4.0, 3.0, 3.0], [3.0, 4.0, 3.0], [3.0, 3.0, 4.0]]
    )
    return prepared, np.asarray(cell.vectors), spread, clustered


def test_source_overflow_is_capacity_and_domain_failure_is_scientific() -> None:
    prepared, _, _, clustered = _replay_fixture()
    overflow = prepared.initialize(prepared.geometry(jnp.asarray(clustered[None])))
    evidence = overflow.views[0].evidence
    np.testing.assert_array_equal(evidence.source_capacity_failure, [True])
    np.testing.assert_array_equal(overflow.scientific_failure, [False])
    np.testing.assert_array_equal(overflow.capacity_only, [True])
    assert prepared.grow_capacities(overflow).source_levels == (1,)

    invalid = clustered.copy()
    invalid[2, 1] = np.nan
    failed = prepared.initialize(prepared.geometry(jnp.asarray(invalid[None])))
    np.testing.assert_array_equal(failed.scientific_failure, [True])
    np.testing.assert_array_equal(failed.capacity_only, [False])
    with pytest.raises(ValueError, match="not capacity-only"):
        prepared.grow_capacities(failed)


def test_minimum_image_pairs_refuse_repeated_images_after_deformation() -> None:
    reference = 6.0 * np.eye(3)
    prepared = _plan((LJ,), (_source("pairs", (LJ,)),)).prepare(
        _system(PeriodicCell(reference))
    )
    fractions = np.asarray(
        [[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.1, 0.6, 0.6], [0.6, 0.6, 0.6]]
    )
    state = prepared.initialize(
        prepared.geometry(jnp.asarray((fractions @ reference)[None]))
    )
    np.testing.assert_array_equal(state.successful, [True])
    compressed = 3.5 * np.eye(3)
    candidate = prepared.refresh(
        state,
        prepared.geometry(
            jnp.asarray((fractions @ compressed)[None]),
            cell_vectors=jnp.asarray(compressed[None]),
        ),
    )
    evidence = candidate.views[0].evidence
    # IDs 3 and 7 are 1.75 apart through two images: a minimum-image pair
    # term would silently drop one; the view refuses it as scientific.
    np.testing.assert_array_equal(evidence.repeated_minimum_image, [True])
    np.testing.assert_array_equal(evidence.scientific_failure, [True])
    np.testing.assert_array_equal(evidence.capacity_failure, [False])
    with pytest.raises(ValueError, match="not capacity-only"):
        prepared.grow_capacities(candidate)


def test_rejected_candidate_leaves_accepted_state_unchanged() -> None:
    prepared = _two_source_plan(pair_skin=0.4, image_skin=0.4).prepare(_system())
    positions = FRACTIONS @ CELL
    accepted = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    snapshot = [np.array(leaf) for leaf in jax.tree.leaves(accepted)]
    invalid = positions + 1.0
    invalid[0, 0] = np.nan
    candidate = prepared.refresh(accepted, prepared.geometry(jnp.asarray(invalid[None])))
    np.testing.assert_array_equal(candidate.scientific_failure, [True])
    for before, after in zip(snapshot, jax.tree.leaves(accepted), strict=True):
        np.testing.assert_array_equal(before, np.asarray(after))
    kept = prepared.select_cases(jnp.asarray([False]), candidate, accepted)
    np.testing.assert_array_equal(kept.successful, [True])
    _leaves_match(kept, accepted)
    committed = prepared.select_cases(jnp.asarray([True]), candidate, accepted)
    np.testing.assert_array_equal(committed.successful, [False])


def test_capacity_replay_from_accepted_state_keeps_epochs_and_routes() -> None:
    prepared, vectors, spread, clustered = _replay_fixture()
    accepted = prepared.initialize(prepared.geometry(jnp.asarray(spread[None])))
    np.testing.assert_array_equal(accepted.successful, [True])
    accepted_routes = _view_routes(accepted.views[0], spread, vectors[None])
    assert len(accepted_routes) == 1
    with pytest.raises(ValueError, match="no capacity failure"):
        prepared.grow_capacities(accepted)
    attempt = prepared.geometry(jnp.asarray(clustered[None]))

    # Source storage overflows first; only the source ladder advances.
    candidate = prepared.refresh(accepted, attempt)
    np.testing.assert_array_equal(candidate.capacity_only, [True])
    grown = prepared.grow_capacities(candidate)
    assert grown.source_levels == (1,) and grown.role_levels == (0,)
    rebased = grown.rebase(accepted)
    np.testing.assert_array_equal(rebased.successful, [True])
    np.testing.assert_array_equal(rebased.epochs, accepted.epochs)
    _assert_routes(_view_routes(rebased.views[0], spread, vectors[None]), accepted_routes)
    assert rebased.views[0].requirement_id == LJ.requirement_id

    # With complete source routes the role storage is the remaining failure.
    replay = grown.refresh(rebased, attempt)
    np.testing.assert_array_equal(replay.views[0].evidence.route_overflow, [True])
    np.testing.assert_array_equal(replay.capacity_only, [True])
    final_bundle = grown.grow_capacities(replay)
    assert final_bundle.source_levels == (1,) and final_bundle.role_levels == (1,)
    final = final_bundle.refresh(final_bundle.rebase(accepted), attempt)
    np.testing.assert_array_equal(final.successful, [True])
    np.testing.assert_array_equal(final.epochs, accepted.epochs + 1)
    expected = _canonical(_oracle(clustered, vectors, 2.0))
    assert len(expected) == 6
    _assert_routes(_view_routes(final.views[0], clustered, vectors[None]), expected)
    with pytest.raises(ValueError, match="another prepared bundle"):
        final_bundle.refresh(accepted, attempt)


def test_uninitialized_cases_and_interaction_site_entities() -> None:
    system = _virtual_site_system()
    tip = Requirement("tip", Role.INTERACTION_SITE_PAIRS, 2.0)
    plan = _plan(
        (LJ, tip),
        (
            _source("atoms", (LJ,), cases=2),
            _source("sites", (tip,), cases=2, entity=SITES),
        ),
        case_count=2,
    )
    prepared = plan.prepare(system)
    first = FRACTIONS @ CELL
    positions = np.stack((first, first + np.asarray([0.3, -0.2, 0.1])))
    sites = jax.vmap(
        lambda value: system.coordinate_map.realize(value, cell=system.cell)
    )(jnp.asarray(positions))
    with pytest.raises(ValueError, match="realized interaction sites"):
        prepared.geometry(jnp.asarray(positions))
    geometry = prepared.geometry(jnp.asarray(positions), sites=sites)
    state = prepared.initialize(geometry, cases=jnp.asarray([True, False]))
    np.testing.assert_array_equal(state.successful, [True, False])
    np.testing.assert_array_equal(state.uninitialized, [False, True])
    np.testing.assert_array_equal(state.capacity_failure, [False, False])
    np.testing.assert_array_equal(state.scientific_failure, [False, False])

    site_positions = np.asarray(sites.positions)
    expected = _canonical(_oracle(site_positions[0], CELL, 2.0, ids=SITE_IDS, case=0))
    assert any(50 in key[1:3] for key in expected)
    _assert_routes(
        _view_routes(prepared.view(state, tip), site_positions, CASE_CELLS), expected
    )

    refreshed = prepared.refresh(state, geometry)
    np.testing.assert_array_equal(refreshed.successful, [True, True])
    np.testing.assert_array_equal(refreshed.rebuilt, [[False, True], [False, True]])
    np.testing.assert_array_equal(refreshed.epochs, [[0, 1], [0, 1]])
    both = {}
    for case in range(2):
        both |= _canonical(
            _oracle(site_positions[case], CELL, 2.0, ids=SITE_IDS, case=case)
        )
    _assert_routes(
        _view_routes(prepared.view(refreshed, tip), site_positions, CASE_CELLS), both
    )

    failed_sites = eqx.tree_at(
        lambda value: value.successful, sites, jnp.asarray([True, False])
    )
    failed = prepared.refresh(
        refreshed, prepared.geometry(jnp.asarray(positions), sites=failed_sites)
    )
    site_evidence = prepared.view(failed, tip).evidence
    np.testing.assert_array_equal(site_evidence.entity_failure, [False, True])
    np.testing.assert_array_equal(site_evidence.successful, [True, False])
    np.testing.assert_array_equal(
        prepared.view(failed, LJ).evidence.successful, [True, True]
    )


COULOMB = Requirement("coulomb", Role.CLASSICAL_MINIMUM_IMAGE_PAIRS, None)
FAR = np.asarray(
    [[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [0.0, 55.0, 0.0], [-30.0, 20.0, 70.0]]
)


def _complete_routes(
    positions: np.ndarray, ids: np.ndarray, active: np.ndarray | None = None
) -> dict[Route, np.ndarray]:
    """Independent complete half: every active pair once, smaller ID as source."""
    mask = np.ones(ids.shape, dtype=np.bool_) if active is None else np.asarray(active)
    routes = {}
    for first in np.flatnonzero(mask):
        for second in np.flatnonzero(mask):
            if ids[first] < ids[second]:
                key = (0, int(ids[first]), int(ids[second]), ())
                routes[key] = positions[second] - positions[first]
    return routes


def test_complete_pairs_cover_every_finite_pair_without_cutoff() -> None:
    assert COULOMB.complete
    assert COULOMB.counting is AtomisticPairCounting.COMPLETE_PAIRS
    plan = _plan(
        (COULOMB,),
        (AtomisticCompletePairSource("all", ATOMS, (COULOMB,), maximum_pairs=6),),
    )
    prepared = plan.prepare(_system(finite=True))
    assert prepared.lattice_rank == 0
    geometry = prepared.geometry(jnp.asarray(FAR[None]))
    assert geometry.cell_vectors.shape == (1, 0, 3)
    state = prepared.initialize(geometry)
    np.testing.assert_array_equal(state.successful, [True])
    view = prepared.view(state, COULOMB)
    assert view.cutoff is None and view.directed_multiplicity == 2
    vectors = np.zeros((1, 0, 3))
    # Separations far beyond any finite radius are all present: no cutoff.
    _assert_routes(_view_routes(view, FAR, vectors), _complete_routes(FAR, IDS))
    assert view.pair_relation is not None
    valid = np.asarray(view.pair_relation.valid)
    assert int(np.count_nonzero(valid)) == 6
    np.testing.assert_array_equal(view.evidence.required_routes, [6])
    assert prepared.certify(state, geometry) == (None,)

    # Motion keeps the complete topology; the active set alone changes it.
    moved = prepared.refresh(state, prepared.geometry(jnp.asarray(FAR[None] + 3.0)))
    np.testing.assert_array_equal(moved.epochs, [[0]])
    np.testing.assert_array_equal(moved.graph_epochs, state.graph_epochs)
    active = np.asarray([True, False, True, True])
    reduced = prepared.refresh(
        state, prepared.geometry(jnp.asarray(FAR[None]), active=jnp.asarray(active[None]))
    )
    np.testing.assert_array_equal(reduced.epochs, [[1]])
    np.testing.assert_array_equal(reduced.graph_epochs, state.graph_epochs + 1)
    _assert_routes(
        _view_routes(prepared.view(reduced, COULOMB), FAR, vectors),
        _complete_routes(FAR, IDS, active),
    )
    stale = prepared.refresh(
        state,
        prepared.geometry(jnp.asarray(FAR[None]), active=jnp.asarray(active[None])),
        allow_rebuild=False,
    )
    np.testing.assert_array_equal(stale.stale, [True])
    np.testing.assert_array_equal(stale.successful, [False])
    invalid = FAR.copy()
    invalid[2, 0] = np.nan
    failed = prepared.refresh(state, prepared.geometry(jnp.asarray(invalid[None])))
    np.testing.assert_array_equal(failed.scientific_failure, [True])
    np.testing.assert_array_equal(failed.capacity_failure, [False])


def test_complete_pairs_are_bounded_finite_and_explicitly_bound() -> None:
    source = AtomisticCompletePairSource("all", ATOMS, (COULOMB,), maximum_pairs=5)
    with pytest.raises(ValueError, match="maximum_pairs"):
        _plan((COULOMB,), (source,)).prepare(_system(finite=True))
    exact = AtomisticCompletePairSource("all", ATOMS, (COULOMB,), maximum_pairs=6)
    with pytest.raises(ValueError, match="unbounded periodic support"):
        _plan((COULOMB,), (exact,)).prepare(_system())
    with pytest.raises(ValueError, match="not a role capacity ladder"):
        AtomisticNeighborhoodBundlePlan(
            (COULOMB,),
            (exact,),
            (AtomisticRoleCapacityLadder(COULOMB, (_role_capacity(),)),),
        )
    with pytest.raises(ValueError, match="AtomisticCompletePairSource"):
        _source("images", (COULOMB,))
    with pytest.raises(ValueError, match="only cutoff=None"):
        AtomisticCompletePairSource("all", ATOMS, (LJ,), maximum_pairs=6)
    with pytest.raises(ValueError, match="pair roles"):
        Requirement("mace", Role.DIRECTED_IMAGE_GRAPH, None)
    assert COULOMB.requirement_id != LJ.requirement_id


def test_complete_site_pairs_with_bounded_atoms_in_a_frame() -> None:
    frame = PeriodicCell(200.0 * np.eye(3), periodic_axes=(False, False, False))
    system = _virtual_site_system(finite=True)
    sites_all = Requirement("tip", Role.INTERACTION_SITE_PAIRS, None)
    near = Requirement("lj", Role.CLASSICAL_MINIMUM_IMAGE_PAIRS, 45.0)
    plan = _plan(
        (sites_all, near),
        (
            AtomisticCompletePairSource("sites", SITES, (sites_all,), maximum_pairs=10),
            _source("pairs", (near,)),
        ),
        frame=frame,
    )
    prepared = plan.prepare(system)
    positions = FAR + 50.0
    sites = jax.vmap(system.coordinate_map.realize)(jnp.asarray(positions[None]))
    state = prepared.initialize(
        prepared.geometry(jnp.asarray(positions[None]), sites=sites)
    )
    np.testing.assert_array_equal(state.successful, [True])
    vectors = np.asarray(frame.vectors)[None]
    site_positions = np.asarray(sites.positions)[0]
    site_view = prepared.view(state, sites_all)
    assert site_view.entity is SITES
    complete = _complete_routes(site_positions, SITE_IDS)
    assert len(complete) == 10
    _assert_routes(
        {
            key[:3] + ((),): value
            for key, value in _view_routes(site_view, site_positions, vectors).items()
        },
        complete,
    )
    np.testing.assert_array_equal(np.asarray(site_view.relation.image_shifts), 0)
    expected = _canonical(
        _oracle(positions, frame.vectors, 45.0, periodic=(False, False, False))
    )
    assert 0 < len(expected) < 6
    _assert_routes(_view_routes(prepared.view(state, near), positions, vectors), expected)


GRAPH_PLAN = StreamedRelationPlan(receiver_tile=4, edge_tile=16)


def _scheduled_bundle() -> tuple[
    AtomisticNeighborhoodBundlePlan, PreparedAtomisticNeighborhoodBundle
]:
    plan = _two_source_plan(pair_skin=0.2, image_skin=1.0)
    return plan, plan.prepare(_system(), graph_plans={GRAPH4.requirement_id: GRAPH_PLAN})


def _topology(relation: ParticleImageRelation) -> tuple[np.ndarray, ...]:
    valid = np.asarray(relation.valid)
    return (
        valid,
        np.where(valid, np.asarray(relation.source_indices), -1),
        np.where(valid, np.asarray(relation.receiver_indices), -1),
    )


def _same(first: ParticleImageRelation, second: ParticleImageRelation) -> bool:
    return all(
        np.array_equal(one, two)
        for one, two in zip(_topology(first), _topology(second), strict=True)
    )


def test_graph_schedules_follow_role_topology_revisions() -> None:
    plan, prepared = _scheduled_bundle()
    with pytest.raises(ValueError, match="not a directed role"):
        plan.prepare(_system(), graph_plans={LJ.requirement_id: GRAPH_PLAN})
    with pytest.raises(ValueError, match="outside this plan"):
        plan.prepare(_system(), graph_plans={MACE.requirement_id: GRAPH_PLAN})
    positions = FRACTIONS @ CELL
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    graph = plan.slot(GRAPH4)
    pairs = plan.slot(LJ)
    np.testing.assert_array_equal(state.graph_epochs, [0, 0])
    schedule = prepared.graph_schedule(state, GRAPH4)
    assert int(schedule.binding.epoch) == 0
    assert state.graph_schedules[pairs] is None
    with pytest.raises(ValueError, match="no graph plan"):
        prepared.graph_schedule(state, LJ)

    # The pair source rebuilds while the graph source is certified: the graph
    # keeps its revision and schedule bitwise (sharing never couples roles).
    small = positions + np.asarray([0.15, 0.0, 0.0])
    retained = prepared.refresh(state, prepared.geometry(jnp.asarray(small[None])))
    np.testing.assert_array_equal(
        retained.rebuilt[_source_index(prepared, "images")], [False]
    )
    assert int(prepared.graph_epoch(retained, GRAPH4)) == 0
    _leaves_match(prepared.graph_schedule(retained, GRAPH4), schedule)
    pair_changed = not _same(retained.views[pairs].relation, state.views[pairs].relation)
    assert int(retained.graph_epochs[pairs]) == int(pair_changed)

    # A half-cell displacement rebuilds the graph source with a new topology.
    moved = positions.copy()
    moved[0] += 0.5 * CELL[0]
    geometry = prepared.geometry(jnp.asarray(moved[None]))
    rebuilt = prepared.refresh(state, geometry)
    assert not _same(rebuilt.views[graph].relation, state.views[graph].relation)
    assert int(prepared.graph_epoch(rebuilt, GRAPH4)) == 1
    assert int(prepared.graph_schedule(rebuilt, GRAPH4).binding.epoch) == 1
    _assert_routes(
        _view_routes(prepared.view(rebuilt, GRAPH4), moved, CELL[None]),
        _oracle(moved, CELL, 4.0),
    )
    compiled = jax.jit(lambda previous, current: prepared.refresh(previous, current))(
        state, geometry
    )
    _leaves_match(rebuilt, compiled)

    # Rejecting the candidate restores the accepted schedule and revision.
    kept = prepared.select_cases(jnp.asarray([False]), rebuilt, state)
    np.testing.assert_array_equal(kept.graph_epochs, state.graph_epochs)
    _leaves_match(kept.graph_schedules, state.graph_schedules)
    committed = prepared.select_cases(jnp.asarray([True]), rebuilt, state)
    np.testing.assert_array_equal(committed.graph_epochs, rebuilt.graph_epochs)
    _leaves_match(committed.graph_schedules, rebuilt.graph_schedules)


def test_graph_schedules_survive_capacity_replay() -> None:
    tight = {GRAPH4.requirement_id: (_role_capacity(8, 256), _role_capacity(256, 256))}
    plan = _plan(
        (LJ, GRAPH4),
        (_source("pairs", (LJ,)), _source("images", (GRAPH4,), skin=1.0)),
        capacities=tight,
    )
    prepared = plan.prepare(_system(), graph_plans={GRAPH4.requirement_id: GRAPH_PLAN})
    positions = FRACTIONS @ CELL
    candidate = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
    np.testing.assert_array_equal(
        prepared.view(candidate, GRAPH4).evidence.route_overflow, [True]
    )
    grown = prepared.grow_capacities(candidate)
    assert grown.role_levels[plan.slot(GRAPH4)] == 1
    replayed = grown.rebase(candidate)
    np.testing.assert_array_equal(replayed.successful, [True])
    np.testing.assert_array_equal(replayed.graph_epochs, candidate.graph_epochs)
    schedule = grown.graph_schedule(replayed, GRAPH4)
    assert (
        schedule.binding.owner_id
        != prepared.graph_schedule(candidate, GRAPH4).binding.owner_id
    )
    _assert_routes(
        _view_routes(grown.view(replayed, GRAPH4), positions, CELL[None]),
        _oracle(positions, CELL, 4.0),
    )


# Site IDs run opposite to particle IDs and site slots permute the DOF slots,
# so neither site order nor slot equality identifies the physical pair.
REORDERED_SITE_IDS = np.asarray([100, 90, 80, 70])
REORDERED_DOFS = np.asarray([2, 0, 3, 1], dtype=np.int64)
EXCEPTIONS = {(3, 7): (0.0, 0.25), (11, 20): (0.5, 0.75), (7, 20): (0.125, 0.0)}


def _exception_topology() -> MolecularTopologyPlan:

    pairs = list(EXCEPTIONS)
    return MolecularTopologyPlan(
        pair_exceptions=np.asarray(pairs),
        lennard_jones_scales=np.asarray([EXCEPTIONS[pair][0] for pair in pairs]),
        electrostatic_scales=np.asarray([EXCEPTIONS[pair][1] for pair in pairs]),
    )


def _reordered_site_system(
    cell: PeriodicCell | None = None, *, finite: bool = False
) -> PreparedAtomisticSystem:
    sites = AtomisticInteractionSitePlan(
        REORDERED_SITE_IDS,
        np.ones((4,), dtype=np.int32),
        np.zeros((4,), dtype=np.int32),
        np.zeros((4,)),
    )
    coordinate_map = AtomisticCoordinateMapPlan(IDS, sites, REORDERED_DOFS)
    return _system(
        cell, finite=finite, coordinate_map=coordinate_map, topology=_exception_topology()
    )


def _site_scales(
    view: AtomisticNeighborhoodView, mask: ArrayLike | None = None
) -> dict[tuple[int, int, int], tuple[float, float]]:
    """``(case, physical id, physical id) -> (lj, coulomb)`` of valid site pairs."""
    particle_of = {
        int(site): int(IDS[dof])
        for site, dof in zip(REORDERED_SITE_IDS, REORDERED_DOFS, strict=True)
    }
    pairs = view.pair_relation
    assert pairs is not None
    assert view.lennard_jones_scales is not None
    assert view.electrostatic_scales is not None
    valid = np.asarray(pairs.valid)
    if mask is not None:
        valid = valid & np.asarray(mask)
    result = {}
    for case, left, right, lj, coulomb in zip(
        np.asarray(view.relation.route_cases)[valid],
        np.asarray(pairs.left_particle_ids)[valid],
        np.asarray(pairs.right_particle_ids)[valid],
        np.asarray(view.lennard_jones_scales)[valid],
        np.asarray(view.electrostatic_scales)[valid],
        strict=True,
    ):
        first, second = sorted((particle_of[int(left)], particle_of[int(right)]))
        key = (int(case), first, second)
        assert key not in result, "physical pair appears twice"
        result[key] = (float(lj), float(coulomb))
    return result


def test_site_pair_exceptions_follow_declared_physical_particles() -> None:
    system = _reordered_site_system(finite=True)
    tip = Requirement(
        "ff",
        Role.INTERACTION_SITE_PAIRS,
        None,
        exclusions=AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS,
    )
    plan = _plan(
        (tip,),
        (AtomisticCompletePairSource("sites", SITES, (tip,), maximum_pairs=6),),
        case_count=2,
    )
    prepared = plan.prepare(system)
    positions = np.stack((FAR, FAR + 1.0))
    sites = jax.vmap(system.coordinate_map.realize)(jnp.asarray(positions))
    state = prepared.initialize(prepared.geometry(jnp.asarray(positions), sites=sites))
    np.testing.assert_array_equal(state.successful, [True, True])
    view = prepared.view(state, tip)
    assert view.entity is SITES
    expected = {}
    for case in range(2):
        for first_index, first in enumerate(IDS):
            for second in IDS[first_index + 1 :]:
                expected[(case, int(first), int(second))] = EXCEPTIONS.get(
                    (int(first), int(second)), (1.0, 1.0)
                )
    assert _site_scales(view) == expected
    np.testing.assert_array_equal(view.evidence.identity_conflict, [False, False])


def test_bounded_site_pair_exceptions_in_a_periodic_cell() -> None:
    system = _reordered_site_system()
    tip = Requirement(
        "ff",
        Role.INTERACTION_SITE_PAIRS,
        2.0,
        exclusions=AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS,
    )
    prepared = _plan((tip,), (_source("sites", (tip,), entity=SITES),)).prepare(system)
    positions = FRACTIONS @ CELL
    sites = jax.vmap(
        lambda value: system.coordinate_map.realize(value, cell=system.cell)
    )(jnp.asarray(positions[None]))
    state = prepared.initialize(
        prepared.geometry(jnp.asarray(positions[None]), sites=sites)
    )
    np.testing.assert_array_equal(state.successful, [True])
    view = prepared.view(state, tip)
    site_positions = np.asarray(sites.positions)[0]
    within = _canonical(_oracle(site_positions, CELL, 2.0, ids=REORDERED_SITE_IDS))
    assert within
    particle_of = dict(zip(REORDERED_SITE_IDS.tolist(), IDS[REORDERED_DOFS].tolist()))
    expected = {
        (0, *sorted((particle_of[source], particle_of[receiver]))): EXCEPTIONS.get(
            tuple(sorted((particle_of[source], particle_of[receiver]))), (1.0, 1.0)
        )
        for _, source, receiver, _ in within
    }
    assert _site_scales(view, view.mask) == expected


def test_virtual_sites_refuse_physical_exceptions_and_keep_unit_scales() -> None:
    tip = Requirement(
        "ff",
        Role.INTERACTION_SITE_PAIRS,
        2.0,
        exclusions=AtomisticPairExclusions.TOPOLOGY_EXCEPTIONS,
    )
    plan = _plan((tip,), (_source("sites", (tip,), entity=SITES),))
    with pytest.raises(ValueError, match="site-space pair-exception contract"):
        plan.prepare(
            _system(coordinate_map=_virtual_site_map(), topology=_exception_topology())
        )
    system = _virtual_site_system()
    prepared = plan.prepare(system)
    positions = FRACTIONS @ CELL
    sites = jax.vmap(
        lambda value: system.coordinate_map.realize(value, cell=system.cell)
    )(jnp.asarray(positions[None]))
    state = prepared.initialize(
        prepared.geometry(jnp.asarray(positions[None]), sites=sites)
    )
    np.testing.assert_array_equal(state.successful, [True])
    view = prepared.view(state, tip)
    assert view.pair_relation is not None
    valid = np.asarray(view.pair_relation.valid)
    ids = np.stack(
        (
            np.asarray(view.pair_relation.left_particle_ids)[valid],
            np.asarray(view.pair_relation.right_particle_ids)[valid],
        )
    )
    # Virtual site 50 pairs are present and unscaled, like every other pair.
    assert np.any(ids == 50)
    np.testing.assert_array_equal(np.asarray(view.lennard_jones_scales)[valid], 1.0)
    np.testing.assert_array_equal(np.asarray(view.electrostatic_scales)[valid], 1.0)
    np.testing.assert_array_equal(np.asarray(view.lennard_jones_scales)[~valid], 0.0)
