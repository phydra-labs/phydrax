import random

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax._geometry_predicates as geometry_predicates
from phydrax.geometry.multiregion_surface import (
    apply_surface_events,
    EdgeCollapseProposal,
    EdgeFlipProposal,
    EdgeSplitProposal,
    MultiRegionRemeshPlan,
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    MultiRegionSurfaceValidationPolicy,
    PreparedMultiRegionSurface,
    propose_remesh,
    remesh_edge_flags,
    seed_double_bubble,
    seed_sphere,
    SurfaceEventKind,
    SurfaceEventPassStatus,
    SurfaceEventPolicy,
    SurfaceEventStatus,
    validate_multiregion_surface,
)


DRY_FOAM = MultiRegionSurfaceValidationPolicy(profile="dry_foam")
SPLITS = SurfaceEventPolicy()
DRY_SPLITS = SurfaceEventPolicy(
    validation=MultiRegionSurfaceValidationPolicy(profile="dry_foam")
)


def _with_fields(
    seed: MultiRegionSurfaceSeed, *, headroom: float = 2.0, events: int = 64
) -> tuple[MultiRegionSurfaceTopology, MultiRegionSurfaceState]:
    """Liquid (nonuniform thickness), surfactant and gas amounts on a seed."""
    plan = seed.capacity_plan(
        resource_id="remesh-test", headroom=headroom, event_capacity=events
    )
    topology = seed.topology(plan)
    base = seed.state(topology)
    prepared = PreparedMultiRegionSurface(topology, base)
    slot_area = np.asarray(prepared.slot_areas(base.positions))
    points = np.asarray(base.positions)
    thickness = 1.0e-3 * (1.0 + 0.3 * points[:, 0] + 0.1 * points[:, 2])[:, None]
    sheet = np.stack((thickness * slot_area, 2.0e-6 * slot_area), axis=2)
    volumes = np.asarray(prepared.region_volumes(base.positions))
    gas = np.where(np.asarray(topology.region_finite), 40.0 * volumes, 0.0)[:, None]
    state = MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=sheet,
        region_fields=gas,
        sheet_field_names=("liquid", "surfactant"),
        region_field_names=("gas",),
    )
    return topology, state


def _edge_ids(topology: MultiRegionSurfaceTopology, edge: int, /) -> tuple[int, int]:
    ids = np.asarray(topology.vertex_global_ids)[np.asarray(topology.edges[edge])]
    return int(ids[0]), int(ids[1])


def _totals(state: MultiRegionSurfaceState, /) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(jnp.sum(state.sheet_fields, axis=(0, 1))),
            np.asarray(jnp.sum(state.region_fields, axis=0)),
        )
    )


def test_split_preserves_geometry_uniform_thickness_and_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(geometry_predicates, "meshcore_available", lambda: False)
    with pytest.raises(ValueError, match="self-intersection validation"):
        SurfaceEventPolicy(
            validation=MultiRegionSurfaceValidationPolicy(check_self_intersection=False)
        )
    seed = seed_sphere(1.0, subdivisions=1)
    plan = seed.capacity_plan(resource_id="split", headroom=2.0, event_capacity=8)
    topology = seed.topology(plan)
    base = seed.state(topology)
    prepared = PreparedMultiRegionSurface(topology, base)
    thickness = 2.5e-6
    liquid = thickness * np.asarray(prepared.slot_areas(base.positions))[:, :, None]
    state = MultiRegionSurfaceState(
        topology, base.positions, sheet_fields=liquid, sheet_field_names=("liquid",)
    )
    edge = _edge_ids(topology, 3)
    result = apply_surface_events(
        topology, state, [EdgeSplitProposal(edge)], policy=SPLITS
    )
    assert result.committed
    assert result.evidence.status is SurfaceEventPassStatus.COMMITTED
    assert not result.evidence.derivative_available
    record = result.evidence.records[0]
    assert record.kind is SurfaceEventKind.SPLIT and record.accepted
    assert len(record.removed_face_ids) == 2 and len(record.created_face_ids) == 4
    new_id = record.created_vertex_ids[0]
    lineage = result.evidence.lineage
    if lineage is None:
        raise AssertionError("A committed split must carry lineage.")
    assert (new_id, tuple(sorted(edge))) in lineage.vertex_parents
    split = result.topology
    assert (split.vertex_count, split.face_count) == (
        topology.vertex_count + 1,
        topology.face_count + 2,
    )
    assert split.epoch == topology.epoch + 1
    assert result.evidence.validation is not None
    assert result.evidence.validation.self_intersection_checked
    assert result.evidence.validation.predicate_mode == "exact"
    assert result.evidence.validation.uncertain_pair_count == 0
    after = PreparedMultiRegionSurface(split, result.state, policy=SPLITS.validation)
    before_volume = float(prepared.region_volumes(base.positions)[0])
    assert float(after.region_volumes(result.state.positions)[0]) == pytest.approx(
        before_volume, rel=1e-14
    )
    assert float(jnp.sum(after.face_areas(result.state.positions))) == pytest.approx(
        float(jnp.sum(prepared.face_areas(base.positions))), rel=1e-14
    )
    slot_area = np.asarray(after.slot_areas(result.state.positions))
    content = np.asarray(result.state.sheet_fields[..., 0])
    active = slot_area > 0.0
    np.testing.assert_allclose(content[active] / slot_area[active], thickness, rtol=1e-12)
    assert float(jnp.sum(result.state.sheet_fields)) == pytest.approx(
        float(jnp.sum(liquid)), rel=1e-14
    )
    transition = result.transition
    assert transition is not None
    moved = transition.apply(state.sheet_fields[..., 0].reshape(-1))
    assert bool(moved.successful) and not bool(moved.differentiation_available)


def test_junction_split_keeps_the_plateau_border() -> None:
    seed = seed_double_bubble(1.0, 0.8, ring_points=12)
    topology, state = _with_fields(seed)
    junction = seed.vertex_indices(seed.vertex_set("junction")[:2])
    ids = tuple(int(value) for value in seed.vertex_global_ids[junction])
    result = apply_surface_events(
        topology, state, [EdgeSplitProposal(ids)], policy=DRY_SPLITS
    )
    assert result.committed
    record = result.evidence.records[0]
    assert len(record.removed_face_ids) == 3 and len(record.created_face_ids) == 6
    evidence = validate_multiregion_surface(
        result.topology, result.state, policy=DRY_SPLITS.validation
    )
    assert evidence.accepted and evidence.maximum_edge_valence == 3
    np.testing.assert_allclose(_totals(result.state), _totals(state), rtol=1e-13)


def test_collapse_restores_region_volumes_and_keeps_topology() -> None:
    seed = seed_double_bubble(1.0, 0.8, ring_points=12)
    topology, state = _with_fields(seed)
    before = validate_multiregion_surface(topology, state, policy=DRY_FOAM)
    prepared = PreparedMultiRegionSurface(topology, state, policy=DRY_FOAM)
    lengths = np.sort(
        np.asarray(
            remesh_edge_flags(
                prepared,
                state.positions,
                MultiRegionRemeshPlan(minimum_edge_length=1e-3, maximum_edge_length=10.0),
            ).edge_lengths[: topology.edge_count]
        )
    )
    plan = MultiRegionRemeshPlan(
        minimum_edge_length=float(lengths[8]),
        maximum_edge_length=10.0,
        operations=("collapse",),
    )
    proposals = propose_remesh(prepared, state, plan)
    assert proposals and all(p.kind is SurfaceEventKind.COLLAPSE for p in proposals)
    policy = SurfaceEventPolicy(validation=DRY_FOAM)
    result = apply_surface_events(topology, state, proposals, policy=policy)
    assert result.committed
    accepted = result.evidence.records_of(SurfaceEventKind.COLLAPSE)
    accepted = tuple(record for record in accepted if record.accepted)
    assert accepted
    assert result.topology.vertex_count == topology.vertex_count - len(accepted)
    after = validate_multiregion_surface(result.topology, result.state, policy=DRY_FOAM)
    assert after.accepted
    assert after.region_euler_characteristics == before.region_euler_characteristics
    np.testing.assert_allclose(after.signed_volumes, before.signed_volumes, rtol=1e-11)
    np.testing.assert_allclose(_totals(result.state), _totals(state), rtol=1e-13)
    sheet = result.evidence.sheet_transfer
    assert sheet is not None and bool(sheet.successful) and bool(sheet.target_nonnegative)
    removed = {vertex for record in accepted for vertex in record.removed_vertex_ids}
    lineage = result.evidence.lineage
    if lineage is None:
        raise AssertionError("A committed collapse must carry lineage.")
    survivors = dict(lineage.vertex_parents)
    assert removed == {p for parents in survivors.values() for p in parents} - set(
        survivors
    )


def test_collapse_guards_features_and_duplicate_faces() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology, state = _with_fields(seed, events=4)
    edge = _edge_ids(topology, 0)
    fixed = SurfaceEventPolicy(fixed_vertex_ids=edge)
    refused = apply_surface_events(
        topology, state, [EdgeCollapseProposal(edge)], policy=fixed
    )
    assert refused.evidence.records[0].status is SurfaceEventStatus.FEATURE_NOT_PRESERVED
    assert refused.topology is topology and refused.state is state

    tetrahedron = MultiRegionSurfaceSeed(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, 0.2, 1.0))),
        np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2))),
        np.zeros((4, 2), dtype=np.int64) + np.asarray((0, 1)),
        ("cell", "ambient"),
        ("finite", "boundary"),
        source="tetrahedron",
    )
    topology, state = _with_fields(tetrahedron, events=4)
    result = apply_surface_events(topology, state, [EdgeCollapseProposal((0, 1))])
    assert result.evidence.records[0].status is SurfaceEventStatus.DUPLICATE_FACE
    assert result.state is state


def _tent_seed(*, pebble: bool) -> MultiRegionSurfaceSeed:
    """Two roof faces over a ridge ``u-v`` and optionally a pebble inside the tent."""
    tent = np.asarray(
        ((0.0, -1.0, 1.0), (0.0, 1.0, 1.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    )
    roof = np.asarray(((0, 1, 2), (1, 0, 3)))
    if not pebble:
        return MultiRegionSurfaceSeed(
            tent,
            roof,
            np.tile((0, 1), (2, 1)),
            ("lower", "upper"),
            ("boundary",) * 2,
            source="tent",
        )
    stone = seed_sphere(0.05, center=(0.0, 0.0, 0.5), subdivisions=0)
    return MultiRegionSurfaceSeed(
        np.concatenate((tent, stone.positions)),
        np.concatenate((roof, stone.faces + 4)),
        np.concatenate(
            (np.tile((1, 2), (2, 1)), np.tile((0, 1), (stone.faces.shape[0], 1)))
        ),
        ("pebble", "lower", "upper"),
        ("finite", "boundary", "boundary"),
        source="tent-with-pebble",
    )


def test_flip_sweep_is_certified_by_ccd() -> None:
    free = _tent_seed(pebble=False)
    topology, state = _with_fields(free, events=2)
    result = apply_surface_events(topology, state, [EdgeFlipProposal((0, 1))])
    assert result.committed
    record = result.evidence.records[0]
    assert record.ccd_certified and record.ccd_time_of_impact == 1.0
    faces = {tuple(sorted(row)) for row in result.topology.host_faces().tolist()}
    assert faces == {(0, 2, 3), (1, 2, 3)}
    np.testing.assert_allclose(_totals(result.state), _totals(state), rtol=1e-14)

    blocked = _tent_seed(pebble=True)
    topology, state = _with_fields(blocked, events=2)
    before = validate_multiregion_surface(topology, state)
    assert before.accepted
    result = apply_surface_events(topology, state, [EdgeFlipProposal((0, 1))])
    record = result.evidence.records[0]
    assert record.status is SurfaceEventStatus.CCD_NOT_CERTIFIED
    assert 0.0 < record.ccd_time_of_impact < 1.0
    assert result.evidence.status is SurfaceEventPassStatus.NO_ACCEPTED_EVENTS
    assert result.topology is topology and result.state is state


def test_failed_transfer_and_capacity_return_the_source_state() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology, state = _with_fields(seed)
    poisoned = np.asarray(state.sheet_fields).copy()
    poisoned[5, 0, 1] = np.nan
    bad = MultiRegionSurfaceState(
        topology,
        state.positions,
        sheet_fields=poisoned,
        region_fields=state.region_fields,
        sheet_field_names=state.sheet_field_names,
        region_field_names=state.region_field_names,
    )
    result = apply_surface_events(
        topology, bad, [EdgeSplitProposal(_edge_ids(topology, 2))], policy=SPLITS
    )
    assert result.evidence.status is SurfaceEventPassStatus.TRANSFER_FAILED
    assert not result.committed and result.transition is None
    assert result.topology is topology and result.state is bad
    sheet = result.evidence.sheet_transfer
    assert sheet is not None and not bool(sheet.finite)

    tight = seed.capacity_plan(resource_id="full", event_capacity=4)
    full_topology = seed.topology(tight)
    full_state = seed.state(full_topology)
    refused = apply_surface_events(
        full_topology, full_state, [EdgeSplitProposal(_edge_ids(full_topology, 2))]
    )
    assert refused.evidence.records[0].status is SurfaceEventStatus.CAPACITY_EXCEEDED
    assert refused.topology is full_topology and refused.state is full_state

    no_budget = seed.capacity_plan(resource_id="no-events", headroom=2.0)
    idle = seed.topology(no_budget)
    idle_state = seed.state(idle)
    held = apply_surface_events(idle, idle_state, [EdgeSplitProposal(_edge_ids(idle, 2))])
    assert held.evidence.records[0].status is SurfaceEventStatus.EVENT_BUDGET_EXCEEDED


def test_remesh_pass_conserves_content_and_is_order_independent() -> None:
    seed = seed_double_bubble(1.1, 0.9, ring_points=12)
    topology, state = _with_fields(seed)
    prepared = PreparedMultiRegionSurface(topology, state, policy=DRY_FOAM)
    probe = MultiRegionRemeshPlan(minimum_edge_length=1e-3, maximum_edge_length=10.0)
    lengths = np.asarray(
        remesh_edge_flags(prepared, state.positions, probe).edge_lengths[
            : topology.edge_count
        ]
    )
    shortest = float(np.quantile(lengths, 0.08))
    longest = max(float(np.quantile(lengths, 0.9)), 2.1 * shortest)
    plan = MultiRegionRemeshPlan(
        minimum_edge_length=shortest, maximum_edge_length=longest
    )
    flags = remesh_edge_flags(prepared, state.positions, plan)
    np.testing.assert_array_equal(
        np.asarray(flags.split[: topology.edge_count]), lengths > longest
    )
    proposals = list(propose_remesh(prepared, state, plan))
    kinds = {proposal.kind for proposal in proposals}
    assert SurfaceEventKind.COLLAPSE in kinds
    policy = DRY_SPLITS
    result = apply_surface_events(topology, state, proposals, policy=policy)
    assert result.committed
    statuses = {record.status for record in result.evidence.records}
    assert SurfaceEventStatus.CONFLICT in statuses
    np.testing.assert_allclose(_totals(result.state), _totals(state), rtol=1e-12)
    before = validate_multiregion_surface(topology, state, policy=DRY_FOAM)
    after = validate_multiregion_surface(
        result.topology, result.state, policy=policy.validation
    )
    assert after.accepted
    np.testing.assert_allclose(after.signed_volumes, before.signed_volumes, rtol=1e-10)
    assert bool(jnp.all(result.state.sheet_fields >= 0.0))

    shuffled = list(proposals)
    random.Random(7).shuffle(shuffled)
    again = apply_surface_events(topology, state, shuffled, policy=policy)
    assert again.topology.topology_id == result.topology.topology_id
    assert again.topology.lineage_id == result.topology.lineage_id
    assert again.evidence.evidence_id == result.evidence.evidence_id
    np.testing.assert_array_equal(
        np.asarray(again.state.sheet_fields), np.asarray(result.state.sheet_fields)
    )
