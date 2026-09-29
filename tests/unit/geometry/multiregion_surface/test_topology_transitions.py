import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.foams import catenoid_critical_parameters
from phydrax.geometry.multiregion_surface import (
    apply_surface_events,
    MergeProposal,
    multiregion_sheet_views,
    MultiRegionRemeshPlan,
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    MultiRegionSurfaceValidationPolicy,
    PinchProposal,
    PreparedMultiRegionSurface,
    propose_merges,
    propose_pinches,
    propose_remesh,
    propose_t1_pops,
    RegionSplitProposal,
    seed_catenoid,
    seed_sphere,
    SurfaceEventKind,
    SurfaceEventPassStatus,
    SurfaceEventPolicy,
    SurfaceEventStatus,
    SurfaceMergePolicy,
    T1PopProposal,
    validate_multiregion_surface,
)
from tests._support.multiregion_foams import dry_foam_capacity_plan, t1_cluster


DRY_FOAM = MultiRegionSurfaceValidationPolicy(profile="dry_foam")


def test_t1_pop_restores_region_graph_completeness_and_conserves_volumes() -> None:
    seed = t1_cluster()
    topology = seed.topology(dry_foam_capacity_plan(seed, "t1"))
    base = seed.state(topology)
    before = validate_multiregion_surface(topology, base, policy=DRY_FOAM)
    assert before.accepted
    gas = np.zeros((topology.region_capacity, 1))
    gas[:5, 0] = 30.0 * np.asarray(before.signed_volumes)
    state = MultiRegionSurfaceState(
        topology, base.positions, region_fields=gas, region_field_names=("gas",)
    )
    prepared = PreparedMultiRegionSurface(topology, state, policy=DRY_FOAM)
    search = propose_t1_pops(prepared, state, maximum_film_diameter=0.2)
    proposals = search.proposals
    if len(proposals) != 1 or not isinstance(proposals[0], T1PopProposal):
        raise AssertionError("Expected one typed T1 proposal.")
    proposal = proposals[0]
    assert proposal.region_ids == ("D", "E")
    assert not search.capacity_exceeded and search.uncertified_count == 0
    result = apply_surface_events(
        topology, state, [proposal], policy=SurfaceEventPolicy(validation=DRY_FOAM)
    )
    assert result.committed
    record = result.evidence.records[0]
    assert (
        record.kind is SurfaceEventKind.T1_POP
        and record.accepted
        and record.ccd_certified
    )
    assert len(record.removed_vertex_ids) == 3 and len(record.created_vertex_ids) == 2
    lineage = result.evidence.lineage
    if lineage is None:
        raise AssertionError("A committed T1 pop must carry lineage.")
    parents = dict(lineage.vertex_parents)
    assert all(
        parents[child] == proposal.film_vertex_ids for child in record.created_vertex_ids
    )
    face_parents = dict(lineage.face_parents)
    assert set(face_parents) == set(record.created_face_ids)
    assert set(lineage.removed_face_ids) == set(record.removed_face_ids)
    assert any(not face_parents[child] for child in record.created_face_ids)
    after = validate_multiregion_surface(result.topology, result.state, policy=DRY_FOAM)
    assert after.accepted and after.incomplete_region_graph_vertex_count == 0
    np.testing.assert_allclose(after.signed_volumes, before.signed_volumes, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(result.state.region_fields), gas, rtol=0.0, atol=0.0
    )
    pairs = {
        tuple(pair)
        for pair in np.asarray(
            result.topology.region_pairs[: result.topology.region_pair_count]
        ).tolist()
    }
    assert (0, 1) not in pairs
    edges = np.asarray(result.topology.edges[: result.topology.edge_count])
    new = np.asarray(result.topology.vertex_global_ids)[edges]
    border = np.all(np.isin(new, record.created_vertex_ids), axis=1)
    faces = np.asarray(result.topology.edge_faces[: result.topology.edge_count])[border]
    assert int(np.sum(faces >= 0)) == 3


def test_catenoid_past_the_stability_limit_pinches_into_two_disks() -> None:
    _, critical, _ = catenoid_critical_parameters()
    ratio = 0.7
    assert ratio > critical
    seed = seed_catenoid(1.0, ratio, ring_points=12, rows=8, neck_radius=0.04)
    plan = seed.capacity_plan(resource_id="pinch", headroom=2.0, event_capacity=32)
    topology = seed.topology(plan)
    base = seed.state(topology)
    prepared = PreparedMultiRegionSurface(topology, base)
    liquid = 2.0e-6 * np.asarray(prepared.slot_areas(base.positions))[:, :, None]
    state = MultiRegionSurfaceState(
        topology, base.positions, sheet_fields=liquid, sheet_field_names=("liquid",)
    )
    rings = seed.vertex_set("ring-lower") + seed.vertex_set("ring-upper")
    policy = SurfaceEventPolicy(fixed_vertex_ids=rings)
    remesh = MultiRegionRemeshPlan(
        minimum_edge_length=0.06, maximum_edge_length=1.0, operations=("collapse",)
    )
    pinched = None
    for _ in range(12):
        prepared = PreparedMultiRegionSurface(topology, state)
        necks = propose_pinches(prepared, state, maximum_neck_perimeter=0.3)
        proposals = [
            *necks,
            RegionSplitProposal("core"),
            *propose_remesh(prepared, state, remesh),
        ]
        result = apply_surface_events(topology, state, proposals, policy=policy)
        assert result.evidence.status in (
            SurfaceEventPassStatus.COMMITTED,
            SurfaceEventPassStatus.NO_ACCEPTED_EVENTS,
        )
        topology, state = result.topology, result.state
        if any(
            r.kind is SurfaceEventKind.PINCH and r.accepted
            for r in result.evidence.records
        ):
            pinched = result
            break
    assert pinched is not None
    lineage = pinched.evidence.lineage
    assert lineage is not None
    assert topology.region_ids == ("core/0", "core/1", "ambient")
    assert dict(lineage.region_parents) == {"core/0": ("core",), "core/1": ("core",)}
    assert lineage.removed_region_ids == ("core",)
    pinch = next(r for r in pinched.evidence.records if r.kind is SurfaceEventKind.PINCH)
    parents = dict(lineage.vertex_parents)
    assert all(len(parents[child]) == 3 for child in pinch.created_vertex_ids)
    assert float(jnp.sum(state.sheet_fields)) == pytest.approx(
        float(np.sum(liquid)), rel=1e-12
    )
    views = multiregion_sheet_views(PreparedMultiRegionSurface(topology, state), state)
    assert not views.nonmanifold_pair_indices
    disks = sorted(views.views, key=lambda view: view.region_ids)
    assert [view.region_ids for view in disks] == [
        ("core/0", "ambient"),
        ("core/1", "ambient"),
    ]
    ring_ids = [set(seed.vertex_set("ring-lower")), set(seed.vertex_set("ring-upper"))]
    ids = np.asarray(topology.vertex_global_ids)
    for view in disks:
        mesh = view.mesh.topology
        assert mesh.euler_characteristic == 1
        loop = ids[
            np.asarray(view.vertex_indices)[np.asarray(mesh.boundary_loop_vertices)]
        ]
        assert set(loop.tolist()) in ring_ids


def test_merge_zips_facing_films_into_a_shared_wall() -> None:
    first = seed_sphere(1.0, subdivisions=1)
    second = seed_sphere(1.0, center=(2.04, 0.0, 0.0), subdivisions=1)
    count = first.positions.shape[0]
    seed = MultiRegionSurfaceSeed(
        np.concatenate((first.positions, second.positions)),
        np.concatenate((first.faces, second.faces + count)),
        np.concatenate(
            (
                np.tile((0, 2), (first.faces.shape[0], 1)),
                np.tile((1, 2), (second.faces.shape[0], 1)),
            )
        ),
        ("left", "right", "ambient"),
        ("finite", "finite", "boundary"),
        source="two-bubbles",
    )
    topology = seed.topology(dry_foam_capacity_plan(seed, "merge"))
    base = seed.state(topology)
    prepared = PreparedMultiRegionSurface(topology, base)
    slot = np.asarray(prepared.slot_areas(base.positions))
    volumes = np.asarray(prepared.region_volumes(base.positions))
    gas = np.where(np.asarray(topology.region_finite), 25.0 * volumes, 0.0)[:, None]
    state = MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=np.stack((2.0e-6 * slot, 3.0e-7 * slot), axis=2),
        region_fields=gas,
        sheet_field_names=("liquid", "surfactant"),
        region_field_names=("gas",),
    )
    policy = SurfaceMergePolicy(("ambient",), merge_distance=0.4)
    search = propose_merges(prepared, state, policy)
    proposals = search.proposals
    if not proposals or not isinstance(proposals[0], MergeProposal):
        raise AssertionError("Expected a typed merge proposal.")
    proposal = proposals[0]
    assert not search.capacity_exceeded
    overflow = propose_merges(prepared, state, policy, candidate_capacity=1)
    assert overflow.capacity_exceeded and overflow.proposals == ()
    assert overflow.candidate_count > overflow.candidate_capacity
    far = SurfaceMergePolicy(("ambient",), merge_distance=0.01)
    refused = apply_surface_events(
        topology, state, [MergeProposal(proposal.face_ids, far)]
    )
    assert refused.evidence.records[0].status is SurfaceEventStatus.NOT_TRIGGERED
    result = apply_surface_events(topology, state, [proposal])
    assert result.committed
    record = result.evidence.records[0]
    assert record.kind is SurfaceEventKind.MERGE and record.accepted
    assert len(record.removed_vertex_ids) == 6 and len(record.created_vertex_ids) == 3
    lineage = result.evidence.lineage
    assert lineage is not None
    wall = [
        face
        for face, parents in lineage.face_parents
        if set(parents) == set(proposal.face_ids)
    ]
    assert len(wall) == 1
    merged = result.topology
    pairs = np.asarray(merged.region_pairs[: merged.region_pair_count]).tolist()
    assert [0, 1] in pairs
    after = validate_multiregion_surface(merged, result.state)
    assert after.accepted and after.maximum_edge_valence == 3
    assert after.region_component_counts[merged.region_index("ambient")] == 1
    np.testing.assert_allclose(after.signed_volumes, volumes[:2], rtol=1e-11)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(result.state.sheet_fields, axis=(0, 1))),
        np.asarray(jnp.sum(state.sheet_fields, axis=(0, 1))),
        rtol=1e-13,
    )
    np.testing.assert_array_equal(np.asarray(result.state.region_fields), gas)


def test_explicit_region_split_and_unknown_region() -> None:
    seed = seed_catenoid(1.0, 0.5, ring_points=12, rows=6)
    plan = seed.capacity_plan(resource_id="split-region", headroom=2.0, event_capacity=4)
    topology = seed.topology(plan)
    state = seed.state(topology)
    result = apply_surface_events(
        topology, state, [RegionSplitProposal("core"), RegionSplitProposal("missing")]
    )
    statuses = sorted(record.status for record in result.evidence.records)
    assert statuses == [
        SurfaceEventStatus.NOT_TRIGGERED,
        SurfaceEventStatus.SUPPORT_INVALID,
    ]
    assert result.evidence.status is SurfaceEventPassStatus.NO_ACCEPTED_EVENTS
    loop = (0, 1, 2)
    refused = apply_surface_events(topology, state, [PinchProposal(loop)])
    assert refused.evidence.records[0].status is not SurfaceEventStatus.ACCEPTED
    assert refused.state is state


def test_t1_certificate_refuses_a_diagonal_film_whose_box_passes() -> None:
    seed = t1_cluster(turn=45.0)
    topology = seed.topology(dry_foam_capacity_plan(seed, "t1-diagonal"))
    state = seed.state(topology)
    film = seed.positions[[1, 4, 7]]
    extent = float(np.max(np.ptp(film, axis=0)))
    diameter = float(np.max(np.linalg.norm(film[:, None] - film[None], axis=2)))
    limit = 0.5 * (extent + diameter)
    assert extent < limit < diameter
    prepared = PreparedMultiRegionSurface(topology, state, policy=DRY_FOAM)
    search = propose_t1_pops(prepared, state, maximum_film_diameter=limit)
    assert search.proposals == () and search.rejected_count == 1
    forced = T1PopProposal(("D", "E"), (1, 4, 7), maximum_film_diameter=limit)
    result = apply_surface_events(
        topology, state, [forced], policy=SurfaceEventPolicy(validation=DRY_FOAM)
    )
    assert result.evidence.records[0].status is SurfaceEventStatus.NOT_TRIGGERED
    assert result.topology is topology and result.state is state
    wide = T1PopProposal(("D", "E"), (1, 4, 7), maximum_film_diameter=1.01 * diameter)
    popped = apply_surface_events(
        topology, state, [wide], policy=SurfaceEventPolicy(validation=DRY_FOAM)
    )
    assert popped.committed


def test_t1_certificate_stays_bounded_on_a_refined_film() -> None:
    seed = t1_cluster().subdivided(2)
    topology = seed.topology(dry_foam_capacity_plan(seed, "t1-refined"))
    state = seed.state(topology)
    prepared = PreparedMultiRegionSurface(topology, state)
    films = propose_t1_pops(prepared, state, maximum_film_diameter=0.09)
    if len(films.proposals) != 1 or not isinstance(films.proposals[0], T1PopProposal):
        raise AssertionError("Expected one typed refined-film T1 proposal.")
    proposal = films.proposals[0]
    assert proposal.region_ids == ("D", "E")
    film = proposal.film_vertex_ids
    assert len(film) == 15
    points = seed.positions[seed.vertex_indices(film)]
    assert float(np.max(np.linalg.norm(points[:, None] - points[None], axis=2))) < 0.09
    assert 1 < films.certificate_work <= 200
    starved = propose_t1_pops(
        prepared, state, maximum_film_diameter=0.09, certificate_capacity=1
    )
    assert starved.proposals == () and starved.uncertified_count == 1


def _tetrahedron(base: np.ndarray, apex: np.ndarray, offset: int, /) -> np.ndarray:
    """Outward faces of the tetrahedron ``base + apex`` with vertex offset."""
    points = np.concatenate((base, apex[None]))
    center = np.mean(points, axis=0)
    faces = []
    for face in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)):
        corners = points[list(face)]
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        ordered = (
            face
            if np.dot(normal, np.mean(corners, axis=0) - center) > 0.0
            else face[::-1]
        )
        faces.append([offset + index for index in ordered])
    return np.asarray(faces)


def test_merge_detects_close_edges_of_large_far_centered_films() -> None:
    def pair(gap: float, /) -> tuple[MultiRegionSurfaceTopology, MultiRegionSurfaceState]:
        lower = np.asarray(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)))
        upper = np.asarray(((10.0, 0.0, gap), (10.0, 10.0, gap), (0.0, 10.0, gap)))
        seed = MultiRegionSurfaceSeed(
            np.concatenate((lower, [(2.0, 2.0, -3.0)], upper, [(8.0, 8.0, 3.0 + gap)])),
            np.concatenate(
                (
                    _tetrahedron(lower, np.asarray((2.0, 2.0, -3.0)), 0),
                    _tetrahedron(upper, np.asarray((8.0, 8.0, 3.0 + gap)), 4),
                )
            ),
            np.asarray(((0, 2),) * 4 + ((1, 2),) * 4),
            ("lower", "upper", "ambient"),
            ("finite", "finite", "boundary"),
            source="edge-close-films",
        )
        topology = seed.topology(dry_foam_capacity_plan(seed, "edge-close"))
        return topology, seed.state(topology)

    policy = SurfaceMergePolicy(("ambient",), merge_distance=0.1)
    topology, state = pair(0.05)
    prepared = PreparedMultiRegionSurface(topology, state)
    search = propose_merges(prepared, state, policy)
    if len(search.proposals) != 1 or not isinstance(search.proposals[0], MergeProposal):
        raise AssertionError("Expected one typed close-edge merge proposal.")
    proposal = search.proposals[0]
    faces = topology.host_faces()
    ids = np.asarray(topology.face_global_ids[: topology.face_count])
    chosen = [int(np.flatnonzero(ids == face)[0]) for face in proposal.face_ids]
    centroids = np.mean(np.asarray(state.positions)[faces[chosen]], axis=1)
    assert np.linalg.norm(centroids[0] - centroids[1]) > 40.0 * policy.merge_distance
    assert proposal.priority == pytest.approx(0.05, rel=1e-12)
    topology, state = pair(0.15)
    assert (
        propose_merges(
            PreparedMultiRegionSurface(topology, state), state, policy
        ).proposals
        == ()
    )
