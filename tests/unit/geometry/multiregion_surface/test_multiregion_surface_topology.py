import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.multiregion_surface import (
    multiregion_cell_complex,
    multiregion_sheet_views,
    MultiRegionSurfaceCapacityError,
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceEvidence,
    MultiRegionSurfacePreparationError,
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    MultiRegionSurfaceStatus,
    MultiRegionSurfaceTopology,
    MultiRegionSurfaceValidationPolicy,
    PreparedMultiRegionSurface,
    seed_double_bubble,
    seed_sphere,
    validate_multiregion_surface,
)
from phydrax.geometry.simplicial import TriangleMesh


DRY_FOAM = MultiRegionSurfaceValidationPolicy(profile="dry_foam")
MANIFOLD_TWO_REGION = MultiRegionSurfaceValidationPolicy(profile="manifold_two_region")


def _built(
    seed: MultiRegionSurfaceSeed,
    policy: MultiRegionSurfaceValidationPolicy | None = None,
) -> tuple[
    MultiRegionSurfaceTopology,
    MultiRegionSurfaceState,
    MultiRegionSurfaceEvidence,
]:
    plan = seed.capacity_plan(resource_id="multiregion-test", headroom=1.25)
    topology = seed.topology(plan)
    state = seed.state(topology)
    return topology, state, validate_multiregion_surface(topology, state, policy=policy)


def _polyhedron_volume(points: np.ndarray, faces: np.ndarray) -> float:
    corners = points[faces]
    return float(np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2])) / 6.0)


def _torus_seed(
    major: float, minor: float, rings: int, sides: int
) -> MultiRegionSurfaceSeed:
    u = 2.0 * np.pi * np.arange(rings) / rings
    v = 2.0 * np.pi * (np.arange(sides) + 0.37) / sides
    uu, vv = np.meshgrid(u, v, indexing="ij")
    # A twisted grid keeps every quad nonplanar (general position for the
    # filtered predicates used without the exact meshcore library).
    vv = vv + 0.21 * uu
    points = np.stack(
        (
            (major + minor * np.cos(vv)) * np.cos(uu),
            (major + minor * np.cos(vv)) * np.sin(uu),
            minor * np.sin(vv),
        ),
        axis=2,
    ).reshape((-1, 3))
    index = np.arange(rings * sides).reshape((rings, sides))
    a = index
    b = np.roll(index, -1, axis=0)
    c = np.roll(b, -1, axis=1)
    d = np.roll(index, -1, axis=1)
    faces = np.concatenate(
        (np.stack((a, b, c), 2).reshape((-1, 3)), np.stack((a, c, d), 2).reshape((-1, 3)))
    )
    return MultiRegionSurfaceSeed(
        points,
        faces,
        np.tile(np.asarray((0, 1)), (faces.shape[0], 1)),
        ("torus", "ambient"),
        ("finite", "boundary"),
        source="test-torus",
    )


def test_closed_sphere_and_torus_topology_and_signed_volume() -> None:
    sphere = seed_sphere(1.5, center=(0.3, -0.2, 0.1), subdivisions=2)
    topology, state, evidence = _built(sphere)
    assert evidence.status is MultiRegionSurfaceStatus.ACCEPTED
    assert evidence.region_euler_characteristics == (2,)
    expected = _polyhedron_volume(sphere.positions, sphere.faces)
    assert evidence.signed_volumes[0] == pytest.approx(expected, rel=1e-12)
    # Inscribed polyhedron: below the sphere volume by the discretization defect.
    assert 0.95 < expected / (4.0 / 3.0 * np.pi * 1.5**3) < 1.0
    prepared = PreparedMultiRegionSurface(topology, state)
    volumes = np.asarray(prepared.region_volumes(state.positions))
    assert volumes[0] == pytest.approx(expected, rel=1e-12)
    assert volumes[1] == 0.0

    torus = _torus_seed(2.0, 0.6, 24, 12)
    topology, state, evidence = _built(torus)
    assert evidence.status is MultiRegionSurfaceStatus.ACCEPTED
    assert evidence.region_euler_characteristics == (0,)
    expected = _polyhedron_volume(torus.positions, torus.faces)
    assert evidence.signed_volumes[0] == pytest.approx(expected, rel=1e-12)
    assert 0.9 < expected / (2.0 * np.pi**2 * 2.0 * 0.6**2) < 1.0


def test_forces_are_the_negative_area_gradient_with_scaling_identity() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology, state, _ = _built(seed)
    prepared = PreparedMultiRegionSurface(topology, state)
    tension = jnp.where(topology.face_active, 0.07, 0.0)
    geometry = prepared.evaluate(state, tension)
    forces = np.asarray(geometry.forces)
    points = np.asarray(state.positions)
    # E(s x) = s^2 E(x) for a surface energy, so sum_v F_v . x_v = -2 E.
    assert np.sum(forces * points) == pytest.approx(-2.0 * float(geometry.surface_energy))
    assert np.max(np.abs(np.sum(forces, axis=0))) < 1e-14
    assert np.max(np.abs(np.sum(np.cross(points, forces), axis=0))) < 1e-14
    slot_total = float(jnp.sum(geometry.slot_areas))
    assert slot_total == pytest.approx(float(jnp.sum(geometry.face_areas)))


def test_face_normals_are_unit_and_preserve_pressure_and_flux_scale() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology, state, _ = _built(seed)
    prepared = PreparedMultiRegionSurface(topology, state)
    geometry = prepared.evaluate(
        state,
        jnp.zeros((topology.face_capacity,), dtype=state.positions.dtype),
    )
    active = np.asarray(topology.face_active)
    normals = np.asarray(geometry.face_normals)
    areas = np.asarray(geometry.face_areas)
    faces = np.maximum(np.asarray(topology.faces, dtype=np.int64), 0)
    corners = np.asarray(state.positions)[faces]
    area_vectors = 0.5 * np.cross(
        corners[:, 1] - corners[:, 0],
        corners[:, 2] - corners[:, 0],
    )
    area_vectors = np.where(active[:, None], area_vectors, 0.0)

    np.testing.assert_allclose(
        np.linalg.norm(normals[active], axis=1),
        1.0,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_array_equal(normals[~active], np.zeros_like(normals[~active]))
    np.testing.assert_allclose(
        normals * areas[:, None],
        area_vectors,
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    pressure = np.where(
        active,
        np.arange(topology.face_capacity, dtype=np.float64) + 1.0,
        0.0,
    )
    pressure_force = np.sum(pressure[:, None] * areas[:, None] * normals, axis=0)
    np.testing.assert_allclose(
        pressure_force,
        np.sum(pressure[:, None] * area_vectors, axis=0),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    velocity = np.asarray((0.4, -0.7, 1.1), dtype=np.float64)
    flux = np.sum(areas * np.sum(normals * velocity, axis=1))
    expected_flux = np.sum(area_vectors * velocity)
    assert flux == pytest.approx(expected_flux, rel=1.0e-14, abs=1.0e-14)


def test_triple_sheet_plateau_border_complex() -> None:
    seed = seed_double_bubble(1.0, 0.75, ring_points=18)
    topology, state, evidence = _built(seed, DRY_FOAM)
    assert evidence.status is MultiRegionSurfaceStatus.ACCEPTED
    assert evidence.maximum_edge_valence == 3
    assert evidence.maximum_vertex_regions == 3
    assert evidence.region_euler_characteristics == (2, 2)
    junction = np.asarray(seed.vertex_set("junction"))
    slots = np.asarray(topology.slot_active)
    assert np.all(np.sum(slots[junction], axis=1) == 3)
    interior = np.setdiff1d(np.arange(topology.vertex_count), junction)
    assert np.all(np.sum(slots[interior], axis=1) == 1)
    valence = np.sum(np.asarray(topology.edge_faces[: topology.edge_count]) >= 0, axis=1)
    assert int(np.count_nonzero(valence == 3)) == junction.size
    complex_ = multiregion_cell_complex(topology)
    assert complex_.dimension == 3
    assert complex_.entity_sets[3].count == 2
    prepared = PreparedMultiRegionSurface(topology, state, policy=DRY_FOAM)
    volumes = np.asarray(prepared.region_volumes(state.positions))[:2]
    np.testing.assert_allclose(volumes, evidence.signed_volumes, rtol=1e-12)


def test_manifold_two_region_profile_admits_only_one_closed_sheet() -> None:
    sphere = seed_sphere(1.0, subdivisions=1)
    _, _, accepted = _built(sphere, MANIFOLD_TWO_REGION)
    assert accepted.status is MultiRegionSurfaceStatus.ACCEPTED
    assert accepted.profile == "manifold_two_region"
    assert accepted.maximum_edge_valence == 2
    assert accepted.maximum_vertex_regions == 2

    double_bubble = seed_double_bubble(1.0, 0.75, ring_points=12)
    _, _, refused = _built(double_bubble, MANIFOLD_TWO_REGION)
    assert refused.status is MultiRegionSurfaceStatus.NONPHYSICAL_VALENCE
    assert not refused.accepted


def test_label_and_orientation_inconsistencies_are_rejected() -> None:
    sphere = seed_sphere(1.0, subdivisions=1)
    flipped_faces = sphere.faces.copy()
    flipped_faces[3] = flipped_faces[3, (0, 2, 1)]
    flipped = MultiRegionSurfaceSeed(
        sphere.positions,
        flipped_faces,
        sphere.face_labels,
        sphere.region_ids,
        sphere.region_kinds,
        source="flipped-sphere",
    )
    topology, state, evidence = _built(flipped)
    assert evidence.status is MultiRegionSurfaceStatus.LABEL_ORIENTATION_INCONSISTENT
    assert not evidence.accepted
    with pytest.raises(MultiRegionSurfacePreparationError) as refused:
        PreparedMultiRegionSurface(topology, state)
    assert refused.value.evidence.status is evidence.status

    bubble = seed_double_bubble(1.0, 0.8, ring_points=12)
    labels = bubble.face_labels.copy()
    outer_first = int(np.flatnonzero((labels[:, 0] == 0) & (labels[:, 1] == 2))[0])
    labels[outer_first] = (1, 2)
    relabeled = MultiRegionSurfaceSeed(
        bubble.positions,
        bubble.faces,
        labels,
        bubble.region_ids,
        bubble.region_kinds,
        source="relabeled-double-bubble",
    )
    _, _, evidence = _built(relabeled)
    assert not evidence.label_orientation_consistent
    assert not evidence.finite_regions_watertight
    assert evidence.status is MultiRegionSurfaceStatus.LABEL_ORIENTATION_INCONSISTENT


def _four_sheet_seed() -> MultiRegionSurfaceSeed:
    angles = np.radians((10.0, 100.0, 190.0, 280.0))
    points = [(0.02, -0.01, 0.0), (0.031, 0.004, 1.0)]
    for angle in angles:
        points += [
            (np.cos(angle), np.sin(angle), 0.0),
            (np.cos(angle), np.sin(angle), 1.0),
        ]
    faces, labels = [], []
    for sheet in range(4):
        bottom, top = 2 + 2 * sheet, 3 + 2 * sheet
        faces += [(0, bottom, top), (0, top, 1)]
        labels += [(sheet, (sheet - 1) % 4)] * 2
    return MultiRegionSurfaceSeed(
        np.asarray(points),
        np.asarray(faces),
        np.asarray(labels),
        ("q0", "q1", "q2", "q3"),
        ("boundary",) * 4,
        source="four-sheet-edge",
    )


def test_nonphysical_valence_and_capacity_are_refused() -> None:
    four = _four_sheet_seed()
    _, _, general = _built(four)
    assert general.status is MultiRegionSurfaceStatus.ACCEPTED
    assert general.maximum_edge_valence == 4
    _, _, dry = _built(four, DRY_FOAM)
    assert dry.status is MultiRegionSurfaceStatus.NONPHYSICAL_VALENCE
    assert dry.nonphysical_edge_count == 1

    bubble = seed_double_bubble(1.0, 0.8, ring_points=12)
    counts = bubble.counts()
    plan = MultiRegionSurfaceCapacityPlan(
        vertex_capacity=counts.vertex,
        edge_capacity=counts.edge,
        face_capacity=counts.face,
        region_capacity=counts.region,
        region_pair_capacity=counts.region_pair,
        maximum_edge_valence=2,
        maximum_vertex_region_pairs=counts.vertex_region_pairs,
        resource_id="manifold-only",
    )
    evidence = plan.capacity_evidence(counts)
    assert not evidence.admitted
    assert evidence.exceeded == ("edge_valence",)
    with pytest.raises(MultiRegionSurfaceCapacityError) as refused:
        bubble.topology(plan)
    assert refused.value.evidence.exceeded == ("edge_valence",)


def test_periodic_domains_are_refused() -> None:
    sphere = seed_sphere(1.0, subdivisions=0)
    plan = sphere.capacity_plan(resource_id="periodic")
    with pytest.raises(ValueError, match="unwrapped periodic coordinates"):
        MultiRegionSurfaceTopology(
            plan,
            sphere.faces,
            sphere.face_labels,
            sphere.region_ids,
            sphere.region_kinds,
            vertex_count=sphere.positions.shape[0],
            domain="periodic",
        )


def test_self_intersection_is_detected_by_exact_narrow_phase() -> None:
    first = seed_sphere(1.0, subdivisions=1)
    second = seed_sphere(1.0, center=(1.1, 0.07, 0.03), subdivisions=1)
    count = first.positions.shape[0]
    overlapping = MultiRegionSurfaceSeed(
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
        source="overlapping-spheres",
    )
    _, _, evidence = _built(overlapping)
    assert evidence.status is MultiRegionSurfaceStatus.SELF_INTERSECTION
    assert evidence.intersecting_pair_count > 0
    separated = MultiRegionSurfaceSeed(
        np.concatenate((first.positions, second.positions + np.asarray((1.5, 0.0, 0.0)))),
        overlapping.faces,
        overlapping.face_labels,
        overlapping.region_ids,
        overlapping.region_kinds,
        source="separated-spheres",
    )
    _, _, evidence = _built(separated)
    assert evidence.status is MultiRegionSurfaceStatus.ACCEPTED
    assert evidence.intersecting_pair_count == 0


def test_sheet_view_matches_independently_built_manifold_mesh() -> None:
    seed = seed_double_bubble(1.0, 0.8, ring_points=18)
    topology, state, _ = _built(seed)
    prepared = PreparedMultiRegionSurface(topology, state)
    views = multiregion_sheet_views(prepared, state)
    assert views.nonmanifold_pair_indices == ()
    wall = next(
        view for view in views.views if view.region_ids == ("bubble-1", "bubble-2")
    )
    selected = np.flatnonzero(
        (seed.face_labels[:, 0] == 0) & (seed.face_labels[:, 1] == 1)
    )
    vertices, local = np.unique(seed.faces[selected], return_inverse=True)
    independent = TriangleMesh(seed.positions[vertices], local.reshape((-1, 3)))
    np.testing.assert_array_equal(np.asarray(wall.vertex_indices), vertices)
    np.testing.assert_allclose(
        np.asarray(wall.mesh.vertices), np.asarray(independent.vertices)
    )
    np.testing.assert_array_equal(
        np.asarray(wall.mesh.faces), np.asarray(independent.faces)
    )
    np.testing.assert_allclose(
        np.asarray(wall.mesh.face_areas), np.asarray(independent.face_areas), rtol=1e-14
    )
    assert wall.mesh.topology.euler_characteristic == 1
    boundary = np.asarray(wall.mesh.topology.boundary_loop_vertices)
    assert set(vertices[boundary].tolist()) == set(seed.vertex_set("junction"))
    # The view's normals point out of the pair's first region (bubble-1 into bubble-2).
    assert float(jnp.mean(wall.mesh.face_normals[:, 2])) > 0.9
    slot_table = np.asarray(topology.vertex_pair_slots).reshape(-1)
    pairs = np.asarray(topology.region_pairs)
    assert np.all(slot_table[np.asarray(wall.slot_indices)] == wall.pair_index)
    assert tuple(pairs[wall.pair_index]) == (0, 1)
