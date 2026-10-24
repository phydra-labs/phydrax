#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import pi

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.contact._participant_adapters import (
    ArticulatedPrimitiveContactParticipant,
    prepare_articulated_contact_scene,
)
from phydrax.discretization.contact._articulated_geometry import (
    ArticulatedCollisionGeometry,
    ArticulatedContactTrajectory,
    ArticulatedLiftedPath,
    ArticulatedPrimitiveKind,
)
from phydrax.discretization.contact._ccd import (
    AdmissiblePathEvidence,
    AdmissiblePathPlan,
    AdmissiblePathStatus,
    CCDStatus,
    certify_admissible_path,
    collision_free_step_limit,
    InclusionCCDPlan,
)
from phydrax.discretization.contact._guarantee import ContactGuaranteeLevel
from phydrax.discretization.contact._participant import ContactParticipantScene
from phydrax.discretization.contact._search import (
    ContactCandidateEpoch,
    DenseContactSearchPlan,
)
from phydrax.discretization.contact._stencils import ContactPrimitiveProfile
from phydrax.discretization.contact._trajectory import (
    AbstractContactTrajectory,
    CubicHermiteContactTrajectory,
    RigidSweepContactTrajectory,
)
from phydrax.discretization.particle._reduced_articulation import (
    PreparedReducedArticulation,
)


def _hinge_articulation() -> PreparedReducedArticulation:
    body_ids = jnp.asarray([10, 11], dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids, jnp.asarray([1.0, 2.0]), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        fixed_mask=jnp.asarray([True, False]),
    ).prepare(particles)
    reference = bodies.kinematics(
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64),
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        jnp.zeros((2, 3)),
    )
    graph = phx.discretization.RigidJointGraphPlan(
        hinge=phx.discretization.HingeJointSetPlan(
            jnp.asarray([20]),
            body_ids[:1],
            body_ids[1:],
            jnp.asarray([[0.0, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.0, 1.0]]),
        )
    ).prepare(bodies, reference)
    return phx.discretization.ReducedArticulationPlan(
        10, jnp.asarray([20]), jnp.asarray([10]), jnp.asarray([11])
    ).prepare(graph, reference)


def _geometry(
    articulation: PreparedReducedArticulation,
    *,
    plane_offset: float = -0.1,
    sphere_center: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> ArticulatedCollisionGeometry:
    # Capsule on the rotating link (radial span 0.5..1.5, core at z=0, radius
    # 0.1) resting on the plane z=-0.1; static sphere on the root.
    return ArticulatedCollisionGeometry(
        articulation,
        primitive_ids=(1, 2),
        primitive_body_ids=(11, 10),
        primitive_kinds=(
            ArticulatedPrimitiveKind.CAPSULE,
            ArticulatedPrimitiveKind.SPHERE,
        ),
        local_first=((-0.5, 0.0, 0.0), sphere_center),
        local_second=((0.5, 0.0, 0.0), sphere_center),
        radii=(0.1, 0.05),
        plane_ids=(7,),
        plane_normals=((0.0, 0.0, 1.0),),
        plane_offsets=(plane_offset,),
    )


def _trajectory(
    geometry: ArticulatedCollisionGeometry,
    articulation: PreparedReducedArticulation,
    start: float,
    velocity: float,
    *,
    step: float = 1.0,
    revision: int = 0,
) -> ArticulatedContactTrajectory:
    path = ArticulatedLiftedPath(
        articulation,
        jnp.asarray([start]),
        jnp.asarray([velocity]),
        step,
        revision=revision,
    )
    return ArticulatedContactTrajectory(geometry, path)


def _samples(
    trajectory: AbstractContactTrajectory, lower: float, upper: float, count: int = 129
) -> jax.Array:
    times = jnp.linspace(lower, upper, count)
    return jax.vmap(trajectory.evaluate)(times)


def _assert_enclosed(
    trajectory: AbstractContactTrajectory, lower: float, upper: float
) -> None:
    enclosure = trajectory.enclosure(lower, upper)
    samples = _samples(trajectory, lower, upper)
    assert bool(enclosure.successful)
    assert bool(jnp.all(samples >= enclosure.lower[None]))
    assert bool(jnp.all(samples <= enclosure.upper[None]))
    midpoint = trajectory.midpoint_bound(lower, upper)
    assert bool(midpoint.successful)
    displacement = jnp.linalg.norm(samples - midpoint.positions[None], axis=-1)
    assert bool(jnp.all(displacement <= midpoint.displacement[None]))


def _search(activation_distance: float) -> DenseContactSearchPlan:
    return DenseContactSearchPlan(
        edge_vertex_capacity=4,
        edge_edge_capacity=4,
        face_vertex_capacity=0,
        vertex_vertex_capacity=4,
        vertex_plane_capacity=4,
        edge_plane_capacity=4,
        activation_distance=activation_distance,
        primitive_profile=ContactPrimitiveProfile.SPHERE_CAPSULE,
    )


def _epoch(
    articulation: PreparedReducedArticulation,
    geometry: ArticulatedCollisionGeometry,
    trajectory: ArticulatedContactTrajectory,
    *,
    activation_distance: float = 0.05,
) -> tuple[ContactParticipantScene, ContactCandidateEpoch]:
    scene = prepare_articulated_contact_scene(articulation, geometry)
    epoch = _search(activation_distance).build(
        scene,
        trajectory.evaluate(0.0),
        path_enclosure=trajectory.enclosure(0.0, 1.0),
    )
    assert bool(epoch.successful)
    return scene, epoch


def test_cubic_bernstein_restriction_encloses_and_contracts() -> None:
    cubic = CubicHermiteContactTrajectory(
        jnp.zeros((1, 3)),
        jnp.asarray(((3.0, 0.0, 0.0),)),
        jnp.asarray(((1.0, 1.0, 0.0),)),
        jnp.asarray(((-3.0, 0.0, 0.0),)),
    )
    assert cubic.guarantee_level == ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE
    widths = []
    for width in (0.5, 0.05, 0.005):
        _assert_enclosed(cubic, 0.3, 0.3 + width)
        enclosure = cubic.enclosure(0.3, 0.3 + width)
        widths.append(float(jnp.max(enclosure.upper - enclosure.lower)))
    assert widths[0] > widths[1] > widths[2]
    assert widths[2] < 0.05 * widths[0]
    controls = cubic.restricted_controls(0.0, 1.0)
    np.testing.assert_allclose(controls, cubic.controls, atol=1.0e-15)


def test_rigid_sweep_full_turn_with_identical_endpoints() -> None:
    sweep = RigidSweepContactTrajectory(
        jnp.asarray(((1.0, 0.0, 0.0),)),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.eye(3),
        jnp.asarray((0.0, 0.0, 2.0 * pi)),
    )
    np.testing.assert_allclose(sweep.evaluate(0.0), sweep.evaluate(1.0), atol=1e-12)
    np.testing.assert_allclose(sweep.evaluate(0.5), ((-1.0, 0.0, 0.0),), atol=1e-12)
    _assert_enclosed(sweep, 0.0, 1.0)
    _assert_enclosed(sweep, 0.4, 0.6)


def test_articulated_full_turn_and_contraction() -> None:
    articulation = _hinge_articulation()
    geometry = _geometry(articulation)
    trajectory = _trajectory(geometry, articulation, 0.0, 2.0 * pi)
    assert trajectory.guarantee_level == ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE
    np.testing.assert_allclose(
        trajectory.evaluate(0.0), trajectory.evaluate(1.0), atol=1.0e-12
    )
    # Endpoints coincide, but the path sweeps the whole circle of radius 1.5.
    full = trajectory.enclosure(0.0, 1.0)
    assert float(full.lower[1, 0]) <= -1.5
    assert float(full.lower[1, 1]) <= -1.5
    _assert_enclosed(trajectory, 0.0, 1.0)
    widths = []
    for width in (0.5, 0.05, 0.005):
        _assert_enclosed(trajectory, 0.25, 0.25 + width)
        enclosure = trajectory.enclosure(0.25, 0.25 + width)
        widths.append(float(jnp.max(enclosure.upper[:2] - enclosure.lower[:2])))
    assert widths[0] > widths[1] > widths[2]
    assert widths[2] < 0.05 * widths[0]


def test_articulated_interior_extremum_and_reversed_path() -> None:
    articulation = _hinge_articulation()
    geometry = _geometry(articulation)
    forward = _trajectory(geometry, articulation, -0.5, 1.0)
    reverse = _trajectory(geometry, articulation, 0.5, -1.0)
    enclosure = forward.enclosure(0.0, 1.0)
    # x = 1.5 cos q peaks at the interior q = 0; endpoints give only 1.5 cos 0.5.
    assert float(enclosure.upper[1, 0]) >= 1.5 - 1.0e-12
    _assert_enclosed(forward, 0.0, 1.0)
    _assert_enclosed(reverse, 0.0, 1.0)
    samples = _samples(forward, 0.0, 1.0)
    reversed_enclosure = reverse.enclosure(0.0, 1.0)
    assert bool(jnp.all(samples >= reversed_enclosure.lower[None]))
    assert bool(jnp.all(samples <= reversed_enclosure.upper[None]))


def test_lifted_path_is_not_wrapped() -> None:
    articulation = _hinge_articulation()
    path = ArticulatedLiftedPath(
        articulation, jnp.asarray([3.0]), jnp.asarray([2.0]), 0.5
    )
    np.testing.assert_allclose(path.end_configuration(), (4.0,))
    np.testing.assert_allclose(path.configuration(0.5), (3.5,))


def test_point_force_pullback_is_velocity_transpose() -> None:
    articulation = _hinge_articulation()
    geometry = _geometry(articulation)
    configuration = jnp.asarray([0.3])
    bodies = jnp.asarray([1, 1, -1], dtype=jnp.int32)
    points = jnp.asarray(((1.0, 0.5, 0.0), (0.2, -0.4, 0.1), (0.0, 0.0, 0.0)))
    forces = jnp.asarray(((0.3, -1.0, 2.0), (1.5, 0.5, -0.2), (4.0, 4.0, 4.0)))
    velocity = jnp.asarray([0.7])
    point_velocity = geometry.point_velocities(
        articulation, configuration, velocity, bodies, points
    )
    pulled = geometry.point_force_pullback(
        articulation, configuration, bodies, points, forces
    )
    np.testing.assert_allclose(
        jnp.sum(forces * point_velocity), jnp.sum(pulled * velocity), rtol=1.0e-13
    )
    np.testing.assert_allclose(point_velocity[2], 0.0)


def test_resting_contact_is_admissible_but_not_strictly_separated() -> None:
    articulation = _hinge_articulation()
    geometry = _geometry(articulation)
    trajectory = _trajectory(geometry, articulation, 0.0, -0.5)
    scene, epoch = _epoch(articulation, geometry, trajectory)
    evidence = eqx.filter_jit(certify_admissible_path)(
        AdmissiblePathPlan(1.0e-9), scene, epoch, trajectory
    )
    assert int(evidence.status) == int(AdmissiblePathStatus.ADMISSIBLE)
    assert bool(evidence.successful)
    np.testing.assert_allclose(evidence.certified_prefix, 1.0)
    np.testing.assert_allclose(evidence.minimum_initial_gap, 0.0, atol=1.0e-12)
    strict = collision_free_step_limit(
        InclusionCCDPlan(maximum_iterations=4096),
        scene,
        epoch,
        trajectory.evaluate(0.0),
        trajectory.evaluate(1.0),
        trajectory_bounds=trajectory,
    )
    assert int(strict.status) == int(CCDStatus.INITIAL_DISTANCE_VIOLATION)


def test_initial_overlap_is_initially_inadmissible() -> None:
    articulation = _hinge_articulation()
    geometry = _geometry(articulation, plane_offset=-0.05)
    trajectory = _trajectory(geometry, articulation, 0.0, -0.5)
    scene, epoch = _epoch(articulation, geometry, trajectory)
    evidence = certify_admissible_path(
        AdmissiblePathPlan(1.0e-9), scene, epoch, trajectory
    )
    assert int(evidence.status) == int(AdmissiblePathStatus.INITIALLY_INADMISSIBLE)
    assert not bool(evidence.successful)
    np.testing.assert_allclose(evidence.minimum_initial_gap, -0.05, atol=1.0e-12)


def test_full_turn_through_obstacle_leaves_only_a_prefix() -> None:
    articulation = _hinge_articulation()
    geometry = _geometry(articulation)
    # Identical separated endpoints; the capsule passes through the static
    # sphere at q = pi/2.
    trajectory = _trajectory(geometry, articulation, 0.0, 2.0 * pi)
    scene, epoch = _epoch(articulation, geometry, trajectory)
    evidence = certify_admissible_path(
        AdmissiblePathPlan(1.0e-9), scene, epoch, trajectory
    )
    assert int(evidence.status) == int(AdmissiblePathStatus.CERTIFIED_PREFIX)
    assert not bool(evidence.successful)
    assert 0.0 <= float(evidence.certified_prefix) < 0.25
    # Strict separation with the plane moved away reports an impact before the
    # obstacle, although both endpoints are identical and separated.
    far = _geometry(articulation, plane_offset=-5.0)
    far_trajectory = _trajectory(far, articulation, 0.0, 2.0 * pi)
    far_scene, far_epoch = _epoch(articulation, far, far_trajectory)
    strict = collision_free_step_limit(
        InclusionCCDPlan(maximum_iterations=4096),
        far_scene,
        far_epoch,
        far_trajectory.evaluate(0.0),
        far_trajectory.evaluate(1.0),
        trajectory_bounds=far_trajectory,
    )
    assert int(strict.status) == int(CCDStatus.SUCCESS)
    assert float(strict.minimum_time_of_impact) < 0.25


def test_coverage_from_another_trajectory_revision_refuses() -> None:
    articulation = _hinge_articulation()
    geometry = _geometry(articulation)
    searched = _trajectory(geometry, articulation, 0.0, -0.5, revision=0)
    scene, epoch = _epoch(articulation, geometry, searched)
    certified = _trajectory(geometry, articulation, 0.0, -0.5, revision=1)
    evidence = certify_admissible_path(
        AdmissiblePathPlan(1.0e-9), scene, epoch, certified
    )
    assert int(evidence.status) == int(AdmissiblePathStatus.COVERAGE_MISMATCH)
    wider = _trajectory(geometry, articulation, 0.0, -2.0, revision=0)
    evidence = certify_admissible_path(AdmissiblePathPlan(1.0e-9), scene, epoch, wider)
    assert int(evidence.status) == int(AdmissiblePathStatus.COVERAGE_MISMATCH)


def test_oblique_separated_features_with_overlapping_boxes_certify() -> None:
    articulation = _hinge_articulation()
    # The static capsule axis lies on y = x; the sphere sits inside the
    # capsule's axis-aligned box but 0.283 from its axis (gap 0.133).
    geometry = _geometry(articulation, sphere_center=(0.9, 0.5, 0.0))
    trajectory = _trajectory(geometry, articulation, pi / 4.0, 0.0)
    enclosure = trajectory.enclosure(0.0, 1.0)
    capsule_lower = jnp.min(enclosure.lower[:2], axis=0)
    capsule_upper = jnp.max(enclosure.upper[:2], axis=0)
    sphere = enclosure.lower[2]
    assert bool(
        jnp.all((sphere[:2] >= capsule_lower[:2]) & (sphere[:2] <= capsule_upper[:2]))
    )
    scene, epoch = _epoch(articulation, geometry, trajectory, activation_distance=0.5)
    assert int(epoch.edge_vertex.actual_count) == 1
    evidence = certify_admissible_path(
        AdmissiblePathPlan(1.0e-9), scene, epoch, trajectory
    )
    assert int(evidence.status) == int(AdmissiblePathStatus.ADMISSIBLE)
    assert int(evidence.interval_count) == 1
    np.testing.assert_allclose(evidence.minimum_lower_gap, 0.0, atol=1.0e-9)


def _fixed_articulation() -> PreparedReducedArticulation:
    body_ids = jnp.asarray([10, 11], dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids, jnp.asarray([1.0, 1.0]), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        fixed_mask=jnp.asarray([True, False]),
    ).prepare(particles)
    reference = bodies.kinematics(
        jnp.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float64),
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        jnp.zeros((2, 3)),
    )
    graph = phx.discretization.RigidJointGraphPlan(
        fixed=phx.discretization.FixedJointSetPlan(
            jnp.asarray([20]), body_ids[:1], body_ids[1:]
        )
    ).prepare(bodies, reference)
    return phx.discretization.ReducedArticulationPlan(
        10, jnp.asarray([20]), jnp.asarray([10]), jnp.asarray([11])
    ).prepare(graph, reference)


def _crossing_capsules(
    articulation: PreparedReducedArticulation, height: float
) -> ArticulatedCollisionGeometry:
    # Root capsule A = (0,0,0)->(1,0,0); fixed-child capsule B crosses A's axis
    # at a 2**-28 slope: (0,-2**-28,h)->(1,2**-28,h) in world coordinates.
    slope = 2.0**-28
    radius = 2.0**-30
    return ArticulatedCollisionGeometry(
        articulation,
        primitive_ids=(1, 2),
        primitive_body_ids=(10, 11),
        primitive_kinds=(ArticulatedPrimitiveKind.CAPSULE,) * 2,
        local_first=((0.0, 0.0, 0.0), (-5.0, -slope, height)),
        local_second=((1.0, 0.0, 0.0), (-4.0, slope, height)),
        radii=(radius, radius),
    )


def test_fixed_only_tree_has_static_enclosure() -> None:
    articulation = _fixed_articulation()
    assert articulation.nq == 0
    geometry = _crossing_capsules(articulation, 0.1)
    trajectory = ArticulatedContactTrajectory(
        geometry, ArticulatedLiftedPath(articulation, jnp.zeros(0), jnp.zeros(0), 1.0)
    )
    positions = trajectory.evaluate(0.5)
    enclosure = trajectory.enclosure(0.0, 1.0)
    assert bool(enclosure.successful)
    assert bool(jnp.all(enclosure.lower <= positions))
    assert bool(jnp.all(positions <= enclosure.upper))
    assert float(jnp.max(enclosure.upper - enclosure.lower)) < 1.0e-12


def test_near_parallel_crossing_capsules_are_not_admissible() -> None:
    articulation = _fixed_articulation()
    plan = AdmissiblePathPlan(1.0e-12)

    def certify(height: float) -> AdmissiblePathEvidence:
        geometry = _crossing_capsules(articulation, height)
        trajectory = ArticulatedContactTrajectory(
            geometry,
            ArticulatedLiftedPath(articulation, jnp.zeros(0), jnp.zeros(0), 1.0),
        )
        # Activation beyond the 0.1 separation keeps the pair a route, so the
        # separated control exercises the interval distance bound.
        scene, epoch = _epoch(articulation, geometry, trajectory, activation_distance=0.5)
        assert int(epoch.edge_edge.actual_count) >= 1
        return certify_admissible_path(plan, scene, epoch, trajectory)

    # The axes intersect, so the true gap is -2**-29 despite the binary64
    # determinant cancellation of the closest-point solve.
    overlapping = certify(0.0)
    assert int(overlapping.status) != int(AdmissiblePathStatus.ADMISSIBLE)
    assert int(overlapping.status) == int(AdmissiblePathStatus.UNRESOLVED_INTERVAL)
    separated = certify(0.1)
    assert int(separated.status) == int(AdmissiblePathStatus.ADMISSIBLE)


def test_non_unit_plane_normal_refuses_certificates() -> None:
    articulation = _hinge_articulation()
    geometry = _geometry(articulation)
    trajectory = _trajectory(geometry, articulation, 0.0, -0.5)
    scene, epoch = _epoch(articulation, geometry, trajectory)
    scaled = eqx.tree_at(
        lambda value: value.planes[0].unit_normal,
        scene,
        jnp.asarray((0.0, 0.0, 2.0)),
    )
    evidence = certify_admissible_path(
        AdmissiblePathPlan(1.0e-9), scaled, epoch, trajectory
    )
    assert int(evidence.status) == int(AdmissiblePathStatus.INVALID_GEOMETRY)
    strict = collision_free_step_limit(
        InclusionCCDPlan(maximum_iterations=4096),
        scaled,
        epoch,
        trajectory.evaluate(0.0),
        trajectory.evaluate(1.0),
        trajectory_bounds=trajectory,
    )
    assert int(strict.status) == int(CCDStatus.INVALID_GEOMETRY)


def _physical_radius(scene: ContactParticipantScene) -> jax.Array:
    participant = scene.participants[0]
    assert isinstance(participant, ArticulatedPrimitiveContactParticipant)
    return participant.geometry.surface_plan.feature_policy.physical_radius


def test_epoch_from_foreign_scene_or_numeric_geometry_refuses() -> None:
    articulation = _hinge_articulation()
    geometry = _geometry(articulation)
    trajectory = _trajectory(geometry, articulation, 0.0, 2.0 * pi)
    excluded = ArticulatedCollisionGeometry(
        articulation,
        primitive_ids=(1, 2),
        primitive_body_ids=(11, 10),
        primitive_kinds=(
            ArticulatedPrimitiveKind.CAPSULE,
            ArticulatedPrimitiveKind.SPHERE,
        ),
        local_first=((-0.5, 0.0, 0.0), (0.0, 1.0, 0.0)),
        local_second=((0.5, 0.0, 0.0), (0.0, 1.0, 0.0)),
        radii=(0.1, 0.05),
        plane_ids=(7,),
        plane_normals=((0.0, 0.0, 1.0),),
        plane_offsets=(-0.1,),
        excluded_body_pairs=((10, 11),),
    )
    excluded_trajectory = ArticulatedContactTrajectory(excluded, trajectory.path)
    _, foreign_epoch = _epoch(articulation, excluded, excluded_trajectory)
    scene = prepare_articulated_contact_scene(articulation, geometry)
    with pytest.raises(ValueError):
        certify_admissible_path(
            AdmissiblePathPlan(1.0e-9), scene, foreign_epoch, trajectory
        )
    resting = _trajectory(geometry, articulation, 0.0, -0.5)
    scene, epoch = _epoch(articulation, geometry, resting)
    grown = eqx.tree_at(_physical_radius, scene, _physical_radius(scene) * 2.0)
    evidence = certify_admissible_path(AdmissiblePathPlan(1.0e-9), grown, epoch, resting)
    assert int(evidence.status) == int(AdmissiblePathStatus.COVERAGE_MISMATCH)


def _serial_articulation(kind: str) -> PreparedReducedArticulation:
    # Root -> body 11 -> body 12 with two joints of one kind. Prismatic joints
    # slide along +z with anchors at the origin; hinges turn about +z with
    # anchors at x = 0 and x = 1.
    body_ids = jnp.asarray([10, 11, 12], dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids, jnp.ones(3), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.broadcast_to(jnp.eye(3), (3, 3, 3)),
        fixed_mask=jnp.asarray([True, False, False]),
    ).prepare(particles)
    positions = (
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5]])
        if kind == "prismatic"
        else jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    )
    reference = bodies.kinematics(
        positions,
        jnp.zeros((3, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 3),
        jnp.zeros((3, 3)),
    )
    axes = jnp.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    if kind == "prismatic":
        joints = phx.discretization.RigidJointGraphPlan(
            prismatic=phx.discretization.PrismaticJointSetPlan(
                jnp.asarray([20, 21]), body_ids[:2], body_ids[1:], jnp.zeros((2, 3)), axes
            )
        )
    else:
        joints = phx.discretization.RigidJointGraphPlan(
            hinge=phx.discretization.HingeJointSetPlan(
                jnp.asarray([20, 21]),
                body_ids[:2],
                body_ids[1:],
                jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                axes,
            )
        )
    graph = joints.prepare(bodies, reference)
    return phx.discretization.ReducedArticulationPlan(
        10, jnp.asarray([20, 21]), body_ids[:2], body_ids[1:]
    ).prepare(graph, reference)


def _body_sphere(
    articulation: PreparedReducedArticulation, radius: float
) -> ArticulatedCollisionGeometry:
    return ArticulatedCollisionGeometry(
        articulation,
        primitive_ids=(1,),
        primitive_body_ids=(12,),
        primitive_kinds=(ArticulatedPrimitiveKind.SPHERE,),
        local_first=((0.0, 0.0, 0.0),),
        local_second=((0.0, 0.0, 0.0),),
        radii=(radius,),
        plane_ids=(7,),
        plane_normals=((0.0, 0.0, 1.0),),
        plane_offsets=(0.0,),
    )


def test_cancelling_prismatic_ancestors_certify_resting_contact() -> None:
    articulation = _serial_articulation("prismatic")
    # The parent slides down while the child slides up: the sphere center stays
    # at z = 0.5 and rests on the plane z = 0 with radius 0.5.
    geometry = _body_sphere(articulation, 0.5)
    trajectory = ArticulatedContactTrajectory(
        geometry,
        ArticulatedLiftedPath(articulation, jnp.zeros(2), jnp.asarray([-1.0, 1.0]), 0.01),
    )
    np.testing.assert_allclose(trajectory.evaluate(0.0), trajectory.evaluate(1.0))
    for width in (1.0, 2.0**-10):
        _assert_enclosed(trajectory, 0.25, 0.25 + width * 0.75)
    full = trajectory.enclosure(0.0, 1.0)
    assert float(jnp.max(full.upper - full.lower)) < 1.0e-12
    scene, epoch = _epoch(articulation, geometry, trajectory)
    evidence = certify_admissible_path(
        AdmissiblePathPlan(1.0e-9), scene, epoch, trajectory
    )
    assert int(evidence.status) == int(AdmissiblePathStatus.ADMISSIBLE)
    assert int(evidence.interval_count) == 1


def test_zero_net_rotation_of_moving_hinges_keeps_curved_enclosure() -> None:
    articulation = _serial_articulation("hinge")
    # Opposite hinge increments keep the child's orientation fixed while its
    # center moves on a circle about the root; endpoint hulls would miss the arc.
    geometry = _body_sphere(articulation, 0.1)
    trajectory = ArticulatedContactTrajectory(
        geometry,
        ArticulatedLiftedPath(articulation, jnp.zeros(2), jnp.asarray([pi, -pi]), 1.0),
    )
    samples = _samples(trajectory, 0.0, 1.0)
    np.testing.assert_allclose(samples[0, 0, :2], (2.0, 0.0), atol=1.0e-12)
    np.testing.assert_allclose(samples[-1, 0, :2], (0.0, 0.0), atol=1.0e-12)
    assert float(jnp.max(samples[:, 0, 1])) > 0.9
    _assert_enclosed(trajectory, 0.0, 1.0)
    _assert_enclosed(trajectory, 0.25, 0.5)
