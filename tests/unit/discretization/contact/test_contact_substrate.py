#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.contact._ccd import (
    CCDStatus,
    CertifiedAABBCCDPlan,
    collision_free_step_limit,
    InclusionCCDPlan,
)
from phydrax.discretization.contact._guarantee import ContactGuaranteeLevel
from phydrax.discretization.contact._inversion import (
    InversionStatus,
    simplex_inversion_step_limit,
    SimplexInversionStepPlan,
)
from phydrax.discretization.contact._participant import (
    ContactParticipantScene,
    LinearContactParticipant,
)


def _segment_scene(gap: Any = 0.05) -> Any:
    source = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    moving_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0, 1), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        body_ids=0,
    )
    moving = phx.discretization.PreparedCollisionSurface(
        moving_plan,
        jnp.asarray(((-0.5, gap), (0.5, gap)), dtype=jnp.float64),
        phx.discretization.selection_collision_operator(
            source, jnp.asarray((0, 1), dtype=jnp.int32)
        ),
    )
    static_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((10, 11), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        body_ids=1,
        static_mask=True,
    )
    static = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0)), dtype=jnp.float64),
        phx.discretization.static_collision_operator(source, 2, 2, dtype=np.float64),
    )
    return source, phx.discretization.PreparedCollisionScene((moving, static))


def test_contact_substrate_scenario_1() -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    cells = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(coordinates, cells)
    space = phx.linalg.ArraySpace((3, 2), dtype=np.float64)
    surface = phx.discretization.prepare_cell_mesh_collision_surface(mesh, space)
    state = jnp.arange(6.0).reshape((3, 2)) / 7.0
    dual = jnp.asarray(((0.2, -0.3), (0.7, 0.1), (-0.4, 0.9)))
    evidence = surface.duality_evidence(state, dual)

    assert surface.plan.edge_count == 3
    assert bool(evidence.valid)
    np.testing.assert_allclose(evidence.residual, 0.0, atol=1.0e-14)
    edge = phx.discretization.point_edge_distance(
        jnp.asarray((0.25, 0.2)),
        jnp.asarray((0.0, 0.0)),
        jnp.asarray((1.0, 0.0)),
    )
    triangle = phx.discretization.point_triangle_distance(
        jnp.asarray((0.2, 0.3, 0.4)),
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
        jnp.asarray((0.0, 1.0, 0.0)),
    )
    edges = phx.discretization.edge_edge_distance(
        jnp.asarray((-1.0, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
        jnp.asarray((0.0, -1.0, 0.5)),
        jnp.asarray((0.0, 1.0, 0.5)),
    )

    np.testing.assert_allclose(edge.squared_distance, 0.04, atol=1.0e-14)
    assert int(edge.feature) == int(phx.discretization.PointEdgeFeature.POINT_EDGE)
    np.testing.assert_allclose(triangle.squared_distance, 0.16, atol=1.0e-14)
    assert int(triangle.feature) == int(
        phx.discretization.PointTriangleFeature.POINT_FACE
    )
    np.testing.assert_allclose(edges.squared_distance, 0.25, atol=1.0e-14)
    assert int(edges.feature) == int(phx.discretization.EdgeEdgeFeature.EDGE_EDGE)
    source, scene = _segment_scene()
    state = source.zeros()
    positions = scene.positions(state)
    dense = phx.discretization.DenseContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    ).build(scene, positions)
    sweep = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    ).build(scene, positions)
    overflow = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=1,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    ).build(scene, positions)

    assert bool(dense.successful)
    assert bool(sweep.successful)
    assert int(dense.candidate_count) == int(sweep.candidate_count)
    np.testing.assert_array_equal(
        np.sort(np.asarray(dense.edge_vertex.route_keys[dense.edge_vertex.valid])),
        np.sort(np.asarray(sweep.edge_vertex.route_keys[sweep.edge_vertex.valid])),
    )
    assert bool(overflow.edge_vertex.overflow)
    assert not bool(overflow.successful)
    assert not bool(jnp.any(overflow.edge_vertex.valid))


def test_conservative_ccd_and_simplex_inversion_limit_motion() -> None:
    source, scene = _segment_scene(gap=0.5)
    start_state = source.zeros()
    end_state = jnp.broadcast_to(jnp.asarray((0.0, -1.0)), source.shape)
    start = scene.positions(start_state)
    end = scene.positions(end_state)
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    epoch = search.build(scene, start, end_positions=end)
    safety = phx.discretization.collision_free_step_limit(
        phx.discretization.InclusionCCDPlan(time_tolerance=1.0e-7),
        scene,
        epoch,
        start,
        end,
    )

    reference = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    inverted = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, -1.0)))
    inversion = phx.discretization.simplex_inversion_step_limit(
        phx.discretization.SimplexInversionStepPlan(
            jnp.asarray(((0, 1, 2),), dtype=jnp.int32), reference
        ),
        reference,
        inverted,
    )

    assert bool(safety.successful)
    assert 0.0 < safety.step_size < 0.5
    assert bool(inversion.successful)
    assert 0.0 < inversion.step_size < 0.5


@pytest.mark.strict_jax
@pytest.mark.parametrize("certified", [False, True], ids=["inclusion", "rounding-aabb"])
def test_participant_scene_ccd_certifies_entire_path_against_exact_segments(
    certified: bool,
) -> None:
    source, shared_scene = _segment_scene(gap=0.5)
    scene = ContactParticipantScene(
        tuple(LinearContactParticipant(surface) for surface in shared_scene.surfaces)
    )
    start = scene.positions((source.zeros(), source.zeros()))
    displacement = jnp.broadcast_to(
        jnp.asarray((0.0, -1.0), dtype=jnp.float64), source.shape
    )
    end = scene.positions((displacement, source.zeros()))
    epoch = phx.discretization.DenseContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    ).build(scene, start, end_positions=end)
    plan = (
        CertifiedAABBCCDPlan(time_tolerance=1.0e-7)
        if certified
        else InclusionCCDPlan(time_tolerance=1.0e-7)
    )
    evidence = eqx.filter_jit(collision_free_step_limit)(plan, scene, epoch, start, end)
    alpha = float(evidence.step_size)
    assert evidence.successful
    assert evidence.guarantee.successful
    assert int(evidence.query_count) > 0
    assert 0.0 < alpha < 0.5
    # Exact, independent predicate for these overlapping horizontal segments:
    # signed separation is 1/2-t; its minimum on [0,alpha] is at alpha.
    assert min(0.5, 0.5 - alpha) > 0.0
    np.testing.assert_allclose(evidence.minimum_time_of_impact, 0.5, atol=2.0e-7)
    # Both *unsigned endpoint* distances are positive despite interior crossing.
    assert abs(0.5 - 1.0) > 0.0


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    "scenario",
    ["nonfinite", "initial-contact", "work-limit", "search-incomplete"],
    ids=["nonfinite", "initial-contact", "work-limit", "search-incomplete"],
)
def test_ccd_refusal_preserves_reason_and_zero_bound(scenario: str) -> None:
    source, scene = _segment_scene(gap=0.0 if scenario == "initial-contact" else 0.5)
    start = scene.positions(source.zeros())
    end = scene.positions(
        jnp.broadcast_to(jnp.asarray((0.0, -1.0), dtype=jnp.float64), source.shape)
    )
    epoch = phx.discretization.DenseContactSearchPlan(
        edge_vertex_capacity=1 if scenario == "search-incomplete" else 16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    ).build(scene, start, end_positions=end)
    if scenario == "nonfinite":
        end = end.at[0, 0].set(jnp.nan)
        expected = CCDStatus.NONFINITE_INPUT
    elif scenario == "initial-contact":
        expected = CCDStatus.INITIAL_DISTANCE_VIOLATION
    elif scenario == "search-incomplete":
        expected = CCDStatus.SEARCH_INCOMPLETE
    else:
        expected = CCDStatus.WORK_LIMIT
    plan = InclusionCCDPlan(maximum_iterations=1 if scenario == "work-limit" else 1000)
    evidence = eqx.filter_jit(collision_free_step_limit)(plan, scene, epoch, start, end)
    assert not evidence.successful
    assert not evidence.guarantee.successful
    assert evidence.step_size == 0.0
    assert int(evidence.status) == int(expected)


@pytest.mark.strict_jax
@pytest.mark.parametrize("dimension", [2, 3], ids=["triangle", "tetrahedron"])
def test_simplex_bound_detects_interior_collapse_with_positive_endpoints(
    dimension: int,
) -> None:
    reference = np.concatenate((np.zeros((1, dimension)), np.eye(dimension)), axis=0)
    reflected = reference.copy()
    reflected[:, :2] *= -1.0
    plan = SimplexInversionStepPlan(
        np.arange(dimension + 1, dtype=np.int32)[None, :],
        reference,
        minimum_ratio=0.0,
    )
    evidence = eqx.filter_jit(simplex_inversion_step_limit)(plan, reference, reflected)
    assert evidence.successful
    assert 0.0 < evidence.step_size < 0.5
    # det(diag(1-2t,1-2t[,1]))=(1-2t)^2. Positive endpoints
    # do not exclude a zero determinant at the exact interior time t=1/2.
    np.testing.assert_allclose(evidence.minimum_initial_ratio, 1.0, atol=1.0e-14)
    np.testing.assert_allclose(evidence.minimum_time, 0.5, atol=1.0e-6)
    assert (1.0 - 2.0 * float(evidence.step_size)) ** 2 > 0.0


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    "scenario",
    ["nonfinite", "initial-inverted", "work-limit"],
    ids=["nonfinite", "initial-inverted", "work-limit"],
)
def test_simplex_bound_refuses_invalid_and_exhausted_paths(scenario: str) -> None:
    reference = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=jnp.float64)
    inverted = reference.at[2, 1].set(-1.0)
    start, end = reference, inverted
    if scenario == "nonfinite":
        end = end.at[0, 0].set(jnp.nan)
        expected = InversionStatus.NONFINITE_INPUT
    elif scenario == "initial-inverted":
        start = inverted
        expected = InversionStatus.INITIAL_INVALID
    else:
        expected = InversionStatus.WORK_LIMIT
    plan = SimplexInversionStepPlan(
        jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
        reference,
        maximum_iterations=1 if scenario == "work-limit" else 1000,
    )
    evidence = eqx.filter_jit(simplex_inversion_step_limit)(plan, start, end)
    assert not evidence.successful
    assert evidence.step_size == 0.0
    assert int(evidence.status) == int(expected)


def _parabolic_segment_bounds(
    reference: Array, lower_time: Array, upper_time: Array, /
) -> tuple[Array, Array, Array]:
    first_height = 0.5 - 4.0 * lower_time * (1.0 - lower_time)
    last_height = 0.5 - 4.0 * upper_time * (1.0 - upper_time)
    minimum_height = jnp.where(
        (lower_time <= 0.5) & (upper_time >= 0.5),
        jnp.asarray(-0.5, dtype=reference.dtype),
        jnp.minimum(first_height, last_height),
    )
    maximum_height = jnp.maximum(first_height, last_height)
    lower = reference.at[:2, 1].set(minimum_height)
    upper = reference.at[:2, 1].set(maximum_height)
    return lower, upper, jnp.asarray(True, dtype=jnp.bool_)


@pytest.mark.strict_jax
def test_curved_scene_path_uses_enclosures_not_separated_endpoints() -> None:
    source, shared_scene = _segment_scene(gap=0.5)
    scene = ContactParticipantScene(
        tuple(LinearContactParticipant(surface) for surface in shared_scene.surfaces)
    )
    positions = scene.positions((source.zeros(), source.zeros()))
    epoch = phx.discretization.DenseContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
        envelope_radius=1.0,
    ).build(scene, positions)
    plan = InclusionCCDPlan(time_tolerance=1.0e-7)
    trajectory_bounds = eqx.Partial(_parabolic_segment_bounds, positions)
    evidence = eqx.filter_jit(collision_free_step_limit)(
        plan,
        scene,
        epoch,
        positions,
        positions,
        trajectory_bounds=trajectory_bounds,
    )
    # Exact moving-segment predicate: height(t)=1/2-4t(1-t). The
    # first crossing is (1-sqrt(1/2))/2, despite identical clear endpoints.
    exact_first_crossing = 0.5 * (1.0 - np.sqrt(0.5))
    assert evidence.successful
    assert evidence.guarantee.successful
    assert int(evidence.guarantee.level) == int(
        ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE
    )
    assert 0.0 < float(evidence.step_size) < exact_first_crossing
    np.testing.assert_allclose(
        evidence.minimum_time_of_impact, exact_first_crossing, atol=2.0e-7
    )
    alpha = float(evidence.step_size)
    assert min(0.5, 0.5 - 4.0 * alpha * (1.0 - alpha)) > 0.0


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    "reason",
    [
        "failed-provider",
        "failed-subinterval",
        "nonfinite",
        "reversed",
        "work-limit",
        "stronger-guarantee",
    ],
    ids=[
        "failed-provider",
        "failed-subinterval",
        "nonfinite",
        "reversed",
        "work-limit",
        "stronger-guarantee",
    ],
)
def test_nonlinear_enclosure_refusals_retain_zero_step_and_reason(reason: str) -> None:
    source, scene = _segment_scene(gap=0.5)
    positions = scene.positions(source.zeros())
    epoch = phx.discretization.DenseContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
        envelope_radius=1.0,
    ).build(scene, positions)

    def bounds(
        reference: Array, lower_time: Array, upper_time: Array, /
    ) -> tuple[Array, Array, Array]:
        lower, upper = reference - 1.0, reference + 1.0
        if reason == "nonfinite":
            lower = lower.at[0, 0].set(jnp.nan)
        elif reason == "reversed":
            lower = upper + 1.0
        successful = (
            upper_time - lower_time >= 1.0
            if reason == "failed-subinterval"
            else jnp.asarray(reason != "failed-provider", dtype=jnp.bool_)
        )
        return lower, upper, successful

    plan = (
        CertifiedAABBCCDPlan()
        if reason == "stronger-guarantee"
        else InclusionCCDPlan(
            maximum_iterations=8 if reason == "failed-subinterval" else 1
        )
    )
    evidence = eqx.filter_jit(collision_free_step_limit)(
        plan,
        scene,
        epoch,
        positions,
        positions,
        trajectory_bounds=eqx.Partial(bounds, positions),
    )
    assert not evidence.successful
    assert not evidence.guarantee.successful
    assert evidence.step_size == 0.0
    expected = (
        CCDStatus.WORK_LIMIT
        if reason == "work-limit"
        else CCDStatus.TRAJECTORY_BOUND_FAILURE
    )
    assert int(evidence.status) == int(expected)
