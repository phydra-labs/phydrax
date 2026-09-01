#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _segment_scene(gap=0.05):
    source = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    moving_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0, 1), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        pair_policy=phx.discretization.ContactPairPolicy(
            2, body_ids=jnp.zeros((2,), dtype=jnp.int64)
        ),
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
        pair_policy=phx.discretization.ContactPairPolicy(
            2,
            body_ids=jnp.ones((2,), dtype=jnp.int64),
            static_mask=jnp.ones((2,), dtype=bool),
        ),
    )
    static = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0)), dtype=jnp.float64),
        phx.discretization.static_collision_operator(source, 2, 2, dtype=np.float64),
    )
    return source, phx.discretization.PreparedCollisionScene((moving, static))


def test_collision_surface_map_has_exact_transpose_and_stable_boundary():
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


def test_piecewise_distance_kernels_select_expected_features():
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


def test_dense_and_sweep_search_match_and_overflow_fails_closed():
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


def test_conservative_ccd_and_simplex_inversion_limit_motion():
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
