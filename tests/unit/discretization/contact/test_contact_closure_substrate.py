#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _two_segment_scene(*, envelope=0.0):
    source = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    moving_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0, 1), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        minimum_separation=jnp.asarray((0.01, 0.02)),
    )
    moving = phx.discretization.PreparedCollisionSurface(
        moving_plan,
        jnp.asarray(((-0.5, 0.5), (0.5, 0.5))),
        phx.discretization.selection_collision_operator(source, jnp.asarray((0, 1))),
    )
    static_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((10, 11), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        pair_policy=phx.discretization.ContactPairPolicy(
            2,
            body_ids=jnp.ones((2,), dtype=jnp.int64),
            material_ids=jnp.ones((2,), dtype=jnp.int64),
            static_mask=jnp.ones((2,), dtype=bool),
        ),
    )
    static = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0))),
        phx.discretization.static_collision_operator(source, 2, 2),
    )
    scene = phx.discretization.PreparedCollisionScene((moving, static))
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
        envelope_radius=envelope,
    )
    return source, scene, search


def _candidate_rows(batch):
    indices = np.asarray(batch.vertex_indices)
    return {tuple(row) for row in indices[np.asarray(batch.valid, dtype=bool)].tolist()}


def test_per_vertex_separation_and_certified_ccd_guarantee():
    source, scene, search = _two_segment_scene()
    np.testing.assert_allclose(scene.minimum_separation[:2], (0.01, 0.02))
    start = scene.positions(source.zeros())
    end_state = jnp.broadcast_to(jnp.asarray((0.0, -1.0)), source.shape)
    end = scene.positions(end_state)
    epoch = search.build(scene, start, end_positions=end)
    safety = phx.discretization.collision_free_step_limit(
        phx.discretization.CertifiedAABBCCDPlan(time_tolerance=1.0e-6),
        scene,
        epoch,
        start,
        end,
    )

    assert bool(safety.successful)
    assert int(safety.guarantee.level) == int(
        phx.discretization.ContactGuaranteeLevel.ROUNDING_CERTIFIED
    )
    assert 0.0 < safety.step_size < 0.5


def test_cached_contact_search_reuses_then_rebuilds_inside_skin():
    source, scene, search = _two_segment_scene(envelope=0.2)
    cache = phx.discretization.CachedContactSearchPlan(search, skin=0.2)
    initial_positions = scene.positions(source.zeros())
    state = cache.initialize(scene, initial_positions)
    small = scene.positions(jnp.broadcast_to(jnp.asarray((0.0, -0.04)), source.shape))
    reused = cache.update(scene, state, small)
    large = scene.positions(jnp.broadcast_to(jnp.asarray((0.0, -0.15)), source.shape))
    rebuilt = cache.update(scene, reused.candidate, large)

    assert bool(reused.reused)
    assert int(reused.candidate.reuse_count) == 1
    assert not bool(rebuilt.reused)
    assert int(rebuilt.candidate.rebuild_count) == 2


def test_independent_participants_search_and_force_duality():
    source_a = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    source_b = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    plan_a = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0, 1)),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),)),
        pair_policy=phx.discretization.ContactPairPolicy(
            2, material_ids=jnp.zeros((2,), dtype=jnp.int64)
        ),
    )
    plan_b = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((2, 3)),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),)),
        pair_policy=phx.discretization.ContactPairPolicy(
            2,
            body_ids=jnp.ones((2,), dtype=jnp.int64),
            material_ids=jnp.ones((2,), dtype=jnp.int64),
        ),
    )
    surface_a = phx.discretization.PreparedCollisionSurface(
        plan_a,
        jnp.asarray(((-0.5, 0.05), (0.5, 0.05))),
        phx.discretization.selection_collision_operator(source_a, jnp.asarray((0, 1))),
    )
    surface_b = phx.discretization.PreparedCollisionSurface(
        plan_b,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0))),
        phx.discretization.selection_collision_operator(source_b, jnp.asarray((0, 1))),
    )
    participant_a = phx.discretization.LinearContactParticipant(surface_a)
    participant_b = phx.discretization.LinearContactParticipant(surface_b)
    scene = phx.discretization.ContactParticipantScene((participant_a, participant_b))
    states = (source_a.zeros(), source_b.zeros())
    positions = scene.positions(states)
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    epoch = search.build(scene, positions)
    force = jnp.arange(4.0).reshape((2, 2)) / 3.0
    evidence = participant_a.duality_evidence(states[0], jnp.ones_like(states[0]), force)

    assert bool(epoch.successful)
    assert epoch.candidate_count > 0
    assert bool(evidence.valid)


def test_proxy_implicit_and_trajectory_bounds_are_explicit():
    source = phx.linalg.ArraySpace((3, 3), dtype=np.float64)
    topology = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0, 1, 2)),
        ambient_dimension=3,
        faces=jnp.asarray(((0, 1, 2),)),
        minimum_separation=0.01,
    )
    proxy = phx.discretization.ContactProxyPlan(
        topology, jnp.asarray((0.001, 0.002, 0.003)), certified=True
    ).prepare(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        phx.discretization.selection_collision_operator(source, jnp.arange(3)),
    )
    sphere = phx.discretization.SphereContactGeometry((0.0, 0.0, 0.0), 1.0)
    sphere_evaluation = sphere.evaluate(jnp.asarray(((2.0, 0.0, 0.0),)))
    cubic = phx.discretization.CubicHermiteContactTrajectory(
        jnp.zeros((1, 3)),
        jnp.asarray(((1.0, 0.0, 0.0),)),
        jnp.asarray(((1.0, 1.0, 0.0),)),
        jnp.asarray(((0.0, 1.0, 0.0),)),
    )
    lower, upper = cubic.bounds(0.0, 1.0)
    samples = jnp.stack(tuple(cubic.evaluate(t) for t in jnp.linspace(0.0, 1.0, 21)))

    np.testing.assert_allclose(
        proxy.surface.plan.vertex_minimum_separation,
        (0.011, 0.012, 0.013),
    )
    np.testing.assert_allclose(sphere_evaluation.signed_distance, 1.0)
    assert bool(proxy.evidence.successful)
    assert bool(jnp.all(samples >= lower) & jnp.all(samples <= upper))


def test_interface_traction_and_distributed_route_ownership_are_balanced():
    interface = phx.discretization.ContactInterfacePlan(
        jnp.asarray(((0, 1),)),
        jnp.asarray(((0.5, 0.5),)),
        jnp.asarray(((0, 1),)),
        jnp.asarray(((0.5, 0.5),)),
        jnp.asarray(((0.0, 1.0),)),
        jnp.asarray((2.0,)),
        plus_node_count=2,
        minus_node_count=2,
    )
    residual = phx.discretization.assemble_contact_interface_traction(
        interface, jnp.asarray(((0.0, 3.0),))
    )
    source, scene, search = _two_segment_scene()
    epoch = search.build(scene, scene.positions(source.zeros()))
    partition = phx.discretization.DistributedContactPartitionPlan(
        jnp.asarray((0, 0, 1, 1)), rank_count=2, halo_capacity=16
    )
    distributed = phx.discretization.partition_contact_epoch(partition, epoch)

    np.testing.assert_allclose(residual.action_reaction_residual, 0.0)
    assert bool(residual.successful)
    assert bool(distributed.complete)


def test_compiled_search_matches_host_candidates_for_small_scene():
    source, scene, search = _two_segment_scene()
    displacement = jnp.broadcast_to(jnp.asarray((0.0, -0.45)), source.shape)
    positions = scene.positions(displacement)
    host = search.build(scene, positions)
    compiled = phx.discretization.CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    ).evaluate(positions)
    lbvh = phx.discretization.LBVHContactSearchPlan(
        scene,
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
        morton_bits=10,
        maximum_tree_depth=16,
        maximum_traversal_visits=64,
    ).evaluate(positions)

    assert bool(compiled.evidence.complete)
    assert int(compiled.evidence.candidate_count) == int(host.candidate_count)
    assert bool(lbvh.evidence.complete)
    assert int(lbvh.evidence.candidate_count) == int(host.candidate_count)

    assert _candidate_rows(lbvh.edge_vertex) == _candidate_rows(compiled.edge_vertex)
    assert _candidate_rows(lbvh.edge_edge) == _candidate_rows(compiled.edge_edge)
    assert _candidate_rows(lbvh.face_vertex) == _candidate_rows(compiled.face_vertex)
    primitive_count = scene.vertex_count + scene.edges.shape[0] + scene.faces.shape[0]
    assert int(lbvh.evidence.node_count) == 2 * primitive_count - 1
    assert int(lbvh.evidence.tree_depth) == 4
    assert 0 < int(lbvh.evidence.traversal_visits) <= 64


def test_lbvh_matches_compiled_three_dimensional_candidates():
    positions = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.05),
            (1.0, 0.0, 0.05),
            (0.0, 1.0, 0.05),
        )
    )
    source = phx.linalg.ArraySpace((6, 3), dtype=np.float64)
    topology = phx.discretization.CollisionSurfacePlan(
        jnp.arange(6, dtype=jnp.int64),
        ambient_dimension=3,
        faces=jnp.asarray(((0, 1, 2), (3, 4, 5)), dtype=jnp.int32),
    )
    surface = phx.discretization.PreparedCollisionSurface(
        topology,
        positions,
        phx.discretization.selection_collision_operator(
            source, jnp.arange(6, dtype=jnp.int32)
        ),
    )
    scene = phx.discretization.PreparedCollisionScene((surface,))
    compiled = phx.discretization.CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=0,
        edge_edge_capacity=64,
        face_vertex_capacity=32,
        activation_distance=0.1,
    ).evaluate(positions)
    lbvh = phx.discretization.LBVHContactSearchPlan(
        scene,
        edge_vertex_capacity=0,
        edge_edge_capacity=64,
        face_vertex_capacity=32,
        activation_distance=0.1,
        maximum_tree_depth=16,
        maximum_traversal_visits=196,
    ).evaluate(positions)

    assert bool(compiled.evidence.complete)
    assert bool(lbvh.evidence.complete)
    assert int(lbvh.evidence.candidate_count) > 0
    assert int(lbvh.evidence.candidate_count) == int(compiled.evidence.candidate_count)
    assert _candidate_rows(lbvh.edge_edge) == _candidate_rows(compiled.edge_edge)
    assert _candidate_rows(lbvh.face_vertex) == _candidate_rows(compiled.face_vertex)
    assert int(lbvh.evidence.node_count) == 27
    assert int(lbvh.evidence.tree_depth) == 5
    assert int(lbvh.evidence.traversal_visits) == 196


def test_lbvh_tiny_visit_budget_bounds_work_and_fails_closed():
    vertex_count = 256
    source = phx.linalg.ArraySpace((vertex_count, 2), dtype=np.float64)
    edge_start = jnp.arange(vertex_count - 1, dtype=jnp.int32)
    edges = jnp.stack((edge_start, edge_start + 1), axis=1)
    topology = phx.discretization.CollisionSurfacePlan(
        jnp.arange(vertex_count, dtype=jnp.int64),
        ambient_dimension=2,
        edges=edges,
    )
    positions = jnp.stack(
        (
            jnp.arange(vertex_count, dtype=jnp.float64),
            jnp.zeros((vertex_count,), dtype=jnp.float64),
        ),
        axis=1,
    )
    surface = phx.discretization.PreparedCollisionSurface(
        topology,
        positions,
        phx.discretization.selection_collision_operator(
            source, jnp.arange(vertex_count, dtype=jnp.int32)
        ),
    )
    scene = phx.discretization.PreparedCollisionScene((surface,))
    plan = phx.discretization.LBVHContactSearchPlan(
        scene,
        edge_vertex_capacity=1,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
        maximum_tree_depth=16,
        maximum_traversal_visits=1,
    )
    result = plan.evaluate(positions)

    primitive_count = vertex_count + edges.shape[0]
    assert plan.tree_left.shape == (primitive_count - 1,)
    assert plan.tree_right.shape == (primitive_count - 1,)
    assert plan.node_leaf_count.shape == (2 * primitive_count - 1,)
    assert int(result.evidence.node_count) == 2 * primitive_count - 1
    assert int(result.evidence.tree_depth) == 10
    assert int(result.evidence.traversal_visits) == 1
    assert bool(result.evidence.visit_overflow)
    assert not bool(result.evidence.stack_overflow)
    assert int(result.evidence.candidate_count) == 0
    assert int(result.evidence.output_overflow) == 0
    assert bool(result.evidence.finite_bounds)
    assert not bool(result.evidence.complete)


def test_triangle_patch_and_hydroelastic_equal_pressure_extraction():
    triangle = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    faces = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)
    patch = phx.discretization.build_triangle_mortar_interface(
        triangle,
        faces,
        triangle,
        faces,
        capacity=4,
    )
    vertices = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    mesh = phx.discretization.CellMesh(
        vertices,
        (
            phx.discretization.CellBlock(
                "tetrahedra",
                "tetrahedron",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
            ),
        ),
    )
    plus = phx.discretization.HydroelasticPressureFieldPlan(mesh)
    minus = phx.discretization.HydroelasticPressureFieldPlan(mesh)
    prepared_patch = phx.discretization.HydroelasticPatchExtractionPlan(
        maximum_overlap_pairs=1,
        maximum_polytope_edges=12,
        patch_capacity=4,
    ).prepare(plus, minus)
    pressure_patch = prepared_patch.evaluate(
        phx.discretization.HydroelasticPressureFieldState(
            jnp.asarray((0.0, 1.0, 1.0, 1.0))
        ),
        phx.discretization.HydroelasticPressureFieldState(0.5 * jnp.ones((4,))),
    )

    assert bool(patch.evidence.successful)
    np.testing.assert_allclose(patch.evidence.total_measure, 0.5)
    assert bool(pressure_patch.evidence.successful)
    assert pressure_patch.evidence.triangle_count == 1
    assert pressure_patch.patch.pressure[0] > 0.0


def test_closed_surface_certificate_and_halo_exchange_are_explicit():
    vertices = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    faces = jnp.asarray(
        ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        dtype=jnp.int32,
    )
    topology = phx.discretization.CollisionSurfacePlan(
        jnp.arange(4), ambient_dimension=3, faces=faces
    )
    certificate = phx.discretization.certify_closed_oriented_surface(topology, vertices)
    source, scene, search = _two_segment_scene()
    epoch = search.build(scene, scene.positions(source.zeros()))
    distributed = phx.discretization.partition_contact_epoch(
        phx.discretization.DistributedContactPartitionPlan(
            jnp.asarray((0, 0, 1, 1)),
            rank_count=2,
            halo_capacity=16,
        ),
        epoch,
    )
    halo_plan = phx.discretization.ContactHaloExchangePlan.from_distributed_epoch(
        distributed, rank_count=2, halo_capacity=16
    )
    values = jnp.ones((halo_plan.route_count, 2))
    payload = phx.discretization.pack_contact_halo(halo_plan, values)
    received_values = payload.values[payload.valid]
    received_indices = payload.route_indices[payload.valid]
    received_valid = jnp.ones(received_indices.shape, dtype=bool)
    reduction = phx.discretization.reduce_contact_halo(
        halo_plan,
        jnp.zeros_like(values),
        received_values,
        received_indices,
        received_valid,
    )

    assert bool(certificate.successful)
    assert bool(payload.successful)
    assert bool(reduction.evidence.successful)
