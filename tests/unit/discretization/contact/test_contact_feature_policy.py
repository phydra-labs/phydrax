#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.contact._compiled_search import (
    CompiledContactSearchPlan,
    LBVHContactSearchPlan,
)
from phydrax.discretization.contact._implicit_geometry import SphereContactGeometry
from phydrax.discretization.contact._kinematics import evaluate_contact_kinematics
from phydrax.discretization.contact._participant import FunctionContactParticipant
from phydrax.discretization.contact._proxy import ContactProxyPlan
from phydrax.discretization.contact._search import DenseContactSearchPlan
from phydrax.discretization.contact._surface import (
    CollisionFeatureKind,
    CollisionFeaturePolicy,
    CollisionSurfacePlan,
    ContactPairPolicy,
    PreparedCollisionScene,
    PreparedCollisionSurface,
    selection_collision_operator,
)
from phydrax.linalg import (
    ArraySpace,
    DiagonalPairing,
    dual_transpose,
    DualSpace,
    FunctionLinearOperator,
)


def _prepared_surface(plan, positions):
    positions = jnp.asarray(positions, dtype=jnp.float64)
    space = ArraySpace(positions.shape, dtype=np.float64)
    operator = selection_collision_operator(
        space, jnp.arange(positions.shape[0], dtype=jnp.int32)
    )
    return PreparedCollisionSurface(plan, positions, operator)


def test_surface_feature_labels_are_canonical_and_deterministic():
    first = CollisionSurfacePlan(
        jnp.asarray((40, 10, 30, 20), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((2, 3), (1, 0), (3, 0)), dtype=jnp.int32),
        participant_ids=4,
        body_ids=8,
        material_ids=3,
        patch_ids=6,
    )
    second = CollisionSurfacePlan(
        jnp.asarray((40, 10, 30, 20), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 3), (0, 1), (3, 2)), dtype=jnp.int32),
        participant_ids=4,
        body_ids=8,
        material_ids=3,
        patch_ids=6,
    )

    np.testing.assert_array_equal(first.edges, second.edges)
    np.testing.assert_array_equal(
        first.feature_policy.feature_ids, second.feature_policy.feature_ids
    )
    assert first.topology_id == second.topology_id


def test_heterogeneous_primitive_ownership_is_rejected():
    with pytest.raises(ValueError, match="participant labels must agree"):
        CollisionSurfacePlan(
            jnp.asarray((0, 1), dtype=jnp.int64),
            ambient_dimension=2,
            edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
            participant_ids=jnp.asarray((0, 1), dtype=jnp.int64),
            body_ids=0,
            material_ids=0,
            patch_ids=0,
        )


def test_radius_clearance_and_proxy_error_remain_distinct():
    plan = CollisionSurfacePlan(
        jnp.asarray((5, 9), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        participant_ids=0,
        body_ids=0,
        material_ids=2,
        patch_ids=1,
        physical_radius=jnp.asarray((0.1, 0.2)),
        solver_clearance=jnp.asarray((0.01, 0.03)),
        proxy_error=jnp.asarray((0.001, 0.004)),
    )

    np.testing.assert_allclose(plan.vertex_physical_radius, (0.1, 0.2))
    np.testing.assert_allclose(plan.vertex_solver_clearance, (0.01, 0.03))
    np.testing.assert_allclose(plan.vertex_proxy_error, (0.001, 0.004))
    np.testing.assert_allclose(
        plan.feature_policy.contact_extent,
        (0.111, 0.234, 0.234),
    )

    base = _prepared_surface(plan, ((0.0, 0.0), (1.0, 0.0)))
    proxy = ContactProxyPlan(plan, jnp.asarray((0.02, 0.05)), certified=True).prepare(
        base.rest_positions, base.displacement_operator
    )
    np.testing.assert_allclose(plan.vertex_proxy_error, (0.001, 0.004))
    np.testing.assert_allclose(
        proxy.surface.plan.vertex_proxy_error,
        (0.021, 0.054),
    )
    np.testing.assert_allclose(
        proxy.evidence.inflated_contact_extent,
        proxy.surface.plan.feature_policy.contact_extent,
    )


def test_pair_policy_allows_same_participant_and_honors_self_exclusions():
    vertices = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.05),
            (1.0, 0.0, 0.05),
            (0.0, 1.0, 0.05),
        ),
        dtype=jnp.float64,
    )
    faces = jnp.asarray(((0, 1, 2), (3, 4, 5)), dtype=jnp.int32)
    allowed = jnp.asarray(((7, 7),), dtype=jnp.int64)
    unrestricted = CollisionSurfacePlan(
        jnp.arange(6, dtype=jnp.int64),
        ambient_dimension=3,
        faces=faces,
        participant_ids=7,
        body_ids=12,
        material_ids=4,
        patch_ids=2,
        pair_policy=ContactPairPolicy(6, allowed_participant_pairs=allowed),
    )
    excluded = CollisionSurfacePlan(
        jnp.arange(6, dtype=jnp.int64),
        ambient_dimension=3,
        faces=faces,
        participant_ids=7,
        body_ids=12,
        material_ids=4,
        patch_ids=2,
        pair_policy=ContactPairPolicy(
            6,
            allowed_participant_pairs=allowed,
            excluded_vertex_pairs=jnp.asarray(
                tuple((left, right) for left in range(3) for right in range(3, 6)),
                dtype=jnp.int64,
            ),
        ),
    )
    search = DenseContactSearchPlan(
        edge_vertex_capacity=32,
        edge_edge_capacity=32,
        face_vertex_capacity=16,
        activation_distance=0.1,
    )

    unrestricted_epoch = search.build(
        PreparedCollisionScene((_prepared_surface(unrestricted, vertices),)), vertices
    )
    excluded_epoch = search.build(
        PreparedCollisionScene((_prepared_surface(excluded, vertices),)), vertices
    )

    assert int(unrestricted_epoch.edge_edge.actual_count) > 0
    assert int(excluded_epoch.edge_edge.actual_count) == 0
    assert int(unrestricted_epoch.edge_vertex.actual_count) == 0


def test_pair_policy_is_reciprocal_and_explicit_about_self_contact():
    policy = ContactPairPolicy(
        4,
        allowed_participant_pairs=jnp.asarray(((1, 3), (7, 7)), dtype=jnp.int64),
    )

    assert policy.allows(1, 3)
    assert policy.allows(3, 1)
    assert policy.allows(7, 7)
    assert not policy.allows(1, 1)
    assert not policy.allows(1, 2)


def test_static_static_features_do_not_generate_routes():
    common = dict(
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        body_ids=1,
        material_ids=2,
        patch_ids=3,
        static_mask=True,
    )
    allowed = jnp.asarray(((0, 1),), dtype=jnp.int64)
    first_plan = CollisionSurfacePlan(
        jnp.asarray((0, 1), dtype=jnp.int64),
        participant_ids=0,
        pair_policy=ContactPairPolicy(2, allowed_participant_pairs=allowed),
        **common,
    )
    second_plan = CollisionSurfacePlan(
        jnp.asarray((10, 11), dtype=jnp.int64),
        participant_ids=1,
        pair_policy=ContactPairPolicy(2, allowed_participant_pairs=allowed),
        **common,
    )
    first = _prepared_surface(first_plan, ((-1.0, 0.0), (1.0, 0.0)))
    second = _prepared_surface(second_plan, ((-1.0, 0.05), (1.0, 0.05)))
    scene = PreparedCollisionScene((first, second))
    positions = scene.positions(first.source_space.zeros())
    epoch = DenseContactSearchPlan(
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    ).build(scene, positions)
    compiled = CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    ).evaluate(positions)
    lbvh = LBVHContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
        maximum_traversal_visits=256,
    ).evaluate(positions)

    assert int(epoch.edge_vertex.actual_count) == 0
    assert int(compiled.edge_vertex.actual_count) == 0
    assert int(lbvh.edge_vertex.actual_count) == 0


def test_dense_compiled_and_lbvh_search_share_feature_policy_semantics():
    allowed = jnp.asarray(((0, 1),), dtype=jnp.int64)
    first_plan = CollisionSurfacePlan(
        jnp.asarray((30, 31), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        edge_ids=jnp.asarray((32,), dtype=jnp.int64),
        participant_ids=0,
        body_ids=5,
        material_ids=7,
        patch_ids=1,
        physical_radius=0.05,
        pair_policy=ContactPairPolicy(2, allowed_participant_pairs=allowed),
    )
    second_plan = CollisionSurfacePlan(
        jnp.asarray((40, 41), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        edge_ids=jnp.asarray((42,), dtype=jnp.int64),
        participant_ids=1,
        body_ids=5,
        material_ids=8,
        patch_ids=2,
        physical_radius=0.05,
        pair_policy=ContactPairPolicy(2, allowed_participant_pairs=allowed),
    )
    first = _prepared_surface(first_plan, ((-1.0, 0.15), (1.0, 0.15)))
    second = _prepared_surface(second_plan, ((-1.0, 0.0), (1.0, 0.0)))
    scene = PreparedCollisionScene((first, second))
    state = first.source_space.zeros()
    positions = scene.positions(state)
    dense = DenseContactSearchPlan(
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.06,
    ).build(scene, positions)
    compiled = CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.06,
    ).evaluate(positions)
    lbvh = LBVHContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.06,
        maximum_traversal_visits=256,
    ).evaluate(positions)

    assert int(dense.edge_vertex.actual_count) > 0
    assert int(compiled.edge_vertex.actual_count) > 0
    assert int(lbvh.edge_vertex.actual_count) > 0
    assert bool(compiled.evidence.complete)
    kinematics = evaluate_contact_kinematics(
        scene,
        dense,
        positions,
        jnp.zeros_like(positions),
        0.01,
        rest_positions=positions,
    )
    active = kinematics.batches[0].valid
    material_pairs = jnp.stack(
        (
            kinematics.batches[0].left_material_ids,
            kinematics.batches[0].right_material_ids,
        ),
        axis=1,
    )
    assert bool(
        jnp.all(
            jnp.where(
                active[:, None],
                jnp.sort(material_pairs, axis=1) == jnp.asarray((7, 8), dtype=jnp.int64),
                True,
            )
        )
    )
    assert bool(lbvh.evidence.complete)


def test_weighted_spaces_obey_algebraic_power_duality():
    configuration_space = ArraySpace((1,), dtype=np.float64)
    tangent_space = ArraySpace(
        (2,),
        dtype=np.float64,
        pairing=DiagonalPairing(jnp.asarray((2.0, 5.0), dtype=jnp.float64)),
    )
    plan = CollisionSurfacePlan(
        jnp.asarray((91,), dtype=jnp.int64),
        ambient_dimension=2,
        allow_isolated_vertices=True,
        participant_ids=3,
        body_ids=4,
        material_ids=5,
        patch_ids=6,
    )

    def positions(configuration):
        return jnp.asarray(((configuration[0], 2.0 * configuration[0]),))

    def velocity(configuration, direction):
        del configuration
        return jnp.asarray(((direction[0] + 3.0 * direction[1], 2.0 * direction[0]),))

    def effort_pullback(configuration, effort):
        del configuration
        return jnp.asarray(
            (effort[0, 0] + 2.0 * effort[0, 1], 3.0 * effort[0, 0]),
            dtype=effort.dtype,
        )

    def scaled_positions(scale):
        def action(configuration):
            return jnp.asarray(((scale * configuration[0], 2.0 * configuration[0]),))

        return action

    for scale in (1.0, 2.0):
        with pytest.raises(
            TypeError,
            match="Opaque callables require explicit",
        ):
            FunctionContactParticipant(
                plan,
                configuration_space,
                scaled_positions(scale),
                tangent_space=configuration_space,
            )

    participant = FunctionContactParticipant(
        plan,
        configuration_space,
        positions,
        tangent_space=tangent_space,
        velocity_action=velocity,
        effort_pullback_action=effort_pullback,
        participant_id="weighted-space-duality",
    )
    state = jnp.asarray((0.25,), dtype=jnp.float64)
    direction = jnp.asarray((0.4, -0.3), dtype=jnp.float64)
    effort = jnp.asarray(((1.7, -0.8),), dtype=jnp.float64)
    pulled = participant.effort_pullback(state, effort)

    left = participant.contact_effort_space.pair(
        effort, participant.velocities(state, direction)
    )
    right = DualSpace(tangent_space).pair(pulled, direction)
    np.testing.assert_allclose(left, right, atol=1.0e-12)
    assert bool(participant.duality_evidence(state, direction, effort).valid)


def test_dual_transpose_preserves_power_on_unequally_weighted_spaces():
    source = ArraySpace(
        (2,),
        dtype=np.float64,
        pairing=DiagonalPairing(jnp.asarray((2.0, 7.0), dtype=jnp.float64)),
    )
    target = ArraySpace(
        (2,),
        dtype=np.float64,
        pairing=DiagonalPairing(jnp.asarray((11.0, 3.0), dtype=jnp.float64)),
    )
    matrix = jnp.asarray(((1.0, -2.0), (4.0, 0.5)), dtype=jnp.float64)
    operator = FunctionLinearOperator(
        lambda vector: matrix @ vector,
        source=source,
        target=target,
        transpose_action=lambda covector: matrix.T @ covector,
    )
    vector = jnp.asarray((0.3, -0.8), dtype=jnp.float64)
    covector = jnp.asarray((1.2, -0.4), dtype=jnp.float64)
    pulled = dual_transpose(operator).mv(covector)

    np.testing.assert_allclose(
        DualSpace(target).pair(covector, operator.mv(vector)),
        DualSpace(source).pair(pulled, vector),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(pulled, matrix.T @ covector, atol=1.0e-12)


def test_implicit_geometry_is_bound_to_one_analytic_feature():
    analytic = CollisionFeaturePolicy(
        jnp.asarray((700,), dtype=jnp.int64),
        jnp.asarray((int(CollisionFeatureKind.ANALYTIC),), dtype=jnp.int32),
        participant_ids=2,
        body_ids=9,
        material_ids=4,
        patch_ids=1,
        physical_radius=0.03,
        solver_clearance=0.01,
        proxy_error=0.002,
        provenance_id="analytic-sphere-source",
    )
    sphere = SphereContactGeometry((0.0, 0.0, 0.0), 1.0, feature_policy=analytic)

    assert sphere.feature_policy.policy_id == analytic.policy_id
    assert sphere.geometry_id != analytic.policy_id
    np.testing.assert_allclose(sphere.feature_policy.contact_extent, (0.042,))
