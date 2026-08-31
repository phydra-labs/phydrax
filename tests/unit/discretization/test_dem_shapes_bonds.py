#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _rigid_pair():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([1, 2]), jnp.ones((2,)), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.asarray([0, 0]), jnp.stack((jnp.eye(3), jnp.eye(3)))
    ).prepare(particles)
    orientation = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    kinematics = bodies.kinematics(
        jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
        orientation,
        jnp.zeros((2, 3)),
    )
    return particles, bodies, kinematics


def test_rigid_body_lie_step_and_clump_owner_component_contracts():
    particles, bodies, kinematics = _rigid_pair()
    load = phx.discretization.RigidBodyLoad(
        jnp.zeros((2, 3)), jnp.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    )
    result = phx.discretization.rigid_body_kick_drift_kick(
        bodies,
        kinematics,
        load,
        jnp.asarray(0.0),
        jnp.asarray(1.0e-3),
        lambda time, state, args: load,
        None,
    )
    assert result.successful
    assert jnp.allclose(jnp.linalg.norm(result.kinematics.orientation, axis=-1), 1.0)

    template = phx.discretization.SphereClumpTemplatePlan(
        jnp.asarray([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        jnp.asarray([0.1, 0.1]),
        jnp.asarray([0.5, 0.5]),
        jnp.asarray([0, 0]),
    )
    clumps = phx.discretization.RigidSphereClumpSetPlan(
        (template,), jnp.asarray([0, 0]), jnp.asarray([0, 0])
    ).prepare(particles)
    relation = (
        phx.discretization.DenseParticleNeighborhoodPlan(1)
        .prepare(particles)
        .build(kinematics.position)
        .pair_relation
    )
    key_space = phx.discretization.ParticlePairKeySpace(particles)
    pair_keys = key_space.keys(relation).keys
    expanded = phx.discretization.expand_clump_owner_pairs(
        clumps, kinematics, relation, pair_keys
    )
    assert expanded.valid.shape == (4,)
    assert jnp.sum(expanded.valid) == 4
    assert jnp.unique(expanded.component_pair_keys, axis=0).shape[0] == 4


def test_triangle_and_shape_agnostic_sphere_contact_geometry():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([3]), jnp.asarray([1.0]), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5]), jnp.asarray([0])
    ).prepare(particles)
    kinematics = bodies.kinematics(jnp.asarray([[0.0, 0.0, 0.4]]), jnp.zeros((1, 3)))
    wall = phx.discretization.TriangleWallPlan(
        jnp.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
        jnp.asarray([[0, 1, 2]]),
        jnp.asarray([0]),
        two_sided=True,
    ).prepare()
    result = phx.discretization.sphere_triangle_contact_geometry(bodies, kinematics, wall)
    assert result.geometry.successful
    assert result.geometry.valid[0]
    assert jnp.isclose(result.geometry.overlap[0], 0.1)
    batch = result.geometry.as_contact_batch()
    assert batch.normal.shape == (1, 3)


def test_fixed_bond_elasticity_damage_and_irreversibility():
    particles, bodies, kinematics = _rigid_pair()
    plan = phx.discretization.FixedBondGraphPlan(
        jnp.asarray([1]),
        jnp.asarray([2]),
        jnp.asarray([100]),
        jnp.zeros((1, 3)),
        jnp.zeros((1, 3)),
        jnp.asarray([[1.0, 0.0, 0.0]]),
        cross_section=jnp.asarray([1.0]),
        normal_stiffness=jnp.asarray([100.0]),
        shear_stiffness=jnp.asarray([50.0]),
        bending_stiffness=jnp.asarray([10.0]),
        twisting_stiffness=jnp.asarray([10.0]),
    ).prepare(bodies)
    state = plan.initialize_state()
    evaluation = phx.discretization.evaluate_bonds(plan, kinematics, state)
    assert evaluation.successful
    assert jnp.allclose(evaluation.net_force, 0.0)
    assert evaluation.stored_energy[0] > 0.0
    damage = phx.discretization.MixedModeBondDamagePlan(
        jnp.asarray([0.05]), jnp.asarray([0.2]), jnp.asarray([2.0])
    )
    updated = damage.update(state, evaluation, jnp.asarray(4, dtype=jnp.int32))
    assert updated.damage[0] > 0.0
    assert updated.damage[0] >= state.damage[0]
    assert updated.cumulative_fracture_energy[0] >= 0.0


def test_fixed_pool_topology_split_conserves_mass_and_momentum():
    plan = phx.discretization.TopologyEventPlan(3, 2, 2, 3)
    state = phx.discretization.TopologyPoolState(
        jnp.asarray([10, 11, 12], dtype=jnp.int64),
        jnp.asarray([True, False, False]),
        -jnp.ones((3,), dtype=jnp.int64),
        jnp.zeros((3, 3)),
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 3),
        jnp.zeros((3, 3)),
        jnp.asarray([2.0, 0.0, 0.0]),
        jnp.stack((jnp.eye(3), jnp.eye(3), jnp.eye(3))),
        jnp.asarray(0, dtype=jnp.int64),
    )
    record = phx.discretization.initialize_topology_event_record(plan)
    result = phx.discretization.split_preallocated_owner(
        plan,
        state,
        record,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray([1, 2]),
        jnp.asarray([True, True]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        jnp.stack((jnp.eye(3), jnp.eye(3))),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.1),
    )
    assert result.successful
    assert not result.accepted_state.active[0]
    assert jnp.all(result.accepted_state.active[1:])
    assert jnp.isclose(jnp.sum(result.accepted_state.mass[1:]), 2.0)
    assert jnp.linalg.norm(result.record.linear_momentum_residual[0]) < 1.0e-12


def _tetrahedron(shift):
    vertices = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ) + jnp.asarray(shift)
    triangles = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return phx.discretization.ConvexShapePlan(vertices, triangles, 0).prepare()


def test_convex_and_implicit_contact_oracles_report_certified_geometry():
    left = _tetrahedron((0.0, 0.0, 0.0))
    right = _tetrahedron((0.0, 0.0, 0.0))
    contact = phx.discretization.convex_sat_contact(
        left,
        right,
        jnp.zeros((3,)),
        jnp.asarray([0.1, 0.1, 0.1]),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.asarray(7, dtype=jnp.int64),
    )
    assert contact.successful
    assert contact.minimum_overlap >= 0.0

    implicit = phx.discretization.ImplicitRigidShapePlan(
        lambda point: jnp.linalg.norm(point, axis=-1) - 0.5,
        lambda point: point / jnp.linalg.norm(point, axis=-1, keepdims=True),
        jnp.asarray([-1.0, -1.0, -1.0]),
        jnp.asarray([1.0, 1.0, 1.0]),
        0,
        shape_id="implicit-sphere",
    )
    result = phx.discretization.sphere_implicit_contact(
        jnp.asarray([0.9, 0.0, 0.0]),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.asarray(0.5),
        implicit,
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        jnp.zeros((3,)),
        jnp.asarray(9, dtype=jnp.int64),
    )
    assert result.successful
    assert jnp.isclose(result.geometry.overlap[0], 0.1)
