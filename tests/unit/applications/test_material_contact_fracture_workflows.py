#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_cpfem_identity_update_is_objective_and_admissible():
    cpfem = phx.applications.crystal_plasticity
    model = cpfem.CrystalPlasticityModel(
        (
            cpfem.CrystalSlipSystem(
                jnp.asarray([1.0, 0.0, 0.0]),
                jnp.asarray([0.0, 1.0, 0.0]),
            ),
        ),
        cpfem.CrystalPlasticityParameters(10.0, 20.0, 0.01, 0.1, 1.0, 2.0),
    )
    result = model.update(jnp.eye(3), model.initial_state(), jnp.eye(3), 0.1)

    assert bool(result.converged)
    assert jnp.linalg.norm(result.first_piola) < 1.0e-12
    assert jnp.allclose(jnp.linalg.det(result.state.plastic_deformation), 1.0)


def test_contact_route_keys_persist_through_accepted_commit():
    contact = phx.applications.contact
    collision = phx.discretization.contact
    source = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    moving_plan = collision.CollisionSurfacePlan(
        jnp.asarray([1, 3]),
        ambient_dimension=2,
        edges=jnp.asarray([[0, 1]], dtype=jnp.int32),
        physical_radius=0.1,
    )
    fixed_plan = collision.CollisionSurfacePlan(
        jnp.asarray([2, 4]),
        ambient_dimension=2,
        edges=jnp.asarray([[0, 1]], dtype=jnp.int32),
        body_ids=1,
        material_ids=0,
        static_mask=True,
        physical_radius=0.1,
    )
    moving = collision.PreparedCollisionSurface(
        moving_plan,
        jnp.asarray([[0.25, 0.1], [0.75, 0.1]]),
        collision.selection_collision_operator(source, jnp.asarray([0, 1])),
    )
    fixed = collision.PreparedCollisionSurface(
        fixed_plan,
        jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
        collision.static_collision_operator(source, 2, 2),
    )
    scene = collision.ContactParticipantScene(
        (
            collision.LinearContactParticipant(moving),
            collision.LinearContactParticipant(fixed),
        )
    )
    search = collision.DenseContactSearchPlan(
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.3,
    )
    materials = contact.ContactMaterialPairTable.uniform(
        normal_stiffness=100.0,
        static_friction=0.0,
        dynamic_friction=0.0,
        restitution=0.0,
        adhesion_energy=0.0,
        thermal_conductance=0.0,
        electrical_conductance=0.0,
        wear_coefficient=0.0,
        hardness=1.0,
        roughness=0.0,
    )
    closure = contact.ContactClosurePlan(contact.CompliantNormalContactLaw(), materials)
    accepted = contact.ContactRouteState.empty(0, 1, closure.closure_id)
    states = (source.zeros(), source.zeros())
    rest = scene.positions(states)
    transaction = contact.evaluate_cross_discretization_contact(
        scene,
        states,
        states,
        search,
        closure,
        accepted,
        1.0,
        rest,
        activation_distance=0.3,
    )
    promoted = transaction.commit()

    np.testing.assert_array_equal(
        promoted.route_keys, transaction.kinematics.batches[0].route_keys
    )
    assert promoted.state_version == accepted.state_version + 2
    assert transaction.rollback() is accepted


def test_diffuse_fracture_history_promotes_only_an_accepted_transaction():
    fracture = phx.applications.fracture
    history = fracture.PhaseFieldHistoryState(
        jnp.zeros((2, 1)),
        jnp.asarray([0.1, 0.2]),
    )
    rejected = history.transaction(
        jnp.asarray([[1.0], [0.5]]),
        jnp.asarray([0.2, 0.2]),
        accepted=False,
    )
    accepted = history.transaction(
        jnp.asarray([[1.0], [0.5]]),
        jnp.asarray([0.2, 0.2]),
        accepted=True,
    )
    promoted = accepted.commit(history)

    assert rejected.commit(history) is history
    assert jnp.all(promoted.history >= history.history)
    assert jnp.all(promoted.accepted_damage >= history.accepted_damage)
