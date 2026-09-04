#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.nn.parameters import ParameterSubspace


contact = phx.applications.contact
collision = phx.discretization.contact


def _material_table():
    return contact.ContactMaterialPairTable.uniform(
        normal_stiffness=100.0,
        static_friction=0.5,
        dynamic_friction=0.4,
        restitution=0.0,
        adhesion_energy=0.0,
        thermal_conductance=0.0,
        electrical_conductance=0.0,
        wear_coefficient=1.0e-4,
        hardness=10.0,
        roughness=1.0e-3,
    )


def _canonical_pair(*, friction=False):
    moving_space = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    static_space = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    moving_plan = collision.CollisionSurfacePlan(
        jnp.asarray([10, 11]),
        ambient_dimension=2,
        edges=jnp.asarray([[0, 1]], dtype=jnp.int32),
        body_ids=0,
        material_ids=0,
        physical_radius=0.05,
    )
    static_plan = collision.CollisionSurfacePlan(
        jnp.asarray([20, 21]),
        ambient_dimension=2,
        edges=jnp.asarray([[0, 1]], dtype=jnp.int32),
        body_ids=1,
        material_ids=0,
        static_mask=True,
        physical_radius=0.05,
    )
    moving = collision.PreparedCollisionSurface(
        moving_plan,
        jnp.asarray([[0.25, 0.05], [0.75, 0.05]]),
        collision.selection_collision_operator(
            moving_space, jnp.asarray([0, 1], dtype=jnp.int32)
        ),
    )
    static = collision.PreparedCollisionSurface(
        static_plan,
        jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
        collision.selection_collision_operator(
            static_space, jnp.asarray([0, 1], dtype=jnp.int32)
        ),
    )
    scene = collision.ContactParticipantScene(
        (
            collision.LinearContactParticipant(moving),
            collision.LinearContactParticipant(static),
        )
    )
    search = collision.DenseContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.2,
    )
    closure = contact.ContactClosurePlan(
        contact.CompliantNormalContactLaw(),
        _material_table(),
        tangential=(contact.RegularizedCoulombContactLaw(1.0e-3) if friction else None),
        evolution=(
            contact.FrictionWearEvolutionLaw(
                critical_slip_distance=0.1,
                damage_onset=0.1,
                damage_completion=0.2,
            )
            if friction
            else None
        ),
    )
    state = contact.ContactRouteState.empty(0, 1, closure.closure_id)
    states = (moving_space.zeros(), static_space.zeros())
    rates = (moving_space.zeros(), static_space.zeros())
    rest = jnp.concatenate((moving.rest_positions, static.rest_positions), axis=0)
    epoch = search.build(scene, scene.positions(states))
    return scene, search, closure, state, states, rates, rest, epoch


def _evaluate(case, states=None, rates=None):
    scene, search, closure, state, initial_states, initial_rates, rest, epoch = case
    return contact.evaluate_cross_discretization_contact(
        scene,
        initial_states if states is None else states,
        initial_rates if rates is None else rates,
        search,
        closure,
        state,
        0.01,
        rest,
        activation_distance=0.2,
        candidate_epoch=epoch,
    )


def test_collision_surface_search_is_deterministic_and_fixed_epoch_residual_is_dense():
    case = _canonical_pair()
    scene, search, _, _, states, rates, rest, epoch = case
    repeated = search.build(scene, scene.positions(states))
    first = _evaluate(case)
    moved = (states[0].at[:, 1].add(-0.02), states[1])
    second = _evaluate(case, states=moved)

    assert epoch.epoch_id == repeated.epoch_id
    np.testing.assert_array_equal(
        epoch.edge_vertex.route_keys, repeated.edge_vertex.route_keys
    )
    np.testing.assert_array_equal(
        first.kinematics.batches[0].route_keys,
        second.kinematics.batches[0].route_keys,
    )
    assert jnp.min(second.kinematics.batches[0].gap) < jnp.min(
        first.kinematics.batches[0].gap
    )
    np.testing.assert_allclose(
        second.assembly.action_reaction_residual, 0.0, atol=1.0e-12
    )

    def residual(moving_state):
        evaluation = contact.evaluate_cross_discretization_contact(
            scene,
            (moving_state, states[1]),
            rates,
            search,
            case[2],
            case[3],
            0.01,
            rest,
            activation_distance=0.2,
            candidate_epoch=epoch,
        )
        return evaluation.generalized_efforts[0]

    _, action = jax.jvp(residual, (states[0],), (jnp.ones_like(states[0]),))
    assert action.shape == states[0].shape
    assert jnp.all(jnp.isfinite(action))


def test_closure_candidate_history_is_committed_or_rolled_back_explicitly():
    case = _canonical_pair(friction=True)
    rates = (
        jnp.broadcast_to(jnp.asarray([0.2, -0.05]), case[4][0].shape),
        case[5][1],
    )
    attempt = _evaluate(case, rates=rates)

    assert attempt.rollback() is case[3]
    committed = attempt.commit()
    assert committed.state_version == case[3].state_version + 2
    assert jnp.linalg.norm(committed.accumulated_slip) > 0.0
    assert case[3].capacity == 0
    np.testing.assert_allclose(
        attempt.assembly.action_reaction_residual, 0.0, atol=1.0e-12
    )


def test_mortar_and_nitsche_evidence_are_derived_from_discrete_actions():
    mortar = contact.ContactMortarSpace(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.75, 0.25], [0.25, 0.75]]),
        jnp.asarray([0.5, 0.5]),
        mortar_id="nonmatching-line",
    )
    evidence = mortar.evaluate(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
        jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
        jnp.asarray([[0.0, 2.0], [0.0, 2.0]]),
    )
    assert bool(evidence.constant_reproduced)
    assert bool(evidence.adjoint_consistent)
    assert bool(evidence.conservative)
    np.testing.assert_allclose(evidence.virtual_work_defect, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(evidence.action_reaction_defect, 0.0, atol=1.0e-12)

    nitsche = contact.NitscheContactPolicy(20.0, 10.0)
    nitsche_evidence = nitsche.evidence(
        jnp.asarray([-0.1, 0.2]),
        jnp.zeros((2,)),
        jnp.ones((2,)),
        jnp.asarray([0.4, 0.1]),
        jnp.asarray([0.4, 0.1]),
    )
    np.testing.assert_allclose(
        nitsche_evidence.projected_pressure, jnp.asarray([2.0, 0.0])
    )
    assert bool(nitsche_evidence.adjoint_consistent)
    assert bool(nitsche_evidence.coercive)


def test_neural_contact_uses_canonical_fixed_manifold_virtual_work():
    case = _canonical_pair()
    scene, search, closure, state, states, _, rest, epoch = case
    functions = {"moving": states[0], "static": states[1]}

    def state_trace(root, args):
        del args
        return root["moving"], root["static"]

    adapter = contact.NeuralContactAdapter(
        scene,
        search,
        closure,
        state,
        epoch,
        rest,
        state_trace,
        adapter_id="neural-obstacle-contact",
        activation_distance=0.2,
    )
    direct = adapter.evaluate(functions)
    for virtual_work, force in zip(
        direct.virtual_work, direct.contact.generalized_efforts, strict=True
    ):
        np.testing.assert_allclose(virtual_work, -force)
    assert direct.contact.candidate_epoch is epoch

    subspace = ParameterSubspace(functions, eqx.is_inexact_array)
    prepared = adapter.prepare_equilibrium(functions, subspace)
    residual = prepared.problem.residual_function(prepared.initial_state, None)
    assert residual.shape == prepared.initial_state.shape
    assert jnp.all(jnp.isfinite(residual))
    assert prepared.formulation == "virtual-work"


def test_mpm_participant_uses_canonical_manifold_and_conservative_pullback():
    query_space = phx.linalg.ArraySpace((1, 2), dtype=np.float64)
    query = contact.prepare_point_contact_participant(
        query_space,
        jnp.asarray([[0.0, 0.05]]),
        vertex_ids=jnp.asarray([1]),
        body_ids=jnp.asarray([0]),
        physical_radius=jnp.asarray([0.05]),
    )
    surface_space = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    surface_plan = collision.CollisionSurfacePlan(
        jnp.asarray([2, 3]),
        ambient_dimension=2,
        edges=jnp.asarray([[0, 1]], dtype=jnp.int32),
        body_ids=1,
        material_ids=0,
        static_mask=True,
        physical_radius=0.05,
    )
    surface = collision.LinearContactParticipant(
        collision.PreparedCollisionSurface(
            surface_plan,
            jnp.asarray([[-1.0, 0.0], [1.0, 0.0]]),
            collision.selection_collision_operator(
                surface_space, jnp.asarray([0, 1], dtype=jnp.int32)
            ),
        )
    )
    scene = collision.ContactParticipantScene((query, surface))
    search = collision.DenseContactSearchPlan(
        edge_vertex_capacity=4,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.2,
    )
    closure = contact.ContactClosurePlan(
        contact.CompliantNormalContactLaw(), _material_table()
    )
    state = contact.ContactRouteState.empty(0, 1, closure.closure_id)
    states = (query_space.zeros(), surface_space.zeros())
    rest = scene.positions(states)
    result = contact.evaluate_cross_discretization_contact(
        scene,
        states,
        (query_space.zeros(), surface_space.zeros()),
        search,
        closure,
        state,
        1.0,
        rest,
        activation_distance=0.2,
    )

    assert isinstance(result.candidate_epoch, collision.ContactCandidateEpoch)
    assert len(result.generalized_efforts) == 2
    np.testing.assert_allclose(
        result.assembly.action_reaction_residual, 0.0, atol=1.0e-12
    )
    assert bool(result.successful)
