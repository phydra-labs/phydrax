#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.nn.parameters import ParameterSubspace


contact = phx.applications.contact


def _surface_pair(*, epoch=0):
    plus = contact.ContactSurface(
        "plus",
        jnp.asarray([10, 11]),
        jnp.asarray([[0.25, -0.1], [0.75, 0.2]]),
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        jnp.asarray([100]),
    )
    minus = contact.ContactSurface(
        "minus",
        jnp.asarray([20, 21]),
        jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        jnp.asarray([200]),
    )
    configuration = contact.ContactConfiguration(
        plus,
        minus,
        epoch=epoch,
        search_radius=1.0,
    )
    return plus, minus, configuration, contact.ContactQueryPlan(configuration).execute()


def test_current_geometry_query_is_exact_deterministic_and_epoch_frozen():
    plus, minus, configuration, first = _surface_pair()
    second = contact.ContactQueryPlan(configuration).execute()

    assert first.query_id == second.query_id
    assert first.patches.pair_ids == second.patches.pair_ids
    np.testing.assert_allclose(first.patches.gaps, jnp.asarray([-0.1, 0.2]))
    np.testing.assert_allclose(first.patches.normals, jnp.asarray([[0.0, 1.0]] * 2))

    moved_plus = plus.current_coordinates.at[:, 1].add(0.05)
    gap, normal, closest = first.current_kinematics(moved_plus, minus.current_coordinates)
    np.testing.assert_allclose(gap, jnp.asarray([-0.05, 0.25]))
    np.testing.assert_allclose(normal, first.patches.normals)
    np.testing.assert_allclose(closest[:, 1], 0.0)

    next_configuration = configuration.next_epoch(
        plus.with_current_coordinates(moved_plus), minus
    )
    assert next_configuration.epoch == 1
    assert next_configuration.plus.surface_id == configuration.plus.surface_id


def test_self_contact_excludes_incident_facets_and_preserves_stable_pairs():
    surface = contact.ContactSurface(
        "loop",
        jnp.asarray([0, 1, 2, 3]),
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        jnp.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=jnp.int32),
        jnp.asarray([10, 11, 12, 13]),
    )
    configuration = contact.ContactConfiguration(
        surface,
        surface,
        epoch=0,
        search_radius=2.0,
        self_contact=True,
        excluded_node_facet_pairs=jnp.asarray([[0, 11]]),
    )
    query = contact.ContactQueryPlan(configuration).execute()
    node_ids = np.asarray(surface.node_ids)
    facets = np.asarray(surface.facets)

    for plus_index, facet_index in zip(
        np.asarray(query.patches.plus_node_indices),
        np.asarray(query.patches.minus_facet_indices),
        strict=True,
    ):
        assert int(node_ids[plus_index]) not in set(
            node_ids[facets[facet_index]].tolist()
        )
    assert (
        query.patches.pair_ids
        == contact.ContactQueryPlan(configuration).execute().patches.pair_ids
    )
    assert int(query.excluded_count) > 0


def test_penalty_fe_adapter_transaction_and_exact_tangent_action():
    plus, minus, _, query = _surface_pair()
    operator = contact.FixedEpochContactOperator(query, contact.PenaltyContactLaw(100.0))
    accepted = operator.accepted_state()
    boundary = contact.FiniteElementContactBoundary(operator)
    assembly = boundary.assemble(accepted)

    np.testing.assert_allclose(accepted.normal_pressure, 0.0)
    np.testing.assert_allclose(assembly.contact.normal_pressure, jnp.asarray([10.0, 0.0]))
    np.testing.assert_allclose(assembly.contact.action_reaction_defect, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        jnp.sum(assembly.plus_residual, axis=0)
        + jnp.sum(assembly.minus_residual, axis=0),
        0.0,
        atol=1.0e-12,
    )
    assert bool(assembly.finite)
    assert bool(assembly.convergence.satisfies_contract)

    plus_direction = jnp.asarray([[0.0, -1.0], [0.0, 0.0]])
    plus_action, minus_action = boundary.tangent_action(
        accepted,
        plus.current_coordinates,
        minus.current_coordinates,
        plus_direction,
        jnp.zeros_like(minus.current_coordinates),
    )
    assert plus_action.shape == plus.current_coordinates.shape
    assert minus_action.shape == minus.current_coordinates.shape
    assert jnp.linalg.norm(plus_action) > 0.0
    np.testing.assert_allclose(
        jnp.sum(plus_action, axis=0) + jnp.sum(minus_action, axis=0),
        0.0,
        atol=1.0e-10,
    )

    transaction = boundary.attempt(accepted)
    assert transaction.rollback() is accepted
    committed = transaction.commit()
    assert committed.state_version == accepted.state_version + 1
    np.testing.assert_allclose(accepted.normal_pressure, 0.0)
    np.testing.assert_allclose(committed.normal_pressure, jnp.asarray([10.0, 0.0]))


def test_pdas_and_augmented_lagrangian_keep_multiplier_updates_transactional():
    _, _, _, query = _surface_pair()
    pdas = contact.FixedEpochContactOperator(
        query, contact.FrictionlessPDASContactLaw(10.0)
    )
    pdas_state = pdas.accepted_state()
    pdas_evaluation = pdas.evaluate(pdas_state, normal_pressure=jnp.asarray([3.0, 0.0]))
    np.testing.assert_array_equal(pdas_evaluation.active, jnp.asarray([True, False]))
    np.testing.assert_allclose(pdas_evaluation.normal_pressure, jnp.asarray([3.0, 0.0]))

    augmented = contact.FixedEpochContactOperator(
        query, contact.AugmentedLagrangianContactLaw(20.0)
    )
    initial = augmented.accepted_state()
    first = augmented.attempt(initial)
    np.testing.assert_allclose(initial.normal_pressure, 0.0)
    np.testing.assert_allclose(first.trial.normal_pressure, jnp.asarray([2.0, 0.0]))
    assert first.rollback() is initial
    committed = first.commit()
    second = augmented.evaluate(committed)
    np.testing.assert_allclose(second.normal_pressure, jnp.asarray([4.0, 0.0]))


def test_coulomb_history_is_committed_only_and_reports_normal_reversal():
    plus, minus, configuration, query = _surface_pair()
    operator = contact.FixedEpochContactOperator(
        query,
        contact.PenaltyContactLaw(100.0),
        friction_law=contact.CoulombContactLaw(0.5, 10.0),
    )
    accepted = operator.accepted_state()
    shifted_plus = plus.current_coordinates.at[0, 0].add(1.0)
    attempt = operator.attempt(accepted, shifted_plus, minus.current_coordinates)

    assert int(attempt.evaluation.mode[0]) == contact.CONTACT_SLIP
    assert attempt.evaluation.dissipation > 0.0
    np.testing.assert_allclose(accepted.accumulated_slip, 0.0)
    committed = attempt.commit()
    assert jnp.linalg.norm(committed.accumulated_slip[0]) > 0.0

    reversed_minus = contact.ContactSurface(
        minus.surface_id,
        minus.node_ids,
        minus.current_coordinates,
        jnp.asarray([[1, 0]], dtype=jnp.int32),
        minus.facet_ids,
    )
    next_configuration = configuration.next_epoch(
        plus.with_current_coordinates(shifted_plus), reversed_minus
    )
    next_query = contact.ContactQueryPlan(next_configuration).execute()
    next_operator = contact.FixedEpochContactOperator(
        next_query,
        contact.PenaltyContactLaw(100.0),
        friction_law=contact.CoulombContactLaw(0.5, 10.0),
    )
    epoch_attempt = next_operator.attempt_epoch(committed)
    assert epoch_attempt.rollback() is committed
    assert epoch_attempt.commit().epoch == next_configuration.epoch
    assert bool(epoch_attempt.evaluation.transport_ambiguous[0])


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


def test_fixed_epoch_neural_virtual_work_builds_parameter_space_root():
    plus, minus, _, query = _surface_pair()
    operator = contact.FixedEpochContactOperator(query, contact.PenaltyContactLaw(100.0))
    accepted = operator.accepted_state()
    functions = {
        "minus": minus.current_coordinates,
        "plus": plus.current_coordinates,
    }

    def plus_trace(root, epoch_coordinates, args):
        del epoch_coordinates, args
        return root["plus"]

    def minus_trace(root, epoch_coordinates, args):
        del epoch_coordinates, args
        return root["minus"]

    adapter = contact.FixedEpochNeuralContactAdapter(
        operator,
        accepted,
        plus_trace,
        minus_trace,
        adapter_id="neural-obstacle-contact",
    )
    direct = adapter.evaluate(functions)
    np.testing.assert_allclose(
        direct.plus_virtual_work, -direct.contact.plus_nodal_forces
    )
    subspace = ParameterSubspace(functions, eqx.is_inexact_array)
    prepared = adapter.prepare_equilibrium(functions, subspace)
    residual = prepared.problem.residual_function(prepared.initial_state, None)
    assert residual.shape == prepared.initial_state.shape
    assert jnp.all(jnp.isfinite(residual))
    assert prepared.formulation == "virtual-work"


def test_neural_pdas_includes_multiplier_complementarity_in_the_vjp_root():
    plus, minus, _, query = _surface_pair()
    operator = contact.FixedEpochContactOperator(
        query, contact.FrictionlessPDASContactLaw(10.0)
    )
    functions = {
        "minus": minus.current_coordinates,
        "plus": plus.current_coordinates,
        "pressure": jnp.asarray([3.0, 0.0]),
    }

    def plus_trace(root, epoch_coordinates, args):
        del epoch_coordinates, args
        return root["plus"]

    def minus_trace(root, epoch_coordinates, args):
        del epoch_coordinates, args
        return root["minus"]

    def pressure_trace(root, query_result, args):
        del query_result, args
        return root["pressure"]

    adapter = contact.FixedEpochNeuralContactAdapter(
        operator,
        operator.accepted_state(),
        plus_trace,
        minus_trace,
        adapter_id="neural-pdas-contact",
        normal_pressure_trace=pressure_trace,
    )
    direct = adapter.evaluate(functions)
    np.testing.assert_allclose(
        direct.normal_pressure_virtual_work,
        direct.contact.complementarity_residual,
    )
    prepared = adapter.prepare_equilibrium(
        functions, ParameterSubspace(functions, eqx.is_inexact_array)
    )
    residual = prepared.problem.residual_function(prepared.initial_state, None)
    assert residual.shape == prepared.initial_state.shape
    assert jnp.all(jnp.isfinite(residual))


def test_deformable_mpm_adapter_is_distinct_and_conserves_route_action():
    plan = contact.DeformableMPMContactPlan(
        jnp.asarray([0]),
        jnp.asarray([[0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0]]),
        activation_distance=0.5,
    )
    prepared = plan.prepare(1)
    adapter = contact.DeformableMPMContactAdapter(
        prepared, contact.PenaltyContactLaw(100.0)
    )
    position = jnp.asarray([[0.0, -0.2]])
    result = adapter.evaluate(
        position,
        jnp.zeros_like(position),
        position,
        jnp.zeros_like(position),
    )

    assert isinstance(result, contact.DeformableMPMContactEvaluation)
    np.testing.assert_allclose(result.normal_pressure, jnp.asarray([20.0]))
    np.testing.assert_allclose(result.transpose.balance_residual, 0.0, atol=1.0e-12)
    assert bool(result.successful)
