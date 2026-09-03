#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

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


def test_contact_pair_ids_persist_through_accepted_commit():
    contact = phx.applications.contact
    plus = contact.ContactSurface(
        "plus",
        jnp.asarray([1, 3]),
        jnp.asarray([[0.25, -0.1], [0.75, 0.1]]),
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        jnp.asarray([11]),
    )
    minus = contact.ContactSurface(
        "minus",
        jnp.asarray([2, 4]),
        jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        jnp.asarray([22]),
    )
    query = contact.ContactQueryPlan(
        contact.ContactConfiguration(plus, minus, epoch=0)
    ).execute()
    operator = contact.FixedEpochContactOperator(query, contact.PenaltyContactLaw(100.0))
    accepted = operator.accepted_state()
    transaction = operator.attempt(accepted)
    promoted = transaction.commit()

    assert bool(transaction.evaluation.active[0])
    assert promoted.pair_ids == query.patches.pair_ids
    assert promoted.state_version == accepted.state_version + 1
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
