# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from typing import assert_type

from jax import Array

from phydrax.applications.contact import (
    AffineBodyParticipant,
    ContactSolvePolicy,
    initialize_monolithic_contact_state,
    MonolithicContactPlan,
    MonolithicContactResult,
    MonolithicContactState,
    prepare_monolithic_contact_equilibrium,
    prepare_monolithic_contact_step,
    PreparedMonolithicContactEquilibrium,
    PreparedMonolithicContactStep,
    solve_monolithic_contact_equilibrium,
    solve_monolithic_contact_step,
)


def consumer(
    plan: MonolithicContactPlan,
    participant: AffineBodyParticipant,
    coordinates: Array,
    history: Array,
) -> MonolithicContactState:
    assert_type(ContactSolvePolicy(merit="residual"), ContactSolvePolicy)
    assert_type(participant.participant_id, str)
    state = initialize_monolithic_contact_state(plan, (coordinates,), (history,))
    assert_type(state, MonolithicContactState)
    prepared = prepare_monolithic_contact_step(
        plan, state, 0.01, args={"load": coordinates}
    )
    assert_type(prepared, PreparedMonolithicContactStep)
    result = solve_monolithic_contact_step(prepared)
    assert_type(result, MonolithicContactResult)
    assert_type(result.commit(), MonolithicContactState)
    assert_type(result.rollback(), MonolithicContactState)
    equilibrium = prepare_monolithic_contact_equilibrium(plan, state)
    assert_type(equilibrium, PreparedMonolithicContactEquilibrium)
    assert_type(
        solve_monolithic_contact_equilibrium(equilibrium), MonolithicContactResult
    )
    return result.commit()
