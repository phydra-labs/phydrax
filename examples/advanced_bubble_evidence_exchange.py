#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Create an evidence-only request from a vanished resolved bubble."""

from typing import Any

import jax.numpy as jnp

import phydrax.bubble_dynamics as bubbles
from phydrax.applications import two_phase_flow as flow


def run() -> dict[str, Any]:
    volume = 4.0e-15
    amount = 1.6e-12
    energy = 5.2e-7
    source = flow.BubbleCompartmentState(
        bubble_id=jnp.asarray((42,), dtype=jnp.int32),
        amount=jnp.asarray((amount,), dtype=jnp.float64),
        internal_energy=jnp.asarray((energy,), dtype=jnp.float64),
        internal=jnp.asarray(((300.0,),), dtype=jnp.float64),
        volume=jnp.asarray((volume,), dtype=jnp.float64),
        pressure=jnp.asarray((101325.0,), dtype=jnp.float64),
        centroid=jnp.asarray(((0.2, 0.3, 0.4),), dtype=jnp.float64),
        epoch=jnp.asarray(6, dtype=jnp.int32),
    )
    transition = flow.BubbleTransitionRecord(
        7,
        "vanish",
        parent_ids=(42,),
        child_ids=(),
        parent_volumes=(volume,),
        child_volumes=(),
        child_slots=(),
        overlaps=(),
    )
    evaluation = flow.BubbleCompartmentEvaluation(
        pressure=jnp.asarray((101325.0,), dtype=jnp.float64),
        temperature=jnp.asarray((300.0,), dtype=jnp.float64),
        compliance=jnp.asarray((volume / (1.4 * 101325.0),), dtype=jnp.float64),
        admissible=jnp.asarray((True,)),
    )
    evidence = flow.resolved_bubble_evidence_records(
        transition,
        source,
        time=0.015,
        source_realization_id="resolved-example-run",
        law_id="caloric-ideal-gas",
        evaluation=evaluation,
        environment=bubbles.BubbleEnvironment(101325.0, 300.0),
        translational_momentum={42: jnp.asarray((2.0e-10, 0.0, 0.0), dtype=jnp.float64)},
        # Liquid impulse is not available from this compartment journal.
        liquid_impulse=None,
    )[0]
    request = flow.reduced_bubble_request(evidence, "single-bubble", "keller-miksis")
    return {
        "bubble_id": evidence.bubble_id,
        "event": evidence.event_kind,
        "gas_amount_mol": evidence.gas_amount,
        "internal_energy_joule": evidence.internal_energy,
        "equivalent_radius_meter": evidence.equivalent_volume_radius,
        "liquid_impulse_available": evidence.liquid_impulse is not None,
        "missing_invariants": int(request.missing_invariants),
        "request_accepted": request.accepted,
        "handoff_ready": request.handoff_ready,
        "stable_fingerprint": request.request_id,
    }


if __name__ == "__main__":
    print(run())
