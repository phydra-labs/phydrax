#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host epoch driver of resolved bubbly flow.

Each step is one compiled `IncompressibleTwoPhaseVOFMethod.step` call. At the
epoch boundary after the call, the driver reads the step's bubble evidence and
executes the host transactions that the plan contract reserves for the host:

1. **Compartment transaction.** A step whose proposed identities differ from
   the registered compartments is refused. The driver proposes the registry
   and ledger transaction through the gas law and repeats the fluid step with
   that candidate registry. The transaction commits only with the repeated
   step, and retry failure rolls the whole candidate back. Repeats are bounded
   by ``maximum_transactions``.
2. **Identity journal.** Every accepted topology event is appended to the
   `BubbleTransitionJournal`.
3. **Drainage-gated merge.** A film that ruptured (`FilmContactStatus.MERGED`)
   recolors the absorbed bubble into the color of the kept one; the kept
   identity is the smaller one, and the atmosphere is always kept. The pair
   is then exempt from conflict recoloring until the geometry joins.
4. **Conflict recoloring.** Same-color close pairs are recolored
   deterministically.

Refusals stop the run with an explicit status. The last accepted state is
returned, with no silent repair.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ..._validation import positive_finite_float, positive_integer
from ...solver import FixedStepResult, MACCompartmentProjectionStatus
from ._bubble_components import (
    ATMOSPHERE_ID,
    BubbleComponentLabels,
    BubbleTransitionJournal,
    transition_records,
)
from ._bubble_thermodynamics import BubbleCompartmentLedger
from ._bubbly_flow import BubblyFlowPlan, BubblyFlowState
from ._coalescence import FilmContactStatus
from ._multi_marker import MarkerProximity, MarkerRecolorStatus, MultiMarkerPlan
from ._step import IncompressibleTwoPhaseVOFMethod, TwoPhaseContinuationState


class BubblyFlowRunStatus(IntEnum):
    """Terminal status of one bubbly-flow run."""

    COMPLETED = 0
    STEP_FAILED = 1
    TRANSACTION_REFUSED = 2
    TRANSACTIONS_EXHAUSTED = 3


class BubblyFlowRun(StrictModule):
    """Result of `run_bubbly_flow`: last accepted state and host records.

    ``recolors`` lists ``(bubble_id, from_color, to_color)`` moves in
    execution order. ``merges`` lists the ``(kept, absorbed)`` identity pairs
    whose film ruptured. ``compartment_ledger`` accumulates the amount and
    energy moved by committed compartment transactions.
    """

    continuation: TwoPhaseContinuationState
    journal: BubbleTransitionJournal
    compartment_ledger: BubbleCompartmentLedger
    time: Array
    steps: int = eqx.field(static=True)
    recolors: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    merges: tuple[tuple[int, int], ...] = eqx.field(static=True)
    status: BubblyFlowRunStatus = eqx.field(static=True)
    message: str = eqx.field(static=True)

    @property
    def completed(self) -> bool:
        return self.status is BubblyFlowRunStatus.COMPLETED


@eqx.filter_jit
def _attempt(
    method: IncompressibleTwoPhaseVOFMethod,
    step_index: Array,
    time: Array,
    state: TwoPhaseContinuationState,
    step_size: Array,
) -> FixedStepResult:
    return method.step(step_index, time, state, step_size, None)


@eqx.filter_jit
def _proximity(plan: MultiMarkerPlan, labels: BubbleComponentLabels) -> MarkerProximity:
    return plan.proximity(labels)


def _bubbles(state: TwoPhaseContinuationState, /) -> BubblyFlowState:
    bubbles = state.bubbles
    if bubbles is None:
        raise ValueError("The continuation carries no bubble state.")
    return bubbles


def _ruptured_pairs(bubbles: BubblyFlowState, /) -> tuple[tuple[int, int], ...]:
    contacts = bubbles.contacts
    if contacts is None:
        return ()
    status = np.asarray(contacts.status)
    first = np.asarray(contacts.first_id)
    second = np.asarray(contacts.second_id)
    merged = np.flatnonzero(status == FilmContactStatus.MERGED.value)
    return tuple(
        sorted((int(first[slot]), int(second[slot])) for slot in merged.tolist())
    )


class _HostRecords:
    """Mutable host bookkeeping of one run (never part of device state)."""

    def __init__(self) -> None:
        self.journal = BubbleTransitionJournal()
        self.ledger = BubbleCompartmentLedger.zeros()
        self.recolors: list[tuple[int, int, int]] = []
        self.merges: list[tuple[int, int]] = []
        self.exempt: set[tuple[int, int]] = set()


def _compartment_transaction(
    plan: BubblyFlowPlan,
    state: TwoPhaseContinuationState,
    result: FixedStepResult,
    /,
) -> tuple[TwoPhaseContinuationState, BubbleCompartmentLedger] | None:
    """Propose the refused step's registry and ledger transaction."""

    compartments = plan.compartments
    source = _bubbles(state)
    if compartments is None or source.compartments is None:
        return None
    candidate: TwoPhaseContinuationState = result.candidate_state
    proposed = _bubbles(candidate)
    evidence = candidate.bubble_evidence
    if evidence is None:
        return None
    events = transition_records(proposed.event, source.identity)
    transaction = compartments.transact(
        source.compartments,
        events,
        evidence.creation_pressure,
        plan.atmosphere_pressure,
        proposed.identity.labels.centroid,
    )
    if not transaction.committed:
        return None
    repaired = eqx.tree_at(
        lambda value: value.bubbles.compartments, state, transaction.state
    )
    return repaired, transaction.ledger


def _marker_transactions(
    plan: BubblyFlowPlan,
    state: TwoPhaseContinuationState,
    records: _HostRecords,
    /,
) -> tuple[TwoPhaseContinuationState, str]:
    """Drainage-gated merges and conflict recoloring on an accepted state."""

    markers = plan.markers
    bubbles = _bubbles(state)
    evidence = state.bubble_evidence
    if markers is None or bubbles.markers is None or evidence is None:
        return state, ""
    ruptured = _ruptured_pairs(bubbles)
    if int(evidence.recolor_conflicts) == 0 and not ruptured:
        return state, ""
    identity = bubbles.identity
    ids = {int(value) for value in np.asarray(identity.slot_ids)}
    merges = tuple(
        (first, second) for first, second in ruptured if first in ids and second in ids
    )
    # The smaller identity is kept; the atmosphere (identity 0) always is.
    result = markers.recolor(
        bubbles.markers,
        identity.labels,
        identity.slot_ids,
        _proximity(markers, identity.labels),
        merge_pairs=merges,
        exempt_pairs=tuple(sorted(records.exempt)),
    )
    if result.status in (
        MarkerRecolorStatus.MARKER_CAPACITY_EXCEEDED,
        MarkerRecolorStatus.PROXIMITY_OVERFLOW,
    ):
        return state, f"marker recoloring refused: {result.status.name}"
    records.merges.extend(merges)
    records.exempt |= set(merges)
    if result.committed:
        records.recolors.extend(result.recolored)
        state = eqx.tree_at(lambda value: value.bubbles.markers, state, result.state)
    return state, ""


def _prune_exempt(records: _HostRecords, bubbles: BubblyFlowState, /) -> None:
    alive = {
        int(value)
        for value in np.asarray(bubbles.identity.slot_ids)
        if int(value) >= ATMOSPHERE_ID
    }
    records.exempt = {
        pair for pair in records.exempt if pair[0] in alive and pair[1] in alive
    }


def _finish(
    state: TwoPhaseContinuationState,
    records: _HostRecords,
    time: float,
    steps: int,
    status: BubblyFlowRunStatus,
    message: str,
    /,
) -> BubblyFlowRun:
    return BubblyFlowRun(
        continuation=state,
        journal=records.journal,
        compartment_ledger=records.ledger,
        time=jnp.asarray(time, dtype=jnp.float64),
        steps=steps,
        recolors=tuple(records.recolors),
        merges=tuple(records.merges),
        status=status,
        message=message,
    )


def _step_failure_message(index: int, result: FixedStepResult, /) -> str:
    evidence = result.candidate_state.bubble_evidence
    if evidence is None:
        return f"step {index} failed"
    projection = MACCompartmentProjectionStatus(int(evidence.projection_status))
    if projection is MACCompartmentProjectionStatus.CONVERGED:
        return f"step {index} failed"
    return f"step {index} failed: {projection.name}"


def run_bubbly_flow(
    method: IncompressibleTwoPhaseVOFMethod,
    continuation: TwoPhaseContinuationState,
    /,
    *,
    steps: int,
    step_size: float,
    start_time: float = 0.0,
    first_step: int = 0,
    maximum_transactions: int = 4,
    observer: Callable[[float, TwoPhaseContinuationState], None] | None = None,
) -> BubblyFlowRun:
    """Advance a bubbly-flow continuation with host epoch transactions.

    ``first_step`` continues the step index (and so the sweep rotation) of an
    earlier run. ``observer(time, state)`` is called on the host after every
    accepted step and its transactions.
    """

    plan = method.bubbles
    if plan is None:
        raise ValueError("method declares no bubbly-flow plan.")
    count = positive_integer(steps, "steps")
    dt = positive_finite_float(step_size, "step_size")
    retries = positive_integer(maximum_transactions, "maximum_transactions")
    records = _HostRecords()
    state = continuation
    time = float(start_time)
    for index in range(count):
        source = state
        attempt_state = source
        candidate_ledger = records.ledger
        result = None
        for transaction_count in range(retries + 1):
            result = _attempt(
                method,
                jnp.asarray(first_step + index, dtype=jnp.int32),
                jnp.asarray(time, dtype=jnp.float64),
                attempt_state,
                jnp.asarray(dt, dtype=jnp.float64),
            )
            if bool(result.successful):
                break
            evidence = result.candidate_state.bubble_evidence
            if evidence is None or not bool(evidence.registry_mismatch):
                return _finish(
                    source,
                    records,
                    time,
                    index,
                    BubblyFlowRunStatus.STEP_FAILED,
                    _step_failure_message(index, result),
                )
            if transaction_count == retries:
                break
            proposal = _compartment_transaction(plan, attempt_state, result)
            if proposal is None:
                return _finish(
                    source,
                    records,
                    time,
                    index,
                    BubblyFlowRunStatus.TRANSACTION_REFUSED,
                    f"compartment transaction refused at step {index}",
                )
            attempt_state, ledger = proposal
            candidate_ledger = jax.tree.map(jnp.add, candidate_ledger, ledger)
        if result is None or not bool(result.successful):
            return _finish(
                source,
                records,
                time,
                index,
                BubblyFlowRunStatus.TRANSACTIONS_EXHAUSTED,
                f"step {index} still requires a compartment transaction",
            )
        records.ledger = candidate_ledger
        accepted: TwoPhaseContinuationState = result.accepted_state
        bubbles = _bubbles(accepted)
        if bool(bubbles.event.topology_changed):
            records.journal = records.journal.extend(
                transition_records(bubbles.event, _bubbles(attempt_state).identity)
            )
        state, refusal = _marker_transactions(plan, accepted, records)
        time += dt
        if refusal:
            return _finish(
                state,
                records,
                time,
                index + 1,
                BubblyFlowRunStatus.TRANSACTION_REFUSED,
                refusal,
            )
        _prune_exempt(records, _bubbles(state))
        if observer is not None:
            observer(time, state)
    return _finish(state, records, time, count, BubblyFlowRunStatus.COMPLETED, "")


__all__ = ["BubblyFlowRun", "BubblyFlowRunStatus", "run_bubbly_flow"]
