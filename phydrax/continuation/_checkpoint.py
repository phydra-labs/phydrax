#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._state_machine import (
    AbstractContinuationAdapter,
    ContinuationAcceptedState,
    ContinuationCandidate,
    ContinuationStepResult,
    ParameterRealization,
)


class ContinuationReplayEvidence(StrictModule):
    """Identity evidence for restoring one committed continuation decision history."""

    runtime_identities_match: Array
    decision_history_matches: Array
    realization_matches: Array
    application_state_matches: Array
    checkpoint_id: str = eqx.field(static=True)
    expected_decision_ids: tuple[str, ...] = eqx.field(static=True)
    observed_decision_ids: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        checkpoint_id: str,
        expected_decision_ids: Sequence[str],
        observed_decision_ids: Sequence[str],
        runtime_identities_match: Any,
        decision_history_matches: Any,
        realization_matches: Any,
        application_state_matches: Any,
    ):
        identifier = str(checkpoint_id)
        expected = tuple(str(value) for value in expected_decision_ids)
        observed = tuple(str(value) for value in observed_decision_ids)
        if not identifier:
            raise ValueError("Replay evidence checkpoint_id must be non-empty.")
        if any(not value for value in expected) or any(not value for value in observed):
            raise ValueError("Replay decision identities must be non-empty.")
        flags = tuple(
            jnp.asarray(value, dtype=bool)
            for value in (
                runtime_identities_match,
                decision_history_matches,
                realization_matches,
                application_state_matches,
            )
        )
        if any(value.shape != () for value in flags):
            raise ValueError("Continuation replay evidence flags must be scalar.")
        evidence_id = canonical_fingerprint(
            {
                "kind": "continuation-checkpoint-replay",
                "checkpoint": identifier,
                "expected_decisions": expected,
                "observed_decisions": observed,
                "flags": tuple(bool(value) for value in flags),
            }
        )
        (
            self.runtime_identities_match,
            self.decision_history_matches,
            self.realization_matches,
            self.application_state_matches,
        ) = flags
        self.checkpoint_id = identifier
        self.expected_decision_ids = expected
        self.observed_decision_ids = observed
        self.evidence_id = evidence_id

    @property
    def matches(self) -> Array:
        return (
            self.runtime_identities_match
            & self.decision_history_matches
            & self.realization_matches
            & self.application_state_matches
        )


class ContinuationCheckpoint(StrictModule):
    """Committed, serializable continuation data plus exact replay identities."""

    candidate: ContinuationCandidate
    application_data: Any
    replay_evidence: ContinuationReplayEvidence
    problem_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    application_state_id: str = eqx.field(static=True)
    accepted_decision_ids: tuple[str, ...] = eqx.field(static=True)
    attempt_decision_ids: tuple[str, ...] = eqx.field(static=True)
    accepted_index: int = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate: ContinuationCandidate,
        application_data: Any,
        /,
        *,
        problem_id: str,
        adapter_id: str,
        plan_id: str,
        prepared_id: str,
        branch_id: str,
        application_state_id: str,
        accepted_decision_ids: Sequence[str],
        attempt_decision_ids: Sequence[str],
        accepted_index: int,
        replay_evidence: ContinuationReplayEvidence | None = None,
    ):
        if not isinstance(candidate, ContinuationCandidate):
            raise TypeError("candidate must be a ContinuationCandidate.")
        identifiers = tuple(
            str(value)
            for value in (
                problem_id,
                adapter_id,
                plan_id,
                prepared_id,
                branch_id,
                application_state_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Continuation checkpoint identities must be non-empty.")
        accepted = tuple(str(value) for value in accepted_decision_ids)
        attempts = tuple(str(value) for value in attempt_decision_ids)
        if not accepted or any(not value for value in accepted + attempts):
            raise ValueError(
                "A checkpoint requires non-empty accepted decision identities."
            )
        if any(value not in attempts for value in accepted):
            raise ValueError("Accepted decisions must be present in the attempt history.")
        if len(set(accepted)) != len(accepted) or len(set(attempts)) != len(attempts):
            raise ValueError("Checkpoint decision histories must not contain duplicates.")
        index = int(accepted_index)
        if index < 0 or index + 1 != len(accepted):
            raise ValueError("Checkpoint accepted index and decision history disagree.")
        checkpoint_id = canonical_fingerprint(
            {
                "kind": "continuation-checkpoint",
                "problem": identifiers[0],
                "adapter": identifiers[1],
                "plan": identifiers[2],
                "prepared": identifiers[3],
                "branch": identifiers[4],
                "application_state": identifiers[5],
                "candidate": candidate.candidate_id,
                "realization": candidate.realization.realization_id,
                "accepted_decisions": accepted,
                "attempt_decisions": attempts,
                "accepted_index": index,
            }
        )
        replay = (
            ContinuationReplayEvidence(
                checkpoint_id=checkpoint_id,
                expected_decision_ids=attempts,
                observed_decision_ids=attempts,
                runtime_identities_match=True,
                decision_history_matches=True,
                realization_matches=True,
                application_state_matches=True,
            )
            if replay_evidence is None
            else replay_evidence
        )
        if not isinstance(replay, ContinuationReplayEvidence):
            raise TypeError("replay_evidence must be ContinuationReplayEvidence or None.")
        if replay.checkpoint_id != checkpoint_id:
            raise ValueError("Replay evidence belongs to another checkpoint.")
        if (
            replay.expected_decision_ids != attempts
            or replay.observed_decision_ids != attempts
            or not bool(replay.matches)
        ):
            raise ValueError(
                "Checkpoint replay evidence must certify the complete attempt history."
            )
        self.candidate = candidate
        self.application_data = application_data
        self.replay_evidence = replay
        (
            self.problem_id,
            self.adapter_id,
            self.plan_id,
            self.prepared_id,
            self.branch_id,
            self.application_state_id,
        ) = identifiers
        self.accepted_decision_ids = accepted
        self.attempt_decision_ids = attempts
        self.accepted_index = index
        self.checkpoint_id = checkpoint_id


def continuation_checkpoint(
    accepted_state: Any,
    steps: Sequence[ContinuationStepResult],
    application_data: Any,
    /,
    *,
    problem_id: str,
    adapter_id: str,
    plan_id: str,
    prepared_id: str,
    branch_id: str,
    prior_accepted_decision_ids: Sequence[str] = (),
    prior_attempt_decision_ids: Sequence[str] = (),
) -> ContinuationCheckpoint:
    """Create a checkpoint only from committed application and decision data."""

    if not isinstance(accepted_state, ContinuationAcceptedState):
        raise TypeError("accepted_state must be a ContinuationAcceptedState.")
    steps_ = tuple(steps)
    if any(not isinstance(step, ContinuationStepResult) for step in steps_):
        raise TypeError("steps must contain ContinuationStepResult values.")
    prior_accepted = tuple(str(value) for value in prior_accepted_decision_ids)
    prior_attempts = tuple(str(value) for value in prior_attempt_decision_ids)
    if any(not value for value in prior_accepted + prior_attempts):
        raise ValueError("Prior continuation decision identities must be non-empty.")
    if any(value not in prior_attempts for value in prior_accepted):
        raise ValueError("Prior accepted decisions must belong to the attempt history.")
    accepted_ids = prior_accepted + tuple(
        step.decision_id for step in steps_ if bool(step.accepted)
    )
    attempt_ids = prior_attempts + tuple(step.decision_id for step in steps_)
    if not accepted_ids or accepted_ids[-1] != accepted_state.decision_id:
        raise ValueError("Final accepted state is not the final accepted step decision.")
    return ContinuationCheckpoint(
        accepted_state.candidate,
        application_data,
        problem_id=problem_id,
        adapter_id=adapter_id,
        plan_id=plan_id,
        prepared_id=prepared_id,
        branch_id=branch_id,
        application_state_id=accepted_state.application_state_id,
        accepted_decision_ids=accepted_ids,
        attempt_decision_ids=attempt_ids,
        accepted_index=accepted_state.accepted_index,
    )


def continuation_replay_evidence(
    checkpoint: ContinuationCheckpoint,
    /,
    *,
    observed_decision_ids: Sequence[str],
    runtime_identities_match: Any,
    realization_matches: Any,
    application_state_matches: Any,
) -> ContinuationReplayEvidence:
    """Compare restored runtime data with the complete checkpoint attempt history."""
    if not isinstance(checkpoint, ContinuationCheckpoint):
        raise TypeError("checkpoint must be a ContinuationCheckpoint.")
    observed = tuple(str(value) for value in observed_decision_ids)
    return ContinuationReplayEvidence(
        checkpoint_id=checkpoint.checkpoint_id,
        expected_decision_ids=checkpoint.attempt_decision_ids,
        observed_decision_ids=observed,
        runtime_identities_match=runtime_identities_match,
        decision_history_matches=observed == checkpoint.attempt_decision_ids,
        realization_matches=realization_matches,
        application_state_matches=application_state_matches,
    )


def restore_continuation_checkpoint(
    checkpoint: ContinuationCheckpoint,
    adapter: Any,
    /,
    *,
    plan_id: str,
    prepared_id: str,
    branch_id: str,
    args: Any = None,
    observed_decision_ids: Sequence[str] | None = None,
) -> tuple[ContinuationAcceptedState, ContinuationReplayEvidence]:
    """Restore and verify the opaque application state of one accepted checkpoint."""

    if not isinstance(checkpoint, ContinuationCheckpoint):
        raise TypeError("checkpoint must be a ContinuationCheckpoint.")
    if not isinstance(adapter, AbstractContinuationAdapter):
        raise TypeError("adapter must be an AbstractContinuationAdapter.")
    restored = adapter.restore_application_state(checkpoint.application_data, args)
    application_state_id = adapter.application_state_identity(restored, args)
    realized = ParameterRealization(
        adapter.parameters(checkpoint.candidate.coordinate, args),
        checkpoint.candidate.coordinate,
        problem_id=adapter.problem_id,
    )
    observed = (
        checkpoint.attempt_decision_ids
        if observed_decision_ids is None
        else tuple(observed_decision_ids)
    )
    evidence = continuation_replay_evidence(
        checkpoint,
        observed_decision_ids=observed,
        runtime_identities_match=(
            adapter.problem_id == checkpoint.problem_id
            and adapter.adapter_id == checkpoint.adapter_id
            and str(plan_id) == checkpoint.plan_id
            and str(prepared_id) == checkpoint.prepared_id
            and str(branch_id) == checkpoint.branch_id
        ),
        realization_matches=(
            realized.realization_id == checkpoint.candidate.realization.realization_id
        ),
        application_state_matches=(
            application_state_id == checkpoint.application_state_id
        ),
    )
    if not bool(evidence.matches):
        raise ValueError("Restored continuation checkpoint identities did not match.")
    accepted_state = ContinuationAcceptedState(
        checkpoint.candidate,
        restored,
        application_state_id=application_state_id,
        decision_id=checkpoint.accepted_decision_ids[-1],
        accepted_index=checkpoint.accepted_index,
    )
    return accepted_state, evidence


__all__ = [
    "ContinuationCheckpoint",
    "ContinuationReplayEvidence",
    "continuation_checkpoint",
    "continuation_replay_evidence",
    "restore_continuation_checkpoint",
]
