#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where


State = TypeVar("State")
Evidence = TypeVar("Evidence")


class TransactionalCandidate(StrictModule, NonTrainableState, Generic[State, Evidence]):
    source: State
    proposed: State
    evidence: Evidence
    accepted: Array
    source_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: State,
        proposed: State,
        evidence: Evidence,
        accepted: Any,
        source_id: str,
        /,
    ):
        identifier = str(source_id).strip()
        if not identifier:
            raise ValueError("source_id must be non-empty.")
        accepted_ = jnp.asarray(accepted, dtype=bool)
        if accepted_.shape != ():
            raise ValueError("accepted must be scalar.")
        self.source = source
        self.proposed = proposed
        self.evidence = evidence
        self.accepted = accepted_
        self.source_id = identifier
        self.candidate_id = canonical_fingerprint(
            {"kind": "transactional-candidate", "source": identifier}
        )


class TransactionalCommit(StrictModule, NonTrainableState, Generic[State, Evidence]):
    state: State
    evidence: Evidence
    committed: Array
    candidate_id: str = eqx.field(static=True)


def commit_candidate(
    candidate: TransactionalCandidate[State, Evidence], /
) -> TransactionalCommit[State, Evidence]:
    if not isinstance(candidate, TransactionalCandidate):
        raise TypeError("candidate must be TransactionalCandidate.")
    state = tree_where(candidate.accepted, candidate.proposed, candidate.source)
    return TransactionalCommit(
        state, candidate.evidence, candidate.accepted, candidate.candidate_id
    )


__all__ = ["TransactionalCandidate", "TransactionalCommit", "commit_candidate"]
