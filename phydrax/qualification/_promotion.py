#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Signed channel chains with transactional CAS and a persistent generation floor."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .._fingerprint import canonical_fingerprint, canonical_json
from ._registry import ReleaseIndex, require_profile
from ._trust import (
    AsymmetricReleaseTrustPolicy,
    QualificationRoleTrust,
    SignedQualificationRecord,
)


@dataclass(frozen=True, slots=True)
class PromotionState:
    channel: str
    generation: int
    index_id: str | None
    previous_state_id: str | None
    action: str
    reason: str
    approver: str
    effective_at: int
    expires_at: int
    distribution_id: str | None
    signature: SignedQualificationRecord

    def __post_init__(self) -> None:
        if self.action not in ("promote", "withdraw", "rollback", "refuse"):
            raise ValueError("Unknown promotion action.")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("Promotion generations are positive integers.")
        if (
            type(self.effective_at) is not int
            or type(self.expires_at) is not int
            or not 0 <= self.effective_at < self.expires_at
        ):
            raise ValueError("Promotion validity interval is invalid.")
        for value in (self.channel, self.reason, self.approver):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("Promotion identifiers must be canonical.")
        active = self.action in ("promote", "rollback")
        if active != (self.index_id is not None) or active != (
            self.distribution_id is not None
        ):
            raise ValueError("Withdrawal/refusal states must be signed empty states.")
        if (self.generation == 1) != (self.previous_state_id is None):
            raise ValueError("Only the initial generation may have no predecessor.")

    def content_record(self) -> dict[str, object]:
        return {
            "kind": "promotion-state",
            "channel": self.channel,
            "generation": self.generation,
            "index_id": self.index_id,
            "previous_state_id": self.previous_state_id,
            "action": self.action,
            "reason": self.reason,
            "approver": self.approver,
            "effective_at": self.effective_at,
            "expires_at": self.expires_at,
            "distribution_id": self.distribution_id,
        }

    @property
    def state_id(self) -> str:
        return canonical_fingerprint(self.content_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self.content_record(),
            "state_id": self.state_id,
            "signature": self.signature.to_record(),
        }

    @classmethod
    def from_record(cls, record, /):
        if record.get("kind") != "promotion-state":
            raise ValueError("Invalid promotion-state record.")
        values = {
            name: record[name] for name in cls.__dataclass_fields__ if name != "signature"
        }
        result = cls(
            **values, signature=SignedQualificationRecord.from_record(record["signature"])
        )
        if result.state_id != record["state_id"]:
            raise ValueError("Promotion state content address is invalid.")
        return result

    def verify(self, trust: QualificationRoleTrust, /, *, at_time: int) -> None:
        if not self.effective_at <= at_time < self.expires_at:
            raise ValueError("Promotion state is stale or not yet active.")
        key = trust.verify(
            self.content_record(),
            self.signature,
            role="channel-promoter",
            at_time=at_time,
        )
        if (
            key != self.approver
            or self.signature.issued_at != self.effective_at
            or self.expires_at > trust.expiry(self.signature)
        ):
            raise ValueError("Promotion approver, timestamp or expiry is mismatched.")


class PromotionConflictError(ValueError):
    """A concurrent transition or a generation-floor violation prevented the CAS."""


class PromotionRepository:
    """One durable SQLite transaction owns immutable log, floor and channel head.

    ``minimum_generation`` lets a separately persisted consumer checkpoint reject
    restoring the entire repository to an older filesystem snapshot.
    """

    def __init__(self, path: str | Path, /):
        self.path = str(path)
        with closing(self._connect()) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS promotion_states (
                    state_id TEXT PRIMARY KEY, channel TEXT NOT NULL,
                    generation INTEGER NOT NULL, record TEXT NOT NULL,
                    UNIQUE(channel, generation));
                CREATE TABLE IF NOT EXISTS promotion_heads (
                    channel TEXT PRIMARY KEY, state_id TEXT NOT NULL,
                    generation_floor INTEGER NOT NULL);
                CREATE TRIGGER IF NOT EXISTS immutable_promotion_update BEFORE UPDATE ON promotion_states
                    BEGIN SELECT RAISE(ABORT, 'immutable promotion log'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_promotion_delete BEFORE DELETE ON promotion_states
                    BEGIN SELECT RAISE(ABORT, 'immutable promotion log'); END;
                CREATE TRIGGER IF NOT EXISTS monotonic_promotion_floor BEFORE UPDATE ON promotion_heads
                    WHEN NEW.generation_floor <= OLD.generation_floor
                    BEGIN SELECT RAISE(ABORT, 'promotion generation regression'); END;
                CREATE TRIGGER IF NOT EXISTS retained_promotion_floor BEFORE DELETE ON promotion_heads
                    BEGIN SELECT RAISE(ABORT, 'promotion floor cannot be deleted'); END;
            """)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def current(
        self,
        channel: str,
        trust: QualificationRoleTrust,
        /,
        *,
        at_time: int,
        minimum_generation: int = 0,
    ) -> PromotionState | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT h.state_id,h.generation_floor,s.record FROM promotion_heads h "
                "LEFT JOIN promotion_states s ON h.state_id=s.state_id WHERE h.channel=?",
                (channel,),
            ).fetchone()
            maximum = connection.execute(
                "SELECT COALESCE(MAX(generation),0) FROM promotion_states WHERE channel=?",
                (channel,),
            ).fetchone()[0]
        if row is None:
            if minimum_generation or maximum:
                raise PromotionConflictError(
                    "Channel head is missing below its generation floor."
                )
            return None
        state_id, floor, raw = row
        if raw is None or floor < minimum_generation or floor != maximum:
            raise PromotionConflictError(
                "Channel head or persistent generation floor was rolled back."
            )
        state = PromotionState.from_record(json.loads(raw))
        if (
            state.state_id != state_id
            or state.generation != floor
            or state.channel != channel
        ):
            raise PromotionConflictError("Channel head and transparency record disagree.")
        state.verify(trust, at_time=at_time)
        return state

    def history(self, channel: str, /) -> tuple[PromotionState, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT record FROM promotion_states WHERE channel=? ORDER BY generation",
                (channel,),
            ).fetchall()
        states = tuple(PromotionState.from_record(json.loads(row[0])) for row in rows)
        previous = None
        for generation, state in enumerate(states, 1):
            if state.generation != generation or state.previous_state_id != previous:
                raise PromotionConflictError(
                    "Promotion transparency chain is incomplete."
                )
            previous = state.state_id
        return states

    def compare_and_swap(
        self,
        state: PromotionState,
        /,
        *,
        expected_state_id: str | None,
        trust_policy: AsymmetricReleaseTrustPolicy,
        at_time: int,
        index: ReleaseIndex | None = None,
        minimum_generation: int = 0,
    ) -> PromotionState:
        state.verify(trust_policy.roles, at_time=at_time)
        if state.index_id is not None:
            _validate_index(index, trust_policy, state.distribution_id, at_time)
            if index.index_id != state.index_id:
                raise ValueError("Promotion index was substituted.")
        elif index is not None:
            raise ValueError("An empty channel transition cannot retain an active index.")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            head = connection.execute(
                "SELECT state_id,generation_floor FROM promotion_heads WHERE channel=?",
                (state.channel,),
            ).fetchone()
            actual, floor = (None, 0) if head is None else head
            maximum = connection.execute(
                "SELECT COALESCE(MAX(generation),0) FROM promotion_states WHERE channel=?",
                (state.channel,),
            ).fetchone()[0]
            if (
                actual != expected_state_id
                or state.previous_state_id != actual
                or floor != maximum
                or floor < minimum_generation
                or state.generation != floor + 1
            ):
                raise PromotionConflictError(
                    "Promotion CAS conflict or generation regression."
                )
            if state.action == "rollback":
                prior = connection.execute(
                    "SELECT record FROM promotion_states WHERE channel=?",
                    (state.channel,),
                ).fetchall()
                if not any(
                    json.loads(row[0])["index_id"] == state.index_id for row in prior
                ):
                    raise ValueError(
                        "Rollback target was never admitted on this channel."
                    )
            connection.execute(
                "INSERT INTO promotion_states VALUES (?,?,?,?)",
                (
                    state.state_id,
                    state.channel,
                    state.generation,
                    canonical_json(state.to_record()),
                ),
            )
            connection.execute(
                "INSERT INTO promotion_heads VALUES (?,?,?) ON CONFLICT(channel) "
                "DO UPDATE SET state_id=excluded.state_id,"
                "generation_floor=excluded.generation_floor",
                (state.channel, state.state_id, state.generation),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return state


def _validate_index(index, trust_policy, distribution_id, at_time):
    if not isinstance(index, ReleaseIndex) or not distribution_id:
        raise ValueError(
            "An active promotion requires an exact signed index and distribution."
        )
    index.require_trusted(trust_policy, at_time)
    for profile in index.profiles:
        for support in profile.support_tuples:
            require_profile(
                index, profile.profile_id, support, trust_policy, at_time=at_time
            )
        for gate in profile.release_evidence:
            proofs = [
                proof
                for proof in trust_policy.proofs
                if proof.gate_id == gate.evidence_id
            ]
            if len(proofs) != 1 or proofs[0].distribution_id != distribution_id:
                raise ValueError(
                    "Promotion proof is missing or belongs to a different distribution."
                )
            proofs[0].verify(trust_policy, at_time=at_time)


def advance_channel(
    repository: PromotionRepository,
    signer,
    trust_policy: AsymmetricReleaseTrustPolicy,
    /,
    *,
    channel: str,
    expected_state_id: str | None,
    action: str,
    reason: str,
    at_time: int,
    expires_at: int,
    index: ReleaseIndex | None = None,
    distribution_id: str | None = None,
    minimum_generation: int = 0,
) -> PromotionState:
    history = repository.history(channel)
    previous = history[-1] if history else None
    if (None if previous is None else previous.state_id) != expected_state_id:
        raise PromotionConflictError("Promotion compare-and-swap predecessor changed.")
    if action in ("promote", "rollback"):
        try:
            _validate_index(index, trust_policy, distribution_id, at_time)
            if action == "rollback" and not any(
                state.index_id == index.index_id for state in history
            ):
                raise ValueError("Rollback target was never admitted on this channel.")
        except (ValueError, TypeError, KeyError, RuntimeError):
            if action != "rollback":
                raise
            action, index, distribution_id = "refuse", None, None
            reason = f"rollback-refused:{reason}"
    elif action not in ("withdraw", "refuse"):
        raise ValueError("Unknown channel action.")
    elif index is not None or distribution_id is not None:
        raise ValueError("Withdrawal and refusal must not retain release coordinates.")
    roots = {record.key_id: record for record in trust_policy.roles.store.records()}
    if (
        signer.key_id not in trust_policy.roles.roles["channel-promoter"]
        or signer.key_id not in roots
    ):
        raise ValueError("Channel signer is not an authorized public trust principal.")
    expiry = min(expires_at, roots[signer.key_id].expires_at)
    if index is not None:
        expiry = min(
            expiry,
            index.issued_at + trust_policy.max_index_age,
            roots[index.signer_id].expires_at,
            *(
                gate.expires_at
                for profile in index.profiles
                for gate in profile.release_evidence
            ),
        )
    content = {
        "kind": "promotion-state",
        "channel": channel,
        "generation": 1 if previous is None else previous.generation + 1,
        "index_id": None if index is None else index.index_id,
        "previous_state_id": expected_state_id,
        "action": action,
        "reason": reason,
        "approver": signer.key_id,
        "effective_at": at_time,
        "expires_at": expiry,
        "distribution_id": distribution_id,
    }
    signature = SignedQualificationRecord.sign(
        content, signer, role="channel-promoter", issued_at=at_time, expires_at=expiry
    )
    state = PromotionState(
        **{key: value for key, value in content.items() if key != "kind"},
        signature=signature,
    )
    return repository.compare_and_swap(
        state,
        expected_state_id=expected_state_id,
        trust_policy=trust_policy,
        at_time=at_time,
        index=index,
        minimum_generation=minimum_generation,
    )
