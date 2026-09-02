#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import ReleaseGateEvidence
from ._platform_support import (
    _identifier,
    _nonnegative_integer,
    _positive_integer,
    TensorNetworkClaim,
    TensorNetworkFailure,
    TensorNetworkMaturity,
    TensorNetworkSupportTuple,
)


class TensorNetworkReleaseGate(StrEnum):
    INTENDED_USE = "intended-use"
    CODE_VERIFICATION = "code-verification"
    SOLUTION_VERIFICATION = "solution-verification"
    ARCHIVE_REPLAY = "archive-replay"
    SECURITY = "security"
    PROVENANCE_LICENSE = "provenance-license"
    INDEPENDENT_APPROVAL = "independent-approval"


def _finite_nonnegative(value: object, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


class TensorNetworkQualificationProfile(StrictModule, NonTrainableState):
    supported_tuples: tuple[TensorNetworkSupportTuple, ...]
    required_claims: tuple[TensorNetworkClaim, ...] = eqx.field(static=True)
    claim_tolerances: tuple[tuple[TensorNetworkClaim, float], ...] = eqx.field(
        static=True
    )
    required_release_gates: tuple[TensorNetworkReleaseGate, ...] = eqx.field(static=True)
    maximum_samples_per_claim: int = eqx.field(static=True)
    target_maturity: TensorNetworkMaturity = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        supported_tuples: Sequence[TensorNetworkSupportTuple],
        claim_tolerances: Mapping[TensorNetworkClaim, float],
        /,
        *,
        required_claims: Sequence[TensorNetworkClaim] | None = None,
        required_release_gates: Sequence[TensorNetworkReleaseGate] = tuple(
            TensorNetworkReleaseGate
        ),
        maximum_samples_per_claim: int = 100_000,
        target_maturity: TensorNetworkMaturity = TensorNetworkMaturity.QUALIFIED,
    ):
        supported = tuple(supported_tuples)
        if not supported or any(
            not isinstance(value, TensorNetworkSupportTuple) for value in supported
        ):
            raise TypeError("Qualification requires explicit typed support tuples.")
        support_ids = tuple(value.support_tuple_id for value in supported)
        if len(set(support_ids)) != len(support_ids):
            raise ValueError("Qualification support tuples must be unique.")
        claims = (
            tuple(TensorNetworkClaim(value) for value in TensorNetworkClaim)
            if required_claims is None
            else tuple(TensorNetworkClaim(value) for value in required_claims)
        )
        if not claims or len(set(claims)) != len(claims):
            raise ValueError("Required qualification claims must be nonempty and unique.")
        if not isinstance(claim_tolerances, Mapping):
            raise TypeError("claim_tolerances must be a mapping.")
        normalized_tolerances = {
            TensorNetworkClaim(claim): _finite_nonnegative(
                tolerance, f"{TensorNetworkClaim(claim).value} tolerance"
            )
            for claim, tolerance in claim_tolerances.items()
        }
        if set(normalized_tolerances) != set(claims):
            raise ValueError("Claim tolerances must exactly cover required claims.")
        tolerances = tuple(
            sorted(normalized_tolerances.items(), key=lambda item: item[0].value)
        )
        gates = tuple(TensorNetworkReleaseGate(value) for value in required_release_gates)
        if not gates or len(set(gates)) != len(gates):
            raise ValueError("Required release gates must be nonempty and unique.")
        maximum = _positive_integer(
            maximum_samples_per_claim, "maximum_samples_per_claim"
        )
        target = TensorNetworkMaturity(target_maturity)
        if target != TensorNetworkMaturity.QUALIFIED:
            raise ValueError("Qualification profiles must target qualified maturity.")
        self.supported_tuples = supported
        self.required_claims = claims
        self.claim_tolerances = tolerances
        self.required_release_gates = gates
        self.maximum_samples_per_claim = maximum
        self.target_maturity = target
        self.profile_id = canonical_fingerprint(
            {
                "kind": "tensor-network-qualification-profile",
                "supported_tuples": support_ids,
                "required_claims": [claim.value for claim in claims],
                "claim_tolerances": {
                    claim.value: tolerance for claim, tolerance in tolerances
                },
                "required_release_gates": [gate.value for gate in gates],
                "maximum_samples_per_claim": maximum,
                "target_maturity": target.value,
            }
        )

    def tolerance_for(self, claim: TensorNetworkClaim, /) -> float:
        claim_ = TensorNetworkClaim(claim)
        for candidate, tolerance in self.claim_tolerances:
            if candidate == claim_:
                return tolerance
        raise KeyError(f"Claim {claim_.value!r} is not required by this profile.")

    def supports(self, support: TensorNetworkSupportTuple, /) -> bool:
        if not isinstance(support, TensorNetworkSupportTuple):
            raise TypeError("support must be TensorNetworkSupportTuple.")
        return support.support_tuple_id in {
            value.support_tuple_id for value in self.supported_tuples
        }


class TensorNetworkClaimEvidence(StrictModule, NonTrainableState):
    profile_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    claim: TensorNetworkClaim = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    nonfinite_count: int = eqx.field(static=True)
    maximum_error: float | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: TensorNetworkQualificationProfile,
        support: TensorNetworkSupportTuple,
        claim: TensorNetworkClaim,
        observations: Any,
        /,
        *,
        source_id: str,
    ):
        if not isinstance(profile, TensorNetworkQualificationProfile) or not isinstance(
            support, TensorNetworkSupportTuple
        ):
            raise TypeError("Claim evidence requires profile and support tuple.")
        if not profile.supports(support):
            raise ValueError("Claim evidence support tuple is outside the profile.")
        claim_ = TensorNetworkClaim(claim)
        tolerance = profile.tolerance_for(claim_)
        leaves = tuple(jax.tree.leaves(observations))
        if not leaves:
            raise ValueError("Claim evidence requires observed numerical values.")
        arrays = tuple(np.asarray(value) for value in leaves)
        if any(
            value.size == 0 or value.dtype.hasobject or value.dtype.kind not in "biufc"
            for value in arrays
        ):
            raise TypeError(
                "Qualification observations must be nonempty numerical arrays."
            )
        sample_count = sum(int(value.size) for value in arrays)
        if sample_count <= 0 or sample_count > profile.maximum_samples_per_claim:
            raise ValueError("Qualification observation count exceeds its finite bound.")
        nonfinite_count = sum(
            int(np.count_nonzero(~np.isfinite(value))) for value in arrays
        )
        maximum_error = (
            None
            if nonfinite_count
            else max(float(np.max(np.abs(value))) for value in arrays)
        )
        passed = bool(
            nonfinite_count == 0
            and maximum_error is not None
            and maximum_error <= tolerance
        )
        source = _identifier(source_id, "source_id")
        self.profile_id = profile.profile_id
        self.support_tuple_id = support.support_tuple_id
        self.claim = claim_
        self.source_id = source
        self.sample_count = sample_count
        self.nonfinite_count = nonfinite_count
        self.maximum_error = maximum_error
        self.tolerance = tolerance
        self.passed = passed
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "tensor-network-claim-evidence",
                "profile": profile.profile_id,
                "support": support.support_tuple_id,
                "claim": claim_.value,
                "source": source,
                "sample_count": sample_count,
                "nonfinite_count": nonfinite_count,
                "maximum_error": maximum_error,
                "tolerance": tolerance,
                "passed": passed,
            }
        )


class TensorNetworkQualificationResult(StrictModule, NonTrainableState):
    profile: TensorNetworkQualificationProfile
    support: TensorNetworkSupportTuple
    evidence: tuple[TensorNetworkClaimEvidence, ...]
    passed: bool = eqx.field(static=True)
    maturity: TensorNetworkMaturity = eqx.field(static=True)
    failure: TensorNetworkFailure = eqx.field(static=True)
    failed_claims: tuple[TensorNetworkClaim, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: TensorNetworkQualificationProfile,
        support: TensorNetworkSupportTuple,
        evidence: Sequence[TensorNetworkClaimEvidence],
        /,
    ):
        if not isinstance(profile, TensorNetworkQualificationProfile) or not isinstance(
            support, TensorNetworkSupportTuple
        ):
            raise TypeError("Qualification result requires profile and support tuple.")
        if not profile.supports(support):
            raise ValueError("Qualification support tuple is outside the profile.")
        evidence_ = tuple(evidence)
        if any(not isinstance(value, TensorNetworkClaimEvidence) for value in evidence_):
            raise TypeError("Qualification evidence must contain typed claim evidence.")
        claims = tuple(value.claim for value in evidence_)
        if len(set(claims)) != len(claims) or set(claims) != set(profile.required_claims):
            raise ValueError(
                "Qualification evidence must cover every claim exactly once."
            )
        if any(
            value.profile_id != profile.profile_id
            or value.support_tuple_id != support.support_tuple_id
            for value in evidence_
        ):
            raise ValueError("Qualification evidence identity changed.")
        ordered = tuple(
            next(value for value in evidence_ if value.claim == claim)
            for claim in profile.required_claims
        )
        failed = tuple(value.claim for value in ordered if not value.passed)
        passed = not failed
        maturity = (
            profile.target_maturity if passed else TensorNetworkMaturity.EXPERIMENTAL
        )
        failure = (
            TensorNetworkFailure.NONE
            if passed
            else TensorNetworkFailure.QUALIFICATION_FAILED
        )
        self.profile = profile
        self.support = support
        self.evidence = ordered
        self.passed = passed
        self.maturity = maturity
        self.failure = failure
        self.failed_claims = failed
        self.result_id = canonical_fingerprint(
            {
                "kind": "tensor-network-qualification-result",
                "profile": profile.profile_id,
                "support": support.support_tuple_id,
                "evidence": [value.evidence_id for value in ordered],
                "passed": passed,
                "maturity": maturity.value,
                "failure": failure.value,
                "failed_claims": [value.value for value in failed],
            }
        )


class TensorNetworkReleaseDecision(StrictModule, NonTrainableState):
    qualification_result_id: str = eqx.field(static=True)
    gate_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    evaluated_at: int = eqx.field(static=True)
    released: bool = eqx.field(static=True)
    maturity: TensorNetworkMaturity = eqx.field(static=True)
    failure: TensorNetworkFailure = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)

    def __init__(
        self,
        qualification_result_id: str,
        gate_evidence_ids: Sequence[str],
        evaluated_at: int,
        released: bool,
        maturity: TensorNetworkMaturity,
        failure: TensorNetworkFailure,
        reasons: Sequence[str],
        /,
    ):
        result_id = _identifier(qualification_result_id, "qualification_result_id")
        evidence_ids = tuple(
            _identifier(value, "gate evidence ID") for value in gate_evidence_ids
        )
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Release gate evidence IDs must be unique.")
        timestamp = _nonnegative_integer(evaluated_at, "evaluated_at")
        released_ = bool(released)
        maturity_ = TensorNetworkMaturity(maturity)
        failure_ = TensorNetworkFailure(failure)
        reasons_ = tuple(_identifier(value, "release reason") for value in reasons)
        if released_:
            if (
                maturity_ != TensorNetworkMaturity.RELEASED
                or failure_ != TensorNetworkFailure.NONE
                or reasons_
            ):
                raise ValueError("Released decisions cannot contain failures.")
        elif (
            maturity_ == TensorNetworkMaturity.RELEASED
            or failure_ != TensorNetworkFailure.RELEASE_GATE_FAILED
            or not reasons_
        ):
            raise ValueError("Refused release decisions require typed reasons.")
        self.qualification_result_id = result_id
        self.gate_evidence_ids = evidence_ids
        self.evaluated_at = timestamp
        self.released = released_
        self.maturity = maturity_
        self.failure = failure_
        self.reasons = reasons_
        self.decision_id = canonical_fingerprint(
            {
                "kind": "tensor-network-release-decision",
                "qualification_result": result_id,
                "gate_evidence": evidence_ids,
                "evaluated_at": timestamp,
                "released": released_,
                "maturity": maturity_.value,
                "failure": failure_.value,
                "reasons": reasons_,
            }
        )

    def require_released(self) -> str:
        if not self.released:
            raise RuntimeError(f"{self.failure.value}: {'; '.join(self.reasons)}")
        return self.decision_id


def evaluate_tensor_network_release(
    result: TensorNetworkQualificationResult,
    gate_evidence: Sequence[ReleaseGateEvidence],
    /,
    *,
    evaluated_at: int,
) -> TensorNetworkReleaseDecision:
    """Evaluate required gates from computed qualification and time-bounded evidence."""

    if not isinstance(result, TensorNetworkQualificationResult):
        raise TypeError("result must be TensorNetworkQualificationResult.")
    evidence = tuple(gate_evidence)
    if any(not isinstance(value, ReleaseGateEvidence) for value in evidence):
        raise TypeError("gate_evidence must contain ReleaseGateEvidence values.")
    names = tuple(value.gate for value in evidence)
    if len(set(names)) != len(names):
        raise ValueError("Release gate evidence contains duplicate gates.")
    required = result.profile.required_release_gates
    recognized = {gate.value for gate in TensorNetworkReleaseGate}
    if any(name not in recognized for name in names):
        raise ValueError("Release evidence contains an unknown tensor-network gate.")
    required_names = {gate.value for gate in required}
    if any(name not in required_names for name in names):
        raise ValueError("Release evidence contains a gate not required by the profile.")
    timestamp = _nonnegative_integer(evaluated_at, "evaluated_at")
    by_name = {value.gate: value for value in evidence}
    reasons: list[str] = []
    if not result.passed:
        reasons.append("qualification claims did not pass")
    if result.maturity == TensorNetworkMaturity.EXPERIMENTAL:
        reasons.append("qualification maturity remains experimental")
    for gate in required:
        value = by_name.get(gate.value)
        if value is None:
            reasons.append(f"missing gate {gate.value}")
        elif not value.is_current(timestamp):
            reasons.append(f"gate {gate.value} is not current")
        elif not value.accepted:
            reasons.append(f"gate {gate.value} is not accepted")
    released = not reasons
    failure = (
        TensorNetworkFailure.NONE
        if released
        else TensorNetworkFailure.RELEASE_GATE_FAILED
    )
    maturity = TensorNetworkMaturity.RELEASED if released else result.maturity
    ordered_evidence_ids = tuple(
        by_name[gate.value].evidence_id for gate in required if gate.value in by_name
    )
    return TensorNetworkReleaseDecision(
        result.result_id,
        ordered_evidence_ids,
        timestamp,
        released,
        maturity,
        failure,
        tuple(reasons),
    )


__all__ = [
    "TensorNetworkClaimEvidence",
    "TensorNetworkQualificationProfile",
    "TensorNetworkQualificationResult",
    "TensorNetworkReleaseDecision",
    "TensorNetworkReleaseGate",
    "evaluate_tensor_network_release",
]
