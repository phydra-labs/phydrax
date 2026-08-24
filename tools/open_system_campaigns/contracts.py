#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._precision import (
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
)
from phydrax._strict import StrictModule
from phydrax.operators.quantum import (
    evaluate_open_system_promotion,
    OpenSystemApproximationEvidence,
    OpenSystemPhysicalityEvidence,
    OpenSystemPromotionDecision,
    OpenSystemPromotionPolicy,
)


PERMANENT_OPEN_SYSTEM_STOP_CLAIMS = (
    "exact-general-interacting-bosonic-dynamics",
    "universal-direct-memory-kernel-complete-positivity",
    "exact-mps-unravelling-without-closure",
    "exact-neural-unravelling-without-closure",
    "infinite-depth-heom-proof",
    "unique-process-recovery-outside-gauge-and-design",
    "physical-arbitrary-mpo-compression",
    "global-steady-state-uniqueness-from-finite-window",
)


class SemanticReplayEvidence(StrictModule):
    variates_equal: Array
    address_schema_equal: Array
    event_time_difference: Array
    channel_disagreement_probability: Array
    observable_difference: Array
    event_time_tolerance: Array
    disagreement_tolerance: Array
    observable_tolerance: Array
    valid: Array

    def __init__(
        self,
        /,
        *,
        variates_equal: ArrayLike,
        address_schema_equal: ArrayLike,
        event_time_difference: ArrayLike,
        channel_disagreement_probability: ArrayLike,
        observable_difference: ArrayLike,
        event_time_tolerance: float,
        disagreement_tolerance: float,
        observable_tolerance: float,
    ):
        self.variates_equal = jnp.asarray(variates_equal, dtype=bool)
        self.address_schema_equal = jnp.asarray(address_schema_equal, dtype=bool)
        self.event_time_difference = jnp.asarray(event_time_difference)
        self.channel_disagreement_probability = jnp.asarray(
            channel_disagreement_probability
        )
        self.observable_difference = jnp.asarray(observable_difference)
        self.event_time_tolerance = jnp.asarray(event_time_tolerance)
        self.disagreement_tolerance = jnp.asarray(disagreement_tolerance)
        self.observable_tolerance = jnp.asarray(observable_tolerance)
        values = (
            self.event_time_difference,
            self.channel_disagreement_probability,
            self.observable_difference,
            self.event_time_tolerance,
            self.disagreement_tolerance,
            self.observable_tolerance,
        )
        if (
            self.variates_equal.shape != ()
            or self.address_schema_equal.shape != ()
            or any(
                value.shape != ()
                or not bool(jnp.isfinite(value))
                or bool(value < 0.0)
                for value in values
            )
        ):
            raise ValueError(
                "Semantic replay values and tolerances must be finite "
                "non-negative scalars."
            )
        self.valid = (
            self.variates_equal
            & self.address_schema_equal
            & (self.event_time_difference <= self.event_time_tolerance)
            & (
                self.channel_disagreement_probability
                <= self.disagreement_tolerance
            )
            & (self.observable_difference <= self.observable_tolerance)
        )


class CampaignPrecisionBundle(StrictModule):
    request: PrecisionRequest = eqx.field(static=True)
    resolution: PrecisionResolution = eqx.field(static=True)
    evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)

    def __init__(
        self,
        domain: str,
        provider: str,
        effective: Mapping[str, object],
        /,
        *,
        children: Mapping[str, PrecisionEvidenceEnvelope] | None = None,
    ):
        request = PrecisionRequest(domain, effective)
        resolution = PrecisionResolution(request, provider, effective)
        evidence = PrecisionEvidenceEnvelope(
            resolution,
            effective,
            children={} if children is None else children,
        )
        self.request = request
        self.resolution = resolution
        self.evidence = evidence


class CampaignCapacityEvidence(StrictModule):
    name: str = eqx.field(static=True)
    used: int = eqx.field(static=True)
    limit: int = eqx.field(static=True)
    saturated: Array
    valid: Array

    def __init__(
        self,
        name: str,
        used: int,
        limit: int,
        /,
        *,
        saturated: ArrayLike,
    ):
        identifier = str(name)
        used_ = int(used)
        limit_ = int(limit)
        saturated_ = jnp.asarray(saturated, dtype=bool)
        if not identifier:
            raise ValueError("Capacity evidence name must be non-empty.")
        if used_ < 0 or limit_ <= 0 or used_ > limit_:
            raise ValueError("Capacity evidence requires 0 <= used <= limit.")
        if saturated_.shape != ():
            raise ValueError("Capacity saturation must be one scalar Boolean.")
        self.name = identifier
        self.used = used_
        self.limit = limit_
        self.saturated = saturated_
        self.valid = ~saturated_


class OpenSystemCampaignRecord(StrictModule):
    approximation: OpenSystemApproximationEvidence
    physicality: OpenSystemPhysicalityEvidence
    precision: CampaignPrecisionBundle
    replay: SemanticReplayEvidence
    execution_success: Array
    capacity_evidence: tuple[CampaignCapacityEvidence, ...]
    capacity_exhausted: Array
    campaign_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    artifact_names: tuple[str, ...] = eqx.field(static=True)
    artifact_arrays: tuple[Array, ...]
    work: tuple[tuple[str, int], ...] = eqx.field(static=True)
    unsupported_claims: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        campaign_id: str,
        representation_id: str,
        approximation: OpenSystemApproximationEvidence,
        physicality: OpenSystemPhysicalityEvidence,
        precision: CampaignPrecisionBundle,
        replay: SemanticReplayEvidence,
        /,
        *,
        execution_success: ArrayLike,
        capacity_evidence: Sequence[CampaignCapacityEvidence],
        artifact_arrays: Mapping[str, ArrayLike],
        work: Mapping[str, int],
        unsupported_claims: Sequence[str] = (),
    ):
        campaign = str(campaign_id)
        representation = str(representation_id)
        execution = jnp.asarray(execution_success, dtype=bool)
        capacities = tuple(capacity_evidence)
        if not campaign or not representation:
            raise ValueError("Campaign and representation IDs must be non-empty.")
        if not isinstance(approximation, OpenSystemApproximationEvidence):
            raise TypeError("approximation must be OpenSystemApproximationEvidence.")
        if not isinstance(physicality, OpenSystemPhysicalityEvidence):
            raise TypeError("physicality must be OpenSystemPhysicalityEvidence.")
        if not isinstance(precision, CampaignPrecisionBundle):
            raise TypeError("precision must be CampaignPrecisionBundle.")
        if not isinstance(replay, SemanticReplayEvidence):
            raise TypeError("replay must be SemanticReplayEvidence.")
        if execution.shape != ():
            raise ValueError("Campaign execution gate must be scalar.")
        if not capacities or any(
            not isinstance(value, CampaignCapacityEvidence) for value in capacities
        ):
            raise ValueError("Campaigns require explicit capacity evidence.")
        capacity_names = tuple(value.name for value in capacities)
        if len(set(capacity_names)) != len(capacity_names):
            raise ValueError("Campaign capacity names must be unique.")
        arrays = tuple(
            (str(name), jnp.asarray(value))
            for name, value in sorted(artifact_arrays.items())
        )
        if not arrays or any(not name for name, _ in arrays):
            raise ValueError("Campaign record requires named solver/reference arrays.")
        work_ = tuple((str(name), int(value)) for name, value in sorted(work.items()))
        if not work_ or any(not name or value < 0 for name, value in work_):
            raise ValueError("Campaign work must contain non-negative named counters.")
        unsupported = tuple(str(value) for value in unsupported_claims)
        if any(not value for value in unsupported) or len(set(unsupported)) != len(
            unsupported
        ):
            raise ValueError("Unsupported claims must be unique and non-empty.")
        self.campaign_id = campaign
        self.representation_id = representation
        self.approximation = approximation
        self.physicality = physicality
        self.precision = precision
        self.replay = replay
        self.execution_success = execution
        self.capacity_evidence = capacities
        self.capacity_exhausted = jnp.any(
            jnp.stack([value.saturated for value in capacities])
        )
        self.unsupported_claims = unsupported
        self.artifact_names = tuple(name for name, _ in arrays)
        self.artifact_arrays = tuple(value for _, value in arrays)
        self.work = work_

    def evaluate(
        self,
        policy: OpenSystemPromotionPolicy,
        /,
        *,
        archive_verified: ArrayLike,
    ) -> OpenSystemPromotionDecision:
        return evaluate_open_system_promotion(
            policy,
            self.approximation,
            self.physicality,
            execution_success=self.execution_success & self.replay.valid,
            capacity_exhausted=self.capacity_exhausted,
            archive_verified=archive_verified,
        )


class VerifiedOpenSystemCampaign(StrictModule):
    record: OpenSystemCampaignRecord
    artifact_sha256: str = eqx.field(static=True)
    reproduction_verified: Array
    valid: Array

    def __init__(
        self,
        record: OpenSystemCampaignRecord,
        artifact_sha256: str,
        /,
        *,
        reproduction_verified: ArrayLike,
    ):
        digest = str(artifact_sha256)
        reproduced = jnp.asarray(reproduction_verified, dtype=bool)
        if not isinstance(record, OpenSystemCampaignRecord):
            raise TypeError("record must be an OpenSystemCampaignRecord.")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("artifact_sha256 must be one lowercase SHA-256 digest.")
        if reproduced.shape != ():
            raise ValueError("reproduction_verified must be one scalar Boolean.")
        self.record = record
        self.artifact_sha256 = digest
        self.reproduction_verified = reproduced
        self.valid = reproduced

    def evaluate(
        self, policy: OpenSystemPromotionPolicy, /
    ) -> OpenSystemPromotionDecision:
        return self.record.evaluate(policy, archive_verified=self.valid)


class OpenSystemGraduationResult(StrictModule):
    promoted: Array
    decisions: tuple[OpenSystemPromotionDecision, ...]
    campaign_ids: tuple[str, ...] = eqx.field(static=True)
    stop_claims: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        campaigns: Sequence[VerifiedOpenSystemCampaign],
        policies: Sequence[OpenSystemPromotionPolicy],
        /,
    ):
        campaigns_ = tuple(campaigns)
        policies_ = tuple(policies)
        if len(campaigns_) != len(policies_) or not campaigns_:
            raise ValueError("Graduation requires one policy per verified campaign.")
        campaign_ids = tuple(value.record.campaign_id for value in campaigns_)
        if len(set(campaign_ids)) != len(campaign_ids):
            raise ValueError("Graduation campaign IDs must be unique.")
        decisions = tuple(
            campaign.evaluate(policy)
            for campaign, policy in zip(campaigns_, policies_, strict=True)
        )
        self.promoted = jnp.all(jnp.stack([decision.promoted for decision in decisions]))
        self.decisions = decisions
        self.campaign_ids = campaign_ids
        self.stop_claims = PERMANENT_OPEN_SYSTEM_STOP_CLAIMS


__all__ = [
    "CampaignCapacityEvidence",
    "CampaignPrecisionBundle",
    "OpenSystemCampaignRecord",
    "OpenSystemGraduationResult",
    "PERMANENT_OPEN_SYSTEM_STOP_CLAIMS",
    "SemanticReplayEvidence",
    "VerifiedOpenSystemCampaign",
]
