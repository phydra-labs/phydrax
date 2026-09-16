#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import (
    CapabilityProfile,
    ReleaseGateEvidence,
    ScientificClaimProfile,
    ScientificMetricCriterion,
    SupportDependency,
    SupportTuple,
)


HEP_RELEASE_GATES = (
    "contract",
    "source",
    "numerical",
    "interchange",
    "physics",
    "uncertainty",
    "derivative",
    "resource",
    "preservation",
)


class HEPQualificationBundle(StrictModule, NonTrainableState):
    support: SupportTuple
    capability: CapabilityProfile
    scientific_claim: ScientificClaimProfile
    bundle_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: SupportTuple,
        capability: CapabilityProfile,
        scientific_claim: ScientificClaimProfile,
        /,
    ):
        if (
            not isinstance(support, SupportTuple)
            or not isinstance(capability, CapabilityProfile)
            or not isinstance(scientific_claim, ScientificClaimProfile)
        ):
            raise TypeError(
                "HEP qualification bundle requires typed support, capability, and claim values."
            )
        if (
            support.support_tuple_id
            not in {value.support_tuple_id for value in capability.support_tuples}
            or scientific_claim.support.support_tuple_id != support.support_tuple_id
        ):
            raise ValueError(
                "HEP capability and scientific claim must share exact support."
            )
        self.support = support
        self.capability = capability
        self.scientific_claim = scientific_claim
        self.bundle_id = canonical_fingerprint(
            {
                "kind": "hep-qualification-bundle",
                "support": support.support_tuple_id,
                "capability": capability.profile_id,
                "claim": scientific_claim.claim_id,
            }
        )


def build_hep_qualification_bundle(
    *,
    capability_name: str,
    support_attributes: Mapping[str, str | int | bool],
    profile_name: str,
    provider: str,
    provider_version: str,
    release_evidence: Sequence[ReleaseGateEvidence],
    dependencies: Sequence[str | SupportDependency] = (),
    released: bool,
    campaign_id: str,
    observable_ids: Sequence[str],
    condition_domain_ids: Sequence[str],
    required_stage_ids: Sequence[str],
    criteria: Sequence[ScientificMetricCriterion],
    frozen_criteria_ids: Sequence[str],
    abstention_policy_id: str,
    invalidation_triggers: Sequence[str],
    required_release_gates: Sequence[str] = HEP_RELEASE_GATES,
) -> HEPQualificationBundle:
    """Build one exact governed HEP capability and scientific-claim pair."""
    support = SupportTuple(capability_name, support_attributes)
    gates = tuple(str(value).strip() for value in required_release_gates)
    if (
        not gates
        or any(not value for value in gates)
        or len(set(gates)) != len(gates)
        or not set(gates) <= set(HEP_RELEASE_GATES)
    ):
        raise ValueError(
            "HEP release gates must be a distinct non-empty subset of the governed gate vocabulary."
        )
    capability = CapabilityProfile(
        profile_name,
        provider,
        provider_version,
        (support,),
        dependencies=dependencies,
        required_gates=gates,
        release_evidence=release_evidence,
        released=released,
    )
    claim = ScientificClaimProfile(
        capability_name,
        support,
        observable_ids,
        condition_domain_ids,
        campaign_id,
        required_stage_ids,
        criteria,
        abstention_policy_id,
        invalidation_triggers,
        frozen_criteria_ids=frozen_criteria_ids,
    )
    return HEPQualificationBundle(support, capability, claim)


__all__ = [
    "HEPQualificationBundle",
    "HEP_RELEASE_GATES",
    "build_hep_qualification_bundle",
]
