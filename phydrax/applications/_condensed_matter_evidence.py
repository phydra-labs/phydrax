#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Derived condensed-matter closure evidence; never an umbrella capability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .._fingerprint import canonical_fingerprint
from ..chemistry._campaigns import periodic_chemistry_qualification_campaigns
from ..chemistry.periodic._embedding_qualification import (
    green_embedding_candidate_campaigns,
    green_embedding_candidate_profiles,
)
from ..chemistry.periodic._lattice_frontier import (
    candidate_lattice_campaigns,
    candidate_lattice_profiles,
)
from ..chemistry.periodic._lattice_qualification import (
    lattice_material_candidate_campaigns,
    lattice_material_candidate_profiles,
)
from ..chemistry.periodic._qualification import periodic_candidate_profiles
from ..chemistry.periodic._transport_campaigns import (
    periodic_transport_candidate_campaigns,
)
from ..chemistry.spectroscopy._qualification import (
    material_spectroscopy_candidate_campaigns,
    material_spectroscopy_candidate_profiles,
)
from ..operators.quantum.lattice._qualification import (
    quantum_lattice_candidate_profiles,
)
from ..qualification import (
    CampaignRole,
    CapabilityProfile,
    ReleaseIndex,
    ReleaseTrustPolicy,
    require_profile,
    ScientificCampaign,
    ScientificCase,
    SupportDependency,
)
from ._soft_matter_qualification import (
    soft_matter_candidate_campaigns,
    soft_matter_candidate_profiles,
)
from .diagrammatic_field._qualification import (
    LATTICE_PARQUET_CANDIDATE,
    LOW_ORDER_DIAGRAM_MC_CANDIDATE,
    SIGN_FREE_CTINT_CANDIDATE,
)
from .functional_rg._qualification import FERMION_PATCH_FRG_CANDIDATE
from .magnetic_resonance._qualification import (
    magnetic_resonance_candidate_campaigns,
    magnetic_resonance_candidate_profiles,
)
from .magnetism._qualification import magnetism_candidate_profiles
from .nonequilibrium_field._qualification import FERMIONIC_SECOND_BORN_CANDIDATE
from .semiconductor._production_qualification import (
    semiconductor_candidate_profiles,
    semiconductor_detector_campaign,
    semiconductor_quantum_transport_campaign,
)
from .sign_problem._qualification import CONTROLLED_SIGN_STUDY_CANDIDATE
from .superconductivity._qualification import (
    superconductivity_candidate_campaigns,
    superconductivity_candidate_profiles,
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


@dataclass(frozen=True, slots=True)
class CondensedMatterClosureLedger:
    """Exact profile/tuple dependencies whose conjunction defines closure.

    The ledger has no SupportTuple and no release state. Only the supplied global
    ReleaseIndex can establish that every dependency is released and admissible.
    """

    dependencies: tuple[SupportDependency, ...]
    ledger_id: str = field(init=False)

    def __post_init__(self) -> None:
        dependencies = tuple(
            sorted(
                self.dependencies,
                key=lambda item: (
                    item.profile_id,
                    item.support_tuple_id,
                    item.dependency_id,
                ),
            )
        )
        if not dependencies or any(
            not isinstance(item, SupportDependency) for item in dependencies
        ):
            raise TypeError("Closure dependencies must be typed and non-empty.")
        keys = tuple(item.dependency_id for item in dependencies)
        if len(set(keys)) != len(keys):
            raise ValueError("Closure dependencies must be unique.")
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(
            self,
            "ledger_id",
            canonical_fingerprint(
                {
                    "kind": "derived-condensed-matter-closure-ledger",
                    "dependencies": [item.to_record() for item in dependencies],
                    "release_claim": False,
                }
            ),
        )

    @classmethod
    def from_profiles(
        cls, profiles: Sequence[CapabilityProfile], /
    ) -> "CondensedMatterClosureLedger":
        if not isinstance(profiles, Sequence) or isinstance(profiles, str):
            raise TypeError("profiles must be a sequence of CapabilityProfile values.")
        values = tuple(profiles)
        if not values or any(not isinstance(item, CapabilityProfile) for item in values):
            raise TypeError("profiles must contain typed, non-empty profiles.")
        profile_ids = tuple(item.profile_id for item in values)
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("Closure profiles must be unique.")
        return cls(
            tuple(
                SupportDependency(profile.profile_id, support.support_tuple_id)
                for profile in values
                for support in profile.support_tuples
            )
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "derived-condensed-matter-closure-ledger",
            "dependencies": [item.to_record() for item in self.dependencies],
            "release_claim": False,
            "ledger_id": self.ledger_id,
        }

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> "CondensedMatterClosureLedger":
        if not isinstance(record, Mapping) or set(record) != {
            "kind",
            "dependencies",
            "release_claim",
            "ledger_id",
        }:
            raise ValueError("Condensed-matter closure ledger record is malformed.")
        dependencies = record["dependencies"]
        if (
            record["kind"] != "derived-condensed-matter-closure-ledger"
            or record["release_claim"] is not False
            or not isinstance(dependencies, Sequence)
            or isinstance(dependencies, str)
        ):
            raise ValueError("Condensed-matter closure ledger record is invalid.")
        value = cls(tuple(SupportDependency.from_record(item) for item in dependencies))
        if value.ledger_id != record["ledger_id"]:
            raise ValueError(
                "Condensed-matter closure ledger content address is invalid."
            )
        return value


def condensed_matter_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Collect exact owner-local candidates without changing their identities."""

    profiles = (
        *periodic_candidate_profiles(),
        *lattice_material_candidate_profiles(),
        *green_embedding_candidate_profiles(),
        *quantum_lattice_candidate_profiles(),
        *material_spectroscopy_candidate_profiles(),
        *magnetism_candidate_profiles(),
        *superconductivity_candidate_profiles(),
        *magnetic_resonance_candidate_profiles(),
        *semiconductor_candidate_profiles(),
        *soft_matter_candidate_profiles(),
    )
    return tuple(sorted(profiles, key=lambda profile: profile.profile_id))


def condensed_matter_frontier_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Inventory implemented frontier candidates outside the closure ledger."""

    profiles = (
        *candidate_lattice_profiles(),
        FERMION_PATCH_FRG_CANDIDATE,
        LATTICE_PARQUET_CANDIDATE,
        SIGN_FREE_CTINT_CANDIDATE,
        LOW_ORDER_DIAGRAM_MC_CANDIDATE,
        FERMIONIC_SECOND_BORN_CANDIDATE,
        CONTROLLED_SIGN_STUDY_CANDIDATE,
    )
    return tuple(sorted(profiles, key=lambda profile: profile.profile_id))


def condensed_matter_candidate_closure() -> CondensedMatterClosureLedger:
    """Return the evidence-free candidate coverage ledger, not a release claim."""

    return CondensedMatterClosureLedger.from_profiles(
        condensed_matter_candidate_profiles()
    )


def require_condensed_matter_closure(
    index: ReleaseIndex,
    ledger: CondensedMatterClosureLedger,
    trust_policy: ReleaseTrustPolicy,
    /,
    *,
    at_time: int,
) -> tuple[CapabilityProfile, ...]:
    """Require every exact dependency through the global release authority."""

    if not isinstance(index, ReleaseIndex):
        raise TypeError("index must be ReleaseIndex.")
    if not isinstance(ledger, CondensedMatterClosureLedger):
        raise TypeError("ledger must be CondensedMatterClosureLedger.")
    index.require_trusted(trust_policy, at_time)
    profiles = {profile.profile_id: profile for profile in index.profiles}
    admitted = []
    for dependency in ledger.dependencies:
        if dependency.profile_id not in profiles:
            raise ValueError(
                f"Condensed-matter closure is missing dependency {dependency.profile_id}."
            )
        profile = profiles[dependency.profile_id]
        supports = tuple(
            support
            for support in profile.support_tuples
            if support.support_tuple_id == dependency.support_tuple_id
        )
        if len(supports) != 1:
            raise ValueError(
                "Condensed-matter closure dependency does not match exactly one "
                f"support tuple: {dependency.dependency_id}."
            )
        admitted.append(
            require_profile(
                index,
                dependency.profile_id,
                supports[0],
                trust_policy,
                at_time=at_time,
            )
        )
    return tuple(admitted)


@dataclass(frozen=True, slots=True)
class CondensedMatterCampaignReference:
    """One owner and its unchanged leakage-controlled campaign."""

    owner_id: str
    campaign: ScientificCampaign

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _identifier(self.owner_id, "owner_id"))
        if not isinstance(self.campaign, ScientificCampaign):
            raise TypeError("campaign must be ScientificCampaign.")

    def to_record(self) -> dict[str, object]:
        return {
            "owner_id": self.owner_id,
            "campaign": self.campaign.to_record(),
        }

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> "CondensedMatterCampaignReference":
        if not isinstance(record, Mapping) or set(record) != {"owner_id", "campaign"}:
            raise ValueError("Condensed-matter campaign reference is malformed.")
        return cls(
            str(record["owner_id"]),
            ScientificCampaign.from_record(record["campaign"]),
        )


@dataclass(frozen=True, slots=True)
class CondensedMatterCampaignAggregation:
    """Cross-owner campaign inventory that keeps every campaign independent."""

    campaigns: tuple[CondensedMatterCampaignReference, ...]
    aggregation_id: str = field(init=False)

    def __post_init__(self) -> None:
        campaigns = tuple(
            sorted(
                self.campaigns,
                key=lambda item: (item.owner_id, item.campaign.campaign_id),
            )
        )
        if not campaigns or any(
            not isinstance(item, CondensedMatterCampaignReference) for item in campaigns
        ):
            raise TypeError("campaigns must contain typed, non-empty references.")
        campaign_ids = tuple(item.campaign.campaign_id for item in campaigns)
        if len(set(campaign_ids)) != len(campaign_ids):
            raise ValueError("Owner campaign references must be unique.")
        roles_by_coordinate: dict[tuple[str, str], str] = {}
        case_ids: set[str] = set()
        for reference in campaigns:
            role_by_case = {
                case_id: role.name
                for role in reference.campaign.roles
                for case_id in role.case_ids
            }
            for case in reference.campaign.cases:
                if case.case_id in case_ids:
                    raise ValueError("Owner campaigns must use globally unique case IDs.")
                case_ids.add(case.case_id)
                role = role_by_case[case.case_id]
                for name, value in (
                    ("independent_unit_id", case.independent_unit_id),
                    ("preparation_id", case.preparation_id),
                    ("batch_id", case.batch_id),
                ):
                    key = (name, value)
                    previous = roles_by_coordinate.get(key)
                    if previous is not None and previous != role:
                        raise ValueError(
                            f"Campaign coordinate {name}={value!r} crosses roles."
                        )
                    roles_by_coordinate[key] = role
        object.__setattr__(self, "campaigns", campaigns)
        object.__setattr__(
            self,
            "aggregation_id",
            canonical_fingerprint(
                {
                    "kind": "condensed-matter-campaign-aggregation",
                    "campaigns": [
                        {
                            "owner_id": item.owner_id,
                            "campaign_id": item.campaign.campaign_id,
                        }
                        for item in campaigns
                    ],
                }
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "condensed-matter-campaign-aggregation",
            "campaigns": [item.to_record() for item in self.campaigns],
            "aggregation_id": self.aggregation_id,
        }

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> "CondensedMatterCampaignAggregation":
        if not isinstance(record, Mapping) or set(record) != {
            "kind",
            "campaigns",
            "aggregation_id",
        }:
            raise ValueError("Condensed-matter campaign aggregation is malformed.")
        campaigns = record["campaigns"]
        if (
            record["kind"] != "condensed-matter-campaign-aggregation"
            or not isinstance(campaigns, Sequence)
            or isinstance(campaigns, str)
        ):
            raise ValueError("Condensed-matter campaign aggregation is invalid.")
        value = cls(
            tuple(
                CondensedMatterCampaignReference.from_record(item) for item in campaigns
            )
        )
        if value.aggregation_id != record["aggregation_id"]:
            raise ValueError(
                "Condensed-matter campaign aggregation content address is invalid."
            )
        return value


def _frontier_campaign(profile: CapabilityProfile, /) -> ScientificCampaign:
    support = profile.support_tuples[0]
    slug = support.capability.replace(".", "-")
    calibration = ScientificCase(
        f"{slug}-calibration",
        f"{slug}-calibration-unit",
        support.capability,
        "bounded-analytic-control",
        f"{slug}-calibration-preparation",
        f"{slug}-calibration-batch",
        (f"source:{slug}:analytic",),
    )
    locked = ScientificCase(
        f"{slug}-locked",
        f"{slug}-locked-unit",
        support.capability,
        "independent-locked-control",
        f"{slug}-locked-preparation",
        f"{slug}-locked-batch",
        (f"source:{slug}:independent",),
    )
    return ScientificCampaign(
        (calibration, locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", (locked.case_id,)),
        ),
        criteria_ids=profile.required_gates,
    )


def condensed_matter_candidate_campaigns() -> CondensedMatterCampaignAggregation:
    """Reference owner campaigns while retaining all original role boundaries."""

    references = [
        CondensedMatterCampaignReference("chemistry.periodic", campaign)
        for campaign in periodic_chemistry_qualification_campaigns()
    ]
    references.extend(
        CondensedMatterCampaignReference("chemistry.periodic.transport", campaign)
        for campaign in periodic_transport_candidate_campaigns()
    )
    for owner_id, campaigns in (
        ("chemistry.periodic.lattice", lattice_material_candidate_campaigns()),
        ("chemistry.periodic.embedding", green_embedding_candidate_campaigns()),
        ("chemistry.spectroscopy", material_spectroscopy_candidate_campaigns()),
        ("applications.magnetic-resonance", magnetic_resonance_candidate_campaigns()),
        ("applications.soft-matter", soft_matter_candidate_campaigns()),
        ("applications.superconductivity", superconductivity_candidate_campaigns()),
    ):
        references.extend(
            CondensedMatterCampaignReference(owner_id, campaign) for campaign in campaigns
        )
    for owner_id, profiles in (
        ("operators.quantum.lattice", quantum_lattice_candidate_profiles()),
        ("applications.magnetism", magnetism_candidate_profiles()),
    ):
        references.extend(
            CondensedMatterCampaignReference(owner_id, _frontier_campaign(profile))
            for profile in profiles
        )
    references.extend(
        (
            CondensedMatterCampaignReference(
                "applications.semiconductor.detector",
                semiconductor_detector_campaign(),
            ),
            CondensedMatterCampaignReference(
                "applications.semiconductor.quantum",
                semiconductor_quantum_transport_campaign(),
            ),
        )
    )
    return CondensedMatterCampaignAggregation(tuple(references))


def condensed_matter_frontier_candidate_campaigns() -> CondensedMatterCampaignAggregation:
    """Return frontier campaign inventory without entering closure aggregation."""

    references = [
        CondensedMatterCampaignReference("chemistry.periodic.lattice-frontier", campaign)
        for campaign in candidate_lattice_campaigns()
    ]
    for owner_id, profile in (
        ("applications.functional-rg", FERMION_PATCH_FRG_CANDIDATE),
        ("applications.diagrammatic-field", LATTICE_PARQUET_CANDIDATE),
        ("applications.diagrammatic-field", SIGN_FREE_CTINT_CANDIDATE),
        ("applications.diagrammatic-field", LOW_ORDER_DIAGRAM_MC_CANDIDATE),
        ("applications.nonequilibrium-field", FERMIONIC_SECOND_BORN_CANDIDATE),
        ("applications.sign-problem", CONTROLLED_SIGN_STUDY_CANDIDATE),
    ):
        references.append(
            CondensedMatterCampaignReference(owner_id, _frontier_campaign(profile))
        )
    return CondensedMatterCampaignAggregation(tuple(references))


__all__ = [
    "CondensedMatterCampaignAggregation",
    "CondensedMatterCampaignReference",
    "CondensedMatterClosureLedger",
    "condensed_matter_candidate_campaigns",
    "condensed_matter_candidate_closure",
    "condensed_matter_candidate_profiles",
    "condensed_matter_frontier_candidate_campaigns",
    "condensed_matter_frontier_candidate_profiles",
    "require_condensed_matter_closure",
]
